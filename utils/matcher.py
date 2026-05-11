"""
JIB(Job_Insight_Bridge) — 매칭 모듈 (RAG 검색 코어).

구직자 입력 텍스트 ↔ 임베딩 DB 간 코사인 유사도를 계산해 후보 직업을 정렬한다.
다중지표(유사도 / 수요 / 임금) 가중치 채점으로 TOP-N 추천 결과를 확정한다.

핵심 함수:
    1) load_embedding_db()  — 임베딩 파일을 1회 로드, 행렬 캐시
    2) embed_query()        — OpenAI 임베딩으로 사용자 입력 벡터화
    3) find_similar_jobs()  — TOP-K 유사도 후보 추출
    4) filter_top3()        — 가중치 채점으로 최종 TOP-N 확정
    5) search_jobs()        — find_similar_jobs + filter_top3 묶음 (Streamlit용)

모든 외부 설정값(API 키 / 모델 / 가중치 / 임계값)은 config.py 에서 import 한다.
"""

from __future__ import annotations

import re
import sys as _sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 프로젝트 루트를 sys.path 에 추가해 단독 실행 시에도 config import 가 가능하도록 한다.
if str(Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트 파일 로더 — prompts/ 하위 텍스트 파일을 시스템 프롬프트로 사용
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _load_prompt(filename: str) -> str:
    """
    prompts/ 폴더에서 프롬프트 텍스트를 로드한다.

    실행 디렉터리에 무관하게 프로젝트 루트 기준의 prompts/ 를 우선 시도하고,
    그 후 현재 작업 디렉터리 기준의 상대 경로(prompts/<filename>) 도 시도한다.
    어느 쪽에도 없으면 빈 문자열을 반환해 호출 측의 기본 프롬프트로 폴백한다.
    """
    candidates = [
        _PROJECT_ROOT / "prompts" / filename,
        Path("prompts") / filename,
    ]
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 전역 캐시 — 임베딩 DataFrame 과 L2 정규화된 벡터 행렬
# ──────────────────────────────────────────────────────────────────────────────
_emb_df: Optional[pd.DataFrame] = None
_emb_matrix: Optional[np.ndarray] = None

# OpenAI text-embedding-3-small 의 차원수 (쿼리 임베딩 실패 시 zero 반환용).
_DEFAULT_EMBED_DIM: int = 1536


# ──────────────────────────────────────────────────────────────────────────────
# 기능 1) 임베딩 DB 로드 + 행렬 캐시
# ──────────────────────────────────────────────────────────────────────────────
def load_embedding_db() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    임베딩 파일을 로드하고 'embedding_vector' 컬럼에서 L2 정규화된 numpy 행렬을 만든다.

    한 번 로드된 DataFrame/행렬은 모듈 전역 변수에 캐시되어 두 번째 호출부터는
    파일 I/O 없이 즉시 반환된다.

    Returns:
        (emb_df, emb_matrix)
            emb_df    : 임베딩 + 메타데이터 DataFrame (embedder.load_embeddings 결과)
            emb_matrix: shape=(N, dim) L2 정규화된 float32 행렬
    """
    global _emb_df, _emb_matrix
    if _emb_df is not None and _emb_matrix is not None:
        return _emb_df, _emb_matrix

    from utils.embedder import load_embeddings  # 지연 import (순환참조 방지)

    df = load_embeddings()
    if df is None or df.empty or "embedding_vector" not in df.columns:
        print("⚠️ 임베딩 DB 가 비어 있거나 'embedding_vector' 컬럼이 없습니다.")
        _emb_df = pd.DataFrame()
        _emb_matrix = np.zeros((0, _DEFAULT_EMBED_DIM), dtype=np.float32)
        return _emb_df, _emb_matrix

    # 유효 벡터만 필터링한 뒤 원본 DF 와 정렬을 일치시킨다.
    valid_mask = df["embedding_vector"].apply(
        lambda v: isinstance(v, np.ndarray) and v.size > 0
    )
    df_valid = df[valid_mask].reset_index(drop=True)
    if df_valid.empty:
        print("⚠️ 유효한 embedding_vector 가 없습니다.")
        _emb_df = pd.DataFrame()
        _emb_matrix = np.zeros((0, _DEFAULT_EMBED_DIM), dtype=np.float32)
        return _emb_df, _emb_matrix

    matrix = np.array(df_valid["embedding_vector"].tolist(), dtype=np.float32)

    # 코사인 유사도를 내적 한 번으로 계산하기 위한 L2 정규화.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    _emb_df = df_valid
    _emb_matrix = matrix
    print(f"✅ 임베딩 DB 로드: {len(_emb_df)}건, 벡터 차원: {_emb_matrix.shape[1]}")
    return _emb_df, _emb_matrix


def reset_embedding_cache() -> None:
    """임베딩 DB 캐시를 비운다 (테스트나 재로드 용도)."""
    global _emb_df, _emb_matrix
    _emb_df = None
    _emb_matrix = None


# ──────────────────────────────────────────────────────────────────────────────
# 기능 2) 사용자 쿼리 임베딩
# ──────────────────────────────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    """
    구직자 입력 텍스트를 OpenAI 임베딩으로 벡터화하고 L2 정규화한다.
    실패 시 0 벡터를 반환해 검색이 의미 없는 결과(모두 0 유사도)를 내도록 한다.
    """
    safe_text = (text or "").strip()
    if not safe_text:
        print("⚠️ 빈 쿼리 — 0 벡터 반환")
        return np.zeros(_DEFAULT_EMBED_DIM, dtype=np.float32)

    api_key = (getattr(config, "OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        print("⚠️ OPENAI_API_KEY 미설정 — 0 벡터 반환")
        return np.zeros(_DEFAULT_EMBED_DIM, dtype=np.float32)

    try:
        from openai import OpenAI  # 지연 import: 모듈 import 시 키 검증 회피
    except ImportError:
        print("⚠️ openai 패키지 미설치 — pip install openai")
        return np.zeros(_DEFAULT_EMBED_DIM, dtype=np.float32)

    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[safe_text],
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        print(f"⚠️ 쿼리 임베딩 실패: {e}")
        return np.zeros(_DEFAULT_EMBED_DIM, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 역량 키워드 추출 (UI 표시·진단용 — 매칭 점수에는 사용하지 않음)
#   디스패처: 콜론 패턴이 있으면 정규식 (Fast Path),
#             자유서술이면 Claude LLM 추출 (Smart Path) → 실패 시 정규식 폴백.
# ──────────────────────────────────────────────────────────────────────────────
_CERT_INDICATORS = (
    "급", "기사", "산업기사", "기능사", "면허", "사무관", "사례관리사",
    "ADsP", "SQLD", "SQLP", "ADP",
    "컴퓨터활용능력", "워드프로세서", "정보처리", "정보보안",
    "GTQ", "TOEIC", "TOEFL", "TEPS", "OPIc", "HSK", "JLPT",
    "AICPA", "CPA", "CFA", "PMP", "자격", "인증",
)

# 의미 없는 단일어·연결어 — 정규식 추출 후 토큰 단계에서 제외.
_INVALID_TOKENS = frozenset({
    "졸업", "재학", "수료", "있음", "없음", "준비", "관심",
    "있어", "없어", "입니다", "합니다", "하고", "있고", "없고",
    "됨", "있는", "없는", "많음", "좋음",
})

# 콜론 패턴 감지: '전공/학력:', '기술:', '역량:', '자격:', '강점:', '경험:',
# '희망:', '관심:', '고민:', '상담목적:' 등 (Fast Path 라우팅용).
_COLON_PATTERN_RE = re.compile(
    r"(?:전공|학력|기술|역량|자격|강점|장점|성향|경험|"
    r"희망|관심|선호|고민|상담)[/\s]*[:\uff1a]"
)


def _empty_skills() -> Dict[str, List[str]]:
    """5개 카테고리가 빈 리스트로 초기화된 dict 를 반환한다."""
    return {
        "학력": [],
        "자격증": [],
        "기술도구": [],
        "강점성향": [],
        "희망방향": [],
    }


def _clean_tokens(text_segment: str) -> List[str]:
    """세그먼트에서 의미 있는 토큰만 추출 (오탐 필터 + 최소 길이 검사)."""
    out: List[str] = []
    for token in re.split(r"[,/、·∙]|\s{2,}", text_segment):
        token = token.strip(" .;\u00b7")
        if not token or token in _INVALID_TOKENS:
            continue
        if len(token) < 2:
            continue
        out.append(token)
    return out


def extract_user_skills(user_input: str) -> Dict[str, List[str]]:
    """사용자 입력에서 카테고리별 역량 키워드를 추출한다.

    - 콜론(:) 패턴이 있으면 정규식 기반 추출 (Fast Path)
    - 콜론이 없으면 Claude API 로 LLM 추출 (Smart Path)
    - LLM 실패 시 정규식으로 폴백.

    UI 표시 및 진단용으로 사용되며, 검색 점수 계산에는 영향을 주지 않는다.

    Returns:
        dict: {'학력': [...], '자격증': [...], '기술도구': [...],
               '강점성향': [...], '희망방향': [...]}
    """
    if not user_input or not str(user_input).strip():
        return _empty_skills()

    has_colon_pattern = bool(_COLON_PATTERN_RE.search(user_input))
    if has_colon_pattern:
        return _extract_skills_by_regex(user_input)
    return _extract_skills_by_llm(user_input)


def _extract_skills_by_regex(user_input: str) -> Dict[str, List[str]]:
    """콜론 패턴이 명확한 입력에 사용하는 정규식 기반 추출.

    오탐 키워드(`_INVALID_TOKENS`) 와 최소 길이 2자 필터를 적용한다.
    """
    skills = _empty_skills()
    text = str(user_input).strip()

    # 학력: '전공/학력: ~', '학력: ~', '전공: ~' (콜론은 ASCII : 또는 전각 ：)
    for pat in (
        r"전공[/\s]*학력[:\uff1a\s]+([^\n.]+)",
        r"학력[:\uff1a\s]+([^\n.]+)",
        r"전공[:\uff1a\s]+([^\n.]+)",
    ):
        m = re.search(pat, text)
        if m:
            skills["학력"].extend(_clean_tokens(m.group(1)))
            break

    # 학력 키워드 직접 매칭 (의미 있는 학위 표시만 보완)
    edu_keywords = (
        "고졸", "전문대졸", "대졸", "대학원졸", "박사졸",
        "학사", "석사", "박사", "4년제", "2년제",
    )
    edu_joined = " ".join(skills["학력"])
    for kw in edu_keywords:
        if kw in text and kw not in edu_joined:
            skills["학력"].append(kw)

    # 자격증/기술 도구: 단일 섹션에서 자격 인디케이터 유무로 분기.
    # '보유 기술', '보유 역량', '자격증', '자격', '기술', '역량' 라벨 모두 지원.
    for pat in (
        r"(?:보유\s*기술|보유\s*역량|자격증|자격)[:\uff1a\s]+([^\n.]+)",
        r"기술[:\uff1a\s]+([^\n.]+)",
        r"역량[:\uff1a\s]+([^\n.]+)",
    ):
        m = re.search(pat, text)
        if m:
            for token in _clean_tokens(m.group(1)):
                if any(ind in token for ind in _CERT_INDICATORS):
                    skills["자격증"].append(token)
                else:
                    skills["기술도구"].append(token)
            break

    # 강점·성향 — '경험' / '주요 경험' 도 의미상 강점·성향에 흡수.
    for pat in (
        r"강점[:\uff1a\s]+([^\n.]+)",
        r"장점[:\uff1a\s]+([^\n.]+)",
        r"성향[:\uff1a\s]+([^\n.]+)",
        r"(?:주요\s*)?경험[:\uff1a\s]+([^\n.]+)",
    ):
        m = re.search(pat, text)
        if m:
            skills["강점성향"].extend(_clean_tokens(m.group(1)))
            break

    # 희망 방향 — 상담사 모드의 '고민', '상담목적', '고민/상담목적' 라벨도 흡수.
    # 슬래시 결합 라벨('고민/상담목적')이 단독 라벨보다 우선 매칭되도록 먼저 배치.
    for pat in (
        r"고민[/\s]*상담\s*목적[:\uff1a\s]+([^\n.]+)",
        r"희망\s*(?:방향|진로|직무)?[:\uff1a\s]+([^\n.]+)",
        r"(?:관심|선호)[:\uff1a\s]+([^\n.]+)",
        r"(?:고민|상담\s*목적|상담목적)[:\uff1a\s]+([^\n.]+)",
    ):
        m = re.search(pat, text)
        if m:
            skills["희망방향"].extend(_clean_tokens(m.group(1)))
            break

    for key in skills:
        skills[key] = list(dict.fromkeys(skills[key]))
    return skills


def _extract_skills_by_llm(user_input: str) -> Dict[str, List[str]]:
    """자유서술 입력을 Claude API 로 카테고리별 분류한다.

    Claude 호출이 실패하면 정규식 추출로 자동 폴백한다.
    """
    import json

    skills = _empty_skills()

    try:
        # 함수 내부 import — 모듈 로딩 시 순환 import 방지
        from utils.claude_generator import _safe_create_message

        prompt = f"""다음 자유서술 입력에서 청년 구직자의 역량 키워드를 5개 카테고리로 분류하세요.

[입력]
{user_input}

[추출 규칙]
- 학력: 전공명, 학위(고졸·대졸 등), 학교 유형(4년제 등)
- 자격증: 공식 자격증명, 면허, 시험명 (ADsP, 컴활1급, TOEIC 등)
- 기술도구: 프로그래밍 언어, 소프트웨어, 도구명 (Python, Excel, Photoshop 등)
- 강점성향: 본인이 강점·성향으로 언급한 것 (외향성, 꼼꼼함, 분석력, 글쓰기 등)
- 희망방향: 희망 직무·산업·근무 형태 (사무직, IT, 안정적, 창의적 등)

[추출 원칙]
- 명확하게 언급된 키워드만 추출 (추측 금지)
- 의미 없는 단어("졸업", "있음", "준비") 제외
- 각 키워드는 명사 형태로 정리
- 카테고리별 0~5개 키워드

[출력 형식 — JSON만 반환, 다른 설명 없음]
{{
  "학력": ["..."],
  "자격증": ["..."],
  "기술도구": ["..."],
  "강점성향": ["..."],
  "희망방향": ["..."]
}}"""

        raw = _safe_create_message(
            max_tokens=500,
            user_prompt=prompt,
            system_text=(
                "당신은 청년 구직자 역량 분류 전문가입니다. JSON만 반환하세요."
            ),
        )

        if not raw or raw.startswith("Claude 호출 실패"):
            raise RuntimeError(raw or "empty response")

        text = raw.strip()
        # 코드 펜스 제거 (```json ... ``` 형태도 안전 처리)
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("응답이 JSON object 가 아닙니다.")

        for key in skills:
            value = parsed.get(key)
            if isinstance(value, list):
                items = [str(x).strip() for x in value if str(x).strip()]
                # 카테고리별 최대 5개, 입력 순서 유지하며 중복 제거
                skills[key] = list(dict.fromkeys(items))[:5]

        total = sum(len(v) for v in skills.values())
        print(f"  🧠 LLM 역량 추출: 총 {total}개")
        return skills

    except Exception as e:
        print(f"  ⚠️ LLM 역량 추출 실패, 정규식으로 폴백: {e}")
        return _extract_skills_by_regex(user_input)


# ──────────────────────────────────────────────────────────────────────────────
# 학력 적합도 헬퍼
#   사용자 입력 텍스트에서 학력 키워드를 추출해 0~6 학력점수로 환산하고,
#   직업 평균 학력점수와의 거리 기반으로 -0.03 ~ +0.05 범위의 가산 점수를 산출.
# ──────────────────────────────────────────────────────────────────────────────
def extract_education_score(text: str) -> Optional[float]:
    """사용자 입력 텍스트에서 학력 정보를 추출해 학력 점수(0~6) 로 환산한다.

    학력 키워드 매핑 (우선순위: 높은 학력일수록 먼저 매칭):
        박사 / phd / ph.d                          → 6
        대학원 / 석사 / master                       → 5
        대졸 / 학사 / 4년제 / 대학교 / 대학 졸업      → 4
        전문대 / 전문학사 / 2년제 / 3년제             → 3
        고졸 / 고등학교 / 고교 졸업 / 직업교육 / 직업훈련 → 2
        중졸 / 중학교                               → 0

    Returns:
        Optional[float]: 학력 점수(0~6). 키워드를 못 찾으면 None.
    """
    if not text or not isinstance(text, str):
        return None

    s = text.lower()

    priority_keywords = [
        (["박사", "phd", "ph.d"], 6),
        (["대학원", "석사", "master"], 5),
        (["대졸", "학사", "4년제", "4년 제", "대학교", "대학 졸업"], 4),
        (["전문대", "전문학사", "2년제", "3년제", "2년 제", "3년 제"], 3),
        (["고졸", "고등학교", "고교 졸업", "직업교육", "직업훈련"], 2),
        (["중졸", "중학교"], 0),
    ]

    for keywords, score in priority_keywords:
        for kw in keywords:
            if kw in s:
                return float(score)

    return None


def compute_education_fit(
    user_edu_score: Optional[float],
    job_edu_score: Optional[float],
) -> float:
    """사용자 학력 점수와 직업 평균 학력 점수의 적합도를 -0.03 ~ +0.05 범위로 산출한다.

    적용 규칙:
        - 차이 ±1 이내           : +0.05 (가산점)
        - 차이 ±2 이내           : +0.02
        - 직업이 사용자보다 2 초과 → -0.03 (진입 어려움)
        - 사용자가 직업보다 2 초과 →  0.00 (오버스펙 허용)
        - 입력 None              :  0.00 (보너스 미적용)
    """
    if user_edu_score is None or job_edu_score is None:
        return 0.0

    # diff > 0: 직업 요구가 더 높음, diff < 0: 사용자가 더 높음
    diff = job_edu_score - user_edu_score

    if abs(diff) <= 1:
        return 0.05
    if abs(diff) <= 2:
        return 0.02
    if diff > 2:
        return -0.03
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3) 유사도 기반 TOP-K 후보 추출
# ──────────────────────────────────────────────────────────────────────────────
def find_similar_jobs(
    query_text: str,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> pd.DataFrame:
    """
    구직자 입력 텍스트와 유사한 직업 후보 TOP-K 를 코사인 유사도 내림차순으로 반환한다.

    Args:
        query_text:    검색 텍스트
        top_k:         반환 후보 수 (None 이면 config.TOP_K_CANDIDATES)
        min_similarity:최소 유사도 임계값 (None 이면 config.MIN_SIMILARITY)

    Returns:
        pandas.DataFrame: 핵심 컬럼 + '유사도' 컬럼이 포함된 결과 (필터 통과분만).
    """
    top_k = top_k or config.TOP_K_CANDIDATES
    min_similarity = min_similarity if min_similarity is not None else config.MIN_SIMILARITY

    emb_df, emb_matrix = load_embedding_db()
    if emb_df.empty or emb_matrix.size == 0:
        return pd.DataFrame()

    query_vec = embed_query(query_text)
    if not np.any(query_vec):  # 0 벡터인 경우 유사도 모두 0
        print("⚠️ 쿼리 벡터가 비어있어 유의미한 검색 결과를 얻을 수 없습니다.")
        return pd.DataFrame()

    # 차원 호환성 검증.
    if query_vec.shape[0] != emb_matrix.shape[1]:
        print(
            f"⚠️ 쿼리 벡터 차원({query_vec.shape[0]}) ≠ DB 벡터 차원({emb_matrix.shape[1]}). "
            "EMBEDDING_MODEL 일치 여부를 확인하세요."
        )
        return pd.DataFrame()

    # 이미 양쪽 모두 L2 정규화 → 내적 = 코사인 유사도.
    similarities = emb_matrix @ query_vec  # shape: (N,)

    k = min(top_k, similarities.shape[0])
    if k == 0:
        return pd.DataFrame()
    top_indices = np.argsort(similarities)[::-1][:k]
    top_scores = similarities[top_indices]

    result = emb_df.iloc[top_indices].copy()
    result["유사도"] = top_scores.round(4)

    # 임계값 필터 제거 — TOP K 전체를 filter_top3 에 전달.
    # (너무 낮은 유사도는 가중치 채점 후 자연스럽게 탈락하며, filter_top3
    #  마지막 단계에서 0.20 최후 방어선으로 한 번 더 차단한다.)
    # 참고: min_similarity 인자는 호환성 유지를 위해 시그니처에는 남겨둔다.

    return_cols = [
        "직업코드", "직업명", "유사도", "대분류명", "중분류명",
        "전망점수", "평균구인배율", "평균부족률",
        "월평균임금_천원", "신입임금_천원",
        "상위25_임금_천원", "중위_임금_천원", "하위25_임금_천원",
        "주요학력수준", "학력_고졸이하비율", "학력_전문대졸비율",
        "학력_대졸이상비율", "학력점수",
        "직업전망_텍스트", "주요전공계열", "유사직업명_합산",
    ]
    exist_cols = [c for c in return_cols if c in result.columns]
    result = result[exist_cols].reset_index(drop=True)

    # 유사도 상위 50%만 filter_top3 에 전달 (중분류 평균 노이즈 제거).
    # TOP_N_RESULTS × 2 보다 작아지지 않도록 하한선을 둔다.
    half_k = max(config.TOP_N_RESULTS * 2, len(result) // 2)
    return result.head(half_k)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 4) 다중지표 가중치 채점 → TOP-N 확정
# ──────────────────────────────────────────────────────────────────────────────
def filter_top3(
    candidates_df: pd.DataFrame,
    user_wage_floor: Optional[float] = None,
    user_edu_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    find_similar_jobs() 후보에서 다중지표 가중치 채점으로 TOP-N 을 확정한다.

    채점:
        score = 유사도   * config.WEIGHT_SIMILARITY
              + 수요점수 * config.WEIGHT_DEMAND
              + 임금점수 * config.WEIGHT_WAGE
              + 학력적합도 (-0.03 ~ +0.05, user_edu_score 미지정 시 0)

    Args:
        candidates_df:  find_similar_jobs() 결과 DataFrame
        user_wage_floor: 사용자가 지정한 임금 하한선(천원). 미지정 시 미적용.
        user_edu_score:  사용자 학력 점수(0~6). compute_education_fit() 로
                         '학력점수' 컬럼과 비교해 -0.03 ~ +0.05 범위의 가산점.

    Returns:
        pandas.DataFrame: 최종 TOP-N + '수요점수/임금점수/학력적합도/최종점수/추천순위'.
    """
    if candidates_df is None or candidates_df.empty:
        return candidates_df if candidates_df is not None else pd.DataFrame()

    df = candidates_df.copy()

    # ── 관리직(임원·부서장) 사전 필터 ────────────────────────────────
    # 청년 구직자 대상 서비스에서 관리직 중분류는 진입 현실성이 없으므로
    # 후보에서 제거. 단 전체 후보가 관리직뿐이거나 비관리직이 TOP-N 미만이면
    # 사용자 경험을 위해 필터를 해제한다.
    if "중분류명" in df.columns:
        mid_series = df["중분류명"].fillna("").astype(str)
        non_manager_mask = ~mid_series.str.contains(
            "관리직|임원|부서장", na=False
        )
        non_manager = df[non_manager_mask]
        if len(non_manager) >= config.TOP_N_RESULTS:
            removed = len(df) - len(non_manager)
            df = non_manager.reset_index(drop=True)
            print(f"  [필터] 관리직 제외: {removed}건 제거 → {len(df)}개 후보")

    # 유사도 최소 기준 — 무관한 직업이 임금/수요 점수만으로 상위권 진입하는 걸 방지.
    # 동적 하한선: max(최고 유사도의 70%, 0.25). 한국어 임베딩 분포가 매번
    # 달라지는 점을 반영해 후보군이 매우 좋을 땐 더 가파르게 컷, 그렇지 않으면
    # 절대 하한 0.25 를 보장. 단, 필터 결과가 TOP_N_RESULTS 미만이면 하한선을
    # 무시하고 유사도 상위 N*2 개를 강제 확보해 TOP-N 결과를 보장한다.
    if "유사도" in df.columns:
        sim_pre = pd.to_numeric(df["유사도"], errors="coerce").fillna(0.0)
        df = df.assign(_sim_num=sim_pre)
        max_sim = float(df["_sim_num"].max()) if not df.empty else 0.0
        dynamic_floor = max(max_sim * 0.70, 0.25)
        before = len(df)
        filtered = df[df["_sim_num"] >= dynamic_floor]
        if len(filtered) < config.TOP_N_RESULTS:
            pool_size = max(config.TOP_N_RESULTS * 2, len(filtered))
            df = df.nlargest(pool_size, "_sim_num")
        else:
            df = filtered
        df = df.drop(columns="_sim_num").reset_index(drop=True)
        print(
            f"  [필터] 유사도 하한선: {dynamic_floor:.3f} "
            f"({before}→{len(df)}개 후보)"
        )
    if df.empty:
        return pd.DataFrame()

    # ── 수요점수: 구인배율(0~1.5 → 0~1) + 부족률(0~30% → 0~1) 평균 ───
    # 구인배율 1.0 초과는 중분류 평균값 노이즈일 가능성이 높아 0.4 캡 처리하여
    # 판금·용접·제조 등 고배율 직종이 유사도가 낮아도 TOP-N 에 진입하는 것을
    # 방지한다. 값 0.0 은 고용24·인력부족률 통계 미집계(공채·헤드헌팅 위주
    # 직종) 케이스라 부당한 하락을 막기 위해 중간값 0.4 를 부여한다.
    def calc_demand_score(row: pd.Series) -> float:
        scores = []
        ratio = row.get("평균구인배율", None)
        shortage = row.get("평균부족률", None)
        if pd.notna(ratio):
            try:
                r = float(ratio)
                if r == 0.0:
                    scores.append(0.4)
                elif r > 1.0:
                    scores.append(0.4)
                elif r > 0:
                    scores.append(r / 1.5)
            except (TypeError, ValueError):
                pass
        if pd.notna(shortage):
            try:
                s = float(shortage)
                if s == 0.0:
                    scores.append(0.4)
                elif s > 0:
                    scores.append(min(s / 30.0, 1.0))
            except (TypeError, ValueError):
                pass
        # 수요 데이터가 없으면 중간값 0.4 부여 (미집계 케이스와 동일 처리).
        return sum(scores) / len(scores) if scores else 0.4

    # ── 임금점수: 월평균임금(0~7000천원 → 0~1) ─────────────────────
    # 7,000천원(월 700만원) 이상은 캡 적용. 관리자·임원급 중분류 평균이
    # 12,000천원대로 과도하게 높아 청년 구직자 현실과 괴리되는 문제를
    # 방지한다 (예: 관리직(임원·부서장) 12,223천원이 임금점수 1.0 을 받아
    # 유사도가 낮아도 TOP-N 에 진입하는 케이스 차단).
    def calc_wage_score(row: pd.Series) -> float:
        wage = row.get("월평균임금_천원", None)
        if pd.notna(wage):
            try:
                w = float(wage)
                if w > 0:
                    w_capped = min(w, 7000.0)
                    return min(w_capped / 7000.0, 1.0)
            except (TypeError, ValueError):
                pass
        # 임금 데이터가 없으면 중간값 0.5 부여.
        # IT·디자인·신흥직종처럼 임금통계 미집계 직종이 구조적으로
        # 불이익 받는 문제를 방지하기 위함.
        return 0.5

    # ── 임금 하한선 필터 (선택적) ──────────────────────────────────
    if user_wage_floor is not None and "월평균임금_천원" in df.columns:
        wage_series = pd.to_numeric(df["월평균임금_천원"], errors="coerce")
        has_wage = wage_series.notna() & (wage_series > 0)
        below_floor = has_wage & (wage_series < float(user_wage_floor))
        df_filtered = df[~below_floor]
        if df_filtered.empty:
            # 필터 결과가 비면 사용자 경험을 위해 원본 후보 유지.
            print(f"  ℹ️ 임금 하한선({user_wage_floor}천원) 적용 시 후보 0건 — 필터 미적용")
        else:
            df = df_filtered

    df["수요점수"] = df.apply(calc_demand_score, axis=1)
    df["임금점수"] = df.apply(calc_wage_score, axis=1)

    # ── 학력 적합도 (사용자 입력 학력 ↔ 직업 평균 학력점수) ──────────
    # 가산 점수 형태로 -0.03 ~ +0.05 범위를 갖는다. user_edu_score 가
    # None 이거나 후보에 '학력점수' 컬럼이 없으면 0.0 으로 미적용.
    if user_edu_score is not None and "학력점수" in df.columns:
        edu_series = pd.to_numeric(df["학력점수"], errors="coerce")
        df["학력적합도"] = edu_series.apply(
            lambda x: compute_education_fit(user_edu_score, float(x))
            if pd.notna(x) else 0.0
        )
        print(f"  🎓 학력 적합도 평균: {df['학력적합도'].mean():+.3f}")
    else:
        df["학력적합도"] = 0.0

    # 유사도가 누락된 행 대비 안전 변환.
    sim_series = pd.to_numeric(df.get("유사도", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    df["최종점수"] = (
        sim_series * config.WEIGHT_SIMILARITY
        + df["수요점수"] * config.WEIGHT_DEMAND
        + df["임금점수"] * config.WEIGHT_WAGE
        + df["학력적합도"]
    )

    top_n = config.TOP_N_RESULTS

    # ── 유사도 상위 N개 후보 보장 ───────────────────────────────────
    # 임금·수요 점수가 가중 합산되며 본래 가장 의미적으로 가까운 직업이
    # TOP-N 풀에서 밀려나는 케이스를 방지. 유사도 상위 TOP_N_RESULTS 개는
    # 최종점수와 무관하게 후보로 보장하고, 나머지는 최종점수 기준으로
    # 후보 풀(TOP_N_RESULTS * 2)을 채운다.
    if not df.empty and "유사도" in df.columns:
        top_sim_jobs = df.nlargest(top_n, "유사도")
        remaining = df[~df.index.isin(top_sim_jobs.index)]
        pool_size = top_n * 2
        extra_needed = max(0, pool_size - len(top_sim_jobs))
        if extra_needed > 0 and not remaining.empty:
            extra = remaining.nlargest(extra_needed, "최종점수")
            df = pd.concat([top_sim_jobs, extra]).drop_duplicates()
        else:
            df = top_sim_jobs

    result = df.nlargest(top_n, "최종점수").reset_index(drop=True)

    # 최소 1개 결과 보장: 후보 풀이 비어버린 경우 유사도 1위 직업으로 fallback.
    # 동적 하한선과 유사도 상위 보장 로직으로 이미 후보를 걸렀으므로,
    # 추가 하한선 없이 TOP-N 결과를 그대로 반환한다.
    if result.empty and not df.empty and "유사도" in df.columns:
        result = df.nlargest(1, "유사도").reset_index(drop=True)

    result["추천순위"] = range(1, len(result) + 1)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 기능 5) 쿼리 확장 (기법④ 프롬프트 체이닝 — STEP 1)
# ──────────────────────────────────────────────────────────────────────────────
def expand_query_with_claude(query_text: str) -> str:
    """
    [기법④ 프롬프트 체이닝 — STEP 1]
    구직자 원문 입력을 직업 탐색 언어로 변환.
    자격증명·경험 서술을 직무 키워드와 직업군 언어로 확장.
    변환된 텍스트로 임베딩 매칭 시 정확도 향상.
    """
    import anthropic
    import config

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # 시스템 프롬프트는 prompts/parse_resume.txt 우선, 없으면 기본 안내 사용
    parse_system = _load_prompt("parse_resume.txt")
    if not parse_system:
        parse_system = "구직자 입력을 직업 탐색 키워드로 변환합니다."

    user_prompt = (
        "아래 구직자 정보를 RAG 검색 키워드로 변환해주세요:\n"
        f"{query_text}"
    )

    try:
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=200,
            system=parse_system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        expanded = message.content[0].text.strip()
        combined = f"{query_text} {expanded}"
        print(f"  🔄 쿼리 확장: {len(query_text)}자 → {len(combined)}자")
        return combined
    except Exception as e:
        print(f"  ⚠️ 쿼리 확장 실패, 원문 사용: {e}")
        return query_text


# ──────────────────────────────────────────────────────────────────────────────
# 기능 6) Streamlit 메인 인터페이스
# ──────────────────────────────────────────────────────────────────────────────
def search_jobs(
    query_text: str,
    user_wage_floor: Optional[float] = None,
    use_query_expansion: bool = True,
) -> Dict[str, Any]:
    """find_similar_jobs() + filter_top3() 를 묶은 원스톱 검색 함수.

    Streamlit 페이지에서 직접 호출하는 메인 인터페이스.
    use_query_expansion=True (기본값) 시 expand_query_with_claude() 로
    쿼리를 한 번 확장한 뒤 임베딩 매칭에 사용한다 (기법④ STEP 1).

    Returns:
        dict: {
            'results' (pd.DataFrame): 최종 TOP-N 추천 결과 (실패/없음 시 빈 DF).
            'extracted_skills' (dict[str, list[str]]): 사용자 입력에서 추출한
                카테고리별 역량 키워드 (학력 / 자격증 / 기술도구 / 강점성향 / 희망방향).
            'expanded_query' (str): 확장된 검색 쿼리 (확장 미사용 시 원본).
        }
    """
    # UI/진단용 카테고리별 역량 추출 (점수에는 사용하지 않음).
    extracted_skills = extract_user_skills(query_text)

    # 학력 키워드는 Claude 확장 단계에서 의역·삭제될 수 있으므로
    # 항상 원본 query_text 기준으로 추출한다.
    user_edu_score = extract_education_score(query_text)
    if user_edu_score is not None:
        print(f"  🎓 사용자 학력 점수: {user_edu_score} (텍스트에서 자동 추출)")

    if use_query_expansion:
        print("  📝 구직자 입력을 직업 탐색 언어로 변환 중...")
        search_text = expand_query_with_claude(query_text)
    else:
        search_text = query_text

    expanded_query = search_text or ""
    preview = expanded_query.strip().replace("\n", " ")
    print(
        f"🔍 검색: '{preview[:60]}...' "
        f"(상위 {config.TOP_K_CANDIDATES}개 후보 탐색)"
    )

    candidates = find_similar_jobs(expanded_query)
    if candidates.empty:
        print("⚠️ 유사도 임계값 이상의 후보가 없습니다.")
        return {
            "results": pd.DataFrame(),
            "extracted_skills": extracted_skills,
            "expanded_query": expanded_query,
        }

    result_df = filter_top3(
        candidates,
        user_wage_floor=user_wage_floor,
        user_edu_score=user_edu_score,
    )
    print(f"✅ TOP {len(result_df)} 직업 선정 완료")
    return {
        "results": result_df,
        "extracted_skills": extracted_skills,
        "expanded_query": expanded_query,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 단독 실행 테스트 블록
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        (
            "이공계 데이터 분석가 지망생",
            "Python과 SQL을 활용한 데이터 분석 경험이 있고 통계 분석을 잘합니다. "
            "IT 대기업 공채를 준비했습니다.",
        ),
        (
            "법학 전공 공채 준비생",
            "법학을 전공하고 사법시험을 준비했으나 방향을 바꾸고 싶습니다. "
            "논리적 분석과 문서 작성에 강점이 있습니다.",
        ),
        (
            "디자인 전공 취업 준비생",
            "시각디자인을 전공했고 Adobe 툴을 잘 다룹니다. "
            "브랜드 디자인과 UI 작업 경험이 있습니다.",
        ),
    ]

    for title, query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"시나리오: {title}")
        print(f"원문: {query[:80]}")
        print("-" * 40)

        # 학력 점수는 원본 query 에서 추출 (확장 후엔 키워드가 의역될 수 있음).
        user_edu_score = extract_education_score(query)
        if user_edu_score is not None:
            print(f"  🎓 사용자 학력 점수: {user_edu_score} (텍스트에서 자동 추출)")

        # STEP 1) 쿼리 확장 (Claude 호출 1회) — 결과를 출력 후 매칭에 재사용.
        expanded = expand_query_with_claude(query)
        print(f"확장 결과: {expanded[:150]}...")
        print("-" * 40)

        # STEP 2) 확장된 쿼리로 매칭 진단을 위해 search_jobs 를 풀어 호출.
        # (OpenAI 임베딩은 1회만 사용)
        print(
            f"🔍 검색: 확장 쿼리 사용 "
            f"(상위 {config.TOP_K_CANDIDATES}개 후보 탐색)"
        )
        candidates = find_similar_jobs(expanded)
        if not candidates.empty:
            print(
                f"  [진단] 유사도 분포: 최고={candidates['유사도'].max():.4f}, "
                f"최저={candidates['유사도'].min():.4f}, "
                f"평균={candidates['유사도'].mean():.4f}"
            )
            print(
                f"  [진단] 상위 5개:\n"
                f"{candidates.head(5)[['직업명', '유사도']].to_string(index=False)}"
            )
            result = filter_top3(candidates, user_edu_score=user_edu_score)
            print(f"✅ TOP {len(result)} 직업 선정 완료")
        else:
            result = pd.DataFrame()
            print("⚠️ 후보가 없습니다.")

        if not result.empty:
            for _, row in result.iterrows():
                print(
                    f"  {row['추천순위']}위: {row.get('직업명', '?')} "
                    f"(유사도:{row.get('유사도', 0):.3f}, "
                    f"최종점수:{row.get('최종점수', 0):.3f})"
                )
                print(
                    f"       {row.get('대분류명', '?')} > {row.get('중분류명', '?')}"
                )
                print(
                    f"       전망:{row.get('전망점수', '?')} | "
                    f"구인배율:{row.get('평균구인배율', '?')} | "
                    f"임금:{row.get('월평균임금_천원', '?')}천원"
                )
        else:
            print("  결과 없음")
