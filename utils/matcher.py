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

import sys as _sys
from pathlib import Path
from typing import Optional, Tuple

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
        "전망점수", "평균구인배율", "평균부족률", "월평균임금_천원",
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
) -> pd.DataFrame:
    """
    find_similar_jobs() 후보에서 다중지표 가중치 채점으로 TOP-N 을 확정한다.

    채점:
        score = 유사도   * config.WEIGHT_SIMILARITY
              + 수요점수 * config.WEIGHT_DEMAND
              + 임금점수 * config.WEIGHT_WAGE

    Args:
        candidates_df:  find_similar_jobs() 결과 DataFrame
        user_wage_floor: 사용자가 지정한 임금 하한선(천원). 미지정 시 미적용.

    Returns:
        pandas.DataFrame: 최종 TOP-N + '수요점수/임금점수/최종점수/추천순위' 컬럼.
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

    # 유사도가 누락된 행 대비 안전 변환.
    sim_series = pd.to_numeric(df.get("유사도", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    df["최종점수"] = (
        sim_series * config.WEIGHT_SIMILARITY
        + df["수요점수"] * config.WEIGHT_DEMAND
        + df["임금점수"] * config.WEIGHT_WAGE
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
) -> pd.DataFrame:
    """
    find_similar_jobs() + filter_top3() 를 묶은 원스톱 검색 함수.
    Streamlit 페이지에서 직접 호출하는 메인 인터페이스.

    use_query_expansion=True (기본값) 시 expand_query_with_claude() 로
    쿼리를 한 번 확장한 뒤 임베딩 매칭에 사용한다 (기법④ STEP 1).
    """
    if use_query_expansion:
        print("  📝 구직자 입력을 직업 탐색 언어로 변환 중...")
        search_text = expand_query_with_claude(query_text)
    else:
        search_text = query_text

    preview = (search_text or "").strip().replace("\n", " ")
    print(
        f"🔍 검색: '{preview[:60]}...' "
        f"(상위 {config.TOP_K_CANDIDATES}개 후보 탐색)"
    )

    candidates = find_similar_jobs(search_text)
    if candidates.empty:
        print("⚠️ 유사도 임계값 이상의 후보가 없습니다.")
        return pd.DataFrame()

    result = filter_top3(candidates, user_wage_floor=user_wage_floor)
    print(f"✅ TOP {len(result)} 직업 선정 완료")
    return result


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
            result = filter_top3(candidates)
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
