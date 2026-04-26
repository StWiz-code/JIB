"""
JIB(Job_Insight_Bridge) — Claude(Anthropic) 기반 LLM 생성 모듈.

JIB RAG 엔진의 마지막 단계로, matcher.search_jobs() 가 산출한 TOP-3 직업 후보를
Claude(Anthropic) 모델로 자연어 인사이트로 변환한다.

적용 프롬프트 기법:
    ① 구조화 프롬프팅 — 시스템/유저 분리 + 단계별 섹션 구분
    ③ Chain-of-Thought — 단계별 사고 과정을 프롬프트에 명시
    ④ 프롬프트 체이닝 — generate_ncs_translation → generate_job_insight 파이프
    ⑤ 자기일관성   — 3관점 분석 후 교차검증
    ⑥ Tree-of-Thoughts — 가지치기 / 평가 / 결정의 ToT 구조
    ⑦ 이종산업 비유 — NCS 용어를 이종 산업 예시로 확장
    ⑨ 메타 프롬프팅 — 상담사용 열린 질문 설계
    ⑩ 스텝백 프롬프팅 — 직업 근본 역량 → 구직자 적합성
    ⑪ 생성된 지식 프롬프팅 — 직업 시장 배경 지식 사전 생성

모든 외부 설정값(API 키 / 모델명)은 config.py 에서 import 한다.
"""

from __future__ import annotations

import json  # noqa: F401  (외부에서 사용할 수 있도록 노출)
import sys as _sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import pandas as pd

# 프로젝트 루트를 sys.path 에 추가해 단독 실행 시에도 config import 가 가능하도록 한다.
if str(Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _load_prompt(filename: str) -> str:
    """
    prompts/ 폴더에서 프롬프트 텍스트를 로드한다.

    실행 중인 작업 디렉터리에 무관하게 프로젝트 루트 기준의 prompts/ 를 우선 시도하고,
    그 후 현재 작업 디렉터리 기준의 상대 경로(prompts/<filename>)도 시도한다.
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


def _get_client() -> anthropic.Anthropic:
    """Anthropic 클라이언트 인스턴스를 생성한다 (config.ANTHROPIC_API_KEY 사용)."""
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _fmt_top3_summary(top3_df: pd.DataFrame) -> str:
    """TOP-3 DataFrame 을 Claude 프롬프트에 삽입할 텍스트 블록으로 변환한다."""
    PROSPECT_LABEL = {5: "증가", 4: "다소 증가", 3: "유지", 2: "다소 감소", 1: "감소"}
    summary = ""
    for _, row in top3_df.iterrows():
        직업명 = row.get("직업명", "알 수 없음")
        순위 = row.get("추천순위", "-")
        전망 = row.get("전망점수", None)
        구인배율 = row.get("평균구인배율", None)
        부족률 = row.get("평균부족률", None)
        임금 = row.get("월평균임금_천원", None)
        대분류 = row.get("대분류명", "")
        중분류 = row.get("중분류명", "")
        전망텍스트 = str(row.get("직업전망_텍스트", ""))[:150]
        유사도 = row.get("유사도", 0)
        최종점수 = row.get("최종점수", 0)

        try:
            전망_key = int(전망) if pd.notna(전망) else 0
        except (TypeError, ValueError):
            전망_key = 0
        전망_text = PROSPECT_LABEL.get(전망_key, "정보 없음")

        배율_text = (
            f"{float(구인배율):.2f}"
            if pd.notna(구인배율) and float(구인배율) > 0
            else "정보 없음"
        )
        부족_text = (
            f"{float(부족률):.1f}%"
            if pd.notna(부족률) and float(부족률) > 0
            else "정보 없음"
        )
        임금_text = (
            f"월 {int(float(임금)):,}천원"
            if pd.notna(임금) and float(임금) > 0
            else "정보 없음 (워크넷에서 확인 권장)"
        )

        try:
            유사도_f = float(유사도)
        except (TypeError, ValueError):
            유사도_f = 0.0
        try:
            최종점수_f = float(최종점수)
        except (TypeError, ValueError):
            최종점수_f = 0.0

        summary += f"""
[{순위}순위] {직업명}
- 직무 분야: {대분류} > {중분류}
- 역량 적합도(유사도): {유사도_f:.1%}
- 종합 추천 점수: {최종점수_f:.3f}
- 고용 전망: {전망_text}
- 구인 수요(구인배율): {배율_text}
- 인력 부족률: {부족_text}
- 평균 임금: {임금_text}
- 직업 전망 요약: {전망텍스트}
"""
    return summary


def _safe_create_message(
    *,
    max_tokens: int,
    user_prompt: str,
    system_text: Optional[str] = None,
) -> str:
    """
    공통 Claude messages.create 래퍼. API 키 누락 / 패키지 오류 / 통신 오류를 일관되게 처리한다.

    Returns:
        모델이 생성한 텍스트. 실패 시 한국어 안내 문자열을 반환한다.
    """
    api_key = (getattr(config, "ANTHROPIC_API_KEY", "") or "").strip()
    if not api_key:
        return "Claude 호출 실패: ANTHROPIC_API_KEY 가 설정되지 않았습니다 (.env 확인)."

    try:
        client = _get_client()
        kwargs: Dict[str, Any] = {
            "model": config.CLAUDE_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_text:
            kwargs["system"] = system_text
        message = client.messages.create(**kwargs)
        return message.content[0].text
    except Exception as e:
        return f"Claude 호출 실패: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# 기능 1) ToT + 자기일관성 기반 직업 인사이트 생성
# ──────────────────────────────────────────────────────────────────────────────
def generate_job_insight(
    query_text: str,
    top3_df: pd.DataFrame,
    user_profile: Optional[dict] = None,
    mode: str = "jobseeker",
) -> dict:
    """
    구직자(jobseeker) 또는 상담사(counselor)를 위한 직업 인사이트 생성.

    [기법①③④⑤⑥] 단일 Claude 호출 안에서
    [가지치기: 3관점 분석] → [평가: 교차검증] → [결정: 최종 인사이트]
    의 ToT 구조를 완성한다.
    """
    # STEP 1 — TOP-3 요약
    jobs_summary = _fmt_top3_summary(top3_df) if (top3_df is not None and not top3_df.empty) else ""

    # STEP 2 — 시스템 프롬프트 (외부 파일 우선, 없으면 기본값) [기법①]
    system_text = _load_prompt("system_prompt.txt")
    if not system_text:
        system_text = """당신은 JIB(Job Insight Bridge)의 AI 직업 컨설턴트입니다.
고용노동부 공공데이터를 기반으로 구직자의 강점을 발견하고
대안 직무 경로를 안내합니다.

핵심 원칙:
- 보유한 강점을 기반으로 설명하고, 결핍을 지적하지 않습니다
- 데이터에 근거한 현실적 정보를 과장 없이 제공합니다
- 구직자가 스스로 판단하고 선택할 수 있도록 정보를 제시합니다
- 한국어로 자연스럽고 따뜻하게, 전문성 있게 작성합니다
- "즉시", "완벽히", "반드시" 같은 단정적 표현을 피합니다

출력 형식 규칙 (반드시 준수):
- 응답 첫 줄과 마지막 줄에 형식 안내, 메타 설명, 토큰 수 언급 절대 금지
- "제공하신 형식에 맞춰", "이 형식을 따라", "토큰 이내" 등의 문구 출력 금지
- 요청된 내용만 바로 출력하고 전후 설명 없이 시작할 것
- ## STEP 1. 로 시작하여 다음 단계 제안으로 끝낼 것"""

    tdf = top3_df if top3_df is not None else pd.DataFrame()

    # STEP 3 — 모드별 유저 프롬프트
    if mode == "jobseeker":
        user_prompt = f"""구직자 입력:
{query_text}

추천 직업 TOP 3 (고용노동부 공공데이터 기반):
{jobs_summary}

아래 형식으로 작성하세요.
모든 항목을 개조식(bullet)으로 작성하여 토큰을 절약하고
정보 밀도를 높입니다.

─────────────────────────────────────
## STEP 1. 3관점 분석
─────────────────────────────────────
각 직업을 3가지 관점에서 bullet 1개씩만 작성 (총 9개 bullet)

**역량 관점** (각 직업별 bullet 1개)
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [어떤 역량이 → 어떻게 적용]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [어떤 역량이 → 어떻게 적용]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [어떤 역량이 → 어떻게 적용]

**시장 관점** (구인배율·전망 데이터 기반, 없으면 "데이터 없음" 명시)
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [시장 진입 현실]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [시장 진입 현실]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [시장 진입 현실]

**임금·안정성 관점** (중분류 평균 기준, 없으면 "데이터 없음" 명시)
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [임금 수준·전망 등급]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [임금 수준·전망 등급]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [임금 수준·전망 등급]

─────────────────────────────────────
## STEP 2. 교차검증 결과
─────────────────────────────────────
- 종합 평가: [3관점 교차 후 강력 추천 / 조건부 추천 / 탐색 권유 분류]
- 순위 조정: [RAG 순위 유지 또는 조정 이유, 1~2문장]

─────────────────────────────────────
## STEP 3. 최종 인사이트
─────────────────────────────────────

**보유 역량 요약**
- [핵심 강점 1]: [구체적 설명]
- [핵심 강점 2]: [구체적 설명]

**{tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}**
- 탐색 이유: [역량 연결 2문장]
- 고려사항①: [진입 과정 현실]
- 고려사항②: [시장 처우·업무 환경]

**{tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}**
- 탐색 이유: [역량 연결 2문장]
- 고려사항①: [진입 과정 현실]
- 고려사항②: [시장 처우·업무 환경]

**{tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}**
- 탐색 이유: [역량 연결 2문장]
- 고려사항①: [진입 과정 현실]
- 고려사항②: [시장 처우·업무 환경]

**다음 단계 제안**
- 단기(지금 바로): [스펙업 없이 할 수 있는 탐색 행동]
- 중기(3~6개월, 선택): [역량 강화 방향, 필수 아님 명시]

─────────────────────────────────────
작성 규칙 (반드시 준수):
- 모든 항목을 bullet(•) 개조식으로 작성
- 산문 문단 금지 (한 bullet당 최대 2문장)
- "직접", "이미", "완벽히", "즉시" 사용 금지
- 데이터 없는 항목: "관련 데이터 없음" 명시
- 추가 스펙업은 중기 선택 항목에만 언급
- 전체 응답 토큰이 3900을 넘지 않도록 간결하게 작성
─────────────────────────────────────"""
    else:
        # counselor 모드
        user_prompt = f"""내담자 입력:
{query_text}

역량 매칭 결과 TOP 3:
{jobs_summary}

아래 형식으로 작성하세요.
모든 항목을 개조식(bullet)으로 작성합니다.

─────────────────────────────────────
## STEP 1. 3관점 분석
─────────────────────────────────────

**역량 번역 관점**
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [내담자 역량 → 현장 언어 번역]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [내담자 역량 → 현장 언어 번역]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [내담자 역량 → 현장 언어 번역]

**시장 진입 가능성 관점**
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [진입 현실·예상 소요기간]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [진입 현실·예상 소요기간]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [진입 현실·예상 소요기간]

**내담자 특성 적합성 관점**
- {tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}: [강점 기반 적합성]
- {tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}: [강점 기반 적합성]
- {tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}: [강점 기반 적합성]

─────────────────────────────────────
## STEP 2. 교차검증
─────────────────────────────────────
- 상담 우선순위: [1순위→2순위→3순위 순서와 이유]
- RAG 순위 조정: [유지 또는 변경 이유]

─────────────────────────────────────
## STEP 3. 상담 가이드
─────────────────────────────────────

**내담자 역량 번역** (채용공고 언어로)
- [역량1]: [번역 결과]
- [역량2]: [번역 결과]

**{tdf.iloc[0].get('직업명','1순위') if len(tdf)>0 else '1순위'}**
- 역량 연결: [구체적 연결 고리]
- 데이터 근거: [전망·배율·임금 강조 또는 "추가 조사 필요"]
- 상담 질문: [열린 질문 1개]

**{tdf.iloc[1].get('직업명','2순위') if len(tdf)>1 else '2순위'}**
- 역량 연결: / • 데이터 근거: / • 상담 질문:

**{tdf.iloc[2].get('직업명','3순위') if len(tdf)>2 else '3순위'}**
- 역량 연결: / • 데이터 근거: / • 상담 질문:

**상담 주의사항**
- [주의사항 1]
- [주의사항 2]

─────────────────────────────────────
작성 규칙:
- 모든 항목 bullet(•) 개조식
- 산문 문단 금지
- 첫 제안이 추가 교육이 되지 않도록
- 데이터 없으면 "추가 조사 필요" 명시
- 전체 응답 3900 토큰 이내
─────────────────────────────────────"""

    # STEP 4 — Claude API 호출
    api_key = (getattr(config, "ANTHROPIC_API_KEY", "") or "").strip()
    if not api_key:
        insight_text = "인사이트 생성 실패: ANTHROPIC_API_KEY 가 설정되지 않았습니다 (.env 확인)."
        print(f"⚠️ {insight_text}")
    else:
        try:
            client = _get_client()
            message = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=4000,
                system=system_text,
                messages=[{"role": "user", "content": user_prompt}],
            )
            insight_text = message.content[0].text
            try:
                used = message.usage.output_tokens
            except AttributeError:
                used = "?"
            print(f"✅ 인사이트 생성 완료 (mode={mode}, {used} tokens)")
        except Exception as e:
            print(f"⚠️ Claude API 호출 실패: {e}")
            insight_text = f"인사이트 생성 실패: {e}"

    # STEP 5 — 결과 반환
    return {
        "insight_text": insight_text,
        "top3_df": top3_df,
        "query": query_text,
        "mode": mode,
        "jobs_summary": jobs_summary,
        "user_profile": user_profile,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 기능 2) NCS 용어 → 현장 언어 번역 [기법⑦ + ④]
# ──────────────────────────────────────────────────────────────────────────────
def generate_ncs_translation(ncs_terms: List[str]) -> str:
    """NCS 용어 리스트를 민간 채용공고 언어로 번역한다 (이종 산업 예시 포함)."""
    if not ncs_terms:
        return ""

    terms_text = "\n".join([f"- {t}" for t in ncs_terms])
    prompt = f"""아래 NCS(국가직무능력표준) 용어들을 민간 채용공고에서 실제로 사용하는 현장 언어로 번역해주세요.

NCS 용어 목록:
{terms_text}

출력 형식 (각 줄):
NCS용어 → 현장언어 (실무 예시: 실제 업무 1가지)

조건:
- 실제 채용공고에서 쓰이는 표현 사용
- 이종 산업으로 적용될 수 있는 경우 이종 산업 예시도 1개 추가
  예) "데이터베이스 구현 → SQL 기반 데이터 파이프라인 구축 (스마트팜 센서 데이터 수집에도 동일 적용 가능)"
- 과장하지 말고 현실적으로 작성"""

    return _safe_create_message(max_tokens=600, user_prompt=prompt)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3) 상담사용 열린 질문 3가지 [기법⑨ 메타 프롬프팅]
# ──────────────────────────────────────────────────────────────────────────────
def generate_counselor_questions(
    job_name: str,
    user_background: str,
) -> List[str]:
    """상담사가 내담자에게 던질 열린 질문 3가지를 생성해 리스트로 반환한다."""
    prompt = f"""당신은 최고의 직업상담 프롬프트 엔지니어입니다.
직업상담사가 '{job_name}' 방향을 내담자와 탐색할 때
내담자 스스로 강점과 동기를 발견하도록 유도하는
열린 질문 3가지를 설계해주세요.

내담자 배경:
{user_background}

설계 조건:
- Yes/No로 답할 수 없는 열린 질문
- 내담자의 과거 경험을 자원으로 활용하는 질문
- 판단·평가하는 표현 금지
- '{job_name}'에 대한 내담자의 자연스러운 호기심을 유발
- 각 질문은 한 문장, 간결하게

출력: 질문만 3줄 (번호, 설명 없이)"""

    text = _safe_create_message(max_tokens=300, user_prompt=prompt)
    if text.startswith("Claude 호출 실패"):
        return [text]

    lines = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip() and len(line.strip()) > 5
    ]
    return lines[:3] if lines else ["질문 생성 결과가 비어 있습니다."]


# ──────────────────────────────────────────────────────────────────────────────
# 기능 4) 스텝백 상세 분석 [기법⑩]
# ──────────────────────────────────────────────────────────────────────────────
def generate_stepback_analysis(
    job_name: str,
    query_text: str,
) -> str:
    """
    1단계로 직업의 핵심 특성을 정리한 뒤, 2단계로 입력 경험·기술과의 연결을 설명한다.
    구직자 모드 결과에서 특정 직업 클릭 시 상세 분석용으로 사용한다.
    """
    prompt = f"""아래 2단계로 '{job_name}'에 대한 심층 분석을 수행해주세요.

1단계 [상위 개념 도출]:
'{job_name}'이라는 직업의 근본적인 핵심 역량은 무엇인지,
그리고 이 직업이 실제로 어떤 문제를 해결하는 일인지
3~4문장으로 먼저 정리해주세요.

2단계 [나와의 연결점]:
1단계에서 정리한 이 직업의 특성을 바탕으로,
입력하신 경험과 기술이 이 직업과 어떤 방식으로 연결될 수 있는지 설명해주세요.

작성 조건:
- '귀하의', '귀하께서' 표현 사용 금지, '작성하신 내용을 보면' 등으로 대체
- '직접', '이미', '강력한', '완벽히' 등 단정적 표현 사용 금지
- 연결 가능성을 열린 표현으로 서술 ('활용될 수 있습니다', '연결될 수 있습니다')
- 현실성이 높은 내용을 통해 연결 적합성이 논리적이어야 함.

출력 마지막에 아래 문장을 반드시 포함할 것:
"이 내용은 공공데이터를 바탕으로 한 참고 정보이며,
보다 자세한 상담은 가까운 고용센터 또는 관련 기관의 커리어 상담을 통해 확인해 보세요."

구직자 정보:
{query_text}

출력:
#### 이 직업의 핵심 특성
(1단계 내용)

#### 나와의 연결점
(2단계 내용, 3~5문장)"""

    return _safe_create_message(max_tokens=1000, user_prompt=prompt)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 5) 생성된 지식 컨텍스트 [기법⑪]
# ──────────────────────────────────────────────────────────────────────────────
def generate_knowledge_context(job_name: str) -> str:
    """직업 추천 전 해당 직업의 시장 현황을 모델이 먼저 정리해 컨텍스트로 사용한다."""
    prompt = f"""직업 인사이트 생성을 위한 배경 지식을 먼저 정리해주세요.

대상 직업: {job_name}

1) [지식 생성] 현재 대한민국 노동시장에서 '{job_name}'에 대한
   핵심 팩트 4가지를 정리해주세요:
   - 주요 취업 경로 및 일반적인 진입 조건
   - 최근 3년간 수요 변화 추이
   - 이 직업에서 실제로 하는 핵심 업무 2가지
   - 성장하기 위해 일반적으로 필요한 역량 방향

형식: 팩트 번호 + 한 문장 (과장 없이 현실적으로)"""

    return _safe_create_message(max_tokens=400, user_prompt=prompt)


# ──────────────────────────────────────────────────────────────────────────────
# 단독 실행 테스트 블록
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from utils.matcher import search_jobs

    print("=" * 60)
    print("=== 테스트 1: 구직자 모드 (ToT + 자기일관성 포함) ===")
    query = (
        "Python과 SQL을 활용한 데이터 분석 경험이 있고 "
        "통계 분석을 잘합니다. IT 대기업 공채를 준비했으나 "
        "방향을 바꾸고 싶습니다."
    )
    top3 = search_jobs(query)
    if not top3.empty:
        result = generate_job_insight(query, top3, mode="jobseeker")
        print("\n[구직자용 인사이트]")
        print(result["insight_text"])
    else:
        print("검색 결과 없음")

    print("\n" + "=" * 60)
    print("=== 테스트 2: 상담사 모드 ===")
    query2 = (
        "법학을 전공하고 사법시험을 준비했으나 방향을 바꾸고 싶습니다. "
        "논리적 분석과 문서 작성에 강점이 있습니다."
    )
    top3_2 = search_jobs(query2)
    if not top3_2.empty:
        result2 = generate_job_insight(query2, top3_2, mode="counselor")
        print("\n[상담사용 가이드]")
        print(result2["insight_text"])

    print("\n" + "=" * 60)
    print("=== 테스트 3: NCS 번역 [기법⑦] ===")
    ncs_terms = ["데이터베이스 구현", "통계 기반 데이터 분석", "정보시스템 운용"]
    print(generate_ncs_translation(ncs_terms))

    print("\n" + "=" * 60)
    print("=== 테스트 4: 상담사 열린 질문 [기법⑨] ===")
    questions = generate_counselor_questions(
        "빅데이터전문가",
        "Python·SQL 경험 보유, IT 대기업 공채 준비 후 방향 전환 희망",
    )
    for q in questions:
        print(f"  • {q}")

    print("\n" + "=" * 60)
    print("=== 테스트 5: 스텝백 상세 분석 [기법⑩] ===")
    stepback = generate_stepback_analysis("통계사무원", query)
    print(stepback)

    print("\n" + "=" * 60)
    print("=== 테스트 6: 생성된 지식 [기법⑪] ===")
    knowledge = generate_knowledge_context("데이터베이스개발자")
    print(knowledge)
