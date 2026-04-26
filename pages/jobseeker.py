import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_stepback_analysis,
    generate_ncs_translation,
)

PROSPECT_LABEL = {
    5: ("📈 증가", "#1B7A4A"),
    4: ("↗️ 다소 증가", "#2E8B7A"),
    3: ("➡️ 유지", "#7D6B00"),
    2: ("↘️ 다소 감소", "#B05E00"),
    1: ("📉 감소", "#B03030"),
}


def _prospect_badge(score):
    try:
        key = int(score) if pd.notna(score) else 0
    except (TypeError, ValueError):
        key = 0
    label, color = PROSPECT_LABEL.get(
        key,
        ("정보 없음", "#888")
    )
    return f'<span style="color:{color}; font-weight:600;">{label}</span>'


def render():
    # 상단 헤더
    st.markdown(f"""
<div style="background:linear-gradient(90deg,#2E8B7A,#3DAA98);
            border-radius:10px; padding:1.2rem 1.8rem; margin-bottom:1.5rem;">
<h2 style="color:white; margin:0;">👤 역량 기반 직무 탐색</h2>
<p style="color:#D4EDE8; margin:0.3rem 0 0;">
보유하신 경험과 기술을 바탕으로 적합한 직업 경로를 탐색합니다.
</p>
</div>
""", unsafe_allow_html=True)

    # 개인정보 안내 배너
    st.markdown("""
<div class="privacy-banner">
⚠️ <b>개인정보 입력 주의</b> — 이름·연락처·주민등록번호 등 식별 가능한 정보는
입력하지 마세요. 보유 역량과 경험 중심으로 작성해주시면 됩니다.
</div>
""", unsafe_allow_html=True)

    # 입력 폼
    with st.form("js_form"):
        st.markdown("#### 📝 역량 및 경험 입력")

        free_text = ""
        tab1, tab2 = st.tabs(["📌 항목별 입력", "✏️ 자유 서술"])
        with tab1:
            st.caption(
                "처음 이용하시거나 어떻게 입력할지 막막하다면 항목별 입력을 선택하세요. "
                "빈 항목은 건너뛰어도 됩니다."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                field_major = st.text_input(
                    "전공 / 학력",
                    placeholder="예: 통계학 학사, 경영학 석사"
                )
                field_skills = st.text_input(
                    "보유 기술 · 도구",
                    placeholder="예: Python, SQL, R, Excel, Adobe"
                )
                field_experience = st.text_area(
                    "주요 경험 · 프로젝트",
                    placeholder="예: 공채 준비 3년, 데이터 분석 인턴 6개월",
                    height=80
                )
            with col_b:
                field_strength = st.text_area(
                    "본인이 생각하는 강점",
                    placeholder="예: 논리적 분석, 문서 작성, 수치 처리에 강함",
                    height=80
                )
                field_direction = st.text_area(
                    "희망 방향 · 고민",
                    placeholder="예: IT 대기업 공채를 준비했으나 방향을 바꾸고 싶음. 분석 기획 쪽 관심",
                    height=80
                )
        with tab2:
            st.caption(
                "이미 정리된 내용이 있거나, 이력서 내용을 그대로 붙여넣고 싶을 때 "
                "자유 서술을 이용하세요."
            )
            free_text = st.text_area(
                "자유 서술",
                placeholder="""작성 예시:
Python과 SQL을 활용한 데이터 분석 경험이 있고 통계 분석을 잘합니다.
통계학을 전공했으며 공공 빅데이터 공모전에서 입상한 경험이 있습니다.
IT 대기업 공채를 준비했으나 방향을 바꾸고 싶습니다.
개발보다는 데이터 분석·기획 직무에 관심이 있습니다.

※ 이름, 연락처 등 개인 식별 정보는 입력하지 마세요.""",
                height=200
            )

        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            wage_input_str = st.text_input(
                "희망 최소 월 평균 임금",
                value="",
                placeholder="예: 350  →  월 350만원 이상 (빈칸이면 제한 없음)",
                help="만원 단위로 입력하세요. 예: 300 = 월 300만원 이상",
            )
            try:
                wage_floor_manwon = float(wage_input_str) if str(wage_input_str).strip() else 0
                wage_floor = wage_floor_manwon * 100  # 만원 → 천원
                if wage_floor_manwon > 0:
                    st.caption(
                        f"→ 월 {wage_floor_manwon:,.0f}만원 ({int(wage_floor):,}천원) 이상 직업만 탐색합니다."
                    )
            except ValueError:
                wage_floor = 0
                st.caption("숫자만 입력해주세요. 예: 350")
        with col2:
            show_ncs = st.checkbox(
                "NCS → 현장 언어 번역 포함",
                value=False,
                help="NCS 용어를 민간 채용공고에서 실제로 쓰는 언어로 번역해서 보여줍니다.",
            )

        submitted = st.form_submit_button(
            "🔍 직무 탐색 시작",
            type="primary",
            use_container_width=True
        )

    # 쿼리 텍스트 조합 (자유 서술이 비어 있지 않으면 자유 서술 우선)
    if submitted:
        if (free_text or "").strip():
            query_text = free_text.strip()
        else:
            parts = []
            if field_major: parts.append(f"전공/학력: {field_major}")
            if field_skills: parts.append(f"보유 기술: {field_skills}")
            if field_experience: parts.append(f"주요 경험: {field_experience}")
            if field_strength: parts.append(f"강점: {field_strength}")
            if field_direction: parts.append(f"희망 방향: {field_direction}")
            query_text = "\n".join(parts)

        if not query_text.strip():
            st.warning("최소 한 가지 이상 입력해주세요.")
            return

        # 캐시 확인
        if st.session_state.get("js_last_query") != query_text:
            with st.spinner(
                "입력하신 내용을 직업 탐색 언어로 변환하고 "
                "적합한 직업을 탐색하는 중입니다... (15~30초 소요)"
            ):
                wage_input = float(wage_floor) if wage_floor > 0 else None
                top3 = search_jobs(
                    query_text,
                    user_wage_floor=wage_input,
                    use_query_expansion=True,
                )

            if top3.empty:
                st.warning("입력하신 내용과 적합한 직업을 찾지 못했습니다. 더 구체적으로 작성해주세요.")
                return

            with st.spinner("AI가 인사이트를 생성하는 중입니다... (30~60초 소요)"):
                result = generate_job_insight(query_text, top3, mode="jobseeker")

            st.session_state.js_last_query = query_text
            st.session_state.js_top3 = top3
            st.session_state.js_result = result
            st.session_state.js_show_ncs = show_ncs
        else:
            st.session_state.js_show_ncs = show_ncs

    # 결과 표시
    top3 = st.session_state.get("js_top3", pd.DataFrame())
    result = st.session_state.get("js_result", {})
    show_ncs_flag = st.session_state.get("js_show_ncs", False)

    if not top3.empty and result:
        # TOP 3 카드
        st.markdown("""
<div class="section-header-js">📊 추천 직업 TOP 3</div>
""", unsafe_allow_html=True)

        cols = st.columns(3, gap="medium")
        for idx, (_, row) in enumerate(top3.iterrows()):
            직업명 = row.get('직업명', '알 수 없음')
            유사도 = row.get('유사도', 0)
            전망점수 = row.get('전망점수', None)
            구인배율 = row.get('평균구인배율', None)
            부족률 = row.get('평균부족률', None)
            임금 = row.get('월평균임금_천원', None)
            대분류 = row.get('대분류명', '')
            중분류 = row.get('중분류명', '')

            def clean_str(val):
                s = str(val).strip()
                return '' if s in ('nan', 'None', '') else s

            대분류_clean = clean_str(대분류)
            중분류_clean = clean_str(중분류)

            if 대분류_clean and 중분류_clean:
                분류_text = f"{대분류_clean} > {중분류_clean}"
            elif 대분류_clean:
                분류_text = 대분류_clean
            elif 중분류_clean:
                분류_text = 중분류_clean
            else:
                분류_text = "직업 분야 정보 없음"

            try:
                구인_f = float(구인배율) if pd.notna(구인배율) else 0.0
            except (TypeError, ValueError):
                구인_f = 0.0
            try:
                부족_f = float(부족률) if pd.notna(부족률) else 0.0
            except (TypeError, ValueError):
                부족_f = 0.0
            try:
                임금_f = float(임금) if pd.notna(임금) else 0.0
            except (TypeError, ValueError):
                임금_f = 0.0
            if pd.notna(구인배율):
                if 구인_f == 0.0:
                    배율_text = "0.00 (미집계)"
                elif 구인_f > 0:
                    배율_text = f"{구인_f:.2f}"
                else:
                    배율_text = "—"
            else:
                배율_text = "—"
            if pd.notna(부족률):
                if 부족_f == 0.0:
                    부족_text = "0.0% (미집계)"
                elif 부족_f > 0:
                    부족_text = f"{부족_f:.1f}%"
                else:
                    부족_text = "—"
            else:
                부족_text = "—"
            if pd.notna(임금) and 임금_f > 0:
                _iw = int(임금_f)
                임금_html = (
                    f"월 {_iw:,}천원"
                    f'<small style="color:#888;"> (중분류 평균)</small>'
                )
            else:
                임금_html = "—"
            prospect_html = _prospect_badge(전망점수)

            try:
                유사도_f = float(유사도)
            except (TypeError, ValueError):
                유사도_f = 0.0

            with cols[idx]:
                rank_emoji = ["🥇", "🥈", "🥉"][idx]
                st.markdown(f"""
<div class="job-card-js">
<div style="font-size:1.4rem;">{rank_emoji}</div>
<div style="font-size:1.15rem; font-weight:700; color:#1A1A1A; margin:0.3rem 0;">
{직업명}
</div>
<div style="font-size:0.85rem; color:#555; margin-bottom:0.7rem;">
{분류_text}
</div>
<div style="font-size:0.88rem; margin-bottom:0.3rem;">
🎯 <b>역량 적합도</b> {유사도_f:.1%}
</div>
<div style="font-size:0.88rem; margin-bottom:0.3rem;">
📈 <b>고용 전망</b> {prospect_html}
</div>
<div style="font-size:0.88rem; margin-bottom:0.3rem;">
📊 <b>구인배율</b> {배율_text}
</div>
<div style="font-size:0.88rem; margin-bottom:0.3rem;">
⚡ <b>부족률</b> {부족_text}
</div>
<div style="font-size:0.88rem;">
💰 <b>평균 임금</b> {임금_html}
</div>
</div>
""", unsafe_allow_html=True)

                if st.button(
                    f"🔍 '{직업명}' 상세 분석",
                    key=f"js_stepback_{idx}",
                    use_container_width=True
                ):
                    with st.spinner(f"{직업명} 상세 분석 중..."):
                        detail = generate_stepback_analysis(
                            직업명,
                            st.session_state.get("js_last_query", "")
                        )
                    st.session_state[f"js_stepback_result_{idx}"] = detail

                if f"js_stepback_result_{idx}" in st.session_state:
                    with st.expander(f"📋 {직업명} 상세 분석 결과", expanded=True):
                        detail_text = st.session_state[f"js_stepback_result_{idx}"]
                        detail_text = detail_text.replace(
                            "## 이 직업의 근본 역량",
                            "#### 이 직업의 핵심 특성",
                        )
                        detail_text = detail_text.replace(
                            "## 이 직업의 핵심 특성",
                            "#### 이 직업의 핵심 특성",
                        )
                        detail_text = detail_text.replace(
                            "## 당신과의 연결점",
                            "#### 나와의 연결점",
                        )
                        detail_text = detail_text.replace(
                            "## 나와의 연결점",
                            "#### 나와의 연결점",
                        )
                        detail_text = detail_text.replace("귀하의", "작성하신 내용을 보면")
                        detail_text = detail_text.replace("귀하께서", "작성하신 내용을 보면")
                        st.markdown(detail_text)

        with st.expander("📌 표시 데이터 출처 및 해석 안내", expanded=False):
            st.markdown(
                "- **평균 임금 (중분류 평균)**: 한국고용정보원 KNOW 임금통계 기준, "
                "해당 직업이 속한 직종 중분류의 평균값입니다. "
                "실제 개별 기업·경력·지역에 따라 크게 달라질 수 있으며, "
                "정확한 임금 정보는 [워크넷](https://www.work24.go.kr) 에서 확인하세요.\n"
                "- **구인배율**: EIS 고용행정통계 기준 최근 6개월 평균 "
                "(구인인원 ÷ 구직건수). 1.0 이상이면 인력 수요 우위입니다. "
                "값이 `0.00 (미집계)`로 표시된 경우는 해당 직종의 채용이 "
                "고용24 플랫폼보다 공채·헤드헌팅 등 다른 경로로 주로 이루어져 "
                "통계에 잡히지 않는 케이스입니다.\n"
                "- **부족률**: 한국고용정보원 인력부족률 통계의 분기 평균값입니다. "
                "값이 높을수록 인력 채용에 어려움이 큰 직종이며, "
                "`0.0% (미집계)`는 동일 사유로 통계에 잡히지 않은 경우입니다.\n"
                "- **고용 전망**: 한국고용정보원 직업전망 분류를 5단계 점수로 환산한 값입니다.\n"
                "- 일부 직업은 분류 매핑 한계로 값이 \"—\"로 표시될 수 있습니다."
            )
            st.caption(
                "표시된 임금은 해당 직업이 속한 직종 중분류의 평균값으로, "
                "신입부터 경력자까지 전체 재직자를 포함한 평균입니다. "
                "실제 초임 또는 개인별 임금은 기업 규모·경력 연차·지역에 따라 "
                "크게 달라질 수 있습니다.\n\n"
                "구인배율 '미집계'는 해당 직종의 채용이 고용24 플랫폼보다 "
                "공채·헤드헌팅 등 다른 경로로 주로 이루어져 "
                "통계에 잡히지 않는 경우입니다."
            )

        st.markdown("""
<div style="background:#F0F7FF; border-radius:8px;
            padding:0.7rem 1rem; margin:0.8rem 0;
            font-size:0.84rem; color:#4A5568;
            border-left:3px solid #2E8B7A;">
💡 이 분석은 공공데이터 기반의 참고 정보입니다.
보다 정밀한 직업 탐색과 개인 맞춤 상담은
가까운 <b>고용센터</b> 또는 관련 기관의 <b>커리어 상담</b>을 이용해 보세요.
</div>
""", unsafe_allow_html=True)

        # AI 인사이트
        st.markdown(
            '<div class="section-header-js">🤖 AI 역량 인사이트</div>',
            unsafe_allow_html=True,
        )
        insight_text = result.get("insight_text", "")
        if insight_text:
            st.markdown(insight_text)
        else:
            st.warning("인사이트 생성에 실패했습니다.")

        # NCS 번역 (선택적)
        if show_ncs_flag and st.session_state.get("js_last_query"):
            st.markdown('<div class="section-header-js">🔄 NCS → 현장 언어 번역</div>',
                       unsafe_allow_html=True)
            ncs_terms = ["데이터 분석", "통계 처리", "정보시스템 운용"]
            with st.spinner("번역 중..."):
                translation = generate_ncs_translation(ncs_terms)
            trans_html = (translation or '').replace('\n', '<br>')
            st.markdown(
                f'<div class="insight-box">{trans_html}</div>',
                unsafe_allow_html=True
            )
