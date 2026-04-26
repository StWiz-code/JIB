import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_counselor_questions,
    generate_ncs_translation,
)

PROSPECT_LABEL = {
    5: ("📈 증가", "#1B5E7A"),
    4: ("↗️ 다소 증가", "#2C4F8A"),
    3: ("➡️ 유지", "#5D4037"),
    2: ("↘️ 다소 감소", "#6A1F00"),
    1: ("📉 감소", "#7B0000"),
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
    st.markdown("""
<div style="background:linear-gradient(90deg,#2C4F8A,#3D6BB0);
            border-radius:10px; padding:1.2rem 1.8rem; margin-bottom:1.5rem;">
<h2 style="color:white; margin:0;">👥 상담사 전문 대시보드</h2>
<p style="color:#D4E0F0; margin:0.3rem 0 0;">
내담자 역량 데이터 분석 · 상담 가이드 · 열린 질문 생성
</p>
</div>
""", unsafe_allow_html=True)

    # 개인정보 안내 배너
    st.markdown("""
<div class="privacy-banner">
⚠️ <b>개인정보 입력 주의</b> — 내담자의 이름·연락처·주민등록번호 등 식별 가능한 정보는
입력하지 마세요. 역량·경험·희망사항 중심으로 작성해주시면 됩니다.
</div>
""", unsafe_allow_html=True)

    with st.form("cs_form"):
        st.markdown("#### 📋 내담자 정보 입력")

        free_text = ""
        tab1, tab2 = st.tabs(["📌 항목별 입력", "✏️ 자유 서술"])
        with tab1:
            st.caption(
                "상담 전에 내담자 정보를 체계적으로 정리하고 싶다면 항목별 입력을 선택하세요."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                c_background = st.text_input(
                    "전공 / 학력 / 주요 이력",
                    placeholder="예: 법학 전공, 사법시험 준비 3년"
                )
                c_skills = st.text_input(
                    "보유 기술 · 역량",
                    placeholder="예: 법률 문서 작성, 논리적 분석, 문서 작성"
                )
                c_experience = st.text_area(
                    "주요 경험",
                    placeholder="예: 법률 공부 3년, 취업 경험 없음, 20대 후반",
                    height=80
                )
            with col_b:
                c_strength = st.text_area(
                    "내담자 강점 (관찰 또는 자기 서술)",
                    placeholder="예: 분석력 우수, 꼼꼼한 문서 작업, 장기 목표 지속력",
                    height=80
                )
                c_concern = st.text_area(
                    "내담자 고민 · 상담 목적",
                    placeholder="예: 법조계 외 방향 전환 희망, 어떤 직업을 탐색해야 할지 막막함",
                    height=80
                )
        with tab2:
            st.caption(
                "상담 노트나 메모한 내용을 그대로 붙여넣거나 자유롭게 서술하고 싶을 때 "
                "자유 서술을 이용하세요."
            )
            free_text = st.text_area(
                "자유 서술",
                placeholder="""작성 예시:
법학을 전공하고 사법시험을 준비했으나 방향을 바꾸고 싶어합니다.
논리적 분석과 문서 작성에 강점이 있습니다.
법조계 외 분야로 전환을 원하며, 어떤 직업을 탐색해야 할지 모르는 상태입니다.
연령: 20대 후반, 취업 경험 없음.

※ 내담자 이름, 연락처 등 개인 식별 정보는 입력하지 마세요.""",
                height=200
            )

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_guide = st.checkbox("상담 가이드 생성", value=True)
        with col2:
            show_questions = st.checkbox("열린 질문 생성", value=True)
        with col3:
            show_ncs = st.checkbox("NCS 번역 포함", value=True)

        target_job = st.text_input(
            "특정 직업에 대한 열린 질문 생성 (선택, 비우면 1순위 직업 기준)",
            placeholder="예: 법무사"
        )

        submitted = st.form_submit_button(
            "📊 상담 가이드 생성",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if (free_text or "").strip():
            query_text = free_text.strip()
        else:
            parts = []
            if c_background: parts.append(f"학력/이력: {c_background}")
            if c_skills: parts.append(f"보유 역량: {c_skills}")
            if c_experience: parts.append(f"경험: {c_experience}")
            if c_strength: parts.append(f"강점: {c_strength}")
            if c_concern: parts.append(f"고민/상담목적: {c_concern}")
            query_text = "\n".join(parts)

        if not query_text.strip():
            st.warning("내담자 정보를 최소 한 가지 이상 입력해주세요.")
            return

        if st.session_state.get("cs_last_query") != query_text:
            with st.spinner(
                "입력하신 내용을 직업 탐색 언어로 변환하고 "
                "적합한 직업을 탐색하는 중입니다... (15~30초 소요)"
            ):
                top3 = search_jobs(
                    query_text,
                    use_query_expansion=True,
                )

            if top3.empty:
                st.warning("적합한 직업을 찾지 못했습니다.")
                return

            result = {}
            if show_guide:
                with st.spinner("상담 가이드를 생성하는 중... (30~60초 소요)"):
                    result = generate_job_insight(query_text, top3, mode="counselor")

            st.session_state.cs_last_query = query_text
            st.session_state.cs_top3 = top3
            st.session_state.cs_result = result

    top3 = st.session_state.get("cs_top3", pd.DataFrame())
    result = st.session_state.get("cs_result", {})

    if not top3.empty:
        # 데이터 대시보드
        st.markdown('<div class="section-header-cs">📊 역량 매칭 데이터 대시보드</div>',
                   unsafe_allow_html=True)

        cols = st.columns(3, gap="medium")
        for idx, (_, row) in enumerate(top3.iterrows()):
            직업명 = row.get('직업명', '?')
            유사도 = row.get('유사도', 0)
            최종점수 = row.get('최종점수', 0)
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
                유사도_f = float(유사도)
            except (TypeError, ValueError):
                유사도_f = 0.0
            try:
                최종_f = float(최종점수)
            except (TypeError, ValueError):
                최종_f = 0.0
            try:
                g = float(구인배율) if pd.notna(구인배율) else 0.0
            except (TypeError, ValueError):
                g = 0.0
            try:
                b = float(부족률) if pd.notna(부족률) else 0.0
            except (TypeError, ValueError):
                b = 0.0
            try:
                w = float(임금) if pd.notna(임금) else 0.0
            except (TypeError, ValueError):
                w = 0.0

            전망_html = _prospect_badge(전망점수)
            if pd.notna(구인배율):
                if g == 0.0:
                    배율 = "0.00 (미집계)"
                elif g > 0:
                    배율 = f"{g:.2f}"
                else:
                    배율 = "—"
            else:
                배율 = "—"
            if pd.notna(부족률):
                if b == 0.0:
                    부족 = "0.0% (미집계)"
                elif b > 0:
                    부족 = f"{b:.1f}%"
                else:
                    부족 = "—"
            else:
                부족 = "—"
            if pd.notna(임금) and w > 0:
                임금_t = (
                    f"월 {int(w):,}천원"
                    f'<br><small style="color:#888;">(중분류 평균 기준)</small>'
                )
            else:
                임금_t = "—"

            with cols[idx]:
                rank_num = ["1️⃣", "2️⃣", "3️⃣"][idx]
                st.markdown(f"""
<div class="job-card-cs">
<div style="font-size:1.3rem;">{rank_num}</div>
<div style="font-size:1.15rem; font-weight:700; color:#1A1A1A; margin:0.3rem 0;">
{직업명}
</div>
<div style="font-size:0.85rem; color:#555; margin-bottom:0.8rem;">
{분류_text}
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem; font-size:0.85rem;">
<div>🎯 <b>적합도</b><br>{유사도_f:.1%}</div>
<div>📌 <b>종합점수</b><br>{최종_f:.3f}</div>
<div>📈 <b>전망</b><br>{전망_html}</div>
<div>📊 <b>구인배율</b><br>{배율}</div>
<div>⚡ <b>부족률</b><br>{부족}</div>
<div>💰 <b>평균임금</b><br>{임금_t}</div>
</div>
</div>
""", unsafe_allow_html=True)

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
                "- **종합점수**: 유사도·수요(구인배율)·임금에 가중치를 적용해 산출한 값으로, "
                "내담자 적합도 비교 시 참고용으로만 활용해 주세요.\n"
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

        # 상담 가이드
        if result:
            st.markdown(
                '<div class="section-header-cs">🤖 AI 상담 가이드</div>',
                unsafe_allow_html=True,
            )
            insight_text = result.get("insight_text", "")
            if insight_text:
                st.markdown(insight_text)
            else:
                st.warning("상담 가이드 생성에 실패했습니다.")

        # NCS 번역
        if show_ncs and st.session_state.get("cs_last_query"):
            st.markdown('<div class="section-header-cs">🔄 역량 번역 (NCS → 현장 언어)</div>',
                       unsafe_allow_html=True)
            keywords = [
                line.strip("- ").split(",")[0].split(":")[0]
                for line in st.session_state.cs_last_query.split("\n")
                if line.strip() and len(line.strip()) > 3
            ][:3]
            if keywords:
                with st.spinner("번역 중..."):
                    translation = generate_ncs_translation(keywords)
                st.markdown(
                    f'<div class="insight-box">'
                    f'{translation.replace(chr(10),"<br>")}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # 열린 질문
        if show_questions:
            st.markdown('<div class="section-header-cs">💬 상담 유도 질문</div>',
                       unsafe_allow_html=True)
            job_for_q = (target_job.strip() if target_job.strip()
                        else top3.iloc[0].get('직업명', ''))
            if job_for_q:
                with st.spinner(f"'{job_for_q}' 관련 상담 질문 생성 중..."):
                    questions = generate_counselor_questions(
                        job_for_q,
                        st.session_state.get("cs_last_query", "")
                    )
                st.markdown(
                    f'<div class="insight-box">'
                    f'<b>"{job_for_q}" 탐색을 위한 열린 질문</b><br><br>'
                    + "".join([
                        f'<div style="padding:0.4rem 0; border-bottom:1px solid #EEE;">'
                        f'• {q}</div>'
                        for q in questions
                    ])
                    + '</div>',
                    unsafe_allow_html=True
                )
