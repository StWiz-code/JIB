import html
import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_counselor_questions,
    generate_ncs_translation,
    fetch_worknet_supplementary,
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

            # 워크넷 OpenAPI 실시간 보강 (212L01 직업정보 + 212L50 직업사전).
            # TOP3 직업명 기준 hash 키로 세션 캐시 → 동일 결과 재조회 방지.
            worknet_cache_key = (
                f"worknet_data_{hash(tuple(top3['직업명'].tolist()))}"
            )
            if worknet_cache_key not in st.session_state:
                with st.spinner("관련 직업 정보 조회 중..."):
                    st.session_state[worknet_cache_key] = (
                        fetch_worknet_supplementary(top3, max_related=3)
                    )

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
        # 세션 캐시에서 워크넷 보강 데이터 조회 (없으면 빈 dict 폴백 → expander 미노출)
        worknet_cache_key = (
            f"worknet_data_{hash(tuple(top3['직업명'].tolist()))}"
        )
        worknet_data = st.session_state.get(worknet_cache_key, {})

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
            신입임금 = row.get('신입임금_천원', None)
            대분류 = row.get('대분류명', '')
            중분류 = row.get('중분류명', '')

            # 학력 분포 파생 컬럼
            주요학력 = row.get('주요학력수준')
            대졸이상비율 = row.get('학력_대졸이상비율')
            전문대졸비율 = row.get('학력_전문대졸비율')
            고졸이하비율 = row.get('학력_고졸이하비율')
            학력적합도 = row.get('학력적합도', 0.0)
            try:
                학력적합도 = float(학력적합도) if pd.notna(학력적합도) else 0.0
            except (TypeError, ValueError):
                학력적합도 = 0.0

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
            월평균 = 임금
            if pd.notna(월평균) and w > 0:
                avg_str = f"**{int(w):,}천원**"
                try:
                    신입_f = float(신입임금) if pd.notna(신입임금) else 0.0
                except (TypeError, ValueError):
                    신입_f = 0.0
                if pd.notna(신입임금) and 신입_f > 0:
                    # 청년 임금 격차 계산
                    격차_pct = round((1 - 신입_f / w) * 100)
                    격차_color = "#888" if 격차_pct < 30 else "#c44"
                    wage_html = (
                        f"💰 직종 평균 {avg_str} | "
                        f"신입(1~3년) **{int(신입_f):,}천원** "
                        f"<span style='color:{격차_color}; font-size:0.88em;'>"
                        f"(격차 {격차_pct}%)"
                        f"</span>"
                    )
                else:
                    wage_html = f"💰 직종 평균 {avg_str}"
            else:
                wage_html = "💰 <span style='color:#999;'>임금 정보 없음</span>"

            with cols[idx]:
                rank_num = ["1️⃣", "2️⃣", "3️⃣"][idx]
                st.markdown(f"""
<div class="job-card-cs">
<div style="font-size:clamp(1rem, 4vw, 1.3rem);">{rank_num}</div>
<div style="font-size:clamp(0.95rem, 3vw, 1.15rem); font-weight:700; color:#1A1A1A; margin:0.3rem 0;">
{직업명}
</div>
<div style="font-size:clamp(0.8rem, 2.5vw, 0.85rem); color:#555; margin-bottom:0.8rem;">
{분류_text}
</div>
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(100px, 1fr)); gap:0.4rem; font-size:clamp(0.8rem, 2.5vw, 0.85rem);">
<div>🎯 <b>적합도</b><br>{유사도_f:.1%}</div>
<div>📌 <b>종합점수</b><br>{최종_f:.3f}</div>
<div>📈 <b>전망</b><br>{전망_html}</div>
<div>📊 <b>구인배율</b><br>{배율}</div>
<div>⚡ <b>부족률</b><br>{부족}</div>
</div>
</div>
""", unsafe_allow_html=True)

                st.markdown(wage_html, unsafe_allow_html=True)

                # 분위 임금 분포 (wage_by_job.csv 매칭된 직업만)
                상위25 = row.get('상위25_임금_천원')
                중위 = row.get('중위_임금_천원')
                하위25 = row.get('하위25_임금_천원')
                if pd.notna(상위25) and pd.notna(중위) and pd.notna(하위25):
                    범위_html = (
                        f"📊 임금 분포: "
                        f"<span style='color:#888;'>하위 {int(하위25):,}</span> · "
                        f"<b>중위 {int(중위):,}</b> · "
                        f"<span style='color:#5a8a7e;'>상위 {int(상위25):,}</span> 천원"
                    )
                    st.markdown(범위_html, unsafe_allow_html=True)

                # 학력 분포 inline 표시 (주요학력수준 + 비중 + 적합도 라벨)
                if pd.notna(주요학력):
                    if 학력적합도 >= 0.05:
                        fit_label = "<span style='color:#5a8a7e;'>✓ 학력 적합</span>"
                    elif 학력적합도 >= 0.02:
                        fit_label = "<span style='color:#9c8a4a;'>△ 약간 차이</span>"
                    elif 학력적합도 < 0:
                        fit_label = "<span style='color:#c44;'>⚠ 진입 어려움</span>"
                    else:
                        fit_label = ""

                    if 주요학력 in ('대졸', '대학원졸', '박사졸') and pd.notna(대졸이상비율):
                        ratio_str = f"({int(round(float(대졸이상비율) * 100))}%)"
                    elif 주요학력 == '전문대졸' and pd.notna(전문대졸비율):
                        ratio_str = f"({int(round(float(전문대졸비율) * 100))}%)"
                    elif 주요학력 in ('고졸', '중졸이하') and pd.notna(고졸이하비율):
                        ratio_str = f"({int(round(float(고졸이하비율) * 100))}%)"
                    else:
                        ratio_str = ""

                    edu_html = (
                        f"🎓 주요 학력: <b>{주요학력}</b> {ratio_str} {fit_label}".strip()
                    )
                    st.markdown(edu_html, unsafe_allow_html=True)

                    with st.expander("🎓 학력 분포 상세"):
                        if pd.notna(고졸이하비율):
                            st.markdown(
                                f"- **고졸 이하**: {int(round(float(고졸이하비율) * 100))}%"
                            )
                        if pd.notna(전문대졸비율):
                            st.markdown(
                                f"- **전문대졸**: {int(round(float(전문대졸비율) * 100))}%"
                            )
                        if pd.notna(대졸이상비율):
                            st.markdown(
                                f"- **대졸 이상**: {int(round(float(대졸이상비율) * 100))}%"
                            )
                        st.caption(
                            "출처: 한국고용정보원 KNOW 직업정보 — "
                            "직업별 교육훈련 및 학력 분포"
                        )
                        st.caption("※ 본 직업 종사자의 실제 학력 분포 (응답자 기준)")

                # 워크넷 OpenAPI(212L01/212L50)로 가져온 관련 직업 정보 표시
                worknet_info = worknet_data.get(직업명, {})
                if worknet_info.get("related_jobs") or worknet_info.get("official_name"):
                    with st.expander("🔗 관련 직업 정보 (워크넷 실시간)"):
                        official = worknet_info.get("official_name", "")
                        if official and official != 직업명:
                            st.markdown(f"**워크넷 공식 직업명**: {official}")
                        category = worknet_info.get("category_name", "")
                        if category:
                            st.markdown(f"**직업 분류**: {category}")
                        related = worknet_info.get("related_jobs") or []
                        if related:
                            st.markdown(
                                f"**관련 세부 직업**: {', '.join(related)}"
                            )
                        st.caption(
                            "출처: 한국고용정보원 워크넷 직업정보 API (212L01) "
                            "+ 직업사전 API (212L50)"
                        )

        with st.expander("💡 임금 정보 — 상담 활용 가이드", expanded=False):
            st.markdown('''
**📊 직종 평균**: 표준직업분류 7차 기준 5년치 평균 (전경력 종사자)
**🌱 신입 임금**: 경력 1~3년 미만 종사자 평균
**📉 격차**: 신입과 평균의 비율

#### 상담 활용 포인트
- **격차 30% 이상**: 입직 후 일정 기간 임금 상승이 큰 직종 → 장기 비전 강조
- **격차 20% 이하**: 신입과 베테랑 차이가 적은 안정 직종 → 초기 진입 매력 강조
- **임금 정보 없음**: 표준분류에 없는 신생 직군 또는 매핑 보완 필요

> *출처: 고용노동부 고용노동통계포털 — 직종별 임금 및 근로시간 (2020~2025년)*
> *내담자에게 안내 시 "동일 직종 내에서도 기업·지역·역량에 따라 편차가 큽니다" 부연 권장*
''')

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
            # 모바일에서 마크다운 가독성 보강 (헤딩/문단/리스트 폰트 축소)
            st.markdown("""
<style>
@media (max-width: 640px) {
    .stMarkdown p {
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
    }
    .stMarkdown h2, .stMarkdown h3 {
        font-size: 1.05rem !important;
    }
    .stMarkdown li {
        font-size: 0.88rem !important;
        line-height: 1.6 !important;
    }
}
</style>
""", unsafe_allow_html=True)
            insight_text = result.get("insight_text", "")
            if insight_text:
                steps = insight_text.split("---")

                if len(steps) >= 3:
                    with st.expander("🔍 STEP 1. AI 3관점 분석 과정 (클릭하여 펼치기)", expanded=False):
                        st.markdown(steps[0] if steps[0].strip() else "")
                        st.caption(
                            "역량·시장·임금 3가지 관점에서 독립적으로 분석한 결과입니다."
                        )

                    with st.expander("⚖️ STEP 2. 교차검증 결과 (클릭하여 펼치기)", expanded=False):
                        st.markdown(steps[1] if len(steps) > 1 else "")
                        st.caption(
                            "3관점 결과를 교차검증하여 추천 우선순위를 조정한 결과입니다."
                        )

                    st.markdown(
                        '<div class="section-header-cs">📋 최종 상담 가이드</div>',
                        unsafe_allow_html=True,
                    )
                    final_content = "---".join(steps[2:]) if len(steps) > 2 else insight_text
                    st.markdown(final_content)
                else:
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
                    f'<p style="font-weight:600; color:#1A1A1A; margin:0 0 0.5rem 0;">'
                    f'"{html.escape(job_for_q)}" 탐색을 위한 열린 질문</p>',
                    unsafe_allow_html=True,
                )
                st.markdown("""
<div style="background:#EEF3FF; border-radius:8px;
            padding:0.8rem 1rem; margin-bottom:0.8rem;
            font-size:0.85rem; color:#2C4F8A;
            border-left:3px solid #2C4F8A;">
<b>💡 열린 질문 활용 가이드</b><br>
아래 질문들은 내담자가 스스로 강점을 발견하도록 돕는 도구입니다.<br>
- <b>첫 번째 질문</b>: 내담자의 과거 경험에서 강점을 끌어냅니다<br>
- <b>두 번째 질문</b>: 해당 직업에 대한 자연스러운 관심을 탐색합니다<br>
- <b>세 번째 질문</b>: 현실적 장애물을 함께 탐색합니다<br><br>
<i>질문 후 내담자의 답변을 충분히 경청하고, 답변 내용을 다음 질문의 근거로 활용하세요.</i>
</div>
""", unsafe_allow_html=True)

                hints = [
                    "경청 포인트: 내담자가 언급하는 감정 단어에 주목하세요",
                    "탐색 포인트: 구체적 경험이나 사례를 더 물어보세요",
                    "현실화 포인트: 내담자의 현실적 우려를 인정하며 가능성을 함께 탐색하세요",
                ]
                for i, q in enumerate(questions):
                    hint = hints[i] if i < len(hints) else ""
                    q_safe = html.escape(str(q))
                    st.markdown(f"""
<div style="background:white; border-radius:8px; padding:0.8rem 1rem;
            margin-bottom:0.6rem; border-left:3px solid #3D6BB0;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);">
<div style="font-weight:600; color:#1A1A1A; margin-bottom:0.3rem;">
Q{i + 1}. {q_safe}
</div>
<div style="font-size:0.8rem; color:#666; font-style:italic;">
💬 {hint}
</div>
</div>
""", unsafe_allow_html=True)
