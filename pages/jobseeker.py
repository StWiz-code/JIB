import html
import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_stepback_analysis,
    generate_ncs_translation,
    fetch_worknet_supplementary,
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
        # 세션 캐시에서 워크넷 보강 데이터 조회 (없으면 빈 dict 폴백 → expander 미노출)
        worknet_cache_key = (
            f"worknet_data_{hash(tuple(top3['직업명'].tolist()))}"
        )
        worknet_data = st.session_state.get(worknet_cache_key, {})

        # TOP 3 카드
        st.markdown("""
<div class="section-header-js">📊 추천 직업 TOP 3</div>
""", unsafe_allow_html=True)

        with st.expander("📖 카드 항목 설명", expanded=False):
            st.markdown("""
| 항목 | 설명 |
|---|---|
| 🎯 역량 적합도 | 입력하신 경험·기술과 이 직업의 임베딩 벡터 간 코사인 유사도입니다. 높을수록 역량이 더 잘 맞습니다. |
| 📈 고용 전망 | 향후 5년간 고용 변화 예측입니다 (증가/다소 증가/유지/다소 감소/감소). 한국고용정보원 직업전망 기준. |
| 📊 구인배율 | 구인인원 ÷ 구직건수입니다. 1.0 이상이면 일자리가 구직자보다 많고, 미집계는 공채·헤드헌팅 위주 직종을 의미합니다. |
| ⚡ 부족률 | 현재 해당 직종의 인력 부족 비율입니다. 높을수록 인력 수요가 공급보다 많습니다. 직종별사업체노동력조사 기준. |
| 💰 평균 임금 | 한국표준직업분류 7차 기준 직종의 월평균 임금이며, 신입 임금이 함께 집계된 경우 (신입 1~3년: …천원)으로 추가 표시합니다. 실제 개인별 임금은 경력·기업 규모에 따라 다릅니다. |
""")

        st.session_state.setdefault("feedback", [])

        cols = st.columns(3, gap="medium")
        for idx, (_, row) in enumerate(top3.iterrows()):
            직업명 = row.get('직업명', '알 수 없음')
            유사도 = row.get('유사도', 0)
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
            월평균 = 임금
            if pd.notna(월평균) and 임금_f > 0:
                avg_str = f"**{int(임금_f):,}천원**"
                try:
                    신입_f = float(신입임금) if pd.notna(신입임금) else 0.0
                except (TypeError, ValueError):
                    신입_f = 0.0
                if pd.notna(신입임금) and 신입_f > 0:
                    wage_html = (
                        f"💰 월 평균 {avg_str} "
                        f"<span style='color:#5a8a7e; font-size:0.92em;'>"
                        f"(신입 1~3년 {int(신입_f):,}천원)"
                        f"</span>"
                    )
                else:
                    wage_html = f"💰 월 평균 {avg_str}"
            else:
                wage_html = "💰 <span style='color:#999;'>임금 정보 없음</span>"
            prospect_html = _prospect_badge(전망점수)

            try:
                유사도_f = float(유사도)
            except (TypeError, ValueError):
                유사도_f = 0.0

            with cols[idx]:
                rank_emoji = ["🥇", "🥈", "🥉"][idx]
                st.markdown(f"""
<div class="job-card-js">
<div style="font-size:clamp(1rem, 4vw, 1.4rem);">{rank_emoji}</div>
<div style="font-size:clamp(0.95rem, 3vw, 1.15rem); font-weight:700; color:#1A1A1A; margin:0.3rem 0;">
{직업명}
</div>
<div style="font-size:clamp(0.8rem, 2.5vw, 0.85rem); color:#555; margin-bottom:0.7rem;">
{분류_text}
</div>
<div style="font-size:clamp(0.82rem, 2.5vw, 0.88rem); margin-bottom:0.3rem;">
🎯 <b>역량 적합도</b> {유사도_f:.1%}
</div>
<div style="font-size:clamp(0.82rem, 2.5vw, 0.88rem); margin-bottom:0.3rem;">
📈 <b>고용 전망</b> {prospect_html}
</div>
<div style="font-size:clamp(0.82rem, 2.5vw, 0.88rem); margin-bottom:0.3rem;">
📊 <b>구인배율</b> {배율_text}
</div>
<div style="font-size:clamp(0.82rem, 2.5vw, 0.88rem);">
⚡ <b>부족률</b> {부족_text}
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

                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    if st.button(
                        "👍 적합해 보여요",
                        key=f"fb_pos_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.feedback.append({
                            "job": 직업명,
                            "type": "positive",
                            "rank": idx + 1,
                        })
                        st.toast(f"'{직업명}' 피드백 감사합니다! 🎉")
                with fb_col2:
                    if st.button(
                        "👎 맞지 않아요",
                        key=f"fb_neg_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.feedback.append({
                            "job": 직업명,
                            "type": "negative",
                            "rank": idx + 1,
                        })
                        st.toast("피드백 반영했습니다. 다른 방향을 탐색해 보세요.")

        with st.expander("💡 임금 정보 안내", expanded=False):
            st.markdown('''
**📊 평균 임금**
한국표준직업분류 7차 기준 직종별 평균 월급여
(전경력 종사자 기준, 단위: 천원)

**🌱 신입 임금**
경력 1~3년 미만 종사자의 평균 월급여
청년 구직자가 입직 직후 받을 가능성이 높은 임금 수준

> *출처: 고용노동부 고용노동통계포털 — 직종별 임금 및 근로시간 (2020~2025년)*
> *단위: 천원/월 (월임금총액 기준)*
> *주의: 동일 직종 내에서도 기업 규모·지역·개인 역량에 따라 편차가 있을 수 있습니다.*
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
            st.markdown(insight_text)
        else:
            st.warning("인사이트 생성에 실패했습니다.")

        feedback = st.session_state.get("feedback", [])
        neg_jobs = [f["job"] for f in feedback if f.get("type") == "negative"]
        if len(neg_jobs) >= 2:
            neg_display = ", ".join(html.escape(str(j)) for j in neg_jobs)
            st.markdown(f"""
<div style="background:#FFF8E1; border-radius:8px;
            padding:0.8rem 1rem; margin-top:1rem;
            border-left:3px solid #F59E0B; font-size:0.88rem;">
💡 <b>다른 방향 탐색 제안</b><br>
{neg_display}이(가) 맞지 않으신다면,
희망 방향 항목을 더 구체적으로 작성하시면 다른 직업을 탐색할 수 있습니다.
<br><small style="color:#888;">예: "사람을 직접 만나는 일보다 데이터를 다루는 일이 더 좋습니다"</small>
</div>
""", unsafe_allow_html=True)

        # 세션 피드백 현황 표시 (선순환 데이터 시각화)
        if feedback:
            st.markdown("---")
            st.markdown(
                '<div class="section-header-js">🔄 탐색 피드백 현황</div>',
                unsafe_allow_html=True,
            )
            pos = [f["job"] for f in feedback if f.get("type") == "positive"]
            neg = [f["job"] for f in feedback if f.get("type") == "negative"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**👍 관심 있는 직업**")
                for job in pos:
                    st.markdown(f"• {job}")
                if not pos:
                    st.caption("아직 없음")
            with col_b:
                st.markdown("**👎 맞지 않는 직업**")
                for job in neg:
                    st.markdown(f"• {job}")
                if not neg:
                    st.caption("아직 없음")

            st.caption(
                "이 탐색 기록은 현재 세션에만 유지됩니다. "
                "피드백이 쌓이면 더 정밀한 탐색이 가능합니다."
            )
            st.markdown("""
<div style="background:#F0F7FF; border-radius:8px;
            padding:0.7rem 1rem; font-size:0.82rem; color:#4A5568;
            border-left:3px solid #2E8B7A;">
🔄 <b>선순환 데이터 구조</b>: 탐색 피드백 → RAG 가중치 보정 → 더 정확한 직업 매칭
<br><small>현재는 세션 내 피드백을 수집하며, 향후 누적 데이터로 매칭 엔진을 고도화할 예정입니다.</small>
</div>
""", unsafe_allow_html=True)

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
