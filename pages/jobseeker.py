import html
import re
import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_stepback_analysis,
    generate_ncs_translation,
    fetch_worknet_supplementary,
    generate_language_redefinition,
)


# ─────────────────────────────────────────────
# UI 헬퍼 — 안전한 마크다운 렌더링 + 인사이트 파싱
# ─────────────────────────────────────────────


def _safe_render_markdown(text: str) -> str:
    """Streamlit 마크다운 특수 문자에 의한 오작동 방지.

    - 라인 시작 ``---`` / ``~~~`` (수평선) → 빈 줄
    - 중간 ``~~`` (취소선) → en dash
    - 숫자 사이 ``~`` (예: ``4~6개월``) → en dash
    - 인라인 ``•`` → 줄바꿈 + 마크다운 bullet ``-``
    - 카드 안에서 폭주하는 헤딩(``#``/``##``/``###``) → ``**굵게**``
    """
    if not text:
        return ""

    # 1. 라인 시작의 연속 하이픈/물결 (수평선 회피)
    text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\~\~\~+\s*$", "", text, flags=re.MULTILINE)

    # 2. 중간 ~~ (취소선) → en dash, 숫자 사이 단일 ~도 안전 처리
    text = text.replace("~~", "‒")
    text = re.sub(r"(\d)\~(\d)", r"\1‒\2", text)

    # 3. 인라인 "•" → 줄바꿈 + 마크다운 bullet으로 분리
    #    예: "역량 관점 • A: ... • B: ... • C: ..."
    #       → "역량 관점\n- A: ...\n- B: ...\n- C: ..."
    if "•" in text:
        new_lines = []
        for line in text.split("\n"):
            if "•" in line:
                parts = line.split("•")
                first = parts[0].strip()
                if first:
                    new_lines.append(first)
                for part in parts[1:]:
                    part = part.strip()
                    if part:
                        new_lines.append(f"- {part}")
            else:
                new_lines.append(line)
        text = "\n".join(new_lines)

    # 4. 마크다운 헤딩 다운그레이드 (카드 안 폭주 방지)
    text = re.sub(r"^###\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)

    return text


def _inline_md_to_html(text: str) -> str:
    """인라인 마크다운(``**bold**``, ``` `code` ```)을 HTML로 변환."""
    if not text:
        return ""
    text = re.sub(r"\*\*([^\*]+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"`([^`]+?)`",
        r"<code style=\"background:#eef2f6; padding:1px 5px; "
        r"border-radius:3px; font-size:0.9em;\">\1</code>",
        text,
    )
    # 인라인 안전 — `_safe_render_markdown` 통과 후에도 남아 있는 헤딩 방어
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<strong>\1</strong>", text)
    return text


def _convert_markdown_to_card_html(text: str) -> str:
    """마크다운 텍스트를 카드 내부에 안전하게 표시할 HTML로 변환.

    처리:
    - ``**bold**`` → ``<strong>``
    - ```code``` → ``<code>``
    - ``- bullet`` / ``* bullet`` → ``<ul><li>``
    - ``1. 항목`` (숫자 리스트) → ``<ol><li>``
    - 빈 줄 → 단락 분리(작은 spacer)
    - 마크다운 헤딩(``#``) → 굵은 텍스트
    - 일반 라인 → ``<div>`` 래핑

    LLM 출력은 신뢰하는 컨텍스트이므로 별도 HTML escape는 적용하지 않는다.
    """
    if not text:
        return ""

    # 0. 공통 안전 처리 (수평선 / 취소선 / 인라인 • / 헤딩 강등)
    text = _safe_render_markdown(text)

    lines = text.split("\n")
    html_lines: list = []
    in_ul = False
    in_ol = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append("<div style=\"height:0.6em;\"></div>")
            continue

        bullet_match = re.match(r"^[\-\*]\s+(.+)$", stripped)
        if bullet_match:
            content = _inline_md_to_html(bullet_match.group(1))
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append(
                    "<ul style=\"margin:0.3em 0 0.6em 1.2em; "
                    "padding-left:0.5em; line-height:1.75;\">"
                )
                in_ul = True
            html_lines.append(
                f"<li style=\"margin-bottom:0.3em;\">{content}</li>"
            )
            continue

        ol_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ol_match:
            content = _inline_md_to_html(ol_match.group(2))
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append(
                    "<ol style=\"margin:0.5em 0 0.6em 1.2em; "
                    "padding-left:0.5em; line-height:1.75;\">"
                )
                in_ol = True
            html_lines.append(
                "<li style=\"margin-bottom:0.4em; font-weight:600;\">"
                f"{content}</li>"
            )
            continue

        # 일반 텍스트 라인 — 열린 리스트가 있으면 먼저 닫음
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

        content = _inline_md_to_html(stripped)
        html_lines.append(
            f"<div style=\"margin:0.3em 0; line-height:1.75;\">{content}</div>"
        )

    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")

    return "\n".join(html_lines)


def _fixed_height_item(content_html: str, min_height: str = "3em") -> str:
    """카드 내 정보 항목을 고정 높이 div로 감싸 줄 맞춤을 보장한다.

    `st.columns(3)` 으로 TOP 3 카드를 가로 배치할 때, 카드별로 정보 분량이
    달라도 같은 항목 라인이 가로로 정렬되도록 ``min-height`` 를 강제한다.
    """
    return (
        f"<div style='min-height:{min_height}; display:flex; "
        f"flex-direction:column; justify-content:flex-start; "
        f"padding:0.3em 0; margin-bottom:0.3em;'>"
        f"{content_html}"
        f"</div>"
    )


def _parse_insight_sections(text: str) -> dict:
    """AI 인사이트 텍스트를 STEP 1·2·3 섹션으로 파싱.

    프롬프트 출력 예시::

        ═════════════════════════════════════
        ## STEP 1. 3관점 분석
        ═════════════════════════════════════
        ...

        ═════════════════════════════════════
        ## STEP 2. 교차검증
        ═════════════════════════════════════
        ...
    """
    if not text:
        return {
            "step1": "",
            "step2": "",
            "step3": "",
            "raw": "",
            "has_steps": False,
        }

    result = {
        "step1": "",
        "step2": "",
        "step3": "",
        "raw": text,
        "has_steps": False,
    }

    step_pattern = re.compile(
        r"##\s*STEP\s*([1-3])[\.\s]*([^\n]*)\n(.*?)(?=##\s*STEP\s*[1-3]|$)",
        re.DOTALL | re.IGNORECASE,
    )

    for step_num, _step_title, step_content in step_pattern.findall(text):
        cleaned = re.sub(
            r"^[═━─\=]+\s*$", "", step_content, flags=re.MULTILINE
        )
        cleaned = _safe_render_markdown(cleaned.strip())
        result[f"step{step_num}"] = cleaned
        result["has_steps"] = True

    return result


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
                search_result = search_jobs(
                    query_text,
                    user_wage_floor=wage_input,
                    use_query_expansion=True,
                )
                top3 = search_result.get("results", pd.DataFrame())
                extracted_skills = search_result.get("extracted_skills", {})
                expanded_query = search_result.get("expanded_query", "")

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
            st.session_state.js_extracted_skills = extracted_skills
            st.session_state.js_expanded_query = expanded_query
        else:
            st.session_state.js_show_ncs = show_ncs

    # 결과 표시
    top3 = st.session_state.get("js_top3", pd.DataFrame())
    result = st.session_state.get("js_result", {})
    show_ncs_flag = st.session_state.get("js_show_ncs", False)
    extracted_skills = st.session_state.get("js_extracted_skills", {})

    if not top3.empty and result:
        # 세션 캐시에서 워크넷 보강 데이터 조회 (없으면 빈 dict 폴백 → expander 미노출)
        worknet_cache_key = (
            f"worknet_data_{hash(tuple(top3['직업명'].tolist()))}"
        )
        worknet_data = st.session_state.get(worknet_cache_key, {})

        # JIB 3단계 자기이해 사이클 안내 (결과 화면 최상단 — 청록 톤)
        st.markdown(
            "<div style='background:linear-gradient(90deg, #5A8A7E10 0%, "
            "#4A6B8A10 100%); border-radius:10px; padding:18px 22px; "
            "margin-bottom:1.5em; border-left:5px solid #5A8A7E;'>"
            "<div style='color:#5A8A7E; font-size:0.85em; font-weight:600; "
            "letter-spacing:0.5px; margin-bottom:8px;'>"
            "🏠 JIB의 3단계 자기이해 사이클</div>"
            "<div style='display:flex; gap:12px; flex-wrap:wrap; "
            "align-items:stretch;'>"
            # STEP 1
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#5A8A7E; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 1</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>강점 인식</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "AI가 분석한 당신의 역량 카드를 통해 자기 객관화</div>"
            "</div>"
            # 화살표
            "<div style='display:flex; align-items:center; color:#5A8A7E; "
            "font-size:1.3em;'>→</div>"
            # STEP 2
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#5A8A7E; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 2</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>언어 재정의</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "NCS 행정 용어를 시장 채용공고 언어로 변환</div>"
            "</div>"
            # 화살표
            "<div style='display:flex; align-items:center; color:#5A8A7E; "
            "font-size:1.3em;'>→</div>"
            # STEP 3
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#5A8A7E; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 3</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>근거 있는 탐색</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "공공데이터 17종 기반 역량 적용 가능성 분석</div>"
            "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────
        # STEP 1. 강점 인식 — 역량 카드 섹션
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#5A8A7E; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 1</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>강점 인식 — AI가 분석한 당신의 역량</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1em;'>"
            "입력하신 내용에서 추출한 핵심 역량을 정리했습니다. "
            "직업 추천은 이 역량들을 기반으로 진행됩니다.</p>",
            unsafe_allow_html=True,
        )

        SKILL_CATEGORIES = [
            ("학력", "🎓", "#5A8A7E", "전공·학력"),
            ("자격증", "📜", "#C9A87A", "보유 자격"),
            ("기술도구", "⚙️", "#4A6B8A", "기술·도구"),
            ("강점성향", "✨", "#8A7BA8", "강점·성향"),
            ("희망방향", "🎯", "#C97A6E", "희망 방향"),
        ]

        skill_cols = st.columns(len(SKILL_CATEGORIES))
        for col, (key, icon, color, label) in zip(skill_cols, SKILL_CATEGORIES):
            items = extracted_skills.get(key, []) if extracted_skills else []
            with col:
                st.markdown(
                    f"<div style='background:linear-gradient(135deg, "
                    f"{color}1A 0%, {color}0D 100%); "
                    f"border-left:4px solid {color}; border-radius:8px; "
                    f"padding:14px 16px; min-height:120px; "
                    f"box-shadow:0 2px 6px rgba(0,0,0,0.04);'>"
                    f"<div style='font-size:0.85em; color:{color}; "
                    f"font-weight:600; margin-bottom:8px;'>{icon} {label}</div>",
                    unsafe_allow_html=True,
                )
                if items:
                    for item in items[:5]:
                        st.markdown(
                            f"<div style='display:inline-block; background:white; "
                            f"color:#2c3e50; padding:4px 10px; margin:3px 4px 3px 0; "
                            f"border-radius:14px; font-size:0.82em; "
                            f"border:1px solid {color}40;'>{item}</div>",
                            unsafe_allow_html=True,
                        )
                    if len(items) > 5:
                        st.markdown(
                            f"<div style='color:#999; font-size:0.75em; "
                            f"margin-top:6px;'>외 {len(items) - 5}개</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<div style='color:#bbb; font-size:0.82em; "
                        "font-style:italic;'>정보 없음</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        total_skills = sum(
            len(items) for items in extracted_skills.values()
        ) if extracted_skills else 0
        if total_skills > 0:
            st.markdown(
                f"<div style='background:#f5f9f7; border-radius:8px; "
                f"padding:12px 16px; margin-top:1em; color:#5A8A7E; "
                f"font-size:0.88em;'>"
                f"💡 총 <b>{total_skills}개</b>의 역량 키워드가 추출되었습니다. "
                f"이 역량들은 521개 직업과의 의미적 유사도 계산에 활용됩니다."
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#fff8e1; border-left:4px solid #d68910; "
                "border-radius:8px; padding:14px 18px; margin-top:1em;'>"
                "<b style='color:#9a6500;'>📝 더 정확한 분석을 위한 안내</b><br/>"
                "<span style='color:#666; font-size:0.9em;'>"
                "입력 내용에서 역량 키워드를 명확히 추출하지 못했습니다. "
                "다음 형식으로 입력해보세요:</span><br/>"
                "<code style='background:white; padding:6px 10px; margin-top:8px; "
                "display:inline-block; border-radius:6px; font-size:0.85em; "
                "color:#2c3e50;'>"
                "전공: 영어영문학 / 자격증: ADsP / 강점: 외향성, 글쓰기 / 희망: 사무직"
                "</code></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ─────────────────────────────────────────────
        # STEP 2 안내 — 언어 재정의 종합 안내
        # ─────────────────────────────────────────────
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#5A8A7E; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 2</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>"
            "언어 재정의 — NCS 행정 용어를 시장 언어로</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1em;'>"
            "추천된 각 직업의 NCS 행정 용어를 채용공고·자기소개서에서 활용 가능한 "
            "시장 언어로 변환합니다. 직업 카드 내부의 "
            "<b>✏️ STEP 2. 언어 재정의</b> 섹션을 펼쳐 확인하세요.</p>"
            "<div style='background:#f5f9f7; border-radius:8px; padding:10px 14px; "
            "color:#5A8A7E; font-size:0.85em; margin-bottom:1.5em;'>"
            "💬 추천된 각 직업마다 Before(NCS 원문) / After(시장 언어) 쌍이 "
            "제공됩니다. 채용공고 검색, 자기소개서 작성, 면접 답변 준비 시 "
            "활용하세요."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ─────────────────────────────────────────────
        # STEP 3. 근거 있는 탐색 — TOP 3 직업 카드 진입 헤더
        # ─────────────────────────────────────────────
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#5A8A7E; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 3</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>근거 있는 탐색 — 역량 적용 가능성 분석</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1.5em;'>"
            "당신의 역량이 어떤 직업에서 어떻게 활용될 수 있는지, "
            "공공데이터 17종을 기반으로 분석한 결과입니다. "
            "각 직업의 추천 근거를 함께 확인하세요.</p>",
            unsafe_allow_html=True,
        )

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

            # HTML 컨텍스트 안전성 — '>' 는 엔티티로 출력
            if 대분류_clean and 중분류_clean:
                분류_text = f"{대분류_clean} &gt; {중분류_clean}"
            elif 대분류_clean:
                분류_text = 대분류_clean
            elif 중분류_clean:
                분류_text = 중분류_clean
            else:
                분류_text = "분류 정보 없음"

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
<div style="min-height:5.5em; display:flex; flex-direction:column;
            justify-content:flex-start; padding:0.4em 0 0.6em 0;
            border-bottom:1px solid #e8ecef; margin-bottom:0.8em;">
    <div style="font-size:clamp(0.95rem, 3vw, 1.15rem); font-weight:700;
                color:#1A1A1A; line-height:1.3; margin-bottom:0.3em;
                display:-webkit-box; -webkit-line-clamp:2;
                -webkit-box-orient:vertical; overflow:hidden;
                min-height:1.5em;"
         title="{직업명}">
        <span style="font-size:0.95em; margin-right:0.25em;">{rank_emoji}</span>{직업명}
    </div>
    <div style="font-size:clamp(0.72rem, 2.2vw, 0.8rem); color:#666;
                line-height:1.4;
                display:-webkit-box; -webkit-line-clamp:2;
                -webkit-box-orient:vertical; overflow:hidden;
                min-height:2.2em;"
         title="{분류_text}">
        {분류_text}
    </div>
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

                # 💰 임금 — 고정 높이(4em) 컨테이너로 카드 간 줄 맞춤
                wage_html_wrapped = (
                    f"<div style='font-size:0.92em; line-height:1.5;'>{wage_html}</div>"
                )
                st.markdown(
                    _fixed_height_item(wage_html_wrapped, "4em"),
                    unsafe_allow_html=True,
                )

                # 📊 분위 임금 분포 — 데이터 유무와 관계없이 동일 높이(3em)
                상위25 = row.get('상위25_임금_천원')
                중위 = row.get('중위_임금_천원')
                하위25 = row.get('하위25_임금_천원')
                if pd.notna(상위25) and pd.notna(중위) and pd.notna(하위25):
                    range_html_inner = (
                        f"<div style='font-size:0.9em; line-height:1.5;'>"
                        f"📊 임금 분포 "
                        f"<span style='color:#888;'>하위 {int(하위25):,}</span> · "
                        f"<b>중위 {int(중위):,}</b> · "
                        f"<span style='color:#5a8a7e;'>상위 {int(상위25):,}</span>"
                        f"<span style='color:#888; font-size:0.85em;'> 천원</span>"
                        f"</div>"
                    )
                else:
                    range_html_inner = (
                        "<div style='font-size:0.85em; color:#bbb;'>"
                        "📊 임금 분포: <i>해당 직업 데이터 없음</i></div>"
                    )
                st.markdown(
                    _fixed_height_item(range_html_inner, "3em"),
                    unsafe_allow_html=True,
                )

                # 추천 근거 카드 (CoT 요약 — 유사도·수요·학력 적합도 종합)
                최종점수 = row.get('최종점수', 0.0)
                try:
                    최종점수_f = float(최종점수)
                except (TypeError, ValueError):
                    최종점수_f = 0.0
                try:
                    유사도_f = float(유사도) if pd.notna(유사도) else 0.0
                except (TypeError, ValueError):
                    유사도_f = 0.0
                try:
                    학력적합도_f = float(학력적합도) if pd.notna(학력적합도) else 0.0
                except (TypeError, ValueError):
                    학력적합도_f = 0.0

                if 최종점수_f >= 0.6:
                    추천_강도, 추천_색상 = "강력 추천", "#5A8A7E"
                elif 최종점수_f >= 0.45:
                    추천_강도, 추천_색상 = "적극 검토", "#7BA098"
                else:
                    추천_강도, 추천_색상 = "참고 검토", "#9CB3AC"

                근거_요인 = []
                if 유사도_f >= 0.50:
                    근거_요인.append(
                        f"역량 유사도 <b>{유사도_f * 100:.0f}%</b> (높음)"
                    )
                elif 유사도_f >= 0.40:
                    근거_요인.append(
                        f"역량 유사도 <b>{유사도_f * 100:.0f}%</b> (보통)"
                    )
                else:
                    근거_요인.append(
                        f"역량 유사도 <b>{유사도_f * 100:.0f}%</b>"
                    )

                # 구인 수요 4단계 분기 — 0.00 미집계는 공채·헤드헌팅 위주로 안내
                if pd.notna(구인배율):
                    try:
                        구인배율_f = float(구인배율)
                    except (TypeError, ValueError):
                        구인배율_f = 0.0
                    try:
                        부족률_f = float(부족률) if pd.notna(부족률) else 0.0
                    except (TypeError, ValueError):
                        부족률_f = 0.0

                    if 구인배율_f >= 1.0:
                        근거_요인.append(
                            f"구인 수요 <b>활발</b> (배율 {구인배율_f:.2f}, 구인&gt;구직)"
                        )
                    elif 구인배율_f >= 0.5:
                        근거_요인.append(
                            f"구인 수요 <b>양호</b> (배율 {구인배율_f:.2f})"
                        )
                    elif 구인배율_f > 0:
                        근거_요인.append(
                            f"구인 수요 <b>제한적</b> (배율 {구인배율_f:.2f})"
                        )
                    else:
                        # 0.00은 월별 채용 통계 미집계 직종 — 부족률로 수요 신호 보강
                        if 부족률_f > 2.0:
                            근거_요인.append(
                                f"구인 수요 <b>통계 미집계</b> "
                                f"(부족률 {부족률_f:.1f}%, 수요↑)"
                            )
                        elif 부족률_f > 0:
                            근거_요인.append(
                                f"구인 수요 <b>통계 미집계</b> "
                                f"(부족률 {부족률_f:.1f}%)"
                            )
                        else:
                            근거_요인.append(
                                "구인 수요 <b>통계 미집계</b> 직종"
                            )

                if 학력적합도_f >= 0.05:
                    근거_요인.append("학력 적합도 <b>✓ 적합</b>")
                elif 학력적합도_f >= 0.02:
                    근거_요인.append("학력 적합도 <b>△ 약간 차이</b>")
                elif 학력적합도_f < 0:
                    근거_요인.append("학력 적합도 <b>⚠ 진입 어려움</b>")

                근거_html = "<br>".join([f"• {요인}" for 요인 in 근거_요인])

                st.markdown(
                    f"<div style='background:linear-gradient(135deg, "
                    f"{추천_색상}15 0%, {추천_색상}08 100%); "
                    f"border-left:4px solid {추천_색상}; border-radius:8px; "
                    f"padding:14px 18px; margin:1em 0; min-height:14.5em;'>"
                    f"<div style='display:flex; justify-content:space-between; "
                    f"align-items:center; margin-bottom:10px;'>"
                    f"<span style='color:{추천_색상}; font-weight:700; "
                    f"font-size:0.95em;'>🎯 추천 근거</span>"
                    f"<span style='background:{추천_색상}; color:white; "
                    f"padding:3px 10px; border-radius:10px; font-size:0.78em; "
                    f"font-weight:600;'>{추천_강도}</span>"
                    f"</div>"
                    f"<div style='color:#2c3e50; font-size:0.86em; "
                    f"line-height:1.65;'>{근거_html}</div>"
                    f"<div style='color:#888; font-size:0.78em; margin-top:10px; "
                    f"padding-top:10px; border-top:1px solid {추천_색상}20;'>"
                    f"📊 활용: KNOW 직업전망 · 학력 분포 · "
                    f"고용24·EIS 구인구직 · 임금통계"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # 🎓 학력 적합도 inline — 데이터 유무와 관계없이 동일 높이(3em)
                if pd.notna(주요학력) and 주요학력:
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

                    edu_html_inner = (
                        f"<div style='font-size:0.92em; line-height:1.5;'>"
                        f"🎓 주요 학력: <b>{주요학력}</b> "
                        f"<span style='color:#666;'>{ratio_str}</span> "
                        f"{fit_label}"
                        f"</div>"
                    )
                else:
                    edu_html_inner = (
                        "<div style='font-size:0.88em; color:#bbb;'>"
                        "🎓 주요 학력: <i>데이터 없음</i></div>"
                    )
                st.markdown(
                    _fixed_height_item(edu_html_inner, "3em"),
                    unsafe_allow_html=True,
                )

                # 🎓 학력 분포 상세 expander — 데이터 유무와 관계없이 슬롯 유지
                if pd.notna(주요학력) and 주요학력:
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
                else:
                    # 데이터 없을 때도 동일한 expander 구조 사용 (라벨만 다름)
                    with st.expander("🎓 학력 분포 상세 (데이터 없음)", expanded=False):
                        st.markdown(
                            "<div style='color:#666; font-size:0.88em; "
                            "padding:0.6em 0;'>"
                            "<i>해당 직업의 학력 분포 데이터를 찾을 수 없습니다.</i><br>"
                            "<span style='color:#888; font-size:0.85em; "
                            "margin-top:0.4em; display:inline-block;'>"
                            "KNOW 직업정보의 학력 분포 응답 표본이 부족하거나 "
                            "마스터 데이터셋에 학력 컬럼이 누락된 경우입니다."
                            "</span>"
                            "</div>",
                            unsafe_allow_html=True,
                        )

                # 언어 재정의 expander — NCS 행정 용어 → 시장 채용공고 표현
                lang_cache_key = f"js_lang_redef_{직업명}"
                with st.expander("✏️ STEP 2. 언어 재정의 — NCS 용어를 시장 언어로"):
                    if lang_cache_key not in st.session_state:
                        with st.spinner("언어 변환 중..."):
                            st.session_state[lang_cache_key] = (
                                generate_language_redefinition(
                                    job_name=str(직업명),
                                    job_category=str(
                                        row.get("중분류명", "")
                                        or row.get("대분류명", "")
                                    ),
                                    user_skills=extracted_skills or {},
                                    max_pairs=4,
                                )
                            )
                    pairs = st.session_state[lang_cache_key]

                    if not pairs:
                        st.caption("언어 변환 데이터를 생성하지 못했습니다.")
                    else:
                        st.markdown(
                            "<div style='color:#666; font-size:0.85em; "
                            "margin-bottom:1em;'>"
                            "공공 데이터의 NCS 행정 용어를 채용공고·자기소개서에서 "
                            "활용 가능한 시장 언어로 변환했습니다. "
                            "자기 표현 시 참고하세요.</div>",
                            unsafe_allow_html=True,
                        )

                        for pair in pairs:
                            before_html = html.escape(pair.get("before", ""))
                            after_html = html.escape(pair.get("after", ""))
                            st.markdown(
                                f"<div style='display:flex; align-items:stretch; "
                                f"gap:0; margin-bottom:0.8em; border-radius:10px; "
                                f"overflow:hidden; border:1px solid #e0e6eb;'>"
                                f"<div style='flex:1; background:#E8ECEF; "
                                f"padding:12px 16px;'>"
                                f"<div style='font-size:0.72em; color:#666; "
                                f"font-weight:600; letter-spacing:0.5px; "
                                f"margin-bottom:6px;'>BEFORE — NCS 원문</div>"
                                f"<div style='font-size:0.92em; color:#2c3e50; "
                                f"line-height:1.5;'>{before_html}</div></div>"
                                f"<div style='display:flex; align-items:center; "
                                f"padding:0 8px; background:white; color:#5A8A7E; "
                                f"font-size:1.3em; font-weight:bold;'>→</div>"
                                f"<div style='flex:1; background:#E8F4F0; "
                                f"padding:12px 16px;'>"
                                f"<div style='font-size:0.72em; color:#5A8A7E; "
                                f"font-weight:600; letter-spacing:0.5px; "
                                f"margin-bottom:6px;'>AFTER — 시장 언어</div>"
                                f"<div style='font-size:0.92em; color:#2c3e50; "
                                f"line-height:1.5;'>{after_html}</div></div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            context_text = (pair.get("context") or "").strip()
                            if context_text:
                                st.markdown(
                                    f"<div style='color:#888; font-size:0.8em; "
                                    f"margin:-0.5em 0 1em 0; padding-left:1em;'>"
                                    f"💬 {html.escape(context_text)}</div>",
                                    unsafe_allow_html=True,
                                )

                        st.markdown(
                            "<div style='background:#f5f9f7; border-radius:6px; "
                            "padding:10px 14px; margin-top:0.5em; color:#5A8A7E; "
                            "font-size:0.82em;'>"
                            "💡 위 표현들은 채용공고 검색, 자기소개서 작성, "
                            "면접 답변 준비 시 활용할 수 있습니다."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                # 🔗 고용24 관련 직업 정보 — 데이터 유무와 관계없이 expander 슬롯 유지
                worknet_info = worknet_data.get(직업명, {})
                if worknet_info.get("related_jobs") or worknet_info.get("official_name"):
                    with st.expander("🔗 관련 직업 정보 (고용24 실시간)"):
                        st.markdown(
                            "<div style='color:#666; font-size:0.85em; "
                            "margin-bottom:0.6em;'>"
                            "고용24 직업사전에서 실시간으로 가져온 관련 "
                            "직업 정보입니다."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        official = worknet_info.get("official_name", "")
                        if official and official != 직업명:
                            st.markdown(
                                f"<div style='padding:4px 0; color:#2c3e50;'>"
                                f"<b>고용24 공식 직업명</b>: {official}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        category = worknet_info.get("category_name", "")
                        if category:
                            st.markdown(
                                f"<div style='padding:4px 0; color:#2c3e50;'>"
                                f"<b>직업 분류</b>: {category}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        related = worknet_info.get("related_jobs") or []
                        if related:
                            st.markdown(
                                "<div style='padding:6px 0 2px 0; "
                                "color:#2c3e50; font-weight:600;'>"
                                "관련 세부 직업</div>",
                                unsafe_allow_html=True,
                            )
                            for related_name in related:
                                st.markdown(
                                    "<div style='padding:4px 0 4px 0.6em; "
                                    f"color:#2c3e50;'>• {related_name}</div>",
                                    unsafe_allow_html=True,
                                )
                        st.caption(
                            "출처: 한국고용정보원 고용24 직업정보 (212L01) "
                            "+ 직업사전 (212L50)"
                        )
                else:
                    # 데이터 없을 때도 동일한 expander 구조 사용 (라벨만 다름)
                    with st.expander("🔗 관련 직업 정보 (고용24 없음)", expanded=False):
                        st.markdown(
                            "<div style='color:#666; font-size:0.88em; "
                            "padding:0.6em 0;'>"
                            "<i>해당 직업에 대한 고용24 관련 직업 데이터를 "
                            "찾을 수 없습니다.</i><br>"
                            "<span style='color:#888; font-size:0.85em; "
                            "margin-top:0.4em; display:inline-block;'>"
                            "이는 다음 중 하나의 이유일 수 있습니다:"
                            "</span>"
                            "<ul style='color:#888; font-size:0.85em; "
                            "margin:0.3em 0 0 1.2em; padding:0; line-height:1.6;'>"
                            "<li>고용24 직업사전에 해당 직업이 등록되지 않음</li>"
                            "<li>해당 직업명이 고용24 분류 체계와 일치하지 않음</li>"
                            "<li>실시간 API 호출 일시적 장애</li>"
                            "</ul>"
                            "</div>",
                            unsafe_allow_html=True,
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
                "정확한 임금 정보는 "
                "[고용24](https://www.work24.go.kr/cm/main.do) "
                "또는 채용공고를 통해 확인하세요.\n"
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

        # ─────────────────────────────────────────────
        # 🤖 AI 역량 인사이트 — 단계별 카드 렌더링
        # ─────────────────────────────────────────────
        st.markdown("---")

        # 모바일에서 마크다운 가독성 보강 (헤딩/문단/리스트 폰트 축소)
        st.markdown(
            """
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
""",
            unsafe_allow_html=True,
        )

        # 카드 헤더
        st.markdown(
            "<div style='display:flex; align-items:center; gap:12px; "
            "margin-bottom:1.2em;'>"
            "<div style='font-size:1.6em;'>🤖</div>"
            "<div>"
            "<div style='font-size:1.25em; font-weight:700; color:#2c3e50;'>"
            "AI 역량 인사이트</div>"
            "<div style='font-size:0.85em; color:#666; margin-top:2px;'>"
            "Claude Opus 4.7이 11가지 프롬프트 기법으로 생성한 단계별 분석"
            "</div></div></div>",
            unsafe_allow_html=True,
        )

        insight_text = result.get("insight_text", "")
        if not insight_text:
            st.warning("인사이트 생성에 실패했습니다.")
        else:
            sections = _parse_insight_sections(insight_text)

            if sections["has_steps"]:
                # STEP 1 카드 — 청록 (3관점 분석)
                if sections["step1"]:
                    step1_html = _convert_markdown_to_card_html(
                        sections["step1"]
                    )
                    st.markdown(
                        f"""
<div style="background:#f5f9f7; border-left:5px solid #5A8A7E;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(90,138,126,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #5A8A7E20;">
        <span style="background:#5A8A7E; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 1
        </span>
        <span style="font-weight:700; color:#5A8A7E; font-size:1em;">
            3관점 분석 — 역량 · 시장 · 임금
        </span>
    </div>
    <div style="color:#2c3e50; font-size:0.95em; line-height:1.75;">
        {step1_html}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # STEP 2 카드 — 황금색 (교차검증)
                if sections["step2"]:
                    step2_html = _convert_markdown_to_card_html(
                        sections["step2"]
                    )
                    st.markdown(
                        f"""
<div style="background:#fef9e7; border-left:5px solid #d68910;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(214,137,16,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #d6891020;">
        <span style="background:#d68910; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 2
        </span>
        <span style="font-weight:700; color:#9a6500; font-size:1em;">
            교차검증 결과 — 3관점 종합 평가
        </span>
    </div>
    <div style="color:#2c3e50; font-size:0.95em; line-height:1.75;">
        {step2_html}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # STEP 3 카드 — 남색 (최종 인사이트)
                if sections["step3"]:
                    step3_html = _convert_markdown_to_card_html(
                        sections["step3"]
                    )
                    st.markdown(
                        f"""
<div style="background:linear-gradient(135deg, #E8ECF4 0%, #f0f4fa 100%);
            border-left:5px solid #2C4F8A;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(44,79,138,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #2C4F8A20;">
        <span style="background:#2C4F8A; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 3
        </span>
        <span style="font-weight:700; color:#2C4F8A; font-size:1em;">
            최종 인사이트 — 종합 분석 및 다음 단계
        </span>
    </div>
    <div style="color:#2c3e50; font-size:0.95em; line-height:1.75;">
        {step3_html}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )
            else:
                # 파싱 실패 fallback — 전체 텍스트 단일 HTML 카드
                fallback_html = _convert_markdown_to_card_html(
                    sections["raw"]
                )
                st.markdown(
                    f"""
<div style="background:#f7f9fb; border-left:5px solid #5A8A7E;
            border-radius:12px; padding:18px 22px; color:#2c3e50;
            font-size:0.95em; line-height:1.75;">
    {fallback_html}
</div>
""",
                    unsafe_allow_html=True,
                )

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

        # ─────────────────────────────────────────────
        # 🔄 NCS → 현장 언어 번역 — 단일 HTML 블록 통합 렌더링
        # ─────────────────────────────────────────────
        if show_ncs_flag and st.session_state.get("js_last_query"):
            st.markdown("---")

            ncs_terms = ["데이터 분석", "통계 처리", "정보시스템 운용"]
            with st.spinner("번역 중..."):
                translation = generate_ncs_translation(ncs_terms)

            translation_html = _convert_markdown_to_card_html(translation or "")

            # 헤더 + 본문 카드를 단일 HTML 블록으로 통합 (시각적 단절 제거)
            st.markdown(
                f"""
<div style="margin-bottom:1.2em;">
    <div style="display:flex; align-items:center; gap:12px;
                margin-bottom:1em;">
        <div style="font-size:1.5em;">🔄</div>
        <div>
            <div style="font-size:1.15em; font-weight:700; color:#2c3e50;">
                NCS → 현장 언어 번역
            </div>
            <div style="font-size:0.82em; color:#666; margin-top:2px;">
                공공데이터의 행정 용어를 채용공고·자기소개서에서 활용 가능한
                시장 언어로 변환
            </div>
        </div>
    </div>
    <div style="background:#f7f9fb; border-left:5px solid #6B7280;
                border-radius:12px; padding:18px 22px; color:#2c3e50;
                font-size:0.93em; line-height:1.7;
                box-shadow:0 1px 3px rgba(107,114,128,0.06);">
        {translation_html}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
