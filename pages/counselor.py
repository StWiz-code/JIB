import html
import re
import streamlit as st
import pandas as pd
from utils.matcher import search_jobs
from utils.claude_generator import (
    generate_job_insight,
    generate_counselor_questions,
    generate_ncs_translation,
    fetch_worknet_supplementary,
    generate_language_redefinition,
    generate_counselor_cot_analysis,
    generate_terminology_dictionary,
)


# ─────────────────────────────────────────────
# UI 헬퍼 — 안전한 마크다운 렌더링 + 인사이트 파싱
#   (pages/jobseeker.py 와 동일 구현. 향후 utils/ui_helpers.py 로 통합 검토.)
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
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<strong>\1</strong>", text)
    return text


def _convert_markdown_to_card_html(text: str) -> str:
    """마크다운 텍스트를 카드 내부에 안전하게 표시할 HTML로 변환.

    - ``**bold**`` → ``<strong>``
    - ``` `code` ``` → ``<code>``
    - ``- bullet`` / ``* bullet`` → ``<ul><li>``
    - ``1. 항목`` → ``<ol><li>``
    - 빈 줄 → 단락 분리(작은 spacer)
    - 마크다운 헤딩(``#``) → 굵은 텍스트
    - 일반 라인 → ``<div>`` 래핑
    """
    if not text:
        return ""

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

        ─────────────────────────────────────
        ## STEP 1. 3관점 분석
        ─────────────────────────────────────
        ...
        ## STEP 2. 교차검증
        ...
        ## STEP 3. 상담 가이드
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
                search_result = search_jobs(
                    query_text,
                    use_query_expansion=True,
                )
                top3 = search_result.get("results", pd.DataFrame())
                extracted_skills = search_result.get("extracted_skills", {})
                expanded_query = search_result.get("expanded_query", "")

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
            st.session_state.cs_extracted_skills = extracted_skills
            st.session_state.cs_expanded_query = expanded_query

    top3 = st.session_state.get("cs_top3", pd.DataFrame())
    result = st.session_state.get("cs_result", {})
    extracted_skills = st.session_state.get("cs_extracted_skills", {})

    if not top3.empty:
        # 세션 캐시에서 워크넷 보강 데이터 조회 (없으면 빈 dict 폴백 → expander 미노출)
        worknet_cache_key = (
            f"worknet_data_{hash(tuple(top3['직업명'].tolist()))}"
        )
        worknet_data = st.session_state.get(worknet_cache_key, {})

        # JIB 3단계 자기이해 사이클 안내 (결과 화면 최상단 — 남색 톤)
        st.markdown(
            "<div style='background:linear-gradient(90deg, #2C4F8A10 0%, "
            "#4A6B9A10 100%); border-radius:10px; padding:18px 22px; "
            "margin-bottom:1.5em; border-left:5px solid #2C4F8A;'>"
            "<div style='color:#2C4F8A; font-size:0.85em; font-weight:600; "
            "letter-spacing:0.5px; margin-bottom:8px;'>"
            "🏠 JIB의 3단계 자기이해 사이클</div>"
            "<div style='display:flex; gap:12px; flex-wrap:wrap; "
            "align-items:stretch;'>"
            # STEP 1
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#2C4F8A; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 1</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>강점 인식</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "내담자 역량 카드를 통해 객관적 진단</div>"
            "</div>"
            # 화살표
            "<div style='display:flex; align-items:center; color:#2C4F8A; "
            "font-size:1.3em;'>→</div>"
            # STEP 2
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#2C4F8A; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 2</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>언어 재정의</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "NCS 용어를 내담자가 이해할 수 있는 언어로 변환</div>"
            "</div>"
            # 화살표
            "<div style='display:flex; align-items:center; color:#2C4F8A; "
            "font-size:1.3em;'>→</div>"
            # STEP 3
            "<div style='flex:1; min-width:180px; background:white; "
            "border-radius:8px; padding:12px 14px; "
            "box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            "<div style='color:#2C4F8A; font-weight:700; font-size:0.78em; "
            "margin-bottom:4px;'>STEP 3</div>"
            "<div style='color:#2c3e50; font-weight:600; font-size:0.95em; "
            "margin-bottom:4px;'>근거 있는 탐색</div>"
            "<div style='color:#666; font-size:0.82em; line-height:1.4;'>"
            "공공데이터 17종 기반 추천 + CoT 추론 + 직무 용어 사전</div>"
            "</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────
        # STEP 1. 강점 인식 — 역량 카드 섹션 (상담사 모드: 남색 테마)
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#2C4F8A; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 1</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>강점 인식 — AI가 분석한 내담자 역량</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1em;'>"
            "내담자 입력에서 추출한 핵심 역량을 정리했습니다. "
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
                f"<div style='background:#eef2f8; border-radius:8px; "
                f"padding:12px 16px; margin-top:1em; color:#2C4F8A; "
                f"font-size:0.88em;'>"
                f"💡 내담자의 역량 키워드 <b>{total_skills}개</b>를 추출했습니다. "
                f"STEP 2~3에서 이 역량을 시장 언어로 번역하고 매칭 근거를 확인하세요."
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

        # ─────────────────────────────────────────────
        # CoT 추론 데이터 준비 — 직업 카드 expander에서 활용
        #   step1_data: 역량·시장·임금 3관점 독립 분석 (CoT 추론 1단계)
        #   step2_data: 3관점 교차검증 — 자기일관성 평가 (CoT 추론 2단계)
        # ─────────────────────────────────────────────
        cs_user_input = st.session_state.get("cs_last_query", "")
        cot_cache_key = f"cs_cot_{hash(cs_user_input[:200])}"
        if cot_cache_key not in st.session_state:
            with st.spinner("CoT 추론 분석 중..."):
                st.session_state[cot_cache_key] = generate_counselor_cot_analysis(
                    top3_df=top3.head(3),
                    user_skills=extracted_skills or {},
                    user_input=cs_user_input,
                )
        cot_data = st.session_state[cot_cache_key]
        step1_data = cot_data.get("step1", {}) if isinstance(cot_data, dict) else {}
        step2_data = cot_data.get("step2", "") if isinstance(cot_data, dict) else ""

        # ─────────────────────────────────────────────
        # STEP 2 안내 — 언어 재정의 종합 안내 (상담사용)
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#2C4F8A; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 2</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>"
            "언어 재정의 — NCS 행정 용어를 시장 언어로</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1em;'>"
            "추천된 각 직업의 NCS 행정 용어를 내담자가 이해하기 쉬운 시장 언어로 "
            "변환합니다. 직업 카드 내부의 <b>✏️ STEP 2. 언어 재정의</b> 섹션과 "
            "<b>📚 직무 용어 사전</b>을 활용하세요.</p>"
            "<div style='background:#f0f4fa; border-radius:8px; padding:10px 14px; "
            "color:#2C4F8A; font-size:0.85em; margin-bottom:1.5em;'>"
            "💬 내담자와의 상담 시 NCS 용어를 풀어 설명하거나, 자기소개서 코칭 시 "
            "After 표현을 시작점으로 제시하세요. 직무 용어 사전의 활용 예시도 "
            "참고할 수 있습니다."
            "</div>",
            unsafe_allow_html=True,
        )

        # ─────────────────────────────────────────────
        # STEP 3. 근거 있는 탐색 — TOP 3 직업 카드 진입 헤더 (남색 테마)
        # ─────────────────────────────────────────────
        st.markdown(
            "<div style='margin-top:1em; margin-bottom:0.3em;'>"
            "<span style='background-color:#2C4F8A; color:white; padding:4px 12px; "
            "border-radius:12px; font-size:0.85em; font-weight:600;'>STEP 3</span> "
            "<span style='font-size:1.3em; font-weight:700; color:#2c3e50; "
            "margin-left:8px;'>근거 있는 탐색 — 역량 적용 가능성 분석</span>"
            "</div>"
            "<p style='color:#666; font-size:0.9em; margin-bottom:1.5em;'>"
            "내담자의 보유 역량이 각 직업에서 어떻게 활용될 수 있는지, "
            "공공데이터 17종 기반으로 분석한 결과입니다. "
            "AI의 3단계 추론 과정과 데이터 근거를 함께 확인하세요.</p>",
            unsafe_allow_html=True,
        )

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
<div style="min-height:5.5em; display:flex; flex-direction:column;
            justify-content:flex-start; padding:0.4em 0 0.6em 0;
            border-bottom:1px solid #e8ecef; margin-bottom:0.8em;">
    <div style="font-size:clamp(0.95rem, 3vw, 1.15rem); font-weight:700;
                color:#1A1A1A; line-height:1.3; margin-bottom:0.3em;
                display:-webkit-box; -webkit-line-clamp:2;
                -webkit-box-orient:vertical; overflow:hidden;
                min-height:1.5em;"
         title="{직업명}">
        <span style="font-size:0.95em; margin-right:0.25em;">{rank_num}</span>{직업명}
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
<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(100px, 1fr)); gap:0.4rem; font-size:clamp(0.8rem, 2.5vw, 0.85rem);">
<div>🎯 <b>적합도</b><br>{유사도_f:.1%}</div>
<div>📌 <b>종합점수</b><br>{최종_f:.3f}</div>
<div>📈 <b>전망</b><br>{전망_html}</div>
<div>📊 <b>구인배율</b><br>{배율}</div>
<div>⚡ <b>부족률</b><br>{부족}</div>
</div>
</div>
""", unsafe_allow_html=True)

                # 💰 임금 — 고정 높이(4em)로 카드 간 줄 맞춤
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

                # 추천 근거 카드 (CoT 요약 — 남색 톤, NCS 경력개발경로 포함)
                try:
                    최종점수_f = float(최종점수) if pd.notna(최종점수) else 0.0
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
                    추천_강도, 추천_색상 = "강력 추천", "#2C4F8A"
                elif 최종점수_f >= 0.45:
                    추천_강도, 추천_색상 = "적극 검토", "#4A6B9A"
                else:
                    추천_강도, 추천_색상 = "참고 검토", "#7A8AA8"

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
                    f"📊 활용: KNOW 전망 · 학력 분포 · "
                    f"고용24·EIS · 임금통계 · NCS 경로"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ─────────────────────────────────────────────
                # CoT 추론 과정 ① — 3관점 독립 분석 (역량·시장·임금)
                # ─────────────────────────────────────────────
                with st.expander(
                    f"🔍 추론 과정 ① — {직업명}의 3관점 독립 분석"
                ):
                    competency_text = (step1_data.get("competency") or "").strip()
                    market_text = (step1_data.get("market") or "").strip()
                    wage_text = (step1_data.get("wage") or "").strip()

                    if competency_text:
                        st.markdown(
                            "<div style='background:#f5f9f7; "
                            "border-left:4px solid #5A8A7E; border-radius:6px; "
                            "padding:12px 16px; margin-bottom:0.8em;'>"
                            "<div style='color:#5A8A7E; font-weight:700; "
                            "font-size:0.95em; margin-bottom:6px;'>"
                            "① 역량 관점</div>"
                            f"<div style='color:#2c3e50; font-size:0.9em; "
                            f"line-height:1.6;'>{competency_text}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "📊 활용 데이터: KNOW 직업전망 · 학과 진출 매핑 · "
                            "NCS 경력개발경로"
                        )

                    if market_text:
                        st.markdown(
                            "<div style='background:#fff8e1; "
                            "border-left:4px solid #d68910; border-radius:6px; "
                            "padding:12px 16px; margin:0.8em 0;'>"
                            "<div style='color:#9a6500; font-weight:700; "
                            "font-size:0.95em; margin-bottom:6px;'>"
                            "② 시장 관점</div>"
                            f"<div style='color:#2c3e50; font-size:0.9em; "
                            f"line-height:1.6;'>{market_text}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "📊 활용 데이터: 고용24 단기 구인구직 · EIS 5년치 · "
                            "부족률 8반기 평균"
                        )

                    if wage_text:
                        st.markdown(
                            "<div style='background:#f5e8ec; "
                            "border-left:4px solid #c97a6e; border-radius:6px; "
                            "padding:12px 16px; margin-top:0.8em;'>"
                            "<div style='color:#9c4a4a; font-weight:700; "
                            "font-size:0.95em; margin-bottom:6px;'>"
                            "③ 임금 관점</div>"
                            f"<div style='color:#2c3e50; font-size:0.9em; "
                            f"line-height:1.6;'>{wage_text}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "📊 활용 데이터: 직종별 임금 (경력년수별 6년치) · "
                            "분위 임금 (의·법 전문직)"
                        )

                    if not (competency_text or market_text or wage_text):
                        st.caption("3관점 분석 결과를 생성하지 못했습니다.")

                # ─────────────────────────────────────────────
                # CoT 추론 과정 ② — 3관점 교차검증 (자기일관성)
                # ─────────────────────────────────────────────
                with st.expander(
                    f"⚖️ 추론 과정 ② — {직업명}을 포함한 3관점 교차검증"
                ):
                    if step2_data:
                        st.markdown(
                            "<div style='background:linear-gradient(135deg, "
                            "#2C4F8A12 0%, #2C4F8A06 100%); "
                            "border-left:4px solid #2C4F8A; border-radius:8px; "
                            "padding:14px 18px;'>"
                            "<div style='color:#2C4F8A; font-weight:700; "
                            "font-size:0.95em; margin-bottom:8px;'>"
                            "🎯 교차검증 결과 및 상담 메시지</div>"
                            f"<div style='color:#2c3e50; font-size:0.92em; "
                            f"line-height:1.7;'>{step2_data}</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "📊 활용 기법: ⑤ 자기일관성 "
                            "(3관점 독립 평가 후 교차검증) · "
                            "⑥ ToT 가지치기-평가-결정"
                        )
                    else:
                        st.caption("교차검증 결과를 생성하지 못했습니다.")

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

                # 언어 재정의 expander — NCS 행정 용어 → 시장 채용공고 표현 (상담사 모드: 남색 테마)
                lang_cache_key = f"cs_lang_redef_{직업명}"
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
                            "상담 시 참고하세요.</div>",
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
                                f"padding:0 8px; background:white; color:#2C4F8A; "
                                f"font-size:1.3em; font-weight:bold;'>→</div>"
                                f"<div style='flex:1; background:#E8ECF4; "
                                f"padding:12px 16px;'>"
                                f"<div style='font-size:0.72em; color:#2C4F8A; "
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
                            "<div style='background:#eef2f8; border-radius:6px; "
                            "padding:10px 14px; margin-top:0.5em; color:#2C4F8A; "
                            "font-size:0.82em;'>"
                            "💡 내담자와의 상담 시 위 표현 쌍을 활용하여 "
                            "NCS 용어의 의미를 풀어 설명할 수 있습니다. "
                            "특히 자기소개서 작성 코칭 시 'After' 표현으로 "
                            "시작점을 제시하세요."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                # 직무 용어 사전 expander (상담사 전용 — NCS 능력단위 ↔ 시장 언어 카드)
                term_cache_key = f"cs_terms_{직업명}"
                with st.expander(
                    "📚 직무 용어 사전 — 상담 시 활용 가능한 핵심 용어"
                ):
                    if term_cache_key not in st.session_state:
                        with st.spinner("직무 용어 사전 생성 중..."):
                            # 워크넷 L11 NCS 능력단위 데이터는 현재 fetch_worknet_supplementary 미포함.
                            # 추후 L11 직업별 호출 추가 시 ncs_terms_for_dict 를 채워 전달.
                            ncs_terms_for_dict: list = []
                            st.session_state[term_cache_key] = (
                                generate_terminology_dictionary(
                                    job_name=str(직업명),
                                    job_category=str(
                                        row.get("중분류명", "")
                                        or row.get("대분류명", "")
                                    ),
                                    ncs_terms=ncs_terms_for_dict,
                                    max_terms=5,
                                )
                            )
                    terms = st.session_state[term_cache_key]

                    if not terms:
                        st.caption("직무 용어 사전을 생성하지 못했습니다.")
                    else:
                        st.markdown(
                            "<div style='color:#666; font-size:0.85em; "
                            "margin-bottom:1em;'>"
                            "이 직업에 대한 상담 시 활용할 수 있는 핵심 NCS 직무 "
                            "용어를 정리했습니다. 내담자에게 시장 언어로 풀어 "
                            "설명하거나 활용 예시로 안내하세요.</div>",
                            unsafe_allow_html=True,
                        )

                        for term in terms:
                            ncs_html = html.escape(term.get("ncs_term", ""))
                            market_html = html.escape(term.get("market_term", ""))
                            st.markdown(
                                f"<div style='background:#E8ECF4; "
                                f"border-radius:10px; padding:14px 18px; "
                                f"margin-bottom:0.8em; "
                                f"border-left:4px solid #2C4F8A;'>"
                                f"<div style='display:flex; gap:8px; "
                                f"align-items:flex-start; margin-bottom:10px; "
                                f"flex-wrap:wrap;'>"
                                f"<div style='flex:1; min-width:200px;'>"
                                f"<div style='font-size:0.72em; color:#6B7280; "
                                f"font-weight:600; letter-spacing:0.5px; "
                                f"margin-bottom:3px;'>NCS 원문</div>"
                                f"<div style='color:#2c3e50; font-size:0.95em; "
                                f"font-weight:600;'>{ncs_html}</div>"
                                f"</div>"
                                f"<div style='color:#2C4F8A; font-size:1.2em; "
                                f"padding:5px 4px;'>→</div>"
                                f"<div style='flex:1; min-width:200px;'>"
                                f"<div style='font-size:0.72em; color:#2C4F8A; "
                                f"font-weight:600; letter-spacing:0.5px; "
                                f"margin-bottom:3px;'>시장 언어</div>"
                                f"<div style='color:#2c3e50; font-size:0.95em; "
                                f"font-weight:600;'>{market_html}</div>"
                                f"</div>"
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            explanation = (term.get("explanation") or "").strip()
                            if explanation:
                                st.markdown(
                                    f"<div style='background:white; "
                                    f"border-radius:6px; padding:10px 14px; "
                                    f"margin-bottom:6px;'>"
                                    f"<span style='color:#2C4F8A; "
                                    f"font-weight:600; font-size:0.82em;'>"
                                    f"💡 상담 안내</span> "
                                    f"<span style='color:#2c3e50; "
                                    f"font-size:0.88em; line-height:1.5;'>"
                                    f"{html.escape(explanation)}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                            example = (term.get("example") or "").strip()
                            if example:
                                st.markdown(
                                    f"<div style='background:#FFF8E1; "
                                    f"border-radius:6px; padding:10px 14px; "
                                    f"margin-bottom:6px;'>"
                                    f"<span style='color:#9C7A4A; "
                                    f"font-weight:600; font-size:0.82em;'>"
                                    f"📌 활용 예시</span> "
                                    f"<span style='color:#2c3e50; "
                                    f"font-size:0.88em; line-height:1.5;'>"
                                    f"{html.escape(example)}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        st.markdown(
                            "<div style='background:linear-gradient(135deg, "
                            "#2C4F8A12 0%, #2C4F8A06 100%); border-radius:8px; "
                            "padding:12px 16px; margin-top:0.5em; "
                            "color:#2C4F8A; font-size:0.85em;'>"
                            "🎯 <b>상담 활용 팁:</b> 내담자가 NCS 용어에 익숙하지 "
                            "않을 수 있으니, 위 시장 언어로 먼저 설명한 후 "
                            "NCS 원문은 보조 자료로 활용하세요. 활용 예시는 "
                            "내담자의 실제 경험을 직무 표현으로 변환하는 데 "
                            "참고할 수 있습니다."
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

        # ─────────────────────────────────────────────
        # 🤖 AI 상담 가이드 — 단계별 카드 렌더링
        # ─────────────────────────────────────────────
        if result:
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

            # 카드 헤더 (외부, 좌측 정렬)
            st.markdown(
                """
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1.2em;">
    <div style="font-size:1.6em;">🤖</div>
    <div>
        <div style="font-size:1.25em; font-weight:700; color:#2c3e50;">
            AI 상담 가이드
        </div>
        <div style="font-size:0.85em; color:#666; margin-top:2px;">
            데이터 기반 상담을 위한 단계별 분석과 가이드
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            insight_text = result.get("insight_text", "")
            if not insight_text:
                st.warning("상담 가이드 생성에 실패했습니다.")
            else:
                sections = _parse_insight_sections(insight_text)

                if sections["has_steps"]:
                    # STEP 1 카드 — 남색 (3관점 분석)
                    if sections["step1"]:
                        step1_html = _convert_markdown_to_card_html(
                            sections["step1"]
                        )
                        st.markdown(
                            f"""
<div style="background:#f0f4fa; border-left:5px solid #2C4F8A;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(44,79,138,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #2C4F8A20;">
        <span style="background:#2C4F8A; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 1
        </span>
        <span style="font-weight:700; color:#2C4F8A; font-size:1em;">
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

                    # STEP 2 카드 — 보라색 (교차검증)
                    if sections["step2"]:
                        step2_html = _convert_markdown_to_card_html(
                            sections["step2"]
                        )
                        st.markdown(
                            f"""
<div style="background:#f4f0f9; border-left:5px solid #5A4A8A;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(90,74,138,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #5A4A8A20;">
        <span style="background:#5A4A8A; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 2
        </span>
        <span style="font-weight:700; color:#5A4A8A; font-size:1em;">
            교차검증 — 자기일관성 평가
        </span>
    </div>
    <div style="color:#2c3e50; font-size:0.95em; line-height:1.75;">
        {step2_html}
    </div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                    # STEP 3 카드 — 청록색 (상담 가이드)
                    if sections["step3"]:
                        step3_html = _convert_markdown_to_card_html(
                            sections["step3"]
                        )
                        st.markdown(
                            f"""
<div style="background:linear-gradient(135deg, #f5f9f7 0%, #eef4f1 100%);
            border-left:5px solid #5A8A7E;
            border-radius:12px; padding:18px 22px; margin-bottom:1.2em;
            box-shadow:0 1px 4px rgba(90,138,126,0.08);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:12px; padding-bottom:10px;
                border-bottom:1px solid #5A8A7E20;">
        <span style="background:#5A8A7E; color:white; padding:4px 12px;
                     border-radius:12px; font-size:0.78em; font-weight:600;">
            STEP 3
        </span>
        <span style="font-weight:700; color:#5A8A7E; font-size:1em;">
            상담 가이드 — 열린 질문과 탐색 포인트
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
                    # 파싱 실패 fallback — 단일 카드로 전체 안전 렌더
                    fallback_html = _convert_markdown_to_card_html(
                        sections["raw"]
                    )
                    st.markdown(
                        f"""
<div style="background:#f7f9fb; border-left:5px solid #2C4F8A;
            border-radius:12px; padding:18px 22px; color:#2c3e50;
            font-size:0.95em; line-height:1.75;">
    {fallback_html}
</div>
""",
                        unsafe_allow_html=True,
                    )

        # ─────────────────────────────────────────────
        # 🔄 역량 번역 (NCS → 현장 언어) — 단일 HTML 블록 통합 렌더링
        # ─────────────────────────────────────────────
        if show_ncs and st.session_state.get("cs_last_query"):
            keywords = [
                line.strip("- ").split(",")[0].split(":")[0]
                for line in st.session_state.cs_last_query.split("\n")
                if line.strip() and len(line.strip()) > 3
            ][:3]
            if keywords:
                st.markdown("---")

                with st.spinner("번역 중..."):
                    translation = generate_ncs_translation(keywords)

                translation_html = _convert_markdown_to_card_html(
                    translation or ""
                )

                st.markdown(
                    f"""
<div style="margin-bottom:1.2em;">
    <div style="display:flex; align-items:center; gap:12px;
                margin-bottom:1em;">
        <div style="font-size:1.5em;">🔄</div>
        <div>
            <div style="font-size:1.15em; font-weight:700; color:#2c3e50;">
                역량 번역 (NCS → 현장 언어)
            </div>
            <div style="font-size:0.82em; color:#666; margin-top:2px;">
                내담자에게 NCS 행정 용어를 채용공고·자기소개서 표현으로 풀어
                설명할 때 활용
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

        # ─────────────────────────────────────────────
        # 💬 열린 질문 — TOP 3 직업 통합 표시
        # ─────────────────────────────────────────────
        if show_questions:
            st.markdown("---")

            st.markdown(
                """
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1em;">
    <div style="font-size:1.5em;">💬</div>
    <div>
        <div style="font-size:1.15em; font-weight:700; color:#2c3e50;">
            상담 유도 질문 — 추천 직업별 열린 질문
        </div>
        <div style="font-size:0.82em; color:#666; margin-top:2px;">
            내담자가 스스로 강점을 발견하도록 돕는 메타 프롬프팅 기반 질문
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # 활용 가이드 (공통)
            st.markdown(
                """
<div style="background:#EEF3FF; border-radius:10px; padding:14px 18px;
            margin-bottom:1.2em; color:#2c3e50; font-size:0.88em;
            line-height:1.6;">
    <b style="color:#2C4F8A;">💡 열린 질문 활용 가이드</b><br>
    아래 질문들은 내담자가 스스로 강점을 발견하도록 돕는 도구입니다.<br>
    • <b>첫 번째 질문</b>: 내담자의 과거 경험에서 강점을 끌어냅니다<br>
    • <b>두 번째 질문</b>: 해당 직업에 대한 자연스러운 관심을 탐색합니다<br>
    • <b>세 번째 질문</b>: 현실적 장애물을 함께 탐색합니다<br>
    <i style="color:#666;">질문 후 내담자의 답변을 충분히 경청하고,
    답변 내용을 다음 질문의 근거로 활용하세요.</i>
</div>
""",
                unsafe_allow_html=True,
            )

            # 직업 결정 — target_job 입력 시 그것만, 비어 있으면 TOP 3 모두
            if target_job and target_job.strip():
                job_list = [target_job.strip()]
                st.caption(
                    f"📌 사용자 지정 직업 "
                    f"'{target_job.strip()}'에 대한 질문을 생성합니다."
                )
            else:
                job_list = []
                for i in range(min(3, len(top3))):
                    job_name_i = str(top3.iloc[i].get("직업명", "")).strip()
                    if job_name_i:
                        job_list.append(job_name_i)
                st.caption(
                    f"📌 추천 TOP {len(job_list)} 직업에 대한 질문을 각각 "
                    f"생성합니다."
                )

            # 질문별 부가 힌트 (메타 프롬프팅 가이드)
            hints = [
                "경청 포인트: 내담자가 언급하는 감정 단어에 주목하세요",
                "탐색 포인트: 구체적 경험이나 사례를 더 물어보세요",
                "현실화 포인트: 내담자의 현실적 우려를 인정하며 "
                "가능성을 함께 탐색하세요",
            ]

            cs_query = st.session_state.get("cs_last_query", "")

            for idx, job_name in enumerate(job_list, 1):
                # 직업명·질의 해시 기반 세션 캐시
                q_cache_key = (
                    f"cs_questions_{job_name}_{hash(cs_query[:200])}"
                )
                if q_cache_key not in st.session_state:
                    with st.spinner(f"'{job_name}' 열린 질문 생성 중..."):
                        st.session_state[q_cache_key] = (
                            generate_counselor_questions(job_name, cs_query)
                        )

                questions = st.session_state[q_cache_key] or []

                # 각 질문을 흰색 카드(인라인) 로 조립 — 카드 내부에 임베드
                q_blocks: list = []
                for i, q in enumerate(questions):
                    hint = hints[i] if i < len(hints) else ""
                    q_safe = html.escape(str(q))
                    hint_safe = html.escape(hint)
                    q_blocks.append(
                        "<div style=\"background:white; border-radius:8px; "
                        "padding:0.8rem 1rem; margin-bottom:0.6rem; "
                        "border-left:3px solid #3D6BB0; "
                        "box-shadow:0 1px 4px rgba(0,0,0,0.06);\">"
                        "<div style=\"font-weight:600; color:#1A1A1A; "
                        "margin-bottom:0.3rem; line-height:1.6;\">"
                        f"Q{i + 1}. {q_safe}</div>"
                        "<div style=\"font-size:0.8rem; color:#666; "
                        "font-style:italic;\">"
                        f"💬 {hint_safe}</div>"
                        "</div>"
                    )
                questions_inner_html = (
                    "\n".join(q_blocks)
                    if q_blocks
                    else "<div style=\"color:#888; font-size:0.9em;\">"
                    "질문을 생성하지 못했습니다.</div>"
                )

                job_name_safe = html.escape(job_name)
                st.markdown(
                    f"""
<div style="background:#f0f4fa; border-left:5px solid #2C4F8A;
            border-radius:12px; padding:18px 22px; margin-bottom:1em;
            box-shadow:0 1px 4px rgba(44,79,138,0.06);">
    <div style="display:flex; align-items:center; gap:8px;
                margin-bottom:14px; padding-bottom:10px;
                border-bottom:1px solid #2C4F8A20;">
        <span style="background:#2C4F8A; color:white; padding:4px 10px;
                     border-radius:10px; font-size:0.75em; font-weight:600;">
            TOP {idx}
        </span>
        <span style="font-weight:700; color:#2C4F8A; font-size:1.05em;">
            {job_name_safe}
        </span>
        <span style="color:#666; font-size:0.85em; margin-left:auto;">
            탐색을 위한 열린 질문
        </span>
    </div>
    <div style="color:#2c3e50; font-size:0.92em; line-height:1.75;">
        {questions_inner_html}
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
