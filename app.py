import streamlit as st

import config

st.set_page_config(
    page_title="JIB — Job Insight Bridge",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 전역 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
/* 공통 */
.main { background-color: #F8F9FA; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* 모드 선택 카드 */
.mode-card {
    border-radius: 12px;
    padding: 2rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: box-shadow 0.2s;
}
.mode-card-js {
    background: linear-gradient(135deg, #E8F5F2 0%, #D4EDE8 100%);
    border: 2px solid #2E8B7A;
}
.mode-card-cs {
    background: linear-gradient(135deg, #E8EEF7 0%, #D4E0F0 100%);
    border: 2px solid #2C4F8A;
}
.mode-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }

/* 결과 카드 */
.job-card-js {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
    border-left: 4px solid #2E8B7A;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.job-card-cs {
    background: white;
    border-radius: 10px;
    padding: 1.2rem;
    border-left: 4px solid #2C4F8A;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* 개인정보 안내 배너 */
.privacy-banner {
    background: #FFF8E1;
    border: 1px solid #FFD54F;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #5D4037;
}

/* 섹션 헤더 */
.section-header-js {
    background: linear-gradient(90deg, #2E8B7A, #3DAA98);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
}
.section-header-cs {
    background: linear-gradient(90deg, #2C4F8A, #3D6BB0);
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    margin: 1.5rem 0 0.8rem 0;
}

/* 인사이트 박스 */
.insight-box {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    line-height: 1.8;
}

/* 모바일 반응형 — 좁은 뷰포트에서 컨테이너 패딩과 카드 폰트 축소 */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 1rem !important;
    }
    .mode-card {
        margin-bottom: 1rem;
    }
}

@media (max-width: 640px) {
    .job-card-js, .job-card-cs {
        padding: 0.8rem !important;
        font-size: 0.9rem;
    }
    .section-header-js, .section-header-cs {
        font-size: 0.95rem !important;
        padding: 0.5rem 0.8rem !important;
    }
    .insight-box {
        padding: 1rem !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── API 진단 헬퍼 ─────────────────────────────────────────────
# 사이드바 시스템 상태 진단에서 사용한다. 각 헬퍼는 (emoji, "name: status")
# 형태의 튜플을 반환해 호출 측에서 일관된 형식으로 렌더할 수 있도록 한다.
def _check_worknet_api(
    api_key: str,
    *,
    name: str,
    timeout: float = 5.0,
) -> tuple:
    """워크넷 215L11 직무데이터사전 API 헬스체크."""
    import requests

    if not api_key:
        return ("⚠️", f"{name}: 인증키 없음 (WORKNET_API_KEY)")
    try:
        r = requests.get(
            "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo215L11.do",
            params={
                "authKey": api_key,
                "word": "데이터",
                "limit": 1,
                "returnType": "JSON",
            },
            timeout=timeout,
        )
        if r.status_code == 200 and len(r.text) > 50:
            return ("✅", f"{name}: 정상")
        return ("⚠️", f"{name}: HTTP {r.status_code}")
    except Exception as e:
        return ("❌", f"{name}: {str(e)[:50]}")


def _check_worknet_212_api(
    url: str,
    name: str,
    params_extra: dict,
    *,
    timeout: float = 5.0,
) -> tuple:
    """워크넷 212번대 API (직업정보 212L01, 직업사전 212L50) 헬스체크.

    XML 응답을 정확히 파싱해 다음을 구분한다.
        1) <error> 태그 → 활용 미신청 / 인증 오류 등 명시적 실패
        2) <total> 노드 → 결과 건수와 함께 정상
        3) <jobList>·<dJobList> 노드 → 정상
        4) 그 외 빈 응답 → 정상이지만 결과 없음
    """
    import requests
    import xml.etree.ElementTree as ET

    api_key = (getattr(config, "WORKNET_API_KEY_JOB", "") or "").strip()
    if not api_key:
        return ("⚠️", f"{name}: 인증키 없음 (WORKNET_API_KEY_JOB)")
    try:
        params = {"authKey": api_key, **params_extra}
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return ("⚠️", f"{name}: HTTP {r.status_code}")

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            return ("⚠️", f"{name}: 응답 파싱 실패")

        error_node = root.find(".//error")
        if error_node is not None and (error_node.text or "").strip():
            return ("⚠️", f"{name}: {error_node.text.strip()[:30]}")

        total_node = root.find(".//total")
        if total_node is not None:
            total_text = (total_node.text or "0").strip()
            return ("✅", f"{name}: 정상 (총 {total_text}건)")

        list_nodes = root.findall(".//jobList") + root.findall(".//dJobList")
        if list_nodes:
            return ("✅", f"{name}: 정상")

        return ("✅", f"{name}: 정상 (빈 결과)")
    except Exception as e:
        return ("❌", f"{name}: {str(e)[:50]}")


def _check_openai_api(*, name: str = "OpenAI 임베딩") -> tuple:
    """OpenAI 임베딩 API 헬스체크."""
    api_key = (getattr(config, "OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return ("⚠️", f"{name}: 인증키 없음 (OPENAI_API_KEY)")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=["test"],
        )
        return ("✅", f"{name}: 정상")
    except Exception as e:
        return ("❌", f"{name}: {str(e)[:50]}")


def _check_claude_api(*, name: str = "Claude API") -> tuple:
    """Anthropic Claude API 헬스체크 (ping 토큰 10개)."""
    api_key = (getattr(config, "ANTHROPIC_API_KEY", "") or "").strip()
    if not api_key:
        return ("⚠️", f"{name}: 인증키 없음 (ANTHROPIC_API_KEY)")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        return ("✅", f"{name}: 정상")
    except Exception as e:
        return ("❌", f"{name}: {str(e)[:50]}")


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔗 JIB")
    st.markdown("**Job Insight Bridge**")
    st.markdown("---")
    st.markdown("""
고용노동부 공공데이터와 AI를 연결하여  
보유 역량 기반 직무 탐색을 지원합니다.
""")
    st.markdown("---")

    if "mode" in st.session_state and st.session_state.mode:
        mode_label = "👤 구직자 모드" if st.session_state.mode == "jobseeker" else "👥 상담사 모드"
        st.markdown(f"**현재 모드:** {mode_label}")
        if st.button("← 처음으로", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("---")

    st.markdown("**데이터 출처**")
    st.caption("• 한국고용정보원 고용24")
    st.caption("• 한국산업인력공단 NCS")
    st.caption("• 고용노동부 고용행정통계")
    st.caption("• 고용노동통계포털 임금통계")
    st.markdown("---")
    st.caption("본 서비스는 참고용 정보를 제공하며, 직업 선택의 최종 판단은 이용자 본인에게 있습니다.")

    # ── 시스템 상태 진단 (API 연동 점검) ────────────────────────
    st.markdown("---")
    with st.expander("🔧 시스템 상태", expanded=False):
        if st.button("API 연동 확인", key="api_check"):
            checks = [
                _check_worknet_api(
                    config.WORKNET_API_KEY,
                    name="L11 직무데이터사전",
                ),
                _check_worknet_212_api(
                    "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo212L01.do",
                    "L01 직업정보 (212)",
                    {
                        "returnType": "XML",
                        "target": "JOBCD",
                        "srchType": "K",
                        "keyword": "데이터분석가",
                    },
                ),
                _check_worknet_212_api(
                    "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo212L50.do",
                    "L50 직업사전 (212)",
                    {
                        "returnType": "XML",
                        "target": "dJobCD",
                        "srchType": "K",
                        "keyword": "데이터분석",
                        "startPage": 1,
                        "display": 1,
                    },
                ),
                _check_openai_api(),
                _check_claude_api(),
            ]

            for emoji, msg in checks:
                st.caption(f"{emoji} {msg}")

            # 핵심 API(L11 직무사전 + OpenAI + Claude) 상태로 종합 안내.
            # 212번대 워크넷은 활용 신청 단계 차이로 미연동 케이스가 정상이므로
            # 종합 판정에서는 제외한다.
            l11_ok = checks[0][0] == "✅"
            openai_ok = checks[-2][0] == "✅"
            claude_ok = checks[-1][0] == "✅"

            if l11_ok and claude_ok and openai_ok:
                st.success("핵심 API 모두 정상 연동 중입니다.")
            elif claude_ok and openai_ok:
                st.info("AI 핵심 기능은 정상입니다. 고용24 일부 API는 파일 기반으로 동작합니다.")
            else:
                st.warning("일부 API 연동에 문제가 있습니다. 관리자에게 문의하세요.")

# ── 세션 초기화 ───────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None

# ── 모드 선택 화면 ────────────────────────────────────────────
if st.session_state.mode is None:
    st.markdown("# 🔗 Job Insight Bridge")
    st.markdown("**보유 역량 기반 직무 탐색 플랫폼** — 고용노동부 공공데이터 + AI")
    st.markdown("---")
    st.markdown("### 이용 목적을 선택해주세요")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
<div class="mode-card mode-card-js">
<h3 style="color:#2E8B7A; margin-top:0;">👤 구직자 모드</h3>
<p style="color:#444; line-height:1.7;">
경험과 기술을 입력하면<br>
AI가 적합한 직업 TOP 3를 탐색합니다.<br><br>
역량이 어떻게 활용되는지,<br>
현실적인 고려사항은 무엇인지<br>
데이터 기반으로 안내합니다.
</p>
<p style="color:#2E8B7A; font-weight:600;">→ 구직자·취업 준비생</p>
</div>
""", unsafe_allow_html=True)
        if st.button("구직자 모드 시작 →", use_container_width=True,
                     type="primary", key="btn_js"):
            st.session_state.mode = "jobseeker"
            st.rerun()

    with col2:
        st.markdown("""
<div class="mode-card mode-card-cs">
<h3 style="color:#2C4F8A; margin-top:0;">👥 상담사 모드</h3>
<p style="color:#444; line-height:1.7;">
내담자 정보를 입력하면<br>
AI가 데이터 기반 상담 가이드를 생성합니다.<br><br>
역량 번역, 상담 포인트,<br>
열린 질문까지 자동으로<br>
제공합니다.
</p>
<p style="color:#2C4F8A; font-weight:600;">→ 직업상담사·고용센터</p>
</div>
""", unsafe_allow_html=True)
        if st.button("상담사 모드 시작 →", use_container_width=True,
                     key="btn_cs"):
            st.session_state.mode = "counselor"
            st.rerun()

    st.markdown("---")
    st.markdown("""
<div style="background:#F0F4FF; border-radius:8px; padding:1rem 1.2rem; font-size:0.9rem; color:#555;">
<b>🔒 개인정보 안내</b><br>
입력하신 정보는 AI 분석에만 사용되며 저장되지 않습니다.
이름, 연락처, 주민등록번호 등 식별 가능한 개인정보는 입력하지 마세요.
</div>
""", unsafe_allow_html=True)

elif st.session_state.mode == "jobseeker":
    from pages.jobseeker import render
    render()

elif st.session_state.mode == "counselor":
    from pages.counselor import render
    render()
