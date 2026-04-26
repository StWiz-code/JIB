import streamlit as st

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
</style>
""", unsafe_allow_html=True)

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
    st.caption("• 한국고용정보원 워크넷")
    st.caption("• 한국산업인력공단 NCS")
    st.caption("• 고용노동부 고용행정통계")
    st.caption("• 고용노동통계포털 임금통계")
    st.markdown("---")
    st.caption("본 서비스는 참고용 정보를 제공하며, 직업 선택의 최종 판단은 이용자 본인에게 있습니다.")

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
