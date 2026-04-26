"""
JIB(Job_Insight_Bridge) 프로젝트 초기 구조 자동 생성 스크립트
- 폴더와 빈 파일을 한 번에 생성하고, 각 파일 상단에 역할 주석을 미리 삽입한다.
- 최상위 JIB 폴더는 이미 존재한다고 가정하므로, 본 스크립트는 그 하위 항목만 생성한다.
실행 방법: (JIB 폴더 안에서) python setup_project.py
"""

from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 1) 생성할 폴더 목록
# ──────────────────────────────────────────────────────────────────────────────
DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "data/eis",          # EIS 고용행정통계 수동 다운로드 파일 보관
    "embeddings",
    "utils",
    "pages",
    "prompts",
]


# ──────────────────────────────────────────────────────────────────────────────
# 2) 생성할 파일과 초기 콘텐츠 (역할 설명 주석 포함)
# ──────────────────────────────────────────────────────────────────────────────
FILES = {
    # ── 루트 설정/실행 파일 ──────────────────────────────────────────────────
    ".env": (
        "# 환경 변수 파일\n"
        "# - Anthropic, OpenAI 등 외부 API 키를 보관한다.\n"
        "# - 절대 Git에 커밋하지 않는다 (.gitignore에 등록되어 있음).\n"
        "# 예시:\n"
        "# ANTHROPIC_API_KEY=your_key_here\n"
        "# OPENAI_API_KEY=your_key_here\n"
    ),
    ".gitignore": (
        "# 환경 변수 / 비밀키\n"
        ".env\n\n"
        "# 가상환경\n"
        "venv/\n\n"
        "# 파이썬 캐시\n"
        "__pycache__/\n"
        "*.pyc\n\n"
        "# 원천 데이터 / 임베딩 산출물\n"
        "data/raw/\n"
        "embeddings/\n"
    ),
    "requirements.txt": (
        "streamlit\n"
        "pandas\n"
        "numpy\n"
        "openai\n"
        "anthropic\n"
        "python-dotenv\n"
        "scikit-learn\n"
        "plotly\n"
        "requests\n"
        "openpyxl\n"
        "chardet\n"
    ),
    "app.py": (
        '"""\n'
        "JIB(Job_Insight_Bridge) Streamlit 메인 진입점.\n"
        "- 사이드바에서 구직자(jobseeker) / 상담사(counselor) 페이지로 라우팅한다.\n"
        "- 전역 설정(config.py)과 유틸리티(utils/*) 모듈을 불러와 초기화한다.\n"
        '"""\n'
    ),
    "config.py": (
        '"""\n'
        "프로젝트 전역 설정 모듈.\n"
        "- API 키 로드(.env), 경로 상수, 모델명, 하이퍼파라미터 등을 한 곳에서 관리한다.\n"
        '"""\n'
    ),
    # ── embeddings/ ─────────────────────────────────────────────────────────
    "embeddings/job_embeddings.csv": "",  # 초기에는 빈 CSV
    # ── utils/ ──────────────────────────────────────────────────────────────
    "utils/__init__.py": (
        "# utils 패키지 초기화 파일\n"
        "# - 데이터 로딩, 임베딩, 매칭, 필터링, 생성 모듈을 묶는 패키지.\n"
    ),
    "utils/data_loader.py": (
        '"""\n'
        "범용 데이터 로더 모듈.\n"
        "- NCS 직무 데이터, 자격/훈련 데이터 등 정적 파일을 일관된 형태로 읽어 온다.\n"
        '"""\n'
    ),
    "utils/eis_loader.py": (
        '"""\n'
        "EIS(고용행정통계) / ELDS(고용행정 기초데이터셋) 로더 모듈.\n"
        "- Phase1: EIS 수동 다운로드 CSV/Excel을 정적 파일로 로딩한다.\n"
        "- Phase2: EIS Open API 연동 방식으로 교체 예정.\n"
        "- Phase3: ELDS는 승인 후 연구 협약 기반으로 연동 예정.\n"
        '"""\n'
        "\n"
        "import pandas as pd\n"
        "\n"
        "\n"
        "def load_eis_statistics(file_path: str) -> pd.DataFrame:\n"
        '    """\n'
        "    EIS 고용행정통계 수동 다운로드 CSV 로더 (현재 정적 파일 방식).\n"
        "    Phase2: API 연동 방식으로 교체 예정.\n"
        "\n"
        "    Args:\n"
        "        file_path: EIS에서 다운로드한 통계 파일 경로 (CSV/XLSX).\n"
        "\n"
        "    Returns:\n"
        "        pandas.DataFrame: 정제된 통계 데이터프레임.\n"
        '    """\n'
        "    raise NotImplementedError(\n"
        '        "load_eis_statistics()는 아직 구현되지 않았습니다. '
        'data/eis/ 경로의 파일을 읽도록 구현하세요."\n'
        "    )\n"
        "\n"
        "\n"
        "def load_elds_dataset(dataset_name: str) -> pd.DataFrame:\n"
        '    """\n'
        "    ELDS 고용행정 기초데이터셋 로더.\n"
        "    Phase3: 승인 후 연구 협약 기반 연동 예정.\n"
        "    현재는 안내 메시지를 출력하고 NotImplementedError를 발생시킨다.\n"
        "\n"
        "    Args:\n"
        "        dataset_name: ELDS 데이터셋 이름.\n"
        "\n"
        "    Returns:\n"
        "        pandas.DataFrame: ELDS 데이터프레임 (구현 후).\n"
        '    """\n'
        '    print("[ELDS] 현재 ELDS 데이터셋은 연구 협약 승인 후 이용 가능합니다. (Phase3 예정)")\n'
        "    raise NotImplementedError(\n"
        '        "load_elds_dataset()는 Phase3에서 연구 협약 승인 후 구현될 예정입니다."\n'
        "    )\n"
    ),
    "utils/embedder.py": (
        '"""\n'
        "임베딩 생성 모듈.\n"
        "- 직무/이력서 텍스트를 벡터로 임베딩하여 embeddings/job_embeddings.csv 등에 저장한다.\n"
        '"""\n'
    ),
    "utils/matcher.py": (
        '"""\n'
        "매칭 모듈.\n"
        "- 사용자(이력서/스킬)와 직무 임베딩 간 유사도를 계산해 추천 후보를 정렬한다.\n"
        '"""\n'
    ),
    "utils/filter.py": (
        '"""\n'
        "필터 모듈.\n"
        "- 지역, 경력, 학력, 산업 분야 등 조건 기반으로 후보 직무를 필터링한다.\n"
        '"""\n'
    ),
    "utils/claude_generator.py": (
        '"""\n'
        "Claude(Anthropic) 기반 LLM 생성 모듈.\n"
        "- prompts/ 폴더의 템플릿을 불러와 인사이트/상담 스크립트를 생성한다.\n"
        '"""\n'
    ),
    # ── pages/ ──────────────────────────────────────────────────────────────
    "pages/__init__.py": (
        "# pages 패키지 초기화 파일\n"
        "# - Streamlit multipage 구성을 위한 페이지 모듈을 모은다.\n"
    ),
    "pages/jobseeker.py": (
        '"""\n'
        "구직자(Jobseeker) 페이지.\n"
        "- 이력서 입력 → 직무 매칭 → 인사이트 카드 출력 흐름을 담당한다.\n"
        '"""\n'
    ),
    "pages/counselor.py": (
        '"""\n'
        "상담사(Counselor) 페이지.\n"
        "- 내담자 정보 기반 직무 추천 및 상담 스크립트 생성 화면을 담당한다.\n"
        '"""\n'
    ),
    # ── prompts/ ────────────────────────────────────────────────────────────
    "prompts/system_prompt.txt": (
        "# system_prompt.txt\n"
        "# Claude/LLM 호출 시 공통으로 주입되는 시스템 프롬프트.\n"
        "# - JIB의 역할, 톤앤매너, 답변 형식 등을 정의한다.\n"
    ),
    "prompts/parse_resume.txt": (
        "# parse_resume.txt\n"
        "# 자유 형식 이력서를 구조화 데이터(스킬, 경력, 학력 등)로 파싱하는 프롬프트.\n"
    ),
    "prompts/translate_ncs.txt": (
        "# translate_ncs.txt\n"
        "# 사용자 표현을 NCS 직무 표준 용어로 번역(매핑)하는 프롬프트.\n"
    ),
    "prompts/generate_insight.txt": (
        "# generate_insight.txt\n"
        "# 매칭 결과를 바탕으로 구직자용 직무 인사이트 카드를 생성하는 프롬프트.\n"
    ),
    "prompts/counselor_script.txt": (
        "# counselor_script.txt\n"
        "# 상담사용 직무 추천/상담 스크립트를 생성하는 프롬프트.\n"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 3) 실행 로직
# ──────────────────────────────────────────────────────────────────────────────
def create_directories(base: Path) -> None:
    """필요한 모든 하위 폴더를 생성한다 (이미 있으면 통과)."""
    for rel_dir in DIRECTORIES:
        target = base / rel_dir
        target.mkdir(parents=True, exist_ok=True)
        print(f"[DIR ] {target}")


def create_files(base: Path) -> None:
    """파일을 생성하되, 이미 존재하면 덮어쓰지 않는다."""
    for rel_path, content in FILES.items():
        target = base / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"[SKIP] {target} (이미 존재)")
            continue
        target.write_text(content, encoding="utf-8")
        print(f"[FILE] {target}")


def main() -> None:
    base = Path(__file__).resolve().parent
    print(f"[BASE] {base}\n")
    create_directories(base)
    print()
    create_files(base)
    print("\n[DONE] JIB 프로젝트 구조 생성 완료.")


if __name__ == "__main__":
    main()
