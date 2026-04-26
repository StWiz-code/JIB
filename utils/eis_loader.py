"""
utils/eis_loader.py — EIS / ELDS 데이터 연동 모듈

JIB의 고용 통계 보강 레이어. Phase 단계별로 점진 확장될 수 있도록
실제 작동 코드(Phase1)와 향후 구현 예정 코드(Phase2/3) 영역을 명확히 분리한다.

[Phase 구조]
  - Phase1 (현재 작동): EIS 수동 다운로드 CSV 로더
  - Phase2 (예정)     : EIS 공식 REST API 연동 (eis.work24.go.kr 개방 시)
  - Phase3 (예정)     : ELDS 기초데이터셋 (고용노동부 연구 협약 승인 후)

공개 함수:
  - load_eis_statistics(category)        : Phase1 정적 CSV 로더
  - load_elds_dataset(dataset_name)      : Phase3 stub (안내 메시지 + 빈 DataFrame)
  - merge_eis_to_master(master_df, ...)  : EIS 데이터를 마스터셋에 left join
  - get_eis_status()                     : 현재 Phase1/2/3 연동 상태 점검
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path
from typing import Dict

# `python utils/eis_loader.py` 직접 실행 시에도 프로젝트 루트의 config를
# 임포트할 수 있도록 sys.path를 미리 보정한다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

import config

# Phase1 CSV 로딩은 utils/data_loader의 인코딩 자동 감지 헬퍼를 재사용한다.
# 임포트 실패 시(부분 환경) pandas 기본 read_csv로 폴백한다.
try:
    from utils.data_loader import load_csv as _load_csv
except Exception:  # pragma: no cover — 안전 폴백
    _load_csv = None


# ──────────────────────────────────────────────────────────────────────────────
# 기능 1) load_eis_statistics — Phase1 정적 CSV 로더
# ──────────────────────────────────────────────────────────────────────────────
def load_eis_statistics(category: str = "job_demand") -> pd.DataFrame:
    """
    [Phase1 — 현재 작동]
    EIS 고용행정통계 수동 다운로드 CSV 로더.
    eis.work24.go.kr에서 다운로드한 통계 파일을 읽어 반환한다.

    [Phase2 예정]
    EIS 공식 API 연동으로 교체 (API 개방 시).
    # TODO: Phase2 — EIS REST API 연동
    # endpoint = config.ELDS_BASE_URL + "/api/statistics"
    # params = {"category": category, "serviceKey": config.PUBLIC_DATA_API_KEY}

    동작:
        1) config.EIS_STATISTICS_FILES[category] 경로의 CSV를 로드한다.
        2) 파일이 없거나 카테고리 미정의 시 빈 DataFrame + 안내 메시지를 반환한다.
        3) 성공 시 컬럼명 공백 제거 + object 컬럼의 숫자형 자동 변환을 수행한다.

    Args:
        category: config.EIS_STATISTICS_FILES의 키. 기본값은 'job_demand'.
                  현재 정의된 카테고리: 'job_demand'(직종별 구인구직), 'regional'(지역별 취업동향).

    Returns:
        pandas.DataFrame: 정제된 통계 데이터 (없으면 빈 DataFrame).
    """
    files = getattr(config, "EIS_STATISTICS_FILES", {}) or {}
    if category not in files:
        print(f"⚠️ EIS 카테고리 '{category}'를 config.EIS_STATISTICS_FILES에서 찾을 수 없습니다.")
        print(f"   사용 가능한 카테고리: {list(files.keys())}")
        return pd.DataFrame()

    file_path = Path(files[category])
    if not file_path.exists():
        print(f"⚠️ [{file_path.name}] 없음 — 빈 DataFrame 반환")
        print(f"   eis.work24.go.kr에서 '{category}' 통계를 다운로드해 다음 경로에 저장하세요:")
        print(f"   {file_path}")
        return pd.DataFrame()

    try:
        if _load_csv is not None:
            df = _load_csv(file_path)
        else:
            df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"⚠️ [{file_path.name}] 로드 실패 ({e}) — 빈 DataFrame 반환")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    # 숫자형 자동 변환 — 단, 컬럼명에 '코드'/'code'가 있으면 식별자로 보고 문자열 유지.
    # (직종코드/직업코드 같은 컬럼은 외형은 숫자여도 join 키로 쓰여야 함)
    for col in df.columns:
        if df[col].dtype != "object":
            continue
        if any(token in str(col).lower() for token in ("코드", "code")):
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    print(
        f"✅ [{file_path.name}] EIS 통계 로드 완료: "
        f"{len(df):,}행 × {df.shape[1]}컬럼 (category='{category}')"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 기능 2) load_elds_dataset — Phase3 stub
# ──────────────────────────────────────────────────────────────────────────────
def load_elds_dataset(dataset_name: str) -> pd.DataFrame:
    """
    [Phase3 예정 — 현재 미구현]
    ELDS 고용행정 기초데이터셋 로더.
    고용노동부 연구 협약 승인 후 연동 예정이다.

    사용 가능한 데이터셋 (32종 중 JIB 관련):
        - "job_posting"        : 워크넷 구인공고 기초데이터
        - "jobseeker"          : 구직자 이력 기초데이터
        - "employment_history" : 취업이력 기초데이터
        - "training"           : 직업훈련 참여이력 기초데이터

    [Phase3 구현 계획]
    # TODO: Phase3 — ELDS 연구 협약 후 구현
    # 1. ELDS 가상 분석환경 내 Python API 연동
    # 2. 또는 승인된 데이터셋 로컬 반출 후 처리
    # 3. config.ELDS_BASE_URL / config.ELDS_API_KEY 채워진 뒤 활성화

    Args:
        dataset_name: ELDS 데이터셋 이름.

    Returns:
        pandas.DataFrame: 항상 빈 DataFrame (Phase3 진입 전).
    """
    print(f"⏸ ELDS '{dataset_name}' 데이터셋은 Phase3(고용노동부 연구 협약 승인 후) 연동 예정입니다.")
    print(f"   현재는 공개 공공데이터 기반으로 서비스가 운영됩니다.")
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3) merge_eis_to_master — EIS 보완 데이터 병합
# ──────────────────────────────────────────────────────────────────────────────
def merge_eis_to_master(
    master_df: pd.DataFrame,
    category: str = "job_demand",
) -> pd.DataFrame:
    """
    [Phase2 예정]
    EIS 통계 데이터를 master_job_data에 병합한다.
    현재는 파일이 있으면 병합, 없으면 master_df 원본을 그대로 반환한다.

    조인 키 자동 선택 우선순위:
        (EIS 컬럼, master 컬럼)
        1. ('직종코드', '직업코드')
        2. ('직업코드', '직업코드')
        3. ('직종명',  '직업명')
        4. ('직업명',  '직업명')

    Args:
        master_df: build_master_job_data() 등으로 생성된 마스터 DataFrame.
        category : EIS 통계 카테고리 (config.EIS_STATISTICS_FILES의 키).

    Returns:
        pandas.DataFrame: 병합된 DataFrame. 병합 불가 시 master_df 원본 그대로 반환.
    """
    eis = load_eis_statistics(category)
    if eis.empty:
        print("⏸ EIS 보완 데이터 없음 — 기존 마스터 데이터 사용")
        return master_df

    candidates = [
        ("직종코드", "직업코드"),
        ("직업코드", "직업코드"),
        ("직종명",  "직업명"),
        ("직업명",  "직업명"),
    ]
    chosen = next(
        ((ec, mc) for ec, mc in candidates if ec in eis.columns and mc in master_df.columns),
        None,
    )
    if chosen is None:
        print("⏸ EIS와 master_df 사이에 공통 조인 키 없음 — 기존 마스터 데이터 사용")
        return master_df

    eis_col, master_col = chosen
    if eis_col != master_col:
        eis = eis.rename(columns={eis_col: master_col})

    # 양쪽 key 컬럼의 dtype을 문자열로 정렬해 dtype mismatch 방어 (코드 컬럼은 식별자이므로 안전)
    master_df = master_df.copy()
    master_df[master_col] = master_df[master_col].astype(str)
    eis[master_col] = eis[master_col].astype(str)

    merged = master_df.merge(eis, on=master_col, how="left", suffixes=("", "_eis"))
    print(
        f"✅ EIS '{category}' 데이터 병합 완료: '{master_col}' 기준 "
        f"{len(merged):,}행 × {merged.shape[1]}컬럼"
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 기능 4) get_eis_status — 연동 상태 점검 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def get_eis_status() -> Dict[str, object]:
    """
    EIS / ELDS 연동 상태를 점검해 dict로 반환한다.

    Returns:
        dict:
            - "phase1_files" (dict[str, bool]): EIS 정적 파일별 존재 여부.
            - "phase2_ready" (bool)           : EIS API 연동 준비 여부.
            - "phase3_ready" (bool)           : ELDS 연동 준비 여부.
            - "phase2_note"  (str)            : Phase2 상태 안내 문구.
            - "phase3_note"  (str)            : Phase3 상태 안내 문구.
    """
    files = getattr(config, "EIS_STATISTICS_FILES", {}) or {}
    phase1_files: Dict[str, bool] = {
        Path(path).name: Path(path).exists() for path in files.values()
    }

    elds_base = getattr(config, "ELDS_BASE_URL", "") or ""
    elds_key = getattr(config, "ELDS_API_KEY", "") or ""
    public_key = getattr(config, "PUBLIC_DATA_API_KEY", "") or ""

    phase2_ready: bool = bool(public_key) and bool(elds_base)
    phase3_ready: bool = bool(elds_key) and bool(elds_base)

    return {
        "phase1_files": phase1_files,
        "phase2_ready": phase2_ready,
        "phase3_ready": phase3_ready,
        "phase2_note": "EIS API 공식 개방 대기 중",
        "phase3_note": "고용노동부 연구 협약 승인 필요",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 기능 5) 단독 실행 시: 상태 점검 + Phase3 안내 메시지 확인
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[JIB] eis_loader 단독 실행 — Phase 상태 점검\n")

    status = get_eis_status()

    print("=== EIS / ELDS 연동 상태 ===")
    print("  Phase1 정적 파일:")
    if status["phase1_files"]:
        for name, exists in status["phase1_files"].items():
            mark = "✅" if exists else "⚠️"
            note = "존재" if exists else "없음 (eis.work24.go.kr에서 다운로드 필요)"
            print(f"    {mark} {name}  →  {note}")
    else:
        print("    (config.EIS_STATISTICS_FILES가 비어 있음)")

    print(f"  Phase2 준비됨 : {status['phase2_ready']}  ({status['phase2_note']})")
    print(f"  Phase3 준비됨 : {status['phase3_ready']}  ({status['phase3_note']})")

    print("\n=== ELDS 호출 시 안내 메시지 확인 ===")
    df = load_elds_dataset("job_posting")
    print(f"\n반환 DataFrame shape: {df.shape}  (Phase3 진입 전이므로 빈 DataFrame)")
