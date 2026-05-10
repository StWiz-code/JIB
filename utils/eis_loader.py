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
    EIS 고용행정통계 수동 다운로드 파일 로더.
    CSV 또는 XLSX 형식 모두 지원.
    5년치 데이터를 직종별 평균으로 집계하여 반환.

    [Phase2 예정]
    EIS 공식 API 연동으로 교체 (API 개방 시).
    # TODO: Phase2 — EIS REST API 연동
    # endpoint = config.ELDS_BASE_URL + "/api/statistics"
    # params = {"category": category, "serviceKey": config.PUBLIC_DATA_API_KEY}

    동작:
        1) config.EIS_STATISTICS_FILES[category] 경로의 CSV/XLSX 를 자동 탐지해 로드.
        2) 파일이 없거나 카테고리 미정의 시 빈 DataFrame + 안내 메시지 반환.
        3) XLSX 인 경우 피벗(연도×측정값) 형식을 long form 으로 풀고
           직종별로 5년치 평균 구인인원 / 평균 구직건수 → 평균구인배율_EIS 계산.

    Args:
        category: config.EIS_STATISTICS_FILES 의 키. 기본값 'job_demand'.

    Returns:
        pandas.DataFrame: 직종별 평균구인배율_EIS + 직종명_정제 컬럼 포함.
                          (구인구직 측정값이 없는 일반 통계는 raw DataFrame 그대로 반환)
    """
    base_path = config.EIS_STATISTICS_FILES.get(category, "")
    if not base_path:
        print(f"⚠️ EIS '{category}' 설정 경로 없음")
        return pd.DataFrame()

    # CSV/XLSX 자동 감지 — CSV 우선, 없으면 동일 경로의 .xlsx 폴백
    csv_path = Path(base_path)
    xlsx_path = Path(base_path.replace(".csv", ".xlsx"))

    if csv_path.exists():
        file_path = csv_path
        file_type = "csv"
    elif xlsx_path.exists():
        file_path = xlsx_path
        file_type = "xlsx"
    else:
        print(f"⚠️ [{Path(base_path).name}] 없음 — eis.work24.go.kr에서 다운로드 필요")
        print(f"   저장 경로: {base_path} (또는 .xlsx)")
        return pd.DataFrame()

    try:
        if file_type == "csv":
            import chardet
            with open(file_path, "rb") as f:
                enc = chardet.detect(f.read(10000))["encoding"]
            df = pd.read_csv(file_path, encoding=enc)
        else:
            # EIS XLSX 구조 (employment_demand_supply_5y, 구 직종별_구인구직현황):
            #   row 0~12: 메타데이터 (출처·다운로드시간·필터·조회기간)
            #   row 13  : 헤더 ['마감년월', '직종_중분류', '직종_소분류',
            #                   '구인인원(월)', '구직건수(월)', '취업건수(월)']
            #   row 14+ : 데이터 행 (총계 / 월별·중분류·소분류 단위)
            df_raw = pd.read_excel(file_path, header=None, engine="openpyxl")
            print(f"  📂 EIS XLSX 로드: {df_raw.shape}")

            # 헤더 행 자동 감지 — 한 행에 '마감년월' 과 '구인'/'구직' 키워드가
            # 동시에 등장하는 행을 찾는다.
            header_idx = None
            for i in range(min(50, len(df_raw))):
                row_vals = [str(v) for v in df_raw.iloc[i].dropna()]
                if any("마감년월" in v for v in row_vals) and any(
                    "구인" in v for v in row_vals
                ):
                    header_idx = i
                    break

            if header_idx is None:
                print("⚠️ EIS XLSX 헤더 행 자동 감지 실패 — 빈 DataFrame 반환")
                return pd.DataFrame()

            header_row = df_raw.iloc[header_idx]
            col_index: Dict[str, int] = {}
            for j, v in enumerate(header_row):
                if pd.isna(v):
                    continue
                name = str(v).strip()
                if "마감년월" in name:
                    col_index["기간"] = j
                elif "중분류" in name:
                    col_index["중분류"] = j
                elif "소분류" in name:
                    col_index["소분류"] = j
                elif "구인" in name:
                    col_index["구인"] = j
                elif "구직" in name:
                    col_index["구직"] = j
                elif "취업" in name:
                    col_index["취업"] = j

            required = {"중분류", "소분류", "구인", "구직"}
            missing = required - col_index.keys()
            if missing:
                print(f"⚠️ EIS XLSX 필수 컬럼 누락: {missing} — 빈 DataFrame 반환")
                return pd.DataFrame()

            df_data = df_raw.iloc[header_idx + 1:].copy()
            df_data.columns = range(len(df_data.columns))
            print(f"  📂 데이터 영역: {df_data.shape} (헤더 row {header_idx} 이후)")

            # Long form 레코드 생성: 행마다 직종(소분류>중분류) + 구인/구직 값 추출
            def _strip_prefix(name: str) -> str:
                """`'2018직종_관리직(...)'` → `'관리직(...)'` (prefix 제거)."""
                import re
                return re.sub(r"^\d{4}직종_", "", name).strip()

            records = []
            skipped_total = 0
            for _, row in df_data.iterrows():
                # 마감년월: NaN 이거나 '총계' / '... 전체' 인 행은 합계 라인이므로 제외.
                # (단순 NaN 인 소분류 데이터 행은 통과시켜야 한다.)
                기간_raw = row.iloc[col_index["기간"]] if "기간" in col_index else None
                if pd.notna(기간_raw):
                    기간 = str(기간_raw).strip()
                    if 기간 == "총계" or "전체" in 기간:
                        skipped_total += 1
                        continue

                중분류_raw = row.iloc[col_index["중분류"]]
                소분류_raw = row.iloc[col_index["소분류"]]
                중분류 = (
                    str(중분류_raw).strip()
                    if pd.notna(중분류_raw)
                    else ""
                )
                소분류 = (
                    str(소분류_raw).strip()
                    if pd.notna(소분류_raw)
                    else ""
                )

                # 합계 라인: 중분류명이 '... 전체' 로 끝나면 (소분류 NaN) 중분류 단위 합계.
                if "전체" in 중분류 and not 소분류:
                    skipped_total += 1
                    continue
                if "전체" in 소분류:
                    skipped_total += 1
                    continue

                # 가장 구체적인 직종명: 소분류 우선, 없으면 중분류
                if 소분류 and 소분류 not in ("nan", "None"):
                    직종명 = _strip_prefix(소분류)
                elif 중분류 and 중분류 not in ("nan", "None"):
                    직종명 = _strip_prefix(중분류).replace(" 전체", "")
                else:
                    continue

                if not 직종명 or "전체" in 직종명:
                    continue

                try:
                    구인_v = float(row.iloc[col_index["구인"]])
                    구직_v = float(row.iloc[col_index["구직"]])
                except (TypeError, ValueError):
                    continue

                records.append({
                    "직종명": 직종명,
                    "구인인원": 구인_v,
                    "구직건수": 구직_v,
                })

            df = pd.DataFrame(records)
            print(f"  📊 Long form 변환: {len(df)}행 (총계/전체 {skipped_total}행 제외)")
            # 다운스트림 호환 위해 측정값 컬럼은 그대로 두지 않고
            # 아래 분기에서 직접 평균구인배율_EIS 를 계산해 반환한다.
            if df.empty:
                print("⚠️ EIS XLSX long-form 변환 결과가 비어 있음 — 빈 DataFrame 반환")
                return pd.DataFrame()

            # 직종별 5년치 평균 → 평균구인배율_EIS
            grouped = df.groupby("직종명").agg(
                평균구인인원=("구인인원", "mean"),
                평균구직건수=("구직건수", "mean"),
            )
            grouped["평균구인배율_EIS"] = (
                grouped["평균구인인원"] / grouped["평균구직건수"]
            ).replace([float("inf"), float("nan")], 0).round(3)
            grouped = grouped.reset_index()
            grouped["직종명_정제"] = grouped["직종명"]
            print(
                f"✅ EIS '{category}' 구인배율 계산 완료: "
                f"{len(grouped)}개 직종"
            )
            print(
                grouped[["직종명_정제", "평균구인배율_EIS"]]
                .head(10)
                .to_string(index=False)
            )
            return grouped

    except Exception as e:
        print(f"⚠️ EIS 파일 로드 실패: {e}")
        return pd.DataFrame()

    # 측정값 컬럼이 있으면 직종별 5년치 평균 → 구인배율_EIS 계산
    if "측정값" in df.columns and "직종명" in df.columns:
        구인 = df[df["측정값"].str.contains("구인", na=False)].groupby("직종명")["값"].mean()
        구직 = df[df["측정값"].str.contains("구직", na=False)].groupby("직종명")["값"].mean()

        result = pd.DataFrame({"평균구인인원": 구인, "평균구직건수": 구직}).dropna()
        result["평균구인배율_EIS"] = (
            result["평균구인인원"] / result["평균구직건수"]
        ).replace([float("inf"), float("nan")], 0).round(3)
        result = result.reset_index()

        # 직종명 정제 (앞 숫자 코드 제거: "01 관리직" → "관리직")
        result["직종명_정제"] = (
            result["직종명"].str.replace(r"^\d{2,3}\s*", "", regex=True).str.strip()
        )

        print(f"✅ EIS '{category}' 구인배율 계산 완료: {len(result)}개 직종")
        print(result[["직종명_정제", "평균구인배율_EIS"]].head(10).to_string(index=False))
        return result

    # 기본 전처리 (컬럼명 공백 제거)
    if hasattr(df.columns, "str"):
        df.columns = df.columns.str.strip()
    print(f"✅ EIS '{category}' 로드 완료: {len(df)}행")
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
