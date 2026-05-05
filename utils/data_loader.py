"""
범용 데이터 로더 모듈.

- NCS 직무 데이터, 자격/훈련 데이터, 직업전망 5종, 고용동향, 부족인력, 임금정보 등
  정적 파일과 공공데이터 API를 일관된 형태로 읽어 온다.
- 한국 공공데이터 CSV의 인코딩 혼재 문제(cp949/euc-kr/utf-8/utf-8-sig)를
  chardet 기반 자동 감지 + 폴백 체인으로 해결한다.
- 모든 외부 데이터 함수는 파일/네트워크 실패 시 빈 DataFrame을 반환해
  파이프라인 전체가 중단되지 않도록 한다.

주요 함수:
  핵심 로딩 헬퍼
    - load_csv(path)                        : 인코딩 자동 감지로 단일 CSV 로드
  직업전망 5종 (Phase1 핵심)
    - load_job_prospect_bundle()            : 직업전망 관련 5개 CSV를 dict로 반환
    - merge_job_prospect_bundle(bundle, key): 번들을 공통 키로 머지
  보조 데이터 로더
    - load_employment_trend()               : 고용동향 → 평균구인배율
    - load_labor_shortage()                 : 부족인력 → 평균부족률
    - load_job_wage_from_api()              : 공공데이터포털 임금정보 API
    - load_top_jobs_by_major()              : 전공별 진출직업 상위
    - load_first_job_entry()                : 졸업 후 취업률
    - load_ncs_career_path()                : NCS 경로/전직 정보
  마스터 데이터셋
    - build_master_job_data()               : RAG 엔진용 마스터 CSV 생성 (기준)
    - build_master_with_eis(eis_categories) : 기준 마스터 + EIS 보완 통합 파이프라인
  EIS 보완 (Phase2 예정)
    - load_eis_supplement()                 : EIS 고용행정통계 수동 파일 로더

지원 스키마:
  - KNOW(한국직업정보) 직업전망 데이터: prospect = (KNOW직업코드, 직업전망내용)
    직업명은 직업전망내용 텍스트에서 정규식으로 추출하고, 전망점수는 텍스트에서
    PROSPECT_SCORE_MAP 키워드를 길이 내림차순으로 매칭해 결정한다.
    education_major 의 wide format(35개 전공계열 컬럼)에서 대계열 prefix별로
    합산해 argmax 카테고리를 '주요전공계열'로 채택한다.
    similar_names 는 직업코드별로 group concat 한다.
    subclass / midclass 는 KNOW 분류 코드 도메인이 다르므로 master 결합 대상에서
    제외한다 (별도 reference로만 보관).
  - 일반 스키마: prospect = (직업코드, 직업명, 전망등급) 형태도 자동 감지해 처리.
"""

from __future__ import annotations

import re
import sys as _sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

# `python utils/data_loader.py`로 직접 실행하는 경우에도
# 프로젝트 루트의 config 모듈을 찾을 수 있도록 sys.path를 미리 보정한다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import chardet
import pandas as pd
import requests

import config


# ──────────────────────────────────────────────────────────────────────────────
# 직업전망 번들 구성 (논리키 → config의 파일 경로 상수)
# ──────────────────────────────────────────────────────────────────────────────
JOB_PROSPECT_BUNDLE_PATHS: Dict[str, str] = {
    "prospect": config.JOB_PROSPECT_FILE,                  # 직업전망 (전망등급)
    "education_major": config.JOB_EDUCATION_MAJOR_FILE,    # 교육훈련/학력 전공분포
    "similar_names": config.JOB_SIMILAR_NAMES_FILE,        # KNOW 유사직업명
    "subclass": config.JOB_SUBCLASS_FILE,                  # 직업세세분류
    "midclass": config.JOB_MIDCLASS_FILE,                  # 직업중분류
}


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼: 인코딩 감지 / 안전 CSV 로딩
# ──────────────────────────────────────────────────────────────────────────────
def _detect_encoding(file_path: Union[str, Path], sample_size: int = 65536) -> str:
    """
    chardet으로 파일 인코딩을 자동 감지한다.

    한국 공공데이터 CSV는 EUC-KR / CP949 / UTF-8(-sig)이 혼재하므로
    encoding을 사전에 강제하면 깨지는 경우가 잦다.

    Args:
        file_path: 인코딩을 감지할 파일 경로.
        sample_size: 감지에 사용할 바이트 수 (앞부분만 읽음).

    Returns:
        감지된 인코딩 문자열. EUC-KR류는 호환성을 위해 'cp949'로 보정한다.
    """
    with open(file_path, "rb") as f:
        raw = f.read(sample_size)
    if not raw:
        return "utf-8"

    detected = chardet.detect(raw)
    encoding = (detected.get("encoding") or "utf-8").lower()
    if encoding in {"euc-kr", "ks_c_5601-1987"}:
        return "cp949"
    return encoding


def load_csv(file_path: Union[str, Path], **read_csv_kwargs) -> pd.DataFrame:
    """
    인코딩 자동 감지 + 폴백 체인으로 CSV를 안전하게 로드한다.

    감지 결과로 먼저 시도하고, 실패하면 ('utf-8-sig', 'cp949', 'utf-8') 순으로 재시도한다.
    pandas.read_csv에 전달하고 싶은 추가 인자는 키워드로 그대로 넘길 수 있다.

    Args:
        file_path: CSV 파일 경로.
        **read_csv_kwargs: pandas.read_csv에 그대로 전달되는 추가 인자.

    Returns:
        pandas.DataFrame: 로드된 데이터프레임.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때.
        UnicodeDecodeError: 모든 인코딩 시도가 실패했을 때.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"[data_loader] CSV 파일을 찾을 수 없습니다: {path}")

    encodings_to_try: List[str] = []
    try:
        encodings_to_try.append(_detect_encoding(path))
    except Exception:
        pass
    for fallback in ("utf-8-sig", "cp949", "utf-8"):
        if fallback not in encodings_to_try:
            encodings_to_try.append(fallback)

    last_err: Exception | None = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, **read_csv_kwargs)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue

    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1,
        f"인코딩 자동 감지 실패: {path} (시도한 인코딩: {encodings_to_try})",
    ) from last_err


# ──────────────────────────────────────────────────────────────────────────────
# 직업전망 번들 로더 / 머저
# ──────────────────────────────────────────────────────────────────────────────
def load_job_prospect_bundle(strict: bool = True, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    직업전망 관련 5개 CSV를 한 번에 로드해 dict로 반환한다.

    구성:
        - "prospect"        : 직업전망 (전망등급)                [Phase1 필수]
        - "education_major" : 교육훈련 및 학력 전공분포          [Phase1 필수]
        - "similar_names"   : KNOW 유사직업명                    [Phase1 권장]
        - "subclass"        : 직업세세분류                       [Phase1 권장]
        - "midclass"        : 직업중분류                         [Phase1 보조]

    Args:
        strict: True이면 누락된 파일이 있을 때 FileNotFoundError 발생.
                False이면 누락 파일은 건너뛰고 결과 dict에서 해당 키를 제외한다.
        verbose: True이면 각 파일의 행/열 수를 표준 출력으로 안내한다.

    Returns:
        dict[str, pandas.DataFrame]: 논리키 → DataFrame 매핑.

    Raises:
        FileNotFoundError: strict=True인데 누락 파일이 있을 때.
    """
    bundle: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    for key, path in JOB_PROSPECT_BUNDLE_PATHS.items():
        try:
            df = load_csv(path)
        except FileNotFoundError:
            if strict:
                raise
            missing.append(path)
            if verbose:
                print(f"[load_job_prospect_bundle] [SKIP] {key}: 파일 없음 ({path})")
            continue

        bundle[key] = df
        if verbose:
            print(f"[load_job_prospect_bundle] [OK]  {key}: {len(df):,} rows × {df.shape[1]} cols  ←  {path}")

    if missing and verbose:
        print(f"[load_job_prospect_bundle] 누락 파일 {len(missing)}개: {missing}")

    return bundle


def merge_job_prospect_bundle(
    bundle: Dict[str, pd.DataFrame],
    key: Union[str, Sequence[str]],
    how: str = "left",
    base: str = "prospect",
) -> pd.DataFrame:
    """
    번들 dict의 DataFrame들을 공통 키 컬럼 기준으로 차례로 머지한다.

    동작:
        1) `base`(기본 'prospect')를 시작 DataFrame으로 둔다.
        2) 나머지 DataFrame들을 차례로 left-join한다.
        3) 동일 컬럼명 충돌 시 우측에 '_<논리키>' suffix를 붙여 추적 가능하게 한다.

    Args:
        bundle: load_job_prospect_bundle()의 반환값.
        key: 조인 키 컬럼명(또는 리스트). 모든 DataFrame에 존재해야 한다.
        how: pandas merge 방식 ('left' | 'inner' | 'outer' | 'right').
        base: 시작점으로 사용할 논리키. 기본 'prospect'.

    Returns:
        pandas.DataFrame: 머지된 단일 데이터프레임.

    Raises:
        KeyError: base가 bundle에 없거나 키 컬럼이 누락된 DataFrame이 있을 때.
    """
    if base not in bundle:
        raise KeyError(
            f"[merge_job_prospect_bundle] base='{base}'가 bundle에 없습니다. "
            f"보유 키: {list(bundle)}"
        )

    keys: List[str] = [key] if isinstance(key, str) else list(key)

    for name, df in bundle.items():
        missing_keys = [k for k in keys if k not in df.columns]
        if missing_keys:
            raise KeyError(
                f"[merge_job_prospect_bundle] '{name}' DataFrame에 키 컬럼 누락: "
                f"{missing_keys} (보유 컬럼: {list(df.columns)})"
            )

    merged: pd.DataFrame = bundle[base].copy()
    for name, df in bundle.items():
        if name == base:
            continue
        merged = merged.merge(df, on=keys, how=how, suffixes=("", f"_{name}"))

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼: 안전 로딩 / 컬럼 자동 매핑 / 표준 로그
# ──────────────────────────────────────────────────────────────────────────────
def _empty_df(columns: Union[List[str], None] = None) -> pd.DataFrame:
    """규칙에 맞는 빈 DataFrame을 반환한다 (컬럼명만 미리 지정 가능)."""
    return pd.DataFrame(columns=columns or [])


def _log_ok(filename: str, df: pd.DataFrame) -> None:
    """공통 규칙: '✅ [파일명] 로드 완료: N행' 표준 로그."""
    print(f"✅ [{filename}] 로드 완료: {len(df):,}행")


def _log_missing(filename: str) -> None:
    """공통 규칙: '⚠️ [파일명] 없음 — 빈 DataFrame 반환' 표준 로그."""
    print(f"⚠️ [{filename}] 없음 — 빈 DataFrame 반환")


def _safe_read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    공통 규칙에 따라 CSV를 안전하게 로드한다.

    - 파일 없음 → 경고 출력 후 빈 DataFrame 반환.
    - 인코딩 자동 감지(chardet) → 실패 시 'cp949' → 'utf-8-sig' 폴백.
    - 성공/실패 시 표준 로그 출력.
    """
    path = Path(file_path)
    if not path.exists():
        _log_missing(path.name)
        return _empty_df()
    try:
        df = load_csv(path)
        _log_ok(path.name, df)
        return df
    except Exception as e:
        print(f"⚠️ [{path.name}] 로드 실패 ({e}) — 빈 DataFrame 반환")
        return _empty_df()


def _resolve_excel_path(base_dir: Union[str, Path], stem: str) -> Optional[Path]:
    """
    Excel 파일 경로를 자동 탐지한다.

    공공데이터에서 같은 데이터셋이 .xlsx/.xls 어느 형식으로든 제공될 수 있으므로,
    `stem`(확장자 제외 이름) 기준으로 .xlsx → .xls 우선순위로 실제 존재하는 파일을
    찾는다. 둘 다 없으면 None을 반환한다.
    """
    base = Path(base_dir)
    for ext in (".xlsx", ".xls"):
        candidate = base / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _safe_read_excel(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    공통 규칙에 따라 Excel을 안전하게 로드한다.

    엔진 fallback 체인 (한국 공공데이터의 "확장자 ≠ 실제 형식" 패턴 대응):
        1) 확장자 기반 1차 엔진 시도
            - .xlsx → openpyxl
            - .xls  → xlrd (설치 필요; requirements.txt 참조)
        2) 1차 실패 시 다른 엔진 시도 (.xls 확장자에 .xlsx 콘텐츠 / 그 반대)
        3) 둘 다 실패 시 read_html 시도 (HTML 형식 .xls 대응)
        4) 모두 실패 시 친절한 안내 메시지 출력 후 빈 DataFrame 반환.
    """
    path = Path(file_path)
    if not path.exists():
        _log_missing(path.name)
        return _empty_df()

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        engines: List[str] = ["openpyxl", "xlrd"]
    elif suffix == ".xls":
        engines = ["xlrd", "openpyxl"]
    else:
        engines = ["openpyxl", "xlrd"]

    last_error: Optional[Exception] = None
    for engine in engines:
        try:
            df = pd.read_excel(path, engine=engine)
            if engine != engines[0]:
                print(
                    f"  ↪ [{path.name}] '{engines[0]}' 엔진 실패 → '{engine}' 엔진으로 fallback 성공"
                )
            _log_ok(path.name, df)
            return df
        except ImportError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    try:
        tables = pd.read_html(str(path))
        if tables:
            df = tables[0]
            print(f"  ↪ [{path.name}] HTML 형식으로 감지 → read_html로 fallback 성공")
            _log_ok(path.name, df)
            return df
    except Exception as e:
        last_error = e

    print(
        f"⚠️ [{path.name}] 모든 엔진(xlrd/openpyxl/read_html)에서 로드 실패 "
        f"({last_error}) — 빈 DataFrame 반환\n"
        f"   해법: 원본 파일을 Excel에서 열어 '.xlsx'로 다시 저장하거나, "
        f"공공데이터포털에서 .xlsx 형식 파일을 다시 다운로드하세요."
    )
    return _empty_df()


def _pick_column(
    df: pd.DataFrame,
    exact: str,
    keywords: Union[Sequence[str], None] = None,
) -> Union[str, None]:
    """
    DataFrame에서 컬럼명을 찾는다.

    1) `exact`와 정확히 일치하는 컬럼이 있으면 그 컬럼명을 반환.
    2) 없으면 `keywords`를 **순서대로** 시도하며, 해당 키워드를 부분 문자열로
       포함한 첫 번째 컬럼명을 반환한다. 따라서 호출자는 더 구체적인 키워드를
       앞에 배치해야 한다 (예: ['상위25', '상위'] 처럼 좁은→넓은 순).
    3) 모두 실패하면 None.

    공공데이터 CSV는 같은 의미의 컬럼이라도 표기가 조금씩 달라
    이 헬퍼로 키워드 기반 대체 매핑을 수행한다.
    """
    if exact in df.columns:
        return exact
    if keywords:
        for kw in keywords:
            for col in df.columns:
                if kw in str(col):
                    return col
    return None


def _rename_picked(df: pd.DataFrame, mapping: Dict[str, Union[str, None]]) -> pd.DataFrame:
    """`{표준명: 원본컬럼명}` 매핑으로 DataFrame 컬럼을 표준명으로 리네임한다."""
    rename_map = {orig: std for std, orig in mapping.items() if orig and orig in df.columns}
    return df.rename(columns=rename_map)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 1) 고용동향: 평균구인배율 산출 (EIS 피벗 보고서 형식)
# ──────────────────────────────────────────────────────────────────────────────
def load_employment_trend() -> pd.DataFrame:
    """
    EIS 피벗 보고서 형식의 employment_trend Excel 을 로드해 직종 중분류별 평균구인배율을 산출한다.

    원본 구조 (EIS 직업안정정보망 피벗 다운로드):
        - 행0           : 비어있음
        - 행1 (index=1) : 날짜 헤더 'yyyy년 mm월' — 같은 날짜가 3컬럼마다 반복(구인/구직/취업 묶음)
        - 행2 (index=2) : 측정값명 — '구인인원(월)', '구직건수(월)', '취업건수(월)' 반복
        - 행3~          : 실제 데이터 행
        - col0          : 산업_대분류
        - col1          : 직종_중분류  ('2025직종_NN_NN_…' 또는 '2025직종_NN_NN_전체')
        - col2          : 직종_소분류
        - col3~         : (날짜 × 측정값) 측정 데이터

    처리 흐름:
        1) header=None 으로 전체 시트 읽기.
        2) 행1 의 날짜 forward-fill 후 등장 순서대로 unique 날짜 목록 추출.
        3) 행2 의 측정값명을 col_idx 별로 매핑.
        4) col_idx → (date, measure) 메타 사전 구성 (col 3 부터).
        5) 가장 최근 6개 날짜 슬라이스.
        6) 데이터 행(df_raw.iloc[3:])에서 col1 이 '2025직종_' 으로 시작하고
           '전체' 로 끝나지 않는 행만 유지.
        7) 최근 6개월의 구인인원·구직건수 컬럼 평균을 행별로 산출 후
           직종_중분류 기준 집계.
        8) 평균구인배율 = 평균구인인원 / 평균구직건수 (0 나누기·inf → 0).
        9) 직종_중분류에서 '2025직종_' 접두사 제거하여 직종명 컬럼 생성.

    Returns:
        pandas.DataFrame: ['직종중분류', '직종명', '평균구인배율']
    """
    out_cols = ["직종중분류", "직종명", "평균구인배율"]
    file_path = _resolve_excel_path(config.RAW_DATA_DIR, "employment_trend")
    if file_path is None:
        _log_missing("employment_trend.xlsx/.xls")
        return _empty_df(out_cols)

    try:
        df_raw = pd.read_excel(file_path, header=None, engine="openpyxl")
    except Exception:
        try:
            df_raw = pd.read_excel(file_path, header=None, engine="xlrd")
        except Exception as e:
            print(f"⚠️ [{file_path.name}] EIS 피벗 형식 직접 로드 실패: {e}")
            return _empty_df(out_cols)

    if df_raw.empty or len(df_raw) < 4 or df_raw.shape[1] < 4:
        print(
            f"⚠️ [{file_path.name}] 행/열이 부족합니다 "
            f"(shape={df_raw.shape}) — 빈 DataFrame 반환"
        )
        return _empty_df(out_cols)

    date_row = df_raw.iloc[1].astype("object").where(df_raw.iloc[1].notna(), pd.NA).ffill()
    measure_row = df_raw.iloc[2]

    col_meta: Dict[int, tuple] = {}
    for ci in range(3, df_raw.shape[1]):
        d_val = date_row.iloc[ci]
        m_val = measure_row.iloc[ci]
        if pd.isna(d_val) or pd.isna(m_val):
            continue
        d_str = str(d_val).strip()
        m_str = str(m_val).strip()
        if not d_str or not m_str:
            continue
        col_meta[ci] = (d_str, m_str)

    if not col_meta:
        print(f"⚠️ [{file_path.name}] 측정값 컬럼 메타 추출 실패 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    dates_in_order: List[str] = []
    for ci in sorted(col_meta.keys()):
        d, _ = col_meta[ci]
        if d not in dates_in_order:
            dates_in_order.append(d)
    recent_dates = dates_in_order[-6:]

    recruit_cols: List[int] = []
    seek_cols: List[int] = []
    for ci, (d, m) in col_meta.items():
        if d not in recent_dates:
            continue
        m_clean = m.replace(" ", "")
        if "구인인원" in m_clean:
            recruit_cols.append(ci)
        elif "구직건수" in m_clean:
            seek_cols.append(ci)

    if not recruit_cols or not seek_cols:
        print(
            f"⚠️ [{file_path.name}] 최근 {len(recent_dates)}개월 구인인원/구직건수 컬럼을 "
            f"찾지 못함 — 빈 DataFrame 반환"
        )
        return _empty_df(out_cols)

    data = df_raw.iloc[3:].copy()
    data.columns = list(range(df_raw.shape[1]))

    job_mid = data[1].astype(str).str.strip()
    mask = job_mid.str.startswith("2025직종_") & ~job_mid.str.endswith("전체")
    data = data.loc[mask].copy()
    data["_직종중분류"] = job_mid[mask]

    if data.empty:
        print(f"⚠️ [{file_path.name}] '2025직종_*' 직종_중분류 행이 없음 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    for ci in recruit_cols + seek_cols:
        data[ci] = pd.to_numeric(data[ci], errors="coerce")

    data["_평균구인인원"] = data[recruit_cols].mean(axis=1, skipna=True)
    data["_평균구직건수"] = data[seek_cols].mean(axis=1, skipna=True)

    grouped = (
        data.groupby("_직종중분류", as_index=False)[["_평균구인인원", "_평균구직건수"]]
        .mean()
    )
    denom = grouped["_평균구직건수"].where(grouped["_평균구직건수"] > 0)
    rate = (grouped["_평균구인인원"] / denom).fillna(0.0)
    rate = rate.where(rate.replace([float("inf"), float("-inf")], pd.NA).notna(), 0.0)
    grouped["평균구인배율"] = rate.round(2)

    grouped["직종명"] = (
        grouped["_직종중분류"].astype(str).str.replace(r"^2025직종_", "", regex=True).str.strip()
    )
    result = grouped.rename(columns={"_직종중분류": "직종중분류"})[out_cols].copy()

    period_label = f"{recent_dates[0]}~{recent_dates[-1]}" if recent_dates else "기간미상"
    print(
        f"✅ employment_trend 로드 완료: {len(result)}개 직종, 기간: {period_label}"
    )
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 2) 부족인력: 평균부족률 산출 (CSV/XLSX/XLS 자동 탐지)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_data_path(
    base_dir: Union[str, Path],
    stem: str,
    extensions: Sequence[str] = (".csv", ".xlsx", ".xls"),
) -> Optional[Path]:
    """
    파일 경로를 자동 탐지한다 (확장자 우선순위 적용).

    공공데이터에서 같은 데이터셋이 .csv/.xlsx/.xls 중 어느 형식으로든 제공될 수 있으므로,
    `stem` 기준으로 명시된 우선순위대로 실제 존재하는 파일을 찾는다.
    """
    base = Path(base_dir)
    for ext in extensions:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# 직종별 코드 + 명칭 분리용 정규식.
# labor_shortage.csv의 '직종별' 컬럼은 'NN 직종명' 또는 'NNN 직종명' 형식.
_LABOR_CODE_NAME_PAT = re.compile(r"^\s*(\d+)\s+(.+)$")


def load_labor_shortage() -> pd.DataFrame:
    """
    부족인력 데이터를 로드해 KNOW 중분류 단위의 평균부족률을 산출한다.

    실제 사용자 데이터 스키마(KOSIS 직종별 부족현황 long format CSV):
        ['시도별(17개)', '규모별', '직종별', '항목', '단위', '2025. 1/2', '2025. 2/2', ...]
        - '항목'에 ['현원', '구인인원', '채용인원', '미충원인원', '부족인원', '채용계획인원', '부족률']
          7종이 행으로 들어 있다 (long format).
        - '직종별'은 'NN 직종명' (KNOW 대중분류 2자리) 또는 'NNN 직종명' (소분류 3자리) 형식.
        - 분기/반기 컬럼이 옆으로 펼쳐진 wide × long 혼합 구조.
        - '부족률' 항목이 미리 계산되어 들어 있으므로 직접 산출하지 않는다.

    처리 흐름:
        1) labor_shortage.csv (또는 .xlsx/.xls) 자동 탐지 후 로드.
        2) 시도별=='전국' 행만 추출 (시도별 평균과 혼동 방지).
        3) 항목=='부족률', 단위=='%' 행만 추출.
        4) 직종별의 코드 prefix(2자리/3자리)에서 KNOW직업대분류·KNOW직업중분류 추출.
        5) 분기 평균 후 소수 1자리 반올림.

    Returns:
        pandas.DataFrame:
            - 'KNOW직업대분류' (Int64): KNOW 대분류 코드 (0~9)
            - 'KNOW직업중분류' (Int64): KNOW 중분류 코드 (0~9)
            - '직종코드_원본' (str): 원본 prefix 코드 (예: '01', '011')
            - '직종명' (str): 직종 명칭
            - '평균부족률' (float): %, 소수 1자리
    """
    out_cols = ["KNOW직업대분류", "KNOW직업중분류", "직종코드_원본", "직종명", "평균부족률"]
    file_path = _resolve_data_path(config.RAW_DATA_DIR, "labor_shortage")
    if file_path is None:
        _log_missing("labor_shortage.csv/.xlsx/.xls")
        return _empty_df(out_cols)

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = _safe_read_csv(file_path)
    else:
        df = _safe_read_excel(file_path)
    if df.empty:
        return _empty_df(out_cols)

    region_col = _pick_column(df, "시도별", ["시도별", "지역"])
    job_col = _pick_column(df, "직종별", ["직종별", "직종"])
    item_col = _pick_column(df, "항목", ["항목"])
    unit_col = _pick_column(df, "단위", ["단위"])
    if not (region_col and job_col and item_col):
        print(
            f"⚠️ [{file_path.name}] 핵심 컬럼 누락 "
            f"(시도별={region_col}, 직종별={job_col}, 항목={item_col}) — 빈 DataFrame 반환"
        )
        return _empty_df(out_cols)

    sub = df[df[region_col].astype(str).str.strip() == "전국"].copy()
    sub = sub[sub[item_col].astype(str).str.strip() == "부족률"]
    if unit_col is not None:
        sub = sub[sub[unit_col].astype(str).str.strip() == "%"]
    if sub.empty:
        print(f"⚠️ [{file_path.name}] '전국 / 부족률' 행이 없음 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    period_cols: List[str] = []
    for c in sub.columns:
        if c in (region_col, job_col, item_col, unit_col):
            continue
        if str(c).startswith("Unnamed"):
            continue
        coerced = pd.to_numeric(sub[c], errors="coerce")
        if coerced.notna().any():
            period_cols.append(c)
    if not period_cols:
        print(f"⚠️ [{file_path.name}] 분기/시점 데이터 컬럼을 찾지 못함 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    for c in period_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["_평균부족률"] = sub[period_cols].mean(axis=1, skipna=True)

    def _split(s: object) -> tuple:
        m = _LABOR_CODE_NAME_PAT.match(str(s))
        if not m:
            return ("", str(s).strip())
        return (m.group(1), m.group(2).strip())

    parsed = sub[job_col].apply(_split)
    sub["_원본코드"] = parsed.apply(lambda t: t[0])
    sub["_직종명"] = parsed.apply(lambda t: t[1])

    def _to_main_mid(code: str) -> tuple:
        if not code or len(code) < 2:
            return (pd.NA, pd.NA)
        try:
            main = int(code[0])
            mid = int(code[1])
            return (main, mid)
        except ValueError:
            return (pd.NA, pd.NA)

    main_mid = sub["_원본코드"].apply(_to_main_mid)
    sub["KNOW직업대분류"] = main_mid.apply(lambda t: t[0]).astype("Int64")
    sub["KNOW직업중분류"] = main_mid.apply(lambda t: t[1]).astype("Int64")

    result = sub.rename(
        columns={"_원본코드": "직종코드_원본", "_직종명": "직종명", "_평균부족률": "평균부족률"}
    )[out_cols].copy()
    result["평균부족률"] = result["평균부족률"].astype(float).round(1)
    result = result[result["KNOW직업대분류"].notna() & result["KNOW직업중분류"].notna()]

    print(
        f"✅ [{file_path.name}] 부족률 추출 완료: {len(result):,}행 "
        f"(분기 평균 {len(period_cols)}개 컬럼 사용: {period_cols})"
    )
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3) 임금정보: 공공데이터포털 자동변환 Open API 로더
# ──────────────────────────────────────────────────────────────────────────────
def check_wage_api_response() -> Optional[dict]:
    """
    임금정보 API 응답을 진단용으로 1회만(page=1, perPage=10) 호출한다.

    데이터 적재가 아닌 응답 스키마 점검 전용 헬퍼이며, load_job_wage_from_api()
    호출 전에 응답 형식·총 건수·컬럼명 등을 빠르게 확인할 때 사용한다.
    출력 항목:
        - HTTP 상태 코드
        - 응답 JSON의 전체 키 목록
        - totalCount / total_count(존재 시) 전체 건수
        - 첫 데이터 row의 키 목록 + 첫 row 자체(축약)

    Returns:
        Optional[dict]: 응답 payload (성공 시) 또는 None (실패 시).
    """
    base_url = getattr(config, "JOB_WAGE_API_URL", "") or ""
    if not base_url or "여기에" in base_url:
        print("⚠️ 임금정보 API URL 미설정 — config.py의 JOB_WAGE_API_URL을 입력해주세요.")
        return None

    api_key = getattr(config, "PUBLIC_DATA_API_KEY", "") or ""
    print("\n[check_wage_api_response] 임금정보 API 응답 점검 (page=1, perPage=10)")
    print(f"  • URL: {base_url}")

    try:
        resp = requests.get(
            base_url,
            params={"page": 1, "perPage": 10, "serviceKey": api_key},
            timeout=15,
        )
        print(f"  • HTTP status: {resp.status_code}")
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 임금정보 API 호출 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ 임금정보 API 응답 파싱 실패: {e}")
        return None

    if not isinstance(payload, dict):
        print(f"⚠️ 응답이 dict가 아님 — type={type(payload).__name__}")
        return None

    print(f"  🔎 응답 JSON 전체 키: {list(payload.keys())}")
    total_count = payload.get("totalCount", payload.get("total_count"))
    if total_count is not None:
        print(f"  📊 전체 건수(totalCount): {total_count}")
    else:
        print("  ⚠️ totalCount / total_count 키가 응답에 없음")

    rows = payload.get("data") or []
    print(f"  • data 길이: {len(rows)}")
    if rows and isinstance(rows[0], dict):
        first = rows[0]
        print(f"  🔑 첫 row 키 목록: {list(first.keys())}")
        sample_items = list(first.items())[:8]
        print(f"  🧾 첫 row 샘플(앞 8필드): {sample_items}")
    return payload


def load_job_wage_from_api() -> pd.DataFrame:
    """
    직업별 임금정보 API 로더 (공공데이터포털 자동변환 Open API).

    데이터셋: 한국고용정보원_직업별_임금정보 (namespace=15122500/v1).
    임금 단위는 만원이다.

    파라미터 (모두 config.py에서 읽음):
        - JOB_WAGE_API_URL : 자동변환 API의 UDDI 엔드포인트
        - PUBLIC_DATA_API_KEY : 공공데이터포털 인증키
        - JOB_WAGE_API_PER_PAGE : 페이지당 요청 건수
        - JOB_WAGE_API_MAX_PAGES : 페이지네이션 최대 페이지 수

    동작:
        1) URL이 비어 있거나 placeholder("여기에")를 포함하면 안내 후 빈 DF 반환.
        2) page=1..MAX_PAGES 범위로 페이지네이션:
           - 응답 JSON의 'data' 키에서 레코드 추출.
           - 빈 리스트 응답 시 즉시 루프 종료(마지막 페이지 도달).
           - 페이지마다 진행 로그를 찍고 time.sleep(0.3)으로 부하 방지.
        3) 컬럼 자동 매핑(_pick_column)으로 다양한 응답 표기를 표준화한다.
        4) 숫자형으로 변환하고 NaN은 0으로 채운다(요청 명세).
        5) 요청 자체가 실패하면 에러 메시지 출력 후 빈 DataFrame 반환.

    Returns:
        pandas.DataFrame: ['직종코드', '직업명', '상위임금', '중위임금', '하위임금']
    """
    out_cols = ["직종코드", "직업명", "상위임금", "중위임금", "하위임금"]

    base_url = getattr(config, "JOB_WAGE_API_URL", "") or ""
    if not base_url or "여기에" in base_url:
        print("⚠️ 임금정보 API URL 미설정 — config.py의 JOB_WAGE_API_URL을 입력해주세요.")
        return _empty_df(out_cols)

    api_key = getattr(config, "PUBLIC_DATA_API_KEY", "") or ""
    per_page: int = int(getattr(config, "JOB_WAGE_API_PER_PAGE", 100))
    max_pages: int = int(getattr(config, "JOB_WAGE_API_MAX_PAGES", 20))

    all_rows: List[dict] = []
    try:
        for page in range(1, max_pages + 1):
            resp = requests.get(
                base_url,
                params={
                    "page": page,
                    "perPage": per_page,
                    "serviceKey": api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                print(f"⚠️ 임금정보 API 응답 형식 오류 (page={page}) — 중단")
                break

            # 첫 페이지에서만 응답 메타데이터 진단 출력 (전체 키, 총 건수)
            if page == 1:
                print(f"  🔎 임금정보 API 응답 키 목록: {list(payload.keys())}")
                total_count = payload.get("totalCount", payload.get("total_count"))
                if total_count is not None:
                    print(f"  📊 임금정보 API 전체 건수(totalCount): {total_count}")

            rows = payload.get("data") or []
            print(f"  📄 임금정보 페이지 {page} 로드: {len(rows)}건")
            if not rows:
                break
            all_rows.extend(rows)
            time.sleep(0.3)
    except requests.exceptions.RequestException as e:
        print(f"❌ 임금정보 API 호출 실패: {e} — 빈 DataFrame 반환")
        return _empty_df(out_cols)
    except Exception as e:
        print(f"❌ 임금정보 API 처리 실패: {e} — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    if not all_rows:
        print("⚠️ 임금정보 API 응답 데이터 비어 있음 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    df = pd.DataFrame(all_rows)

    col_map: Dict[str, Union[str, None]] = {
        "직종코드": _pick_column(df, "직종코드", ["직종코드", "직업코드", "코드"]),
        "직업명": _pick_column(df, "직업명", ["직업명", "직종명"]),
        "상위임금": _pick_column(
            df, "상위임금",
            ["상위25", "상위 25", "상위_임금", "상위임금", "75", "p75", "Q3"],
        ),
        "중위임금": _pick_column(
            df, "중위임금",
            ["중위값", "중위_임금", "중위임금", "중위", "50", "median", "p50"],
        ),
        "하위임금": _pick_column(
            df, "하위임금",
            ["하위25", "하위 25", "하위_임금", "하위임금", "25", "p25", "Q1"],
        ),
    }
    if col_map["직업명"] is None:
        print("⚠️ 임금정보 API 응답에서 '직업명' 컬럼을 찾지 못해 빈 DataFrame 반환")
        return _empty_df(out_cols)

    sub = _rename_picked(df, col_map)
    for col in out_cols:
        if col not in sub.columns:
            sub[col] = pd.NA
    sub = sub[out_cols].copy()

    sub["직업명"] = sub["직업명"].astype(str).str.strip()
    sub["직종코드"] = sub["직종코드"].astype(str).str.strip().replace({"nan": ""})

    for col in ("상위임금", "중위임금", "하위임금"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)

    print(f"✅ 임금정보 API 로드 완료: {len(sub):,}건")
    return sub


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3-2) 임금통계: 고용노동통계포털 직종별 월임금총액 (KSCO 기준 CSV)
# ──────────────────────────────────────────────────────────────────────────────
# 한국표준직업분류(개정7차) 끝의 '(코드)' 추출용 정규식.
_WAGE_KSCO_CODE_PAT = re.compile(r"\(([^)]+)\)\s*$")


def load_wage_statistics() -> pd.DataFrame:
    """
    [Phase1] 고용노동통계포털 직종별 임금 통계 로더.

    원본: 118_DT_118N_PAYM47_*.csv (laborstat.moel.go.kr 다운로드).
    한국표준직업분류 개정7차 기준 직종별 월임금총액 (2020~2024년, 단위: 천원).
    파일명을 'wage_statistics.csv' 로 저장한 뒤 사용한다.

    처리 흐름:
        1) wage_statistics.csv 로드 (인코딩 자동 감지: chardet → cp949 → utf-8 fallback).
        2) 필터 적용: 항목=='월임금총액' & 성별=='전체' & 경력구분=='전경력'.
        3) 'YYYY 년' 패턴 컬럼 중 가장 최신 연도(예: '2024 년') 사용 → 월평균임금(천원).
        4) '한국표준직업분류 개정7차' 컬럼 끝의 '(코드)' 를 분리해
           직종코드_KSCO 와 직종명_KSCO(괄호 제거) 컬럼으로 분해.
        5) 코드가 없는 '전직종' 같은 row 와 임금 NaN row 는 제외.

    Returns:
        pandas.DataFrame: ['직종코드_KSCO', '직종명_KSCO', '월평균임금_천원']
    """
    out_cols = ["직종코드_KSCO", "직종명_KSCO", "월평균임금_천원"]
    file_path = Path(config.RAW_DATA_DIR) / "wage_statistics.csv"
    if not file_path.exists():
        _log_missing("wage_statistics.csv")
        return _empty_df(out_cols)

    df = _safe_read_csv(file_path)
    if df.empty:
        return _empty_df(out_cols)

    cls_col = _pick_column(
        df, "한국표준직업분류 개정7차", ["한국표준직업분류", "직업분류", "직종"]
    )
    item_col = _pick_column(df, "항목", ["항목"])
    sex_col = _pick_column(df, "성별", ["성별"])
    career_col = _pick_column(df, "경력구분", ["경력"])
    if not (cls_col and item_col and sex_col and career_col):
        print(
            f"⚠️ [{file_path.name}] 핵심 컬럼 누락 "
            f"(분류={cls_col}, 항목={item_col}, 성별={sex_col}, 경력={career_col}) — 빈 DataFrame 반환"
        )
        return _empty_df(out_cols)

    flt = df[
        (df[item_col].astype(str).str.strip() == "월임금총액")
        & (df[sex_col].astype(str).str.strip() == "전체")
        & (df[career_col].astype(str).str.strip() == "전경력")
    ].copy()
    if flt.empty:
        print(f"⚠️ [{file_path.name}] 필터 조건에 해당하는 행이 없음 — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    year_pat = re.compile(r"^(\d{4})\s*년\s*$")
    year_cols: List[tuple] = []
    for c in flt.columns:
        m = year_pat.match(str(c).strip())
        if m:
            year_cols.append((int(m.group(1)), c))
    if not year_cols:
        print(f"⚠️ [{file_path.name}] 'YYYY 년' 컬럼을 찾지 못함 — 빈 DataFrame 반환")
        return _empty_df(out_cols)
    year_cols.sort(key=lambda t: t[0])
    latest_year, latest_col = year_cols[-1]

    cls_series = flt[cls_col].astype(str).str.strip()

    def _extract_code_name(s: str) -> tuple:
        m = _WAGE_KSCO_CODE_PAT.search(s)
        if m:
            code = m.group(1).strip()
            name = _WAGE_KSCO_CODE_PAT.sub("", s).strip()
            return code, name
        return "", s

    parsed = cls_series.apply(_extract_code_name)
    flt["_직종코드_KSCO"] = parsed.apply(lambda t: t[0])
    flt["_직종명_KSCO"] = parsed.apply(lambda t: t[1])
    flt["_월평균임금_천원"] = pd.to_numeric(flt[latest_col], errors="coerce")

    result = flt[flt["_직종코드_KSCO"] != ""].rename(
        columns={
            "_직종코드_KSCO": "직종코드_KSCO",
            "_직종명_KSCO": "직종명_KSCO",
            "_월평균임금_천원": "월평균임금_천원",
        }
    )[out_cols].copy()
    result = (
        result.dropna(subset=["월평균임금_천원"])
        .drop_duplicates(subset=["직종코드_KSCO"], keep="first")
        .reset_index(drop=True)
    )
    result["월평균임금_천원"] = result["월평균임금_천원"].astype(float).round(0).astype(int)

    print(
        f"✅ [{file_path.name}] 임금통계 로드 완료: {len(result)}건 "
        f"(기준 연도: {latest_year}년, 단위: 천원)"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 기능 4) 전공별 진출직업 상위 (CSV)
# ──────────────────────────────────────────────────────────────────────────────
def load_top_jobs_by_major() -> pd.DataFrame:
    """
    전공별 진출직업 상위 CSV를 로드한다.

    실제 사용자 데이터 스키마: ['계열정보', '학과명', '직업분류명']
        → 진출순위/진출비율 컬럼이 없으므로 두 컬럼은 옵셔널 처리한다.
        → 진출순위 컬럼이 없을 경우 master 단계에서는 학과별 첫 등장 직업을 1순위로 간주.

    Returns:
        pandas.DataFrame: 필수 ['계열명', '학과명', '진출직업명'] +
                          옵셔널 ['진출순위', '진출비율'] (없으면 NA로 채움)
    """
    out_cols = ["계열명", "학과명", "진출직업명", "진출순위", "진출비율"]
    file_name = "top_jobs_by_major.csv"
    file_path = Path(config.RAW_DATA_DIR) / file_name

    df = _safe_read_csv(file_path)
    if df.empty:
        return _empty_df(out_cols)

    required_map: Dict[str, Union[str, None]] = {
        "계열명": _pick_column(df, "계열명", ["계열정보", "계열명", "계열"]),
        "학과명": _pick_column(df, "학과명", ["학과명", "학과"]),
        "진출직업명": _pick_column(
            df, "진출직업명", ["진출직업명", "직업분류명", "진출직업", "직업명"]
        ),
    }
    missing = [k for k, v in required_map.items() if v is None]
    if missing:
        print(f"⚠️ [{file_name}] 핵심 컬럼 누락: {missing} — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    optional_map: Dict[str, Union[str, None]] = {
        "진출순위": _pick_column(df, "진출순위", ["진출순위", "순위"]),
        "진출비율": _pick_column(df, "진출비율", ["진출비율", "비율"]),
    }

    full_map = {**required_map, **{k: v for k, v in optional_map.items() if v is not None}}
    sub = _rename_picked(df, full_map)

    for k, v in optional_map.items():
        if v is None:
            sub[k] = pd.NA

    print(
        f"✅ [{file_name}] 로드 완료: {len(sub):,}행 "
        f"(매핑: {required_map}, 옵셔널 누락: "
        f"{[k for k, v in optional_map.items() if v is None]})"
    )
    return sub[out_cols].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 기능 5) 졸업 후 첫 취업 통계 (CSV)
# ──────────────────────────────────────────────────────────────────────────────
def load_first_job_entry() -> pd.DataFrame:
    """
    졸업 후 첫 취업 통계 CSV를 로드한다.

    실제 사용자 데이터 스키마(long format):
        ['계열명', '학과명', '대학교 구분', '첫일자리진출소요기간 구분', '측정값']
        - '첫일자리진출소요기간 구분' 컬럼에 '6개월', '1년', '1년이상' 등의 카테고리가 들어가고
          '측정값'에 해당 누적 취업률(%)이 들어간다.
        - 본 함수는 long → wide 피벗을 수행하여 카테고리를 컬럼으로 펼친다.

    Returns:
        pandas.DataFrame: ['계열명', '학과명', '대학유형구분'] +
                          가용 카테고리(예: '취업률_6개월', '취업률_1년', '취업률_1년이상')
        - 데이터셋에 카테고리가 부족하면 해당 컬럼은 자동 생략된다.
    """
    file_name = "first_job_entry.csv"
    file_path = Path(config.RAW_DATA_DIR) / file_name

    df = _safe_read_csv(file_path)
    if df.empty:
        return _empty_df(["계열명", "학과명", "대학유형구분"])

    col_map: Dict[str, Union[str, None]] = {
        "계열명": _pick_column(df, "계열명", ["계열명", "계열"]),
        "학과명": _pick_column(df, "학과명", ["학과명", "학과"]),
        "대학유형구분": _pick_column(
            df, "대학유형구분",
            ["대학교 구분", "대학교구분", "대학유형구분", "대학유형", "대학구분"],
        ),
        "기간구분": _pick_column(
            df, "기간구분",
            ["첫일자리진출소요기간 구분", "첫일자리진출소요기간구분", "진출소요기간"],
        ),
        "측정값": _pick_column(df, "측정값", ["측정값", "값"]),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        print(f"⚠️ [{file_name}] 핵심 컬럼 누락: {missing} — 빈 DataFrame 반환")
        return _empty_df(["계열명", "학과명", "대학유형구분"])

    sub = _rename_picked(df, col_map)[
        ["계열명", "학과명", "대학유형구분", "기간구분", "측정값"]
    ].copy()

    sub["측정값"] = pd.to_numeric(sub["측정값"], errors="coerce")
    sub["기간구분"] = sub["기간구분"].astype(str).str.strip()

    # long → wide pivot
    pivoted = (
        sub.pivot_table(
            index=["계열명", "학과명", "대학유형구분"],
            columns="기간구분",
            values="측정값",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None

    rename_pairs: Dict[str, str] = {}
    for c in pivoted.columns:
        if c in ("계열명", "학과명", "대학유형구분"):
            continue
        rename_pairs[c] = f"취업률_{c}"
    pivoted = pivoted.rename(columns=rename_pairs)

    avail_cats = [c for c in pivoted.columns if c.startswith("취업률_")]
    print(
        f"✅ [{file_name}] long→wide 변환 완료: {len(pivoted):,}행, "
        f"카테고리={avail_cats}"
    )
    return pivoted


# ──────────────────────────────────────────────────────────────────────────────
# 기능 6) NCS 경로/전직 정보 (CSV)
# ──────────────────────────────────────────────────────────────────────────────
def load_ncs_career_path() -> pd.DataFrame:
    """
    NCS 경로/전직 정보 CSV를 로드한다.

    실제 사용자 데이터 스키마:
        ['대분류코드', '중분류코드', '소분류코드', '직무코드', '직무명',
         '직무역량코드', '직무역량수준(...)', '직무역량명', '수준(직급수준)', '직급명']
        → 분류 '명' 컬럼과 '전직가능직무명' 컬럼이 직접 존재하지 않는다.
        → 본 함수는 같은 (대분류코드, 중분류코드, 소분류코드) 그룹에 속한
          **자기 자신을 제외한 다른 직무명들**을 합쳐 '전직가능직무명'을 자체 생성한다.

    Returns:
        pandas.DataFrame: ['대분류코드', '중분류코드', '소분류코드',
                           '직무코드', '직무명', '전직가능직무명']
        - '전직가능직무명'이 빈 그룹(같은 소분류 내 직무가 1개 뿐)은 제외.
    """
    out_cols = [
        "대분류코드", "중분류코드", "소분류코드",
        "직무코드", "직무명", "전직가능직무명",
    ]
    file_name = "ncs_career_path.csv"
    file_path = Path(config.RAW_DATA_DIR) / file_name

    df = _safe_read_csv(file_path)
    if df.empty:
        return _empty_df(out_cols)

    col_map: Dict[str, Union[str, None]] = {
        "대분류코드": _pick_column(df, "대분류코드", ["대분류코드", "대분류"]),
        "중분류코드": _pick_column(df, "중분류코드", ["중분류코드", "중분류"]),
        "소분류코드": _pick_column(df, "소분류코드", ["소분류코드", "소분류"]),
        "직무코드": _pick_column(df, "직무코드", ["직무코드", "세분류코드"]),
        "직무명": _pick_column(df, "직무명", ["직무명", "세분류명"]),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        print(f"⚠️ [{file_name}] 핵심 컬럼 누락: {missing} — 빈 DataFrame 반환")
        return _empty_df(out_cols)

    sub = _rename_picked(df, col_map)[
        ["대분류코드", "중분류코드", "소분류코드", "직무코드", "직무명"]
    ].copy()
    sub = sub.dropna(subset=["직무명"])
    sub["직무명"] = sub["직무명"].astype(str).str.strip()
    sub = sub[sub["직무명"] != ""]

    unique_jobs = sub.drop_duplicates(
        subset=["대분류코드", "중분류코드", "소분류코드", "직무명"]
    ).reset_index(drop=True)

    grouped = (
        unique_jobs.groupby(
            ["대분류코드", "중분류코드", "소분류코드"], as_index=False
        )["직무명"]
        .agg(list)
        .rename(columns={"직무명": "_그룹직무명목록"})
    )
    merged = unique_jobs.merge(
        grouped, on=["대분류코드", "중분류코드", "소분류코드"], how="left"
    )

    def _others(row) -> object:
        peers = [j for j in row["_그룹직무명목록"] if j != row["직무명"]]
        return ", ".join(peers) if peers else None

    merged["전직가능직무명"] = merged.apply(_others, axis=1)
    merged = merged.drop(columns=["_그룹직무명목록"])
    merged = merged[merged["전직가능직무명"].notna()]
    merged = merged[merged["전직가능직무명"].astype(str).str.strip() != ""]

    print(
        f"✅ [{file_name}] 전직가능직무명 자체 생성 완료: "
        f"{len(merged):,}행 (원본 {len(df):,}행 → 직무 unique {len(unique_jobs):,})"
    )
    return merged[out_cols].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 기능 7) 마스터 데이터셋 빌드 — KNOW/일반 스키마 정규화 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

# PROSPECT_SCORE_MAP 키워드를 길이 내림차순으로 정렬해 두면
# '다소 증가' 텍스트가 '증가'에 먼저 매치되는 오류를 막을 수 있다.
_PROSPECT_KEYWORDS_BY_LEN: List[str] = sorted(
    config.PROSPECT_SCORE_MAP.keys(), key=len, reverse=True
)

# 직업전망 텍스트에서 직업명을 추출하는 정규식 (우선순위 순서).
# 1) 텍스트 시작 부분의 <직업명> 머리표기.
# 2) "향후 N년간/N년 동안 ○○○의/는/은 고용/일자리/채용" 정형 패턴.
# 3) "향후 ○○○의/는/은 고용..." 단순 패턴 (5년간 누락 케이스 보완).
_JOB_NAME_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*<\s*([^<>\n]+?)\s*>"),
    re.compile(
        r"향후[\s\S]{0,40}?(?:\d+\s*년간|\d+\s*년\s*동안)\s+(.+?)"
        r"\s*(?:의|는|은)\s+(?:고용|일자리|채용|전망)"
    ),
    re.compile(r"향후\s+(.+?)\s*(?:의|는|은)\s+(?:고용|일자리|채용|전망)"),
]


def _extract_job_name_from_text(text: object) -> object:
    """직업전망 텍스트에서 직업명을 추출한다.

    `_JOB_NAME_PATTERNS`를 순서대로 시도하며, 첫 매칭의 group(1)을 strip 후 반환한다.
    추출 결과가 30자를 넘으면 잘못 잡힌 경우로 보고 다음 패턴으로 넘어간다.
    실패 시 pd.NA 반환.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return pd.NA
    s = str(text).strip()
    if not s:
        return pd.NA
    for pat in _JOB_NAME_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        name = m.group(1).strip()
        if 0 < len(name) <= 30:
            return name
    return pd.NA


def _map_prospect_score(text: object) -> object:
    """직업전망 텍스트를 PROSPECT_SCORE_MAP 점수로 매핑.

    키워드를 길이 내림차순으로 시도하므로 '다소 증가' / '다소 감소' 같은
    합성 키워드가 우선 매칭된다.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return pd.NA
    s = str(text)
    if not s:
        return pd.NA
    for kw in _PROSPECT_KEYWORDS_BY_LEN:
        if kw in s:
            return config.PROSPECT_SCORE_MAP[kw]
    return pd.NA


def _aggregate_education_major_to_top_category(em_df: pd.DataFrame) -> pd.DataFrame:
    """education_major wide DataFrame에서 직업코드별 '주요전공계열'을 추출한다.

    education_major는 KNOW 형식인 경우 학과 코드/학력 컬럼 외에
    '인문-언어_문학', '공학-컴퓨터_통신' 같은 35개의 전공계열 wide 컬럼을 갖는다.
    이를 대계열 prefix(예: '공학', '의학')로 그룹 합산 후 argmax를 채택한다.

    Returns:
        pandas.DataFrame[직업코드, 주요전공계열] (직업코드 dedup 완료).
        wide 컬럼이 없거나 키 컬럼이 없으면 빈 DataFrame.
    """
    if em_df is None or em_df.empty:
        return _empty_df(["직업코드", "주요전공계열"])

    df = em_df.copy()
    code_col = _pick_column(df, "직업코드", ["KNOW직업코드", "직업코드", "코드"])
    if not code_col:
        return _empty_df(["직업코드", "주요전공계열"])

    # 학력- 컬럼과 메타 컬럼은 전공계열 후보에서 제외.
    excluded_exact = {code_col, "되는길순번", "교육1코드", "교육2코드", "교육3코드"}
    major_cols = [
        c for c in df.columns
        if c not in excluded_exact
        and not c.startswith("학력-")
        and "-" in c
    ]
    if not major_cols:
        return _empty_df(["직업코드", "주요전공계열"])

    # 모든 wide 컬럼을 numeric으로 변환 (NaN→0).
    nums = df[major_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 컬럼명 prefix(대계열)별로 그룹 합산.
    cat_groups: Dict[str, List[str]] = {}
    for col in major_cols:
        cat = col.split("-", 1)[0]
        cat_groups.setdefault(cat, []).append(col)

    cat_sums = pd.DataFrame(
        {cat: nums[cols].sum(axis=1) for cat, cols in cat_groups.items()},
        index=df.index,
    )

    has_data = cat_sums.sum(axis=1) > 0
    top_cat = cat_sums.idxmax(axis=1).where(has_data, other=pd.NA)

    out = pd.DataFrame({
        "직업코드": df[code_col].astype(str).str.strip(),
        "주요전공계열": top_cat.astype(object),
    }).drop_duplicates(subset=["직업코드"])
    return out.reset_index(drop=True)


def _normalize_prospect_base(prospect_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """prospect DataFrame을 master 베이스 표준 형태로 정규화한다.

    표준 형태는 다음 4개 컬럼을 갖는다:
        - 직업코드 (str)
        - 직업명 (str | NA)
        - 직업전망_텍스트 (str | NA)
        - 전망점수 (float | NA)

    KNOW 스키마(KNOW직업코드 + 직업전망내용)와 일반 스키마(직업코드 + 직업명 + 전망등급)
    를 자동 감지해 모두 처리한다. 처리 불가 시 None.
    """
    if prospect_df is None or prospect_df.empty:
        return None

    df = prospect_df.copy()

    # Case A: KNOW 스키마 — 직업명을 텍스트에서 추출.
    if "KNOW직업코드" in df.columns and "직업전망내용" in df.columns:
        out = pd.DataFrame({
            "직업코드": df["KNOW직업코드"].astype(str).str.strip(),
            "직업전망_텍스트": df["직업전망내용"].astype(object),
        })
        out["직업명"] = out["직업전망_텍스트"].apply(_extract_job_name_from_text)
        out["전망점수"] = out["직업전망_텍스트"].apply(_map_prospect_score)
        return out[["직업코드", "직업명", "직업전망_텍스트", "전망점수"]].copy()

    # Case B: 일반 스키마 — 컬럼명을 추정.
    code_col = _pick_column(df, "직업코드", ["직업코드", "직종코드", "코드"])
    name_col = _pick_column(df, "직업명", ["직업명", "직종명"])
    if not (code_col and name_col):
        return None

    df = df.rename(columns={code_col: "직업코드", name_col: "직업명"})
    df["직업코드"] = df["직업코드"].astype(str).str.strip()

    text_col = _pick_column(
        df, "직업전망_텍스트", ["전망등급", "전망", "직업전망내용", "직업전망"]
    )
    if text_col and text_col != "직업전망_텍스트":
        df = df.rename(columns={text_col: "직업전망_텍스트"})
    if "직업전망_텍스트" not in df.columns:
        df["직업전망_텍스트"] = pd.NA

    df["전망점수"] = df["직업전망_텍스트"].apply(_map_prospect_score)
    return df[["직업코드", "직업명", "직업전망_텍스트", "전망점수"]].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 기능 7) 마스터 데이터셋 빌드
# ──────────────────────────────────────────────────────────────────────────────
def build_master_job_data() -> pd.DataFrame:
    """
    모든 보조 데이터를 병합해 RAG 엔진용 마스터 데이터셋을 생성·저장한다.

    이미 작성된 load_job_prospect_bundle()의 결과를 기준 DataFrame으로 사용한다.
    KNOW 스키마(KNOW직업코드 + 직업전망내용)와 일반 스키마 모두 자동 감지해 처리한다.

    처리 순서:
        1) load_job_prospect_bundle() → 5종 dict
           prospect를 _normalize_prospect_base()로 정규화해 기준 DataFrame 생성
           (직업코드/직업명/직업전망_텍스트/전망점수)
        2) education_major → 직업코드 기준 left join, 주요전공계열 단일값
           (KNOW 형식의 wide 35컬럼은 대계열 prefix별 합산 후 argmax 채택)
        3) similar_names → 직업코드 기준 group concat → 유사직업명_합산
        4) subclass / midclass: KNOW 분류 코드 도메인이 prospect와 다르므로 결합 제외.
           대분류명은 NaN으로 두며, 명시적 매핑표가 추가되면 후속 작업에서 채움.
        5) load_employment_trend() → 직업명 기준 left join (평균구인배율)
        6) load_labor_shortage()   → 직업명 기준 left join (평균부족률)
        7) load_job_wage_from_api()→ 직업명 기준 left join (중위/하위임금)
        8) load_top_jobs_by_major()→ 진출순위 1위만, 직업명 기준 left join (계열명)
        9) load_first_job_entry()  → 계열명 기준 평균 left join (취업률 2종)
       10) load_ncs_career_path()  → 직무명↔직업명 group concat (전직가능직무명)
       11) 직업명 NaN/공백 행 제거
       12) config.MASTER_JOB_FILE 경로에 utf-8-sig CSV로 저장

    Returns:
        pandas.DataFrame: 마스터 데이터셋(핵심 14컬럼 순서로 정렬됨).
    """
    print("\n[build_master_job_data] 시작\n")

    # ── 1) 기준 DataFrame 구축 (KNOW/일반 스키마 자동 정규화) ─────────
    bundle = load_job_prospect_bundle(strict=False, verbose=True)
    if "prospect" not in bundle or bundle["prospect"].empty:
        print("⚠️ prospect 번들이 비어 있어 master 데이터를 생성할 수 없습니다.")
        return _empty_df()

    base = _normalize_prospect_base(bundle["prospect"])
    if base is None or base.empty:
        print("⚠️ prospect 정규화 실패 — 직업코드/직업명을 추출할 수 없습니다.")
        return _empty_df()

    n_total = len(base)
    n_named = int(base["직업명"].notna().sum())
    n_scored = int(base["전망점수"].notna().sum())
    print(
        f"[prospect 정규화] {n_total}행 — "
        f"직업명 추출 {n_named}/{n_total} "
        f"({n_named / max(n_total, 1) * 100:.1f}%), "
        f"전망점수 매핑 {n_scored}/{n_total} "
        f"({n_scored / max(n_total, 1) * 100:.1f}%)"
    )

    # ── 2) 주요전공계열 ← education_major (KNOW wide → 대계열 argmax) ─
    if "education_major" in bundle and not bundle["education_major"].empty:
        em_top = _aggregate_education_major_to_top_category(bundle["education_major"])
        if not em_top.empty:
            base["직업코드"] = base["직업코드"].astype(str)
            em_top["직업코드"] = em_top["직업코드"].astype(str)
            base = base.merge(em_top, on="직업코드", how="left")
    if "주요전공계열" not in base.columns:
        base["주요전공계열"] = pd.NA

    # ── 3) 유사직업명_합산 ← similar_names 보강 (2단계 매칭) ──────────
    #
    # 1차) 직업코드 기준 join (similar_names.KNOW직업코드 ↔ master.직업코드).
    #      진단 결과 두 코드 도메인이 의미적으로 일치하지 않아 매칭 거의 0건이지만,
    #      향후 KNOW직업코드 ↔ KNOW직업명 매핑표가 추가되면 자동으로 채워진다.
    # 2차) 토큰 정확 일치 fallback (코드 도메인 차이 우회):
    #      similar_names.유사직업명 을 콤마/슬래시 기준 토큰화 → master.직업명과 정확 일치
    #      하는 토큰을 anchor 로 삼아, 같은 row 의 나머지 토큰들을 그 master 행의
    #      유사직업명_합산에 합쳐 넣는다. 부분 일치(substring)는 사용하지 않아 잡음 최소화.
    base["유사직업명_합산"] = pd.NA
    similar_match_count = 0
    fallback_match_count = 0
    if "similar_names" in bundle and not bundle["similar_names"].empty:
        sn = bundle["similar_names"].copy()
        sn_code = _pick_column(sn, "직업코드", ["KNOW직업코드", "직업코드", "코드"])
        sn_name = _pick_column(sn, "유사직업명", ["유사직업명", "유사", "직업명"])
        if sn_code and sn_name:
            sn[sn_code] = sn[sn_code].astype(str).str.strip()
            sn[sn_name] = sn[sn_name].fillna("").astype(str).str.strip()
            sn = sn[(sn[sn_name] != "") & (sn[sn_name].str.lower() != "nan")].copy()

            def _join_unique(items: Sequence[object]) -> str:
                seen: Dict[str, None] = {}
                for x in items:
                    if x is None:
                        continue
                    s = str(x).strip()
                    if not s or s.lower() == "nan":
                        continue
                    seen.setdefault(s, None)
                return ", ".join(seen.keys())

            agg = (
                sn.groupby(sn_code)[sn_name]
                .apply(lambda s: _join_unique(s.tolist()))
                .rename("_sim_by_code")
                .reset_index()
                .rename(columns={sn_code: "직업코드"})
            )
            agg = agg[agg["_sim_by_code"].astype(str).str.strip() != ""]
            base["직업코드"] = base["직업코드"].astype(str)
            agg["직업코드"] = agg["직업코드"].astype(str)
            base = base.merge(agg, on="직업코드", how="left")
            similar_match_count = int(base["_sim_by_code"].notna().sum())
            base["유사직업명_합산"] = base["_sim_by_code"]
            base = base.drop(columns=["_sim_by_code"])

            tok_pat = re.compile(r"[,/·•・]| {2,}")
            name_to_tokens: Dict[str, List[List[str]]] = {}
            for txt in sn[sn_name].dropna().tolist():
                tokens = [t.strip() for t in tok_pat.split(str(txt)) if t and t.strip()]
                tokens = [t for t in tokens if 1 < len(t) <= 30]
                if not tokens:
                    continue
                for tk in tokens:
                    name_to_tokens.setdefault(tk, []).append(tokens)

            def _build_fallback(name: str) -> Optional[str]:
                rows = name_to_tokens.get(name)
                if not rows:
                    return None
                seen: Dict[str, None] = {}
                for tokens in rows:
                    for t in tokens:
                        if t == name:
                            continue
                        seen.setdefault(t, None)
                if not seen:
                    return None
                return ", ".join(seen.keys())

            mask_empty = base["유사직업명_합산"].isna()
            fallback = base.loc[mask_empty, "직업명"].apply(
                lambda n: _build_fallback(n) if isinstance(n, str) and n else None
            )
            base.loc[mask_empty, "유사직업명_합산"] = fallback
            fallback_match_count = int(fallback.notna().sum())

    print(
        f"[similar_names] 직업코드 join 매칭 {similar_match_count}건, "
        f"토큰 정확일치 fallback {fallback_match_count}건"
    )

    # ── 4) 대분류명/중분류명: subclass.KNOW직업명 ↔ master.직업명 join ──
    #
    # subclass.csv 에는 'KNOW직업명' 컬럼이 있어 master.직업명과 직접 매칭된다
    # (진단 결과 master.직업명 ∩ subclass.KNOW직업명 ≈ 64.9%).
    # 이 join으로 KNOW직업대분류·KNOW직업중분류 코드(2자리)를 확보하면:
    #   - 대분류명: config.KNOW_MAIN_CATEGORY_NAMES 매핑
    #   - 중분류명: midclass.csv 의 KNOW직업중분류명을 그대로 join
    #   - 부족률(── 6): (대분류, 중분류) 키로 labor_shortage 와 join
    if "subclass" in bundle and not bundle["subclass"].empty:
        sc = bundle["subclass"].copy()
        sc_name_col = _pick_column(sc, "KNOW직업명", ["KNOW직업명", "직업명"])
        sc_main_col = _pick_column(sc, "KNOW직업대분류", ["KNOW직업대분류", "대분류"])
        sc_mid_col = _pick_column(sc, "KNOW직업중분류", ["KNOW직업중분류", "중분류"])
        if sc_name_col and sc_main_col:
            sc[sc_name_col] = sc[sc_name_col].astype(str).str.strip()
            sc[sc_main_col] = pd.to_numeric(sc[sc_main_col], errors="coerce")
            if sc_mid_col:
                sc[sc_mid_col] = pd.to_numeric(sc[sc_mid_col], errors="coerce")
            keep_cols = [sc_name_col, sc_main_col] + ([sc_mid_col] if sc_mid_col else [])
            sc_dedup = (
                sc.dropna(subset=[sc_name_col, sc_main_col])[keep_cols]
                .drop_duplicates(subset=[sc_name_col], keep="first")
                .rename(columns={
                    sc_name_col: "직업명",
                    sc_main_col: "KNOW직업대분류",
                    **({sc_mid_col: "KNOW직업중분류"} if sc_mid_col else {}),
                })
            )
            sc_dedup["KNOW직업대분류"] = sc_dedup["KNOW직업대분류"].astype("Int64")
            if "KNOW직업중분류" in sc_dedup.columns:
                sc_dedup["KNOW직업중분류"] = sc_dedup["KNOW직업중분류"].astype("Int64")
            sc_dedup["대분류명"] = sc_dedup["KNOW직업대분류"].map(
                config.KNOW_MAIN_CATEGORY_NAMES
            )
            base = base.merge(sc_dedup, on="직업명", how="left")
        else:
            print(
                "⚠️ [subclass] KNOW직업명/KNOW직업대분류 컬럼 매핑 실패 — 대/중분류명 NA"
            )

    # 4-2) midclass(KNOW 중분류명) 매핑 — (E)
    if (
        "KNOW직업대분류" in base.columns
        and "KNOW직업중분류" in base.columns
        and "midclass" in bundle
        and not bundle["midclass"].empty
    ):
        mc = bundle["midclass"].copy()
        mc_main = _pick_column(mc, "KNOW직업대분류", ["KNOW직업대분류", "대분류"])
        mc_mid = _pick_column(mc, "KNOW직업중분류", ["KNOW직업중분류", "중분류"])
        mc_name = _pick_column(mc, "KNOW직업중분류명", ["KNOW직업중분류명", "중분류명"])
        if mc_main and mc_mid and mc_name:
            mc_join = mc[[mc_main, mc_mid, mc_name]].rename(columns={
                mc_main: "KNOW직업대분류",
                mc_mid: "KNOW직업중분류",
                mc_name: "중분류명",
            })
            mc_join["KNOW직업대분류"] = pd.to_numeric(
                mc_join["KNOW직업대분류"], errors="coerce"
            ).astype("Int64")
            mc_join["KNOW직업중분류"] = pd.to_numeric(
                mc_join["KNOW직업중분류"], errors="coerce"
            ).astype("Int64")
            mc_join = mc_join.drop_duplicates(
                subset=["KNOW직업대분류", "KNOW직업중분류"], keep="first"
            )
            base = base.merge(
                mc_join, on=["KNOW직업대분류", "KNOW직업중분류"], how="left"
            )

    for col in ("KNOW직업대분류", "KNOW직업중분류", "대분류명", "중분류명"):
        if col not in base.columns:
            base[col] = pd.NA

    # ── 4-2) 대분류명 도메인 추론 fallback ────────────────────────────
    # subclass.KNOW직업명 ↔ master.직업명 매칭에 실패해 대분류명이 NaN 으로
    # 남는 직업(약 35%)들에 대해, config.KNOW_DOMAIN_KEYWORDS 사전을 사용해
    # 직업명 substring 매칭으로 대분류명을 추론 보강한다.
    # 이후 step 5 / 6 의 도메인 평균 fallback 단계가 이 보강 결과를 이용해
    # 평균구인배율·평균부족률을 채울 수 있다.
    domain_kw_map: Dict[str, List[str]] = getattr(
        config, "KNOW_DOMAIN_KEYWORDS", {}
    ) or {}
    if domain_kw_map and "대분류명" in base.columns:
        # (키워드 길이 desc, 도메인 정의 순서) 평탄화
        flat_pairs: List[tuple] = []
        for di, (domain, kws) in enumerate(domain_kw_map.items()):
            for kw in kws or []:
                kw_s = str(kw or "").strip()
                if kw_s:
                    flat_pairs.append((len(kw_s), di, kw_s, domain))
        flat_pairs.sort(key=lambda t: (-t[0], t[1]))

        def _infer_domain(jobname: object) -> object:
            s = str(jobname or "").strip()
            if not s:
                return pd.NA
            for _ln, _di, kw_s, domain in flat_pairs:
                if kw_s in s:
                    return domain
            return pd.NA

        nan_mask = base["대분류명"].isna()
        n_before = int(nan_mask.sum())
        if n_before > 0 and "직업명" in base.columns:
            inferred = base.loc[nan_mask, "직업명"].apply(_infer_domain)
            success = inferred.notna()
            if success.any():
                base["대분류명"] = base["대분류명"].astype("object")
                base.loc[inferred[success].index, "대분류명"] = (
                    inferred[success].astype(str).values
                )
            n_filled = int(success.sum())
            print(
                f"✅ [대분류명 도메인 추론] 직업명 키워드 매칭으로 추가 보강 "
                f"{n_filled}/{n_before}건"
            )

    # ── 5) 고용동향 — 평균구인배율 3단계 매칭 ─────────────────────────
    # 1단계: EIS 직종명이 중분류명·대분류명 문자열에 포함
    # 2단계: 중·대분류명이 EIS 직종명에 포함(역방향)
    # 3단계: KNOW_JOBTYPE_KEYWORDS로 직업명 키워드 → EIS 직종명(rate_map)
    base["평균구인배율"] = pd.NA
    et_df = load_employment_trend()
    if not et_df.empty and "직종명" in et_df.columns:
        et_clean = et_df.dropna(subset=["직종명"]).copy()
        et_clean["직종명"] = et_clean["직종명"].astype(str).str.strip()
        et_clean = et_clean[et_clean["직종명"] != ""]
        et_clean = et_clean.assign(
            _len=et_clean["직종명"].astype(str).str.len()
        ).sort_values("_len", ascending=False)
        rate_pairs: List[tuple] = list(
            zip(et_clean["직종명"].tolist(), et_clean["평균구인배율"].tolist())
        )
        rate_map: Dict[str, object] = dict(rate_pairs)
        kw_map = getattr(config, "KNOW_JOBTYPE_KEYWORDS", None) or {}

        def find_rate(row: pd.Series):
            _top = row.get("대분류명")
            _mid = row.get("중분류명")
            대분류 = "" if pd.isna(_top) else str(_top).strip()
            중분류 = "" if pd.isna(_mid) else str(_mid).strip()
            _jn = row.get("직업명")
            직업명 = "" if pd.isna(_jn) else str(_jn).strip()

            for 직종명, rate in rate_pairs:
                if 직종명 in 중분류 or 직종명 in 대분류:
                    return rate, 1
            for 직종명, rate in rate_pairs:
                if 중분류 and 중분류 in 직종명:
                    return rate, 2
                if 대분류 and 대분류 in 직종명:
                    return rate, 2
            if isinstance(kw_map, dict) and kw_map and 직업명:
                for know_key, keywords in kw_map.items():
                    if know_key not in rate_map:
                        continue
                    if not keywords:
                        continue
                    if any(kw in 직업명 for kw in keywords):
                        return rate_map[know_key], 3
            return None, 0

        _tuples = base.apply(find_rate, axis=1)
        base["평균구인배율"] = [
            t[0] if t[0] is not None else pd.NA for t in _tuples
        ]
        stage_list = [t[1] for t in _tuples]
        matched_1 = stage_list.count(1)
        matched_2 = stage_list.count(2)
        matched_3 = stage_list.count(3)
        total_matched = matched_1 + matched_2 + matched_3
        print(
            f"✅ 구인배율 매핑: {total_matched}건 / {len(base)}건"
        )
        print(
            f"   (1단계 contains: {matched_1}건, "
            f"2단계 역방향: {matched_2}건, "
            f"3단계 키워드: {matched_3}건)"
        )

    # ── 5-2) 평균구인배율 도메인 평균 fallback ────────────────────────
    # 위 3단계로도 매칭되지 않은 행에 대해, 동일 대분류명에 속한 master
    # 다른 직업들의 평균구인배율 평균값을 적용한다 (step 4-2 도메인 추론으로
    # 대분류명이 채워진 행도 함께 대상). 데이터 단위 자체가 직종 중분류라
    # 대분류 평균을 사용하는 것은 fallback 으로서 합리적이다.
    if "대분류명" in base.columns and "평균구인배율" in base.columns:
        nan_mask = base["평균구인배율"].isna()
        n_before = int(nan_mask.sum())
        if n_before > 0:
            domain_avg = (
                base.dropna(subset=["대분류명", "평균구인배율"])
                .assign(평균구인배율=lambda d: pd.to_numeric(
                    d["평균구인배율"], errors="coerce"
                ))
                .dropna(subset=["평균구인배율"])
                .groupby("대분류명")["평균구인배율"]
                .mean()
                .round(2)
            )
            base["평균구인배율"] = base["평균구인배율"].astype("Float64")
            n_filled = 0
            for idx in base.index[nan_mask]:
                d = base.at[idx, "대분류명"]
                if pd.notna(d) and d in domain_avg.index:
                    base.at[idx, "평균구인배율"] = float(domain_avg.loc[d])
                    n_filled += 1
            print(
                f"✅ [구인배율 도메인 fallback] 동일 대분류명 평균 적용 "
                f"{n_filled}/{n_before}건"
            )

    # ── 6) 부족인력 (KNOW 대/중분류 코드 단위 평균부족률) ───────────────
    # labor_shortage.csv 의 '직종별' prefix 코드 2자리 = (대분류, 중분류) 이므로
    # ── 4) 에서 확보한 (KNOW직업대분류, KNOW직업중분류) 코드와 직접 join 한다.
    # 같은 중분류에 속하는 모든 직업은 동일한 부족률을 공유하지만, 이는 원본
    # 데이터 자체가 중분류 단위로 발표되는 한계에서 비롯된 정상 동작이다.
    ls = load_labor_shortage()
    if (
        not ls.empty
        and "KNOW직업대분류" in base.columns
        and "KNOW직업중분류" in base.columns
    ):
        ls_mid = (
            ls.groupby(["KNOW직업대분류", "KNOW직업중분류"], as_index=False)["평균부족률"]
            .mean()
        )
        ls_mid["평균부족률"] = ls_mid["평균부족률"].astype(float).round(1)
        base = base.merge(
            ls_mid, on=["KNOW직업대분류", "KNOW직업중분류"], how="left"
        )
    else:
        base["평균부족률"] = pd.NA

    # ── 6-2) 부족률 fallback: 직업명 ↔ labor_shortage.직종명 substring 매칭 ──
    # KNOW 코드(대/중분류)가 매핑 안 된 직업은 step 6) 에서 NaN 으로 남는다.
    # 사용자 지적대로 대분류명·중분류명이 NaN 이라 코드 join 이 불가능한
    # 케이스인데, labor_shortage 에는 '직종명' 컬럼(예: "음식 서비스직",
    # "디자이너", "예술인")이 있고 그 단위가 (대/중) 코드라서 충분히 의미
    # 있는 평균을 제공한다. NaN 행에 대해서만 직업명 ↔ 직종명 양방향
    # substring 매칭으로 대체값을 채운다.
    if (
        not ls.empty
        and "평균부족률" in base.columns
        and base["평균부족률"].isna().any()
    ):
        ls_by_name = (
            ls.dropna(subset=["직종명"])
            .groupby("직종명", as_index=False)["평균부족률"]
            .mean()
        )
        ls_by_name["평균부족률"] = ls_by_name["평균부족률"].astype(float).round(1)
        ls_by_name["직종명"] = ls_by_name["직종명"].astype(str).str.strip()
        ls_by_name = ls_by_name[ls_by_name["직종명"] != ""]
        ls_by_name = ls_by_name.assign(
            _len=ls_by_name["직종명"].str.len()
        ).sort_values("_len", ascending=False)
        name_pairs = list(
            zip(ls_by_name["직종명"].tolist(), ls_by_name["평균부족률"].tolist())
        )

        def _fallback_shortage(jobname: object) -> object:
            s = str(jobname or "").strip()
            if not s:
                return pd.NA
            for nm, val in name_pairs:
                nm_s = str(nm or "").strip()
                if not nm_s:
                    continue
                if nm_s in s or s in nm_s:
                    return val
            return pd.NA

        unmatched_mask = base["평균부족률"].isna()
        n_before = int(unmatched_mask.sum())
        if n_before > 0:
            base["평균부족률"] = base["평균부족률"].astype("Float64")
            fb_vals = base.loc[unmatched_mask, "직업명"].apply(_fallback_shortage)
            success = fb_vals.notna()
            if success.any():
                base.loc[fb_vals[success].index, "평균부족률"] = (
                    fb_vals[success].astype(float).values
                )
            n_filled = int(success.sum())
            print(
                f"✅ [부족률 fallback] 직업명↔직종명 substring 매칭으로 "
                f"추가 보강 {n_filled}/{n_before}건"
            )

    # ── 6-3) 평균부족률 도메인 평균 fallback ──────────────────────────
    # step 6 / 6-2 까지 거치고도 NaN 으로 남은 행은 KNOW 코드도, 직종명
    # substring 도 매칭되지 않은 케이스다. step 4-2 도메인 추론으로
    # 대분류명이 채워진 행을 포함해, 동일 대분류명에 속한 master 다른
    # 직업들의 평균부족률 평균값으로 채운다.
    if (
        "대분류명" in base.columns
        and "평균부족률" in base.columns
    ):
        nan_mask = base["평균부족률"].isna()
        n_before = int(nan_mask.sum())
        if n_before > 0:
            domain_avg_short = (
                base.dropna(subset=["대분류명", "평균부족률"])
                .assign(평균부족률=lambda d: pd.to_numeric(
                    d["평균부족률"], errors="coerce"
                ))
                .dropna(subset=["평균부족률"])
                .groupby("대분류명")["평균부족률"]
                .mean()
                .round(1)
            )
            base["평균부족률"] = base["평균부족률"].astype("Float64")
            n_filled = 0
            for idx in base.index[nan_mask]:
                d = base.at[idx, "대분류명"]
                if pd.notna(d) and d in domain_avg_short.index:
                    base.at[idx, "평균부족률"] = float(domain_avg_short.loc[d])
                    n_filled += 1
            print(
                f"✅ [부족률 도메인 fallback] 동일 대분류명 평균 적용 "
                f"{n_filled}/{n_before}건"
            )

    # ── 7) 임금정보 API (직업명 기준, 빈 DF면 skip) ────────────────────
    wage = load_job_wage_from_api()
    if not wage.empty:
        wage_sub = wage[["직업명", "중위임금", "하위임금"]].drop_duplicates(subset=["직업명"])
        base = base.merge(wage_sub, on="직업명", how="left")
    else:
        base["중위임금"] = pd.NA
        base["하위임금"] = pd.NA

    # ── 7-2) 임금통계 — KNOW↔KSCO 키워드 매핑 기반 3단계 병합 ─────────
    #
    # [기존 방식의 한계]
    # KNOW 대분류명(예: '미용·여행·숙박·음식·경비·청소·돌봄직')과 KSCO 직종명
    # (예: '음식 서비스직')은 텍스트 표기 자체가 달라 단순 substring contains
    # 매칭으로는 1~2건 외에 거의 매칭이 되지 않는다.
    #
    # [새 방식: KNOW_TO_KSCO_KEYWORDS 기반 3단계 매핑]
    # ① load_wage_statistics() 로 ws_df 확보.
    # ② config.KNOW_TO_KSCO_KEYWORDS 의 (KNOW직종 → KSCO키워드 리스트) 정의를
    #    사용해, KNOW직종별로 ws_df 에서 키워드 OR 매칭되는 행들의 평균 임금을
    #    선계산해 know_avg_wage 사전을 만든다.
    # ③ master 의 각 행에서 중분류명/대분류명에 KNOW직종 키가 substring으로
    #    포함되면 해당 평균 임금을 채택. 충돌 시 가장 긴 KNOW키(가장 구체적인
    #    매칭)를 우선한다.
    #
    # [기존 contains 방식 코드 — 참고용 주석 보존]
    # base["월평균임금_천원"] = pd.NA
    # ws = load_wage_statistics()
    # if not ws.empty:
    #     ws_match = ws.dropna(subset=["직종명_KSCO"]).copy()
    #     ws_match["직종명_KSCO"] = ws_match["직종명_KSCO"].astype(str).str.strip()
    #     ws_match = ws_match[ws_match["직종명_KSCO"] != ""]
    #     ws_match = ws_match.assign(
    #         _len=ws_match["직종명_KSCO"].astype(str).str.len()
    #     ).sort_values("_len", ascending=False)
    #     ws_pairs = list(
    #         zip(ws_match["직종명_KSCO"].tolist(), ws_match["월평균임금_천원"].tolist())
    #     )
    #     def _match_wage(row: pd.Series) -> object:
    #         top = row.get("대분류명")
    #         mid = row.get("중분류명")
    #         targets = [
    #             str(t).strip() for t in (top, mid)
    #             if pd.notna(t) and str(t).strip()
    #         ]
    #         if not targets:
    #             return pd.NA
    #         for ksco_name, val in ws_pairs:
    #             for tgt in targets:
    #                 if ksco_name in tgt or tgt in ksco_name:
    #                     return val
    #         return pd.NA
    #     base["월평균임금_천원"] = base.apply(_match_wage, axis=1)
    #     ws_matched = int(base["월평균임금_천원"].notna().sum())
    #     print(f"✅ 임금통계 병합 완료: {ws_matched}건 매칭")

    base["월평균임금_천원"] = pd.NA
    ws_df = load_wage_statistics()
    know_to_ksco: Dict[str, List[str]] = getattr(
        config, "KNOW_TO_KSCO_KEYWORDS", {}
    ) or {}

    if not ws_df.empty and know_to_ksco:
        ws_names = ws_df["직종명_KSCO"].astype(str).fillna("")

        # 단계 2: KNOW직종별 평균 임금 선계산
        know_avg_wage: Dict[str, float] = {}
        know_match_count: Dict[str, int] = {}
        for know_key, ksco_keywords in know_to_ksco.items():
            if not ksco_keywords:
                continue
            kws = list(ksco_keywords)
            mask = ws_names.apply(
                lambda x, kws=kws: any(kw in x for kw in kws)
            )
            matched_wages = ws_df.loc[mask, "월평균임금_천원"]
            if matched_wages.empty:
                continue
            avg = float(pd.to_numeric(matched_wages, errors="coerce").mean())
            if pd.notna(avg):
                know_avg_wage[know_key] = avg
                know_match_count[know_key] = int(matched_wages.shape[0])

        # 단계 3: master 행에 매핑 (가장 긴 KNOW 키 우선 = 구체성 우선)
        sorted_keys = sorted(know_avg_wage.keys(), key=len, reverse=True)

        def _resolve_wage(row: pd.Series) -> object:
            mid = row.get("중분류명")
            top = row.get("대분류명")
            mid_s = str(mid) if pd.notna(mid) else ""
            top_s = str(top) if pd.notna(top) else ""
            if not mid_s and not top_s:
                return pd.NA
            for know_key in sorted_keys:
                if know_key in mid_s or know_key in top_s:
                    return int(round(know_avg_wage[know_key]))
            return pd.NA

        base["월평균임금_천원"] = base.apply(_resolve_wage, axis=1)
        ws_matched = int(base["월평균임금_천원"].notna().sum())
        print(
            f"✅ KNOW↔KSCO 임금 매핑 완료: {ws_matched}건 "
            f"(전체 {len(base)}건 중 {ws_matched}건 매칭)"
        )

        # 디버그: 매핑된 KNOW직종별 평균임금 상위 10개
        if know_avg_wage:
            top10 = sorted(
                know_avg_wage.items(), key=lambda kv: kv[1], reverse=True
            )[:10]
            print("📊 KNOW직종별 평균임금 상위 10개 (KSCO 키워드 매칭 기반):")
            for know_key, avg in top10:
                n_ksco = know_match_count.get(know_key, 0)
                print(
                    f"   - {know_key:<35s} "
                    f"{avg:>8,.0f}천원  (KSCO {n_ksco}개 평균)"
                )
    elif not ws_df.empty:
        print("⚠️ config.KNOW_TO_KSCO_KEYWORDS 가 비어 있어 임금통계 매핑 생략")

    # ── 8) 전공별 진출직업 (진출순위 1위만, 진출직업명↔직업명) ─────────
    # 사용자 데이터에 진출순위/진출비율이 없을 수 있으므로 옵셔널 처리:
    #   - 진출순위 컬럼이 존재하고 숫자로 변환 가능하면 1위만 채택
    #   - 그 외에는 직업명 기준으로 첫 등장 행을 1순위로 간주 (drop_duplicates)
    tj = load_top_jobs_by_major()
    if not tj.empty:
        tj_top = tj.copy()
        rank_numeric = pd.to_numeric(tj_top.get("진출순위"), errors="coerce")
        if rank_numeric.notna().any():
            tj_top = tj_top[rank_numeric == 1]
        tj_top = (
            tj_top.rename(columns={"진출직업명": "직업명"})[["직업명", "계열명"]]
            .dropna(subset=["직업명"])
            .drop_duplicates(subset=["직업명"], keep="first")
        )
        base = base.merge(tj_top, on="직업명", how="left")
    if "계열명" not in base.columns:
        base["계열명"] = pd.NA

    # ── 9) 첫 취업률 — long format(첫일자리진출소요기간 카테고리) wide pivot 결과를 계열명별 평균 ────
    # 사용자 데이터에는 '졸업전취업률'/'3개월이내취업률' 카테고리가 없고,
    # '취업률_6개월', '취업률_1년', '취업률_1년이상' 등이 들어 있다.
    # → 14 핵심 컬럼 슬롯(졸업전/3개월이내)은 NA로 두고, 가용 카테고리는
    #   master에 추가 컬럼으로 붙인다(동적 확장).
    #
    # 도메인 매핑 (A):
    #   master.주요전공계열 ('공학', '사회', '예체능', '자연', '의학', '인문', '교육')
    #   ↔ first_job_entry.계열명 ('공학계열', '사회계열', ...)
    #   → config.EDU_CATEGORY_TO_FACULTY 로 도메인 변환 후 join.
    fj = load_first_job_entry()
    fj_avail_cols: List[str] = []
    if not fj.empty and "주요전공계열" in base.columns:
        base["_계열명_매칭키"] = (
            base["주요전공계열"].map(config.EDU_CATEGORY_TO_FACULTY)
        )
        cat_cols = [c for c in fj.columns if c.startswith("취업률_")]
        if cat_cols:
            fj_sub = (
                fj.groupby("계열명", as_index=False)[cat_cols]
                .mean(numeric_only=True)
                .rename(columns={"계열명": "_계열명_매칭키"})
            )
            base = base.merge(fj_sub, on="_계열명_매칭키", how="left")
            fj_avail_cols = [c for c in cat_cols if c in base.columns]
        base = base.drop(columns=["_계열명_매칭키"])
    for col in ("졸업전취업률", "3개월이내취업률"):
        if col not in base.columns:
            base[col] = pd.NA

    # ── 10) NCS 경로 (직무명↔직업명, 전직가능직무명 합산) ──────────────
    nc = load_ncs_career_path()
    if not nc.empty:
        nc_sub = (
            nc.rename(columns={"직무명": "직업명"})[["직업명", "전직가능직무명"]]
            .groupby("직업명", as_index=False)["전직가능직무명"]
            .agg(lambda s: ", ".join(map(str, dict.fromkeys(s.astype(str)))))
        )
        base = base.merge(nc_sub, on="직업명", how="left")
    else:
        base["전직가능직무명"] = pd.NA

    # ── 11) 직업명 정리 (NaN/공백 제거) ────────────────────────────────
    n_before_drop = len(base)
    base = base[base["직업명"].notna()]
    base = base[base["직업명"].astype(str).str.strip() != ""]
    n_dropped = n_before_drop - len(base)
    if n_dropped > 0:
        print(f"[직업명 정리] 직업명 추출 실패 행 {n_dropped}건 제거됨")

    # ── 12) 핵심 컬럼만 정렬해 master DataFrame 구성 ─────────────────────
    # 14 핵심 컬럼 + first_job_entry pivot에서 가용한 카테고리 컬럼(동적 확장).
    final_cols = [
        "직업코드", "직업명",
        "직업전망_텍스트", "전망점수",
        "주요전공계열", "유사직업명_합산", "대분류명", "중분류명",
        "평균구인배율", "평균부족률",
        "중위임금", "하위임금", "월평균임금_천원",
        "졸업전취업률", "3개월이내취업률",
        "전직가능직무명",
    ]
    extra_cols = [c for c in fj_avail_cols if c in base.columns and c not in final_cols]
    for col in final_cols:
        if col not in base.columns:
            base[col] = pd.NA
    master = base[final_cols + extra_cols].copy()
    if extra_cols:
        print(f"[master] first_job_entry 추가 카테고리 컬럼 포함: {extra_cols}")

    out_path = Path(config.MASTER_JOB_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== 마스터 데이터 채움률 진단 ===")
    total = len(master)
    for col in [
        "평균구인배율",
        "평균부족률",
        "월평균임금_천원",
        "전망점수",
        "대분류명",
        "중분류명",
    ]:
        if col in master.columns:
            filled = master[col].notna().sum()
            pct = filled / total * 100 if total else 0.0
            status = "✅" if pct > 30 else "⚠️"
            print(f"  {status} {col}: {filled}/{total} ({pct:.1f}%)")

    no_rate = master[
        master["평균구인배율"].isna()
        | (master["평균구인배율"] == 0)
    ].head(5)
    if not no_rate.empty:
        print("\n구인배율 미매핑 샘플 (상위 5개):")
        for _, r in no_rate.iterrows():
            print(
                f"  직업명={r.get('직업명', '?')}, "
                f"대분류={r.get('대분류명', '?')}, "
                f"중분류={r.get('중분류명', '?')}"
            )

    no_short = master[master["평균부족률"].isna()].head(5)
    if not no_short.empty:
        print("\n부족률 미매핑 샘플 (상위 5개):")
        for _, r in no_short.iterrows():
            print(
                f"  직업명={r.get('직업명', '?')}, "
                f"대분류={r.get('대분류명', '?')}, "
                f"중분류={r.get('중분류명', '?')}"
            )

    master.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ master_job_data.csv 저장 완료: {len(master):,}행 × {master.shape[1]}컬럼")
    return master


# ──────────────────────────────────────────────────────────────────────────────
# 기능 8) [Phase2 예정] EIS 보완 데이터 로더
# ──────────────────────────────────────────────────────────────────────────────
def load_eis_supplement() -> Dict[str, pd.DataFrame]:
    """
    [Phase2 예정] EIS 고용행정통계 보완 데이터 로더.

    현재는 data/eis/ 폴더 내 수동 다운로드 파일만 처리한다.
    Phase2에서 EIS API 연동으로 교체될 예정이다.

    동작:
        - config.EIS_STATISTICS_FILES의 각 파일 존재 여부 확인.
        - 존재하면 인코딩 자동 감지로 로드해 dict에 담아 반환.
        - 없으면 표준 경고를 출력하고 dict에서 제외.

    Returns:
        dict[str, pandas.DataFrame]: {논리키 → DataFrame}.
                                     없는 파일만 있으면 빈 dict 반환.
    """
    files: Dict[str, str] = getattr(config, "EIS_STATISTICS_FILES", {}) or {}
    if not files:
        print("⚠️ config.EIS_STATISTICS_FILES가 비어 있습니다 — 빈 dict 반환")
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        p = Path(path)
        if not p.exists():
            _log_missing(p.name)
            continue
        try:
            df = load_csv(p)
            _log_ok(p.name, df)
            result[key] = df
        except Exception as e:
            print(f"⚠️ [{p.name}] 로드 실패 ({e}) — 건너뜀")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 기능 9) 통합 파이프라인: build_master_job_data + EIS 보완 병합
# ──────────────────────────────────────────────────────────────────────────────
def build_master_with_eis(
    eis_categories: Sequence[str] = ("job_demand",),
    save: bool = True,
) -> pd.DataFrame:
    """
    공개 공공데이터 기반 마스터 데이터셋에 EIS 보완 데이터를 병합한다.

    동작 순서:
        1) build_master_job_data() 호출 → 기준 마스터 생성 (이때 1차 CSV 저장됨).
        2) eis_categories에 나열된 EIS 카테고리를 차례로 left join.
           (각 카테고리는 config.EIS_STATISTICS_FILES의 키여야 한다)
        3) save=True이면 EIS 보완 결과를 다시 config.MASTER_JOB_FILE에 덮어쓴다.

    EIS 파일이 없으면 merge_eis_to_master가 안내 메시지를 출력하고
    원본 마스터를 그대로 반환하므로, EIS 미설치 환경에서도 안전하게 동작한다.
    (즉, 이 함수의 출력은 항상 build_master_job_data()의 상위호환이다.)

    Args:
        eis_categories: 병합할 EIS 카테고리 시퀀스. 기본값 ('job_demand',).
                        예: ('job_demand', 'regional')
        save: True이면 EIS 보완 결과를 MASTER_JOB_FILE에 저장한다.

    Returns:
        pandas.DataFrame: EIS 보완이 시도된 마스터 데이터셋.
                          기준 마스터가 비어 있으면 빈 DataFrame.
    """
    # 순환 임포트 방지를 위한 함수 내부 lazy import
    from utils.eis_loader import load_eis_statistics, merge_eis_to_master

    print("\n[build_master_with_eis] (1/2) 기준 마스터 데이터 생성")
    master = build_master_job_data()

    if master.empty:
        print("\n[build_master_with_eis] 기준 마스터가 비어 있어 EIS 병합을 건너뜁니다.")
        return master

    print(f"\n[build_master_with_eis] (2/2) EIS 보완 병합 — 카테고리: {list(eis_categories)}")
    cols_before = master.shape[1]

    for category in eis_categories:
        eis_df = load_eis_statistics(category)

        if eis_df.empty:
            print(f"⏸ EIS '{category}' 데이터 없음 — 건너뜀")
            continue

        # ── 새 로직: 평균구인배율_EIS 컬럼이 있으면 직종명_정제 ↔ 대분류명/중분류명
        # contains 매칭으로 평균구인배율 빈 곳을 보완 ────────────────────
        if "평균구인배율_EIS" in eis_df.columns and "직종명_정제" in eis_df.columns:
            eis_rate_map = dict(
                zip(eis_df["직종명_정제"], eis_df["평균구인배율_EIS"])
            )
            보완_count = 0
            for idx, row in master.iterrows():
                # 기존 평균구인배율이 NaN 이거나 0 인 경우만 보완 대상
                current = row.get("평균구인배율", None)
                try:
                    if pd.notna(current) and float(current) != 0:
                        continue
                except (TypeError, ValueError):
                    pass

                대분류 = str(row.get("대분류명", ""))
                중분류 = str(row.get("중분류명", ""))

                for eis_name, rate in eis_rate_map.items():
                    if not eis_name:
                        continue
                    if (
                        eis_name in 대분류
                        or eis_name in 중분류
                        or (대분류 and 대분류 in eis_name)
                        or (중분류 and 중분류 in eis_name)
                    ):
                        master.at[idx, "평균구인배율"] = rate
                        보완_count += 1
                        break

            print(f"✅ EIS '{category}' 구인배율 보완: {보완_count}건 추가 매핑")
        else:
            # 레거시 경로: 직종코드/직업코드/직종명 join 가능한 형식이면 그대로 병합
            master = merge_eis_to_master(master, category=category)

    cols_added = master.shape[1] - cols_before

    if save:
        out_path = Path(config.MASTER_JOB_FILE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"\n✅ EIS 보완 master_job_data 저장 완료: "
            f"{len(master):,}행 × {master.shape[1]}컬럼 (EIS로 +{cols_added}컬럼)"
        )
    else:
        print(
            f"\n[build_master_with_eis] 메모리 결과만 반환 "
            f"(shape={master.shape}, EIS로 +{cols_added}컬럼)"
        )

    return master


# ──────────────────────────────────────────────────────────────────────────────
# 단독 실행 시: 통합 파이프라인 빌드 + 요약 점검
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[JIB] data_loader 단독 실행")
    print("─" * 70)
    print("STEP 0) 임금정보 API 응답 점검 (check_wage_api_response)")
    check_wage_api_response()
    print("─" * 70)
    print("STEP 0-1) load_employment_trend() 단독 테스트")
    et_df = load_employment_trend()
    if et_df.empty:
        print("(빈 DataFrame)")
    else:
        print(et_df.head(10).to_string(index=False))
    print("─" * 70)
    print("STEP 0-2) load_wage_statistics() 단독 테스트")
    ws_df = load_wage_statistics()
    if ws_df.empty:
        print("(빈 DataFrame)")
    else:
        print(ws_df.head(10).to_string(index=False))
    print("─" * 70)
    print("STEP 1) 통합 파이프라인 빌드 (build_master_with_eis)\n")
    master_df = build_master_with_eis()

    print("\n=== df.head(5) ===")
    if master_df.empty:
        print("(빈 DataFrame)")
    else:
        print(master_df.head(5).to_string(index=False))

    print(f"\n=== df.shape ===\n{master_df.shape}")

    null_counts = master_df.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    print("\n=== null > 0 컬럼 ===")
    if null_counts.empty:
        print("(null이 있는 컬럼 없음)")
    else:
        print(null_counts.to_string())

    if "전망점수" in master_df.columns:
        print("\n=== 전망점수 value_counts() ===")
        print(master_df["전망점수"].value_counts(dropna=False).to_string())
