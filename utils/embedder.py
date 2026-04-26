"""
JIB(Job_Insight_Bridge) — 임베딩 모듈 (파일 기반 v2).

목표:
    - 로컬 개발: 파일(master_job_data.csv + ncs_career_path.csv) 기반 코퍼스로
      OpenAI 임베딩을 생성한다.
    - 배포 후: config.WORKNET_* / config.EMBEDDING_MODEL 등 URL/모델만
      바꾸면 동일 인터페이스를 유지하며 외부 API 전환이 가능하도록 설계한다.
    - chardet 등 선택적 의존성 없이 안정적으로 동작한다 (인코딩은 순차 시도).

기능 구성:
    1) build_corpus_from_files() — master + ncs 파일 병합 → embed_text 코퍼스 생성
    2) generate_embeddings()      — embed_text → OpenAI text-embedding-3-small
    3) load_embeddings()          — 저장된 임베딩을 numpy 벡터로 복원해 로드
"""

from __future__ import annotations

import io
import json
import os
import sys as _sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# 프로젝트 루트를 sys.path 에 추가해 단독 실행 시에도 config import 가 가능하도록 한다.
if str(Path(__file__).resolve().parents[1]) not in _sys.path:
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (sys.path 보정 직후 import)


# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
# OpenAI text-embedding-3-small 가격 (2025년 4월 기준 $0.02 / 1M tokens).
EMBEDDING_PRICE_PER_MTOKEN_USD: float = 0.02
# 비용 추정용 토큰 환산비 (대략 4 chars ≈ 1 token).
CHARS_PER_TOKEN: float = 4.0
# 배치 크기 / 호출 간격.
BATCH_SIZE: int = 50
BATCH_SLEEP_SEC: float = 0.5
# embed_text 최소 길이 (이보다 짧으면 의미 없는 텍스트로 간주).
MIN_EMBED_TEXT_LEN: int = 10
# 임베딩 캐시 파일 무효화 임계 (이 크기 이하면 빈 파일로 간주하고 자동 삭제).
EMPTY_FILE_THRESHOLD_BYTES: int = 100


def _resolve_path(p: Union[str, os.PathLike]) -> Path:
    """프로젝트 루트 기준 상대 경로를 절대 경로로 변환한다."""
    pth = Path(p)
    if pth.is_absolute():
        return pth
    root = Path(__file__).resolve().parents[1]
    return (root / pth).resolve()


def _safe_load_csv(filepath: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    인코딩 자동 감지로 CSV 를 안전하게 로드한다.

    시도 순서: utf-8-sig → cp949 → utf-8 → latin-1.
    각 시도는 errors='replace' 로 디코딩하지만, U+FFFD 치환 문자 비율이
    1% 를 초과하면 잘못된 인코딩으로 간주하고 다음 후보로 넘어간다.
    파일이 없으면 경고 출력 후 빈 DataFrame 을 반환한다.
    """
    path = _resolve_path(filepath)
    if not path.exists():
        print(f"⚠️ 파일 없음: {path}")
        return pd.DataFrame()

    encodings = ("utf-8-sig", "cp949", "utf-8", "latin-1")
    last_err: Optional[Exception] = None
    fallback_df: Optional[pd.DataFrame] = None
    fallback_enc: Optional[str] = None

    for enc in encodings:
        try:
            # open(errors='replace') + StringIO 로 chardet 의존 없이 안전 디코딩.
            with open(path, "r", encoding=enc, errors="replace") as f:
                text = f.read()
            # 디코딩 품질 검증 — replace 로 인해 모든 인코딩이 통과하는 문제 방지.
            replacement_ratio = text.count("\ufffd") / max(len(text), 1)
            df = pd.read_csv(io.StringIO(text))
            if replacement_ratio <= 0.01:
                return df
            # 깨진 문자가 많으면 다음 인코딩 시도. 하나도 못 고르면 마지막 시도 반환용으로 보관.
            if fallback_df is None:
                fallback_df = df
                fallback_enc = enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # CSV 파싱 오류 등은 다음 인코딩 시도로 넘어간다.
            last_err = e
            continue

    if fallback_df is not None:
        print(
            f"⚠️ 모든 인코딩에서 깨진 문자 1% 초과 — 가장 먼저 통과한 '{fallback_enc}' "
            f"결과를 반환합니다: {path}"
        )
        return fallback_df

    print(f"⚠️ 모든 인코딩 시도 실패: {path} (마지막 오류: {last_err})")
    return pd.DataFrame()


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """후보 컬럼명 중 DataFrame에 가장 먼저 존재하는 것을 반환한다."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _is_meaningful(value) -> bool:
    """embed_text 구성 시 의미 있는 값인지 검사 (None / NaN / 'nan' / 빈 문자열 제외)."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s != "" and s.lower() not in ("nan", "none")


# ──────────────────────────────────────────────────────────────────────────────
# 기능 1) 파일 기반 코퍼스 구성
# ──────────────────────────────────────────────────────────────────────────────
def build_corpus_from_files() -> pd.DataFrame:
    """
    master_job_data.csv + ncs_career_path.csv 를 결합해 RAG 임베딩용 코퍼스를 만든다.

    파이프라인:
        STEP 1) master 로드 — 사용 가능 컬럼만 추려낸다.
        STEP 2) ncs_career_path 로드 — 컬럼명 자동 감지 후 표준 스키마로 정규화.
        STEP 3) master.직업명 ↔ ncs.직무명 3단계 매칭으로 NCS 메타데이터 보강.
        STEP 4) embed_text 통합 텍스트 컬럼 생성.
        STEP 5) embed_text 길이 필터 + config.NCS_CORPUS_FILE 저장.
    """
    # ── STEP 1: master 로드 ────────────────────────────────────────────
    print("=== STEP 1: master_job_data.csv 로드 ===")
    master = _safe_load_csv(config.MASTER_JOB_FILE)
    if master.empty:
        print("⚠️ master 가 비어 있어 코퍼스를 만들 수 없습니다.")
        return pd.DataFrame()

    desired_master_cols = [
        "직업코드", "직업명", "대분류명", "중분류명",
        "직업전망_텍스트", "전망점수", "주요전공계열",
        "유사직업명_합산", "전직가능직무명",
        "월평균임금_천원", "평균구인배율",
    ]
    available = [c for c in desired_master_cols if c in master.columns]
    missing = [c for c in desired_master_cols if c not in master.columns]
    master = master[available].copy()
    print(f"  📥 master: {len(master)}행, 사용 컬럼 {len(available)}개")
    if missing:
        print(f"  ℹ️ master 누락 컬럼(skip): {missing}")

    # ── STEP 2: ncs_career_path 로드 + 컬럼 표준화 ────────────────────
    print("\n=== STEP 2: ncs_career_path.csv 로드 ===")
    ncs_path = os.path.join(config.RAW_DATA_DIR, "ncs_career_path.csv")
    ncs_raw = _safe_load_csv(ncs_path)

    std_cols = ("대분류명", "중분류명", "소분류명", "직무명", "전직가능직무명")
    if ncs_raw.empty:
        print("  ⚠️ ncs_career_path.csv 로드 실패 — STEP 3 매칭 생략")
        ncs = pd.DataFrame(columns=list(std_cols))
    else:
        col_candidates = {
            "대분류명":     ["대분류명", "NCS대분류명", "대분류"],
            "중분류명":     ["중분류명", "NCS중분류명", "중분류"],
            "소분류명":     ["소분류명", "NCS소분류명", "소분류"],
            "직무명":       ["직무명", "NCS능력단위명", "능력단위명", "직무"],
            "전직가능직무명": ["전직가능직무명", "전직직무명", "관련직무명"],
        }
        rename_map = {}
        picked_summary = {}
        for std, cands in col_candidates.items():
            picked = _pick_column(ncs_raw, cands)
            picked_summary[std] = picked
            if picked and picked != std:
                rename_map[picked] = std
        ncs = ncs_raw.rename(columns=rename_map).copy()
        for std in std_cols:
            if std not in ncs.columns:
                ncs[std] = ""
        ncs = ncs[list(std_cols)].copy()
        ncs["직무명"] = ncs["직무명"].astype(str).str.strip()
        ncs = ncs[ncs["직무명"] != ""]
        before = len(ncs)
        ncs = ncs.drop_duplicates(subset=["직무명"], keep="first").reset_index(drop=True)
        print(f"  📥 ncs: {before}행 → 직무명 기준 중복 제거 후 {len(ncs)}행")
        print(f"  🔍 컬럼 매핑: {picked_summary}")

    # ── STEP 3: 직업명 ↔ 직무명 3단계 매칭 ───────────────────────────
    print("\n=== STEP 3: master.직업명 ↔ ncs.직무명 매칭 ===")
    for col in ("NCS대분류명", "NCS중분류명", "NCS소분류명"):
        master[col] = pd.NA
    if "전직가능직무명" not in master.columns:
        master["전직가능직무명"] = pd.NA

    n_matched = 0
    if not ncs.empty and "직업명" in master.columns:
        # 1차(정확 일치) 빠른 조회용 인덱스.
        ncs_index = ncs.set_index("직무명")
        ncs_jobnames: List[str] = ncs["직무명"].tolist()

        def _row_from_lookup(jn: str):
            row = ncs_index.loc[jn]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row

        for idx, master_row in master.iterrows():
            raw_name = master_row.get("직업명", "")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            job_name = raw_name.strip()

            ncs_row = None

            # 1차: 정확 일치
            if job_name in ncs_index.index:
                ncs_row = _row_from_lookup(job_name)

            # 2차: 직업명 ⊂ 직무명 (master.직업명이 ncs.직무명에 포함)
            if ncs_row is None:
                for jn in ncs_jobnames:
                    if isinstance(jn, str) and jn and job_name in jn:
                        ncs_row = _row_from_lookup(jn)
                        break

            # 3차: 직무명 ⊂ 직업명 (ncs.직무명이 master.직업명에 포함)
            if ncs_row is None:
                for jn in ncs_jobnames:
                    if isinstance(jn, str) and jn and jn in job_name:
                        ncs_row = _row_from_lookup(jn)
                        break

            if ncs_row is not None:
                master.at[idx, "NCS대분류명"] = ncs_row.get("대분류명", pd.NA)
                master.at[idx, "NCS중분류명"] = ncs_row.get("중분류명", pd.NA)
                master.at[idx, "NCS소분류명"] = ncs_row.get("소분류명", pd.NA)

                # master.전직가능직무명이 비어 있을 때만 ncs 값으로 보강.
                cur = master.at[idx, "전직가능직무명"]
                cur_str = "" if (cur is None or (isinstance(cur, float) and pd.isna(cur))) else str(cur).strip()
                ncs_jeon = ncs_row.get("전직가능직무명", "")
                if not cur_str and isinstance(ncs_jeon, str) and ncs_jeon.strip():
                    master.at[idx, "전직가능직무명"] = ncs_jeon.strip()

                n_matched += 1

    print(f"  🔗 NCS 매칭: {n_matched}건 / {len(master)}건")

    # ── STEP 4: embed_text 통합 텍스트 생성 ──────────────────────────
    print("\n=== STEP 4: embed_text 통합 텍스트 생성 ===")

    def make_embed_text(row: pd.Series) -> str:
        parts: List[str] = []
        직업명 = str(row.get("직업명", ""))
        대분류명 = str(row.get("대분류명", ""))
        중분류명 = str(row.get("중분류명", ""))
        소분류명 = str(row.get("NCS소분류명", ""))
        직업전망 = str(row.get("직업전망_텍스트", ""))[:300]
        주요전공 = str(row.get("주요전공계열", ""))
        유사직업 = str(row.get("유사직업명_합산", ""))
        전직직무 = str(row.get("전직가능직무명", ""))

        for val in (
            직업명, 대분류명, 중분류명, 소분류명,
            직업전망, 주요전공, 유사직업, 전직직무,
        ):
            if val and val.strip() not in ("nan", "None", ""):
                parts.append(val.strip())

        text = " ".join(parts)
        return " ".join(text.split())

    master["embed_text"] = master.apply(make_embed_text, axis=1)

    # ── STEP 5: 길이 필터 + 저장 ──────────────────────────────────────
    print("\n=== STEP 5: embed_text 길이 필터 및 저장 ===")
    before_filter = len(master)
    master = master[master["embed_text"].str.len() >= MIN_EMBED_TEXT_LEN].reset_index(drop=True)
    print(f"  📏 길이 < {MIN_EMBED_TEXT_LEN} 행 제거: {before_filter} → {len(master)}")

    out_path = _resolve_path(config.NCS_CORPUS_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False, encoding="utf-8-sig")

    avg_len = master["embed_text"].str.len().mean() if len(master) > 0 else 0
    print(f"✅ 파일 기반 코퍼스 완성: {len(master)}행, 평균 embed_text 길이: {avg_len:.0f}자")
    print(f"   저장 경로: {out_path}")
    return master


# ──────────────────────────────────────────────────────────────────────────────
# 기능 2) OpenAI 임베딩 생성
# ──────────────────────────────────────────────────────────────────────────────
def generate_embeddings(
    corpus_df: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    NCS 코퍼스 embed_text 를 OpenAI 임베딩 벡터로 변환해 EMBEDDINGS_FILE 에 저장한다.

    Args:
        corpus_df: 입력 코퍼스. None 이면 build_corpus_from_files() 를 호출해 생성.
        force: True 면 기존 임베딩 파일을 덮어쓴다.

    Returns:
        pandas.DataFrame: 코퍼스 + 'embedding' (JSON 문자열) 컬럼이 포함된 DataFrame.
    """
    out_path = _resolve_path(config.EMBEDDINGS_FILE)

    # ── 캐시 처리 ──────────────────────────────────────────────────────
    if out_path.exists() and out_path.stat().st_size > EMPTY_FILE_THRESHOLD_BYTES and not force:
        print(f"⚠️ 임베딩 파일 존재. force=True로 재생성 가능. ({out_path})")
        try:
            df = pd.read_csv(out_path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(out_path)
        return df
    if out_path.exists() and out_path.stat().st_size <= EMPTY_FILE_THRESHOLD_BYTES:
        try:
            out_path.unlink()
            print("  🗑️ 빈 임베딩 파일 삭제, 재생성합니다.")
        except OSError as e:
            print(f"  ⚠️ 빈 임베딩 파일 삭제 실패: {e}")

    # ── 코퍼스 준비 ────────────────────────────────────────────────────
    if corpus_df is None:
        corpus_df = build_corpus_from_files()

    if corpus_df is None or corpus_df.empty:
        print("⚠️ 코퍼스가 비어 있습니다 — 빈 DataFrame 반환")
        return pd.DataFrame()
    if "embed_text" not in corpus_df.columns:
        print("⚠️ embed_text 컬럼이 없습니다 — 빈 DataFrame 반환")
        return pd.DataFrame()

    corpus_df = corpus_df.copy()
    corpus_df["embed_text"] = corpus_df["embed_text"].astype(str).fillna("")
    corpus_df = corpus_df[corpus_df["embed_text"].str.strip() != ""].reset_index(drop=True)
    if corpus_df.empty:
        print("⚠️ embed_text 비어있는 행 제거 후 코퍼스가 비었습니다.")
        return corpus_df

    # ── 비용 사전 안내 ─────────────────────────────────────────────────
    total_chars = int(corpus_df["embed_text"].str.len().sum())
    est_tokens = total_chars / CHARS_PER_TOKEN
    est_cost = est_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MTOKEN_USD
    print(f"📊 임베딩 대상: {len(corpus_df)}행")
    print(f"💰 예상 비용: 약 ${est_cost:.4f} USD (text-embedding-3-small)")
    print("   계속하려면 Enter, 취소하려면 Ctrl+C")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️ 사용자가 취소했습니다 — 빈 DataFrame 반환")
        return pd.DataFrame()

    # ── OpenAI 클라이언트 ──────────────────────────────────────────────
    api_key = (getattr(config, "OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        print("⚠️ OPENAI_API_KEY 미설정 — config.py/.env 를 확인해주세요.")
        return pd.DataFrame()

    try:
        from openai import OpenAI  # 지연 import: 모듈 임포트 시 키 검증 회피
    except ImportError:
        print("⚠️ openai 패키지 미설치 — pip install openai")
        return pd.DataFrame()

    client = OpenAI(api_key=api_key)
    model = getattr(config, "EMBEDDING_MODEL", "text-embedding-3-small")

    # ── 배치 임베딩 ────────────────────────────────────────────────────
    texts: List[str] = corpus_df["embed_text"].tolist()
    n = len(texts)
    n_batches = (n - 1) // BATCH_SIZE + 1
    all_embeddings: List[Optional[List[float]]] = []

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        try:
            response = client.embeddings.create(model=model, input=batch)
            batch_vecs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_vecs)
            print(
                f"  🔢 배치 {i // BATCH_SIZE + 1}/{n_batches} 완료 "
                f"({i + len(batch)}건)"
            )
            time.sleep(BATCH_SLEEP_SEC)
        except Exception as e:
            print(f"  ⚠️ 배치 오류 — 해당 배치 skip: {e}")
            all_embeddings.extend([None] * len(batch))

    # ── 저장 ───────────────────────────────────────────────────────────
    corpus_df["embedding"] = [
        json.dumps(v) if v is not None else None
        for v in all_embeddings
    ]
    corpus_df = corpus_df.dropna(subset=["embedding"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 임베딩 저장 완료: {len(corpus_df)}건 → {out_path}")
    return corpus_df


# ──────────────────────────────────────────────────────────────────────────────
# 기능 3) 임베딩 로드 (numpy 벡터 복원)
# ──────────────────────────────────────────────────────────────────────────────
def load_embeddings() -> pd.DataFrame:
    """
    저장된 EMBEDDINGS_FILE 을 로드하고 'embedding' 컬럼(JSON 문자열)을
    'embedding_vector' 컬럼(numpy 배열)으로 함께 복원한다.
    """
    out_path = _resolve_path(config.EMBEDDINGS_FILE)
    if not out_path.exists():
        print(f"⚠️ 임베딩 파일이 없습니다: {out_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(out_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(out_path)

    if "embedding" not in df.columns:
        print("⚠️ embedding 컬럼이 없습니다.")
        return df

    def _to_vec(x):
        if not isinstance(x, str) or not x.strip():
            return None
        if not pd.notna(x):
            return None
        try:
            return np.array(json.loads(x))
        except Exception:
            return None

    df["embedding_vector"] = df["embedding"].apply(_to_vec)

    valid = df["embedding_vector"].dropna()
    dim = len(valid.iloc[0]) if not valid.empty else 0
    print(f"✅ 임베딩 로드: {len(df)}건, 벡터 차원: {dim}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 단독 실행 테스트 블록
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== STEP 1: 파일 기반 코퍼스 구성 ===")
    corpus = build_corpus_from_files()
    print(f"\n코퍼스 요약: {len(corpus)}행")
    if not corpus.empty:
        print("embed_text 샘플 (상위 3개):")
        for _, row in corpus.head(3).iterrows():
            name = row.get("직업명", "?")
            text = str(row.get("embed_text", ""))[:120]
            print(f"  [{name}] {text}...")

    print("\n=== STEP 2: 임베딩 생성 ===")
    emb_df = generate_embeddings(corpus, force=True)
    print(f"\n최종: {len(emb_df)}행 임베딩 완료")
    if len(emb_df) > 0 and "embedding" in emb_df.columns:
        sample = emb_df["embedding"].iloc[0]
        if isinstance(sample, str) and sample:
            print(f"벡터 차원: {len(json.loads(sample))}")
