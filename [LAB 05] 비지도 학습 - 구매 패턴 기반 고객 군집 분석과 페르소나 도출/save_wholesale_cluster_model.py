# -*- coding: utf-8 -*-
"""
[LAB 05] 구매 패턴 기반 고객 군집 분석 모델을 pkl로 저장하는 스크립트
- 노트북과 동일한 전처리·PCA·KMeans 파이프라인을 적용한 뒤 모델을 저장합니다.
- Tableau + TabPy에서 이 pkl을 불러와 클러스터 ID를 계산할 수 있습니다.

사용 방법:
  1) 노트북에서 사용한 데이터를 CSV로 저장한 뒤, 아래 data_path를 해당 CSV로 지정하거나
  2) 같은 환경에서 노트북을 실행한 후, 이 스크립트 대신 노트북 맨 아래의
     "모델 pkl 저장" 셀을 실행해도 됩니다.

실행: python save_wholesale_cluster_model.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 설정: 데이터 경로 및 출력 pkl 경로
# ---------------------------------------------------------------------------
LAB05_DIR = Path(__file__).resolve().parent
# 데이터: 노트북에서 df (Channel, Region 제거 후 6개 컬럼) 에 해당하는 CSV 경로
# 예: 노트북에서 df.to_csv('wholesale_customers_6cols.csv', index=False) 로 저장 후 사용
data_path = LAB05_DIR / "wholesale_customers_6cols.csv"
# pkl 저장 경로 (TabPy 스크립트에서 같은 경로를 참조해야 함)
pkl_path = LAB05_DIR / "wholesale_cluster_model.pkl"

# 데이터가 없으면 샘플로 6컬럼만 만들어서 학습 (실제로는 CSV 또는 노트북 실행 권장)
RAW_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']


def load_data():
    """노트북의 df에 해당하는 6컬럼 DataFrame 로드."""
    if data_path.exists():
        df = pd.read_csv(data_path)
        for c in RAW_COLS:
            if c not in df.columns:
                raise ValueError(f"CSV에 컬럼이 없습니다: {c}. 필요 컬럼: {RAW_COLS}")
        return df[RAW_COLS].astype(float)
    # 데이터 파일이 없으면 노트북과 동일한 데이터 소스 사용 시도 (hossam)
    try:
        from hossam import load_data
        origin = load_data('wholesale_customers')
        return origin.drop(['Channel', 'Region'], axis=1)
    except Exception:
        raise FileNotFoundError(
            f"데이터를 찾을 수 없습니다. 다음 중 하나를 해주세요.\n"
            f"  1) 노트북에서 df를 CSV로 저장: df.to_csv(r'{data_path}', index=False)\n"
            f"  2) 또는 hossam 패키지로 'wholesale_customers' 데이터를 사용 가능하게 한 뒤 다시 실행"
        )


def build_pipeline(df):
    """노트북과 동일한 전처리 + PCA + KMeans 파이프라인."""
    cols = RAW_COLS.copy()
    df_log = df.copy()
    for i in cols:
        df_log[f'log_{i}'] = np.log1p(df_log[i])

    scaler = StandardScaler()
    sdf = pd.DataFrame(scaler.fit_transform(df_log), columns=df_log.columns)
    log_cols = [f'log_{c}' for c in cols]
    sdf_log = sdf[log_cols]

    pca = PCA(n_components=0.8, random_state=52)
    pca_scores = pca.fit_transform(sdf_log)
    n = pca_scores.shape[1]
    pca_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(n)])

    kmeans = KMeans(n_clusters=5, random_state=52)
    kmeans.fit(pca_df)

    return {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'raw_columns': RAW_COLS,
        'log_columns': log_cols,
    }


def transform_row(model, row_raw):
    """한 행(6개 구매액)에 대해 전처리 -> PCA -> 클러스터 ID 반환."""
    raw = np.asarray(row_raw, dtype=float).reshape(1, -1)
    log = np.log1p(raw)
    full = np.hstack([raw, log])
    scaled = model['scaler'].transform(full)
    scaled_log = scaled[:, len(model['raw_columns']):]
    pc = model['pca'].transform(scaled_log)
    return model['kmeans'].predict(pc)[0]


if __name__ == "__main__":
    print("데이터 로드 중...")
    df = load_data()
    print(f"  행 수: {len(df)}, 컬럼: {list(df.columns)}")

    print("전처리 및 군집 모델 학습 중...")
    model = build_pipeline(df)
    # 검증: 스케일러는 12컬럼(원본+log) 입력, PCA는 스케일된 log 6컬럼만 사용
    full_input = np.hstack([df.values, np.log1p(df.values)])
    scaled_full = model['scaler'].transform(full_input)
    scaled_log = scaled_full[:, len(RAW_COLS):]
    clusters = model['kmeans'].predict(model['pca'].transform(scaled_log))
    print(f"  군집 수: {model['kmeans'].n_clusters}, PCA 주성분 수: {model['pca'].n_components_}")

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"모델 저장 완료: {pkl_path}")

    # TabPy에서 사용할 때와 동일한 변환 함수 참조용 (스크립트에서는 사용하지 않음)
    # TabPy와 동일한 변환 로직 검증
    first_cluster = transform_row(model, df.iloc[0].values)
    assert first_cluster == clusters[0], f"검증 실패: {first_cluster} vs {clusters[0]}"
    print("TabPy 변환 로직 검증 완료.")
