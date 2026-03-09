# -*- coding: utf-8 -*-
"""
Tableau용 CSV 생성
- wholesale_for_tableau.csv: 원본 + PC1, PC2, PC3 + ClusterID (산점도·Biplot 점)
- wholesale_pca_loadings.csv: Biplot 변수 화살표용 (Feature, PointOrder, PC1, PC2)

실행: python create_tableau_csv.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

LAB05_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = LAB05_DIR / "wholesale_for_tableau.csv"
LOADINGS_CSV = LAB05_DIR / "wholesale_pca_loadings.csv"
RAW_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']


def load_full_data():
    """Channel, Region 포함 원본 가능하면 로드."""
    try:
        from hossam import load_data
        origin = load_data('wholesale_customers')
        return origin  # Channel, Region, Fresh, Milk, ...
    except Exception:
        pass
    # 6컬럼 CSV만 있으면 그것만 사용
    path_6 = LAB05_DIR / "wholesale_customers_6cols.csv"
    if path_6.exists():
        return pd.read_csv(path_6)[RAW_COLS].astype(float)
    raise FileNotFoundError(
        f"데이터가 없습니다. 노트북에서 원본 데이터를 export 하거나\n"
        f"  df.to_csv(r'{LAB05_DIR / 'wholesale_customers_6cols.csv'}', index=False)  로 저장 후 다시 실행하세요."
    )


def build_pca_and_cluster(df_raw):
    """노트북과 동일 파이프라인으로 PCA 점수 + ClusterID + fitted PCA 반환."""
    cols = RAW_COLS.copy()
    df_log = df_raw[cols].copy()
    for c in cols:
        df_log[f'log_{c}'] = np.log1p(df_log[c])

    scaler = StandardScaler()
    sdf = pd.DataFrame(scaler.fit_transform(df_log), columns=df_log.columns)
    sdf_log = sdf[[f'log_{c}' for c in cols]]

    pca = PCA(n_components=0.8, random_state=52)
    pca_scores = pca.fit_transform(sdf_log)
    pca_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    kmeans = KMeans(n_clusters=5, random_state=52)
    clusters = kmeans.fit_predict(pca_df)
    return pca_df, clusters, pca


def main():
    print("데이터 로드 중...")
    raw = load_full_data()

    # Tableau용: 원본 금액만 사용 (로그/PCA 컬럼 없음)
    out = pd.DataFrame({'CustomerID': np.arange(1, len(raw) + 1)})
    if 'Channel' in raw.columns:
        out['Channel'] = raw['Channel'].values
    if 'Region' in raw.columns:
        out['Region'] = raw['Region'].values
    for c in RAW_COLS:
        out[c] = raw[c].values

    print("PCA·군집 계산 중 (원본 금액 기준, 로그·PCA는 내부 처리)...")
    pca_df, clusters, pca = build_pca_and_cluster(raw[RAW_COLS].astype(float))
    out['ClusterID'] = clusters
    for c in pca_df.columns:
        out[c] = pca_df[c].values  # PC1, PC2, PC3 → Tableau PCA 산점도용

    out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {OUTPUT_CSV}")
    print(f"  컬럼: {list(out.columns)}, 행 수: {len(out)}")

    # Biplot 변수 화살표용: loadings = pca.components_.T → Feature별 PC1, PC2 로딩
    # Tableau Line mark: (0,0) → (PC1_loading, PC2_loading) 이므로 PointOrder 0/1 두 행
    loadings = pca.components_.T  # (n_features, n_components), feature 순서 = RAW_COLS
    rows = []
    for j, feat in enumerate(RAW_COLS):
        rows.append({"Feature": feat, "PointOrder": 0, "PC1": 0.0, "PC2": 0.0})
        rows.append({"Feature": feat, "PointOrder": 1, "PC1": float(loadings[j, 0]), "PC2": float(loadings[j, 1])})
    pd.DataFrame(rows).to_csv(LOADINGS_CSV, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {LOADINGS_CSV} (Biplot 화살표용 Feature, PointOrder, PC1, PC2)")
    print("\n→ Tableau: 산점도는 wholesale_for_tableau / Biplot 화살표는 wholesale_pca_loadings 를 Line으로 추가.")


if __name__ == "__main__":
    main()
