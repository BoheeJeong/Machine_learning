# -*- coding: utf-8 -*-
"""
TabPy에서 호출할 구매 패턴 군집 예측 스크립트
- Tableau 계산 필드에서 SCRIPT_INT 로 이 스크립트를 호출하면
  각 행의 Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen 에 대해
  군집 ID(0~4)를 반환합니다.

Tableau 계산 필드 예시 (테이블에 행 단위로 적용):
  SCRIPT_INT(
    "
    import sys
    sys.path.append(r'C:\\...\\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
    from tabpy_wholesale_cluster import get_cluster
    return get_cluster(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6)
    ",
    SUM([Fresh]), SUM([Milk]), SUM([Grocery]), SUM([Frozen]),
    SUM([Detergents_Paper]), SUM([Delicassen])
  )

또는 TabPy에 함수를 배포한 경우:
  SCRIPT_INT("return tabpy.query('get_cluster', _arg1, _arg2, _arg3, _arg4, _arg5, _arg6)", ...)

모델 파일 경로: 이 파일과 같은 폴더의 wholesale_cluster_model.pkl

PCA (엑셀 없이 TabPy로 계산):
  get_pca1, get_pca2, get_pca3 → 행별 PC1, PC2, PC3 점수 (SCRIPT_REAL).
  Tableau에서 열: PC1 계산필드, 행: PC2 계산필드, 색상: Cluster ID 로 PCA 산점도.

Biplot (변수 화살표):
  get_biplot_pc1_loadings(Feature), get_biplot_pc2_loadings(Feature) → 변수별 PC1/PC2 로딩 (SCRIPT_REAL).
  Tableau에서 Feature 차원(6개) 행으로 두고, PC1/PC2 로딩을 열/행에 넣어 Line 마크로 화살표.
"""

import os
import pickle
import numpy as np
from pathlib import Path

# 이 스크립트와 같은 폴더의 pkl 경로
_MODEL_PATH = Path(__file__).resolve().parent / "wholesale_cluster_model.pkl"
_model = None

# Biplot 로딩용 변수 순서 (pkl PCA와 동일)
_FEATURE_NAMES = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']


def _feature_name_to_index(name):
    """Tableau 필드명(공백 등)을 인덱스로. Detergents Paper ↔ Detergents_Paper."""
    s = str(name).strip().replace(' ', '_')
    for i, f in enumerate(_FEATURE_NAMES):
        if s == f or s == f.replace('_', ' '):
            return i
    return 0


def _load_model():
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {_MODEL_PATH}\n"
                "먼저 save_wholesale_cluster_model.py 를 실행해 wholesale_cluster_model.pkl 을 생성하세요."
            )
        with open(_MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model


def _to_array(x):
    """Tableau/TabPy는 리스트나 단일 값을 넘길 수 있음."""
    if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        return np.asarray(x, dtype=float)
    return np.asarray([float(x)], dtype=float)


def get_cluster(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """
    TabPy/Tableau에서 호출할 함수.
    각 인자는 (동일 길이의) 리스트 또는 단일 값일 수 있음.
    반환: 각 행에 대한 군집 ID 리스트 (0~4).
    """
    model = _load_model()
    fresh = _to_array(Fresh)
    milk = _to_array(Milk)
    grocery = _to_array(Grocery)
    frozen = _to_array(Frozen)
    det = _to_array(Detergents_Paper)
    deli = _to_array(Delicassen)

    # 길이 맞추기: 단일 값이면 동일한 길이로 브로드캐스트
    n = max(len(fresh), len(milk), len(grocery), len(frozen), len(det), len(deli))
    if n == 0:
        return []
    if len(fresh) == 1 and n > 1:
        fresh = np.full(n, float(fresh.flat[0]))
    if len(milk) == 1 and n > 1:
        milk = np.full(n, float(milk.flat[0]))
    if len(grocery) == 1 and n > 1:
        grocery = np.full(n, float(grocery.flat[0]))
    if len(frozen) == 1 and n > 1:
        frozen = np.full(n, float(frozen.flat[0]))
    if len(det) == 1 and n > 1:
        det = np.full(n, float(det.flat[0]))
    if len(deli) == 1 and n > 1:
        deli = np.full(n, float(deli.flat[0]))

    raw = np.column_stack([fresh, milk, grocery, frozen, det, deli])
    log = np.log1p(raw)
    full = np.hstack([raw, log])
    scaled = model['scaler'].transform(full)
    scaled_log = scaled[:, 6:]
    pc = model['pca'].transform(scaled_log)
    labels = model['kmeans'].predict(pc)
    return labels.tolist()


def _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """
    공통: 6개 구매 컬럼 → (raw+log) → scale → PCA 변환.
    반환: (n_rows, n_components) 배열.
    """
    model = _load_model()
    fresh = _to_array(Fresh)
    milk = _to_array(Milk)
    grocery = _to_array(Grocery)
    frozen = _to_array(Frozen)
    det = _to_array(Detergents_Paper)
    deli = _to_array(Delicassen)

    n = max(len(fresh), len(milk), len(grocery), len(frozen), len(det), len(deli))
    if n == 0:
        return np.zeros((0, 3))
    if len(fresh) == 1 and n > 1:
        fresh = np.full(n, float(fresh.flat[0]))
    if len(milk) == 1 and n > 1:
        milk = np.full(n, float(milk.flat[0]))
    if len(grocery) == 1 and n > 1:
        grocery = np.full(n, float(grocery.flat[0]))
    if len(frozen) == 1 and n > 1:
        frozen = np.full(n, float(frozen.flat[0]))
    if len(det) == 1 and n > 1:
        det = np.full(n, float(det.flat[0]))
    if len(deli) == 1 and n > 1:
        deli = np.full(n, float(deli.flat[0]))

    raw = np.column_stack([fresh, milk, grocery, frozen, det, deli])
    log = np.log1p(raw)
    full = np.hstack([raw, log])
    scaled = model['scaler'].transform(full)
    scaled_log = scaled[:, 6:]
    pc = model['pca'].transform(scaled_log)
    return pc


def get_pca1(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """TabPy: 행별 PC1 점수 리스트 반환 (SCRIPT_REAL용)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return pc[:, 0].tolist()


def get_pca2(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """TabPy: 행별 PC2 점수 리스트 반환 (SCRIPT_REAL용)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return pc[:, 1].tolist()


def get_pca3(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """TabPy: 행별 PC3 점수 리스트 반환 (SCRIPT_REAL용)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return (pc[:, 2].tolist() if pc.shape[1] > 2 else [0.0] * pc.shape[0])


# ---------------------------------------------------------------------------
# Biplot: 변수별 PC1/PC2 로딩 (화살표 방향)
# ---------------------------------------------------------------------------

def _get_biplot_loadings():
    """pkl PCA의 components_ → (PC1로딩 6개, PC2로딩 6개)."""
    model = _load_model()
    comp = model['pca'].components_  # (n_components, n_features), n_features=6
    pc1 = comp[0, :].tolist()
    pc2 = comp[1, :].tolist()
    return pc1, pc2


def get_biplot_pc1_loadings(Feature):
    """
    TabPy: Biplot 화살표용 변수별 PC1 로딩.
    Feature: Tableau에서 넘기는 변수 이름 (단일 또는 리스트). 예: 'Fresh', 'Milk', ...
    반환: Feature 순서대로 PC1 로딩 리스트 (SCRIPT_REAL용).
    """
    pc1_list, _ = _get_biplot_loadings()
    if hasattr(Feature, '__len__') and not isinstance(Feature, (str, bytes)):
        return [float(pc1_list[_feature_name_to_index(f)]) for f in Feature]
    return [float(pc1_list[_feature_name_to_index(Feature)])]


def get_biplot_pc2_loadings(Feature):
    """
    TabPy: Biplot 화살표용 변수별 PC2 로딩.
    Feature: Tableau에서 넘기는 변수 이름 (단일 또는 리스트).
    반환: Feature 순서대로 PC2 로딩 리스트 (SCRIPT_REAL용).
    """
    _, pc2_list = _get_biplot_loadings()
    if hasattr(Feature, '__len__') and not isinstance(Feature, (str, bytes)):
        return [float(pc2_list[_feature_name_to_index(f)]) for f in Feature]
    return [float(pc2_list[_feature_name_to_index(Feature)])]


def get_biplot_arrow_endpoints(Feature):
    """
    TabPy: Biplot 화살표 (0,0)→(PC1,PC2) 의 끝점만 반환.
    반환: (pc1_list, pc2_list) 각각 길이 6. Line 그릴 때 PointOrder 1 인 점.
    """
    return _get_biplot_loadings()
