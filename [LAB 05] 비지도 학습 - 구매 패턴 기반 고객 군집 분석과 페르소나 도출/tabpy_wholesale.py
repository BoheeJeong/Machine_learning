# -*- coding: utf-8 -*-
"""
TabPy에서 호출할 Wholesale 구매 패턴 스크립트 (군집 + 채널 분류)

[1] 군집 예측 (기존)
  - get_cluster: 6개 구매 컬럼 → 군집 ID (0~4)
  - get_pca1, get_pca2, get_pca3, get_biplot_pc1_loadings, get_biplot_pc2_loadings
  - 모델: wholesale_cluster_model.pkl

[2] 채널 분류 (4개 모델: LogisticRegression, XGBoost, CatBoost, RandomForest)
  - get_channel: 6개 구매 + Region → 채널 0/1 (SCRIPT_INT)
  - get_channel_proba: 6개 구매 + Region → 채널 1 확률 (SCRIPT_REAL)
  - 모델: wholesale_scaler.pkl, wholesale_logistic.pkl, wholesale_xgb.pkl,
          wholesale_catboost.pkl, wholesale_rf.pkl
  - Tableau 계산 필드 예시:
    SCRIPT_INT("from tabpy_wholesale import get_channel; return get_channel(_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,'XGBoost')",
      SUM([Fresh]), SUM([Milk]), SUM([Grocery]), SUM([Frozen]), SUM([Detergents_Paper]), SUM([Delicassen]), ATTR([Region]))
    SCRIPT_REAL("from tabpy_wholesale import get_channel_proba; return get_channel_proba(_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,'XGBoost')", ...)
  - ModelName: 'LogisticRegression', 'XGBoost', 'CatBoost', 'RandomForest' 중 하나

추가로 알려주시면 좋은 정보:
  - TabPy 서버를 어디에 두실지 (로컬/서버 주소, 포트). 이 파일은 같은 폴더의 pkl을 쓰므로
    TabPy 작업 디렉터리를 이 스크립트가 있는 [LAB 05] 폴더로 두거나, pkl 경로를 수정해야 합니다.
  - Tableau에서 Region 값이 원본 데이터처럼 1/2/3 숫자인지, 다른 인코딩인지.
  - 채널 분류만 쓰시는 경우 군집용 wholesale_cluster_model.pkl 은 없어도 get_channel / get_channel_proba 는 동작합니다.
"""

import os
import pickle
import numpy as np
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# [1] 군집 모델 (KMeans + PCA + scaler)
# ---------------------------------------------------------------------------
_CLUSTER_MODEL_PATH = _DIR / "wholesale_cluster_model.pkl"
_model = None

_FEATURE_NAMES = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']


def _feature_name_to_index(name):
    """Tableau 필드명(공백 등)을 인덱스로."""
    s = str(name).strip().replace(' ', '_')
    for i, f in enumerate(_FEATURE_NAMES):
        if s == f or s == f.replace('_', ' '):
            return i
    return 0


def _load_model():
    global _model
    if _model is None:
        if not _CLUSTER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {_CLUSTER_MODEL_PATH}\n"
                "먼저 save_wholesale_cluster_model.py 를 실행해 wholesale_cluster_model.pkl 을 생성하세요."
            )
        with open(_CLUSTER_MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model


def _to_array(x):
    """Tableau/TabPy는 리스트나 단일 값을 넘길 수 있음."""
    if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
        return np.asarray(x, dtype=float)
    return np.asarray([float(x)], dtype=float)


def get_cluster(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """
    TabPy/Tableau에서 호출. 6개 구매 컬럼 → 군집 ID (0~4) 리스트.
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
    """6개 구매 컬럼 → scale → PCA. 반환: (n, n_components)."""
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
    """TabPy: 행별 PC1 점수 (SCRIPT_REAL)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return pc[:, 0].tolist()


def get_pca2(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """TabPy: 행별 PC2 점수 (SCRIPT_REAL)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return pc[:, 1].tolist()


def get_pca3(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """TabPy: 행별 PC3 점수 (SCRIPT_REAL)."""
    pc = _transform_to_pc(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen)
    return (pc[:, 2].tolist() if pc.shape[1] > 2 else [0.0] * pc.shape[0])


def _get_biplot_loadings():
    model = _load_model()
    comp = model['pca'].components_
    pc1 = comp[0, :].tolist()
    pc2 = comp[1, :].tolist()
    return pc1, pc2


def get_biplot_pc1_loadings(Feature):
    """TabPy: Biplot 변수별 PC1 로딩 (SCRIPT_REAL)."""
    pc1_list, _ = _get_biplot_loadings()
    if hasattr(Feature, '__len__') and not isinstance(Feature, (str, bytes)):
        return [float(pc1_list[_feature_name_to_index(f)]) for f in Feature]
    return [float(pc1_list[_feature_name_to_index(Feature)])]


def get_biplot_pc2_loadings(Feature):
    """TabPy: Biplot 변수별 PC2 로딩 (SCRIPT_REAL)."""
    _, pc2_list = _get_biplot_loadings()
    if hasattr(Feature, '__len__') and not isinstance(Feature, (str, bytes)):
        return [float(pc2_list[_feature_name_to_index(f)]) for f in Feature]
    return [float(pc2_list[_feature_name_to_index(Feature)])]


def get_biplot_arrow_endpoints(Feature):
    """TabPy: Biplot 화살표 끝점 (pc1_list, pc2_list)."""
    return _get_biplot_loadings()


# ---------------------------------------------------------------------------
# [2] 채널 분류 (4개 모델: LogisticRegression, XGBoost, CatBoost, RandomForest)
# ---------------------------------------------------------------------------
_RAW_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
_LOG_COLS = [f'log_{c}' for c in _RAW_COLS]
_FEATURE_COLS_CHANNEL = _LOG_COLS + ['Region']

_CHANNEL_SCALER_PATH = _DIR / "wholesale_scaler.pkl"
_CHANNEL_MODEL_FILES = {
    "logisticregression": "wholesale_logistic.pkl",
    "xgboost": "wholesale_xgb.pkl",
    "catboost": "wholesale_catboost.pkl",
    "randomforest": "wholesale_rf.pkl",
}

_channel_scaler = None
_channel_models = {}


def _load_channel_scaler():
    global _channel_scaler
    if _channel_scaler is None:
        if not _CHANNEL_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"채널 분류용 scaler가 없습니다: {_CHANNEL_SCALER_PATH}\n"
                "01-XGBoost 등 노트북에서 wholesale_scaler.pkl 저장 후 다시 시도하세요."
            )
        with open(_CHANNEL_SCALER_PATH, 'rb') as f:
            _channel_scaler = pickle.load(f)
    return _channel_scaler


def _load_channel_model(model_name):
    """model_name: 'LogisticRegression', 'XGBoost', 'CatBoost', 'RandomForest' (대소문자 무관)."""
    global _channel_models
    key = str(model_name).strip().lower()
    if key not in _CHANNEL_MODEL_FILES:
        raise ValueError(
            f"지원 모델: {list(_CHANNEL_MODEL_FILES.keys())}. 입력: {model_name}"
        )
    if key not in _channel_models:
        path = _DIR / _CHANNEL_MODEL_FILES[key]
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {path}\n"
                "02/01/07/05 노트북에서 해당 pkl 저장 후 다시 시도하세요."
            )
        with open(path, 'rb') as f:
            _channel_models[key] = pickle.load(f)
    return _channel_models[key]


def _channel_preprocess(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region):
    """
    6개 구매 + Region → 01 노트북과 동일: 13컬럼(df_log) 구성 → scaler.transform → 7컬럼만 반환.
    반환: (n, 7) numpy array.
    """
    scaler = _load_channel_scaler()
    fresh = _to_array(Fresh)
    milk = _to_array(Milk)
    grocery = _to_array(Grocery)
    frozen = _to_array(Frozen)
    det = _to_array(Detergents_Paper)
    deli = _to_array(Delicassen)
    region = _to_array(Region)

    n = max(len(fresh), len(milk), len(grocery), len(frozen), len(det), len(deli), len(region))
    if n == 0:
        return np.zeros((0, 7))
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
    if len(region) == 1 and n > 1:
        region = np.full(n, float(region.flat[0]))

    raw = np.column_stack([fresh, milk, grocery, frozen, det, deli])
    log = np.log1p(raw)
    # 01과 동일: 13컬럼 순서 (RAW 6 + log 6 + Region)
    df_log = np.hstack([raw, log, region.reshape(-1, 1)])
    X_all = scaler.transform(df_log)
    col_ix = [6, 7, 8, 9, 10, 11, 12]
    X = X_all[:, col_ix]
    return X


def get_channel(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost"):
    """
    TabPy/Tableau: 6개 구매 + Region → 채널 0 또는 1 리스트 (SCRIPT_INT).
    ModelName: 'LogisticRegression', 'XGBoost', 'CatBoost', 'RandomForest' 중 하나.
    """
    X = _channel_preprocess(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region)
    if X.shape[0] == 0:
        return []
    clf = _load_channel_model(ModelName)
    pred = clf.predict(X)
    return pred.tolist()


def get_channel_proba(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost"):
    """
    TabPy/Tableau: 6개 구매 + Region → 채널 1(양성) 확률 리스트 (SCRIPT_REAL).
    ModelName: 'LogisticRegression', 'XGBoost', 'CatBoost', 'RandomForest' 중 하나.
    """
    X = _channel_preprocess(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region)
    if X.shape[0] == 0:
        return []
    clf = _load_channel_model(ModelName)
    proba = clf.predict_proba(X)[:, 1]
    return proba.tolist()
