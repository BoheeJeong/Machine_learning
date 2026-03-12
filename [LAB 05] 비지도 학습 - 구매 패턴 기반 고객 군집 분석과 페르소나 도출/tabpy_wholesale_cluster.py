# -*- coding: utf-8 -*-
"""
TabPy에서 호출할 Wholesale 구매 패턴 스크립트 (군집 + 채널 분류)

[0] Spearman 상관계수 (UCI Wholesale 지출 항목용)
  - calculate_spearman: 두 지출 항목 리스트(_arg1, _arg2) → Spearman 상관계수. 테이블 계산용으로 결과를 입력 길이만큼 복제한 리스트 반환.
  - 속도: 함수 속성 OrderedDict 캐시(최대 128개). 동일 (arg1, arg2)면 재계산 없이 반환. TabPy 배포 시 이 함수만 직렬화되므로 캐시·계산 로직은 모두 함수 내부에 있음.
  - Tableau 계산 필드: SCRIPT_REAL("return tabpy.query('calculate_spearman', _arg1, _arg2)['response']", SUM([항목1]), SUM([항목2]))

[1] 군집 예측 (기존)
  - get_cluster: 6개 구매 컬럼 → 군집 ID (0~4)
  - get_pca_coords: 6개 구매 컬럼 → PCA 좌표 리스트 (행별 [PC1, PC2, ...])
  - get_pca1, get_pca2, get_pca3, get_biplot_pc1_loadings, get_biplot_pc2_loadings
  - 모델: wholesale_cluster_model.pkl

[2] 채널 분류 (4개 모델: LogisticRegression, XGBoost, SGDClassifier, RandomForest)
  - get_channel: 6개 구매 + Region(1/2/3, 범주형) → 내부에서 원핫(Region_2, Region_3) → 채널 0/1 (SCRIPT_INT)
  - get_channel_proba: 6개 구매 + Region → 채널 1 확률 (SCRIPT_REAL)
  - predict_customer_channel: 6개 구매 + Region → (예측 리스트, 확률 리스트) 한 번에 (로컬/노트북 검증용)
  - 모델: wholesale_scaler.pkl, wholesale_logistic.pkl, wholesale_xgb.pkl,
          wholesale_sgd.pkl, wholesale_rf.pkl
  - Tableau 계산 필드 예시:
    SCRIPT_INT("from tabpy_wholesale_cluster import get_channel; return get_channel(_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,'XGBoost')",
      SUM([Fresh]), SUM([Milk]), SUM([Grocery]), SUM([Frozen]), SUM([Detergents_Paper]), SUM([Delicassen]), ATTR([Region]))
    SCRIPT_REAL("from tabpy_wholesale_cluster import get_channel_proba; return get_channel_proba(_arg1,_arg2,_arg3,_arg4,_arg5,_arg6,_arg7,'XGBoost')", ...)
  - 고객별로 다른 확률/예측을 받으려면: 뷰에 [Customer ID]를 [행], [열] 또는 [마크-세부 정보]에 넣어야 합니다.
  - ModelName: 'LogisticRegression', 'XGBoost', 'SGDClassifier', 'RandomForest' 중 하나

추가로 알려주시면 좋은 정보:
  - TabPy 서버를 어디에 두실지 (로컬/서버 주소, 포트). 이 파일은 같은 폴더의 pkl을 쓰므로
    TabPy 작업 디렉터리를 이 스크립트가 있는 [LAB 05] 폴더로 두거나, pkl 경로를 수정해야 합니다.
  - Tableau에서 Region 값이 원본 데이터처럼 1/2/3 숫자인지, 다른 인코딩인지.
  - 채널 분류만 쓰시는 경우 군집용 wholesale_cluster_model.pkl 은 없어도 get_channel / get_channel_proba 는 동작합니다.

[배포] 태블로에서 위 함수들을 사용하려면 TabPy 서버에 엔드포인트로 등록해야 합니다.
  - TabPy 서버를 먼저 실행한 뒤, 이 파일을 스크립트로 한 번 실행하세요: python tabpy_wholesale_cluster.py
  - 아래 if __name__ == "__main__": 블록이 tabpy_client로 각 함수를 서버에 deploy 합니다.

[더블체크] 실행 전에 로컬에서 함수·모델이 정상 동작하는지 검사하려면:
  - python tabpy_wholesale_cluster.py --check   (또는 -c)
  - TabPy 서버 없이 pkl 로드 및 get_cluster, get_channel_proba 등 호출을 검증합니다.
  - 배포 시에도 배포 직전에 자동으로 한 번 실행됩니다.
"""

import os
import pickle
import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path
from tabpy.tabpy_tools.client import Client
_DIR = Path(__file__).resolve().parent

# Spearman 상관계수 등 TabPy 콘솔 디버깅용 (데이터 개수 출력)
_logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# [0] Spearman 상관계수 (태블로 테이블 계산용: 두 지출 항목 리스트 → 상관계수)
# TabPy 배포 시 이 함수만 직렬화되므로, 캐시·계산 로직을 모두 이 안에 두었음.
# ---------------------------------------------------------------------------
def calculate_spearman(_arg1, _arg2):
    """
    TabPy/Tableau에서 호출. 두 지출 항목 리스트 → Spearman 상관계수.
    테이블 계산 특성상 결과값을 입력 리스트 길이만큼 복제한 리스트로 반환합니다.
    동일 (arg1, arg2) 조합은 함수 속성 캐시(최대 128개)로 재사용해 속도 향상.
    """
    _MAX_CACHE = 128
    cache = getattr(calculate_spearman, "_spearman_cache", None)
    if cache is None:
        calculate_spearman._spearman_cache = OrderedDict()
        cache = calculate_spearman._spearman_cache

    try:
        arg1 = _arg1 if hasattr(_arg1, '__len__') and not isinstance(_arg1, (str, bytes)) else [_arg1]
        arg2 = _arg2 if hasattr(_arg2, '__len__') and not isinstance(_arg2, (str, bytes)) else [_arg2]
        n = len(arg1)

        _logger.info("calculate_spearman: 데이터 개수 = %d", n)
        print(f"[TabPy] calculate_spearman: 데이터 개수 = {n}")

        if n == 0 or len(arg2) == 0:
            return []

        t1, t2 = tuple(arg1), tuple(arg2)
        cache_key = (t1, t2)
        if cache_key in cache:
            res = cache[cache_key]
        else:
            # 캐시 미스: 상관계수 계산 (로직 인라인, TabPy가 다른 함수를 찾지 않도록)
            try:
                min_len = min(len(t1), len(t2))
                if min_len == 0:
                    res = 0.0
                else:
                    s1 = pd.Series(t1[:min_len], dtype=float)
                    s2 = pd.Series(t2[:min_len], dtype=float)
                    valid = s1.notna() & s2.notna()
                    s1_clean = s1[valid]
                    s2_clean = s2[valid]
                    if len(s1_clean) < 2 or s1_clean.nunique() < 2 or s2_clean.nunique() < 2:
                        res = 0.0
                    else:
                        corr = float(s1_clean.corr(s2_clean, method="spearman"))
                        res = corr if not pd.isna(corr) else 0.0
            except Exception:
                res = 0.0
            if len(cache) >= _MAX_CACHE:
                cache.popitem(last=False)
            cache[cache_key] = res

        return [res] * n
    except Exception as e:
        _logger.exception("calculate_spearman 예외: %s", e)
        print(f"[TabPy] calculate_spearman 예외: {e}")
        n = len(_arg1) if hasattr(_arg1, '__len__') and not isinstance(_arg1, (str, bytes)) else 1
        return [0.0] * max(n, 1)


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


def get_pca_coords(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    """
    TabPy/Tableau에서 호출. 6개 구매 컬럼 → PCA 좌표 리스트 (행별 [PC1, PC2, ...]).
    태블로에서 PC1, PC2 등을 각각 쓰려면 이 리스트의 열을 SCRIPT_REAL 등으로 꺼내 쓰면 됨.
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
    return pc.tolist()


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
# [2] 채널 분류 (4개 모델: LogisticRegression, XGBoost, SGDClassifier, RandomForest)
# ---------------------------------------------------------------------------
# 디버깅용 특이값 (필요 시 Tableau 계산 필드에서 "뭉침" 감지용으로 사용 가능).
# 주의: Tableau가 행 단위로 보낼 때도 호출당 1건이므로, Python만으로는 "전체 1건 뭉침"과 구분 불가.
TABPY_AGGREGATED_SENTINEL_REAL = 999.0   # get_channel_proba (SCRIPT_REAL)
TABPY_AGGREGATED_SENTINEL_INT = 999      # get_channel (SCRIPT_INT)

_RAW_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
_LOG_COLS = [f'log_{c}' for c in _RAW_COLS]
_FEATURE_COLS_CHANNEL = _LOG_COLS + ['Region_2', 'Region_3']  # Region 범주형 원핫 (8개)

_CHANNEL_SCALER_PATH = _DIR / "wholesale_scaler.pkl"
_CHANNEL_MODEL_FILES = {
    "logisticregression": "wholesale_logistic.pkl",
    "xgboost": "wholesale_xgb.pkl",
    "sgdclassifier": "wholesale_sgd.pkl",
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
    """model_name: 'LogisticRegression', 'XGBoost', 'SGDClassifier', 'RandomForest' (대소문자 무관)."""
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
    6개 구매 + Region(1,2,3) → 01 노트북과 동일: Region 범주형 원핫(Region_2, Region_3),
    14컬럼(RAW 6 + log 6 + Region_2 + Region_3) → scaler.transform → 8컬럼만 반환.
    반환: (n, 8) numpy array.
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
        return np.zeros((0, 8))
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
    # Region 범주형 원핫 (기준 1 → Region_2, Region_3)
    region = region.ravel()
    region_2 = (region == 2).astype(float).reshape(-1, 1)
    region_3 = (region == 3).astype(float).reshape(-1, 1)
    df_log = np.hstack([raw, log, region_2, region_3])
    X_all = scaler.transform(df_log)
    col_ix = [6, 7, 8, 9, 10, 11, 12, 13]
    X = X_all[:, col_ix]
    return X


def get_channel(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost"):
    """
    TabPy/Tableau: 6개 구매 + Region → 채널 0 또는 1 리스트 (SCRIPT_INT).
    ModelName: 'LogisticRegression', 'XGBoost', 'SGDClassifier', 'RandomForest' 중 하나.
    태블로 파라미터(드롭다운)에서 리스트로 오면 첫 번째 값 사용.
    """
    if hasattr(ModelName, '__len__') and not isinstance(ModelName, (str, bytes)):
        ModelName = ModelName[0] if len(ModelName) else "XGBoost"
    ModelName = str(ModelName).strip() or "XGBoost"
    X = _channel_preprocess(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region)
    if X.shape[0] == 0:
        return []
    clf = _load_channel_model(ModelName)
    pred = clf.predict(X)
    return pred.tolist()


def get_channel_proba(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost"):
    """
    TabPy/Tableau: 6개 구매 + Region → 채널 1(양성) 확률 리스트 (SCRIPT_REAL).
    ModelName: 'LogisticRegression', 'XGBoost', 'SGDClassifier', 'RandomForest' 중 하나.
    태블로 파라미터(드롭다운)에서 리스트로 오면 첫 번째 값 사용.
    """
    if hasattr(ModelName, '__len__') and not isinstance(ModelName, (str, bytes)):
        ModelName = ModelName[0] if len(ModelName) else "XGBoost"
    ModelName = str(ModelName).strip() or "XGBoost"
    X = _channel_preprocess(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region)
    if X.shape[0] == 0:
        return []
    clf = _load_channel_model(ModelName)
    proba = clf.predict_proba(X)[:, 1]
    return proba.tolist()


def predict_customer_channel(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost"):
    """
    6개 구매 + Region → (예측 클래스 리스트, Channel=1 확률 리스트) 한 번에 반환.
    TabPy/Tableau에서는 get_channel, get_channel_proba를 각각 쓰고,
    로컬/노트북에서 한 번에 (예측, 확률)이 필요할 때 이 함수 사용.
    """
    pred = get_channel(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName)
    proba = get_channel_proba(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName)
    return pred, proba


# 태블로 Feature 필드 값과 정확히 일치. 모델 피처 순서: log 6 + Region_2, Region_3
_FEATURE_IMPORTANCE_ORDER = [
    'log_Fresh', 'log_Milk', 'log_Grocery', 'log_Frozen', 'log_Detergents_Paper', 'log_Delicassen',
    'Region_2', 'Region_3'
]

# EC2/TabPy 배포 시 메모리 부담을 줄이기 위해 CSV가 있으면 모델 로드 없이 CSV만 사용
_FEATURE_IMPORTANCE_CSV = _DIR / "feature_importance_for_tableau.csv"


def _get_feature_importance_dict_from_csv(ModelName):
    """CSV에서 모델별 특성 중요도 읽기 (모델 로드 없음, TabPy 배포/실행 시 가벼움)."""
    if not _FEATURE_IMPORTANCE_CSV.exists():
        return None
    try:
        df = pd.read_csv(_FEATURE_IMPORTANCE_CSV)
        target = ModelName[0] if hasattr(ModelName, '__len__') and not isinstance(ModelName, (str, bytes)) else ModelName
        target = str(target).strip()
        rows = df[df["model_name"].astype(str).str.strip() == target]
        if rows.empty:
            return None
        return dict(zip(rows["feature_name"].astype(str), rows["importance"].astype(float)))
    except Exception:
        return None


def _get_feature_importance_dict(ModelName):
    """모델별 8개 특성 중요도를 이름→값 딕셔너리로 반환 (내부용)."""
    clf = _load_channel_model(ModelName)
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):  # LogisticRegression
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.array([0.0] * 8)
    return {name: float(importances[i]) for i, name in enumerate(_FEATURE_IMPORTANCE_ORDER)}


def get_roc_metrics(model_name, path_index):
    """ROC TPR/AUC용. CSV에서 모델별 FPR/TPR 읽어 101개 지점 보간 후 반환."""
    file_path = _DIR / "roc_curve_multi_models_for_tableau.csv"
    try:
        df = pd.read_csv(file_path)
        target_model = model_name[0] if isinstance(model_name, list) else model_name
        model_df = df[df["model_name"] == target_model].sort_values("FPR")
        if model_df.empty:
            return [0.0] * len(path_index), [0.0] * len(path_index)
        auc_val = model_df["AUC"].iloc[0]
        target_fpr = np.linspace(0, 1, 101)
        interp_tpr = np.interp(target_fpr, model_df["FPR"].values, model_df["TPR"].values)
        return interp_tpr.tolist(), [float(auc_val)] * len(path_index)
    except Exception:
        return [0.0] * len(path_index), [0.0] * len(path_index)


def get_tpr_for_tableau(model_name, path_index):
    tpr_list, _ = get_roc_metrics(model_name, path_index)
    return tpr_list


def get_auc_for_tableau(model_name, path_index):
    _, auc_list = get_roc_metrics(model_name, path_index)
    return auc_list


def get_confusion_metrics(model_name):
    """[TN, FP, FN, TP] 반환 (wide CSV 한 행)."""
    file_path = _DIR / "confusion_matrix_models.csv"
    try:
        df = pd.read_csv(file_path)
        target_model = model_name[0] if isinstance(model_name, list) else model_name
        row = df[df["model_name"] == target_model].iloc[0]
        return [float(row["TN"]), float(row["FP"]), float(row["FN"]), float(row["TP"])]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def get_confusion_matrix_live(model_name):
    file_path = _DIR / "confusion_matrix_models.csv"
    try:
        df = pd.read_csv(file_path)
        # 1. 모델명 필터링
        target_model = model_name[0] if isinstance(model_name, list) else model_name
        model_df = df[df["model_name"] == target_model].copy()
        
        # 2. 실제(actual)와 예측(predicted) 순으로 정렬 
        # (0,0)->TN, (0,1)->FP, (1,0)->FN, (1,1)->TP 순서가 보장됩니다.
        model_df = model_df.sort_values(["actual", "predicted"])
        
        # 3. value 컬럼의 값 4개를 리스트로 반환
        return model_df["value"].astype(float).tolist()
    except Exception as e:
        print(f"Error: {e}")
        return [0.0, 0.0, 0.0, 0.0]


def get_feature_importance(Feature=None, ModelName="XGBoost"):
    """
    TabPy/Tableau: 이름 기반 특성 중요도 반환 (인덱스 순서 무관).

    - Feature 생략: 전체 딕셔너리 반환 (테스트/하이브리드용).
    - Feature 단일 값: 해당 이름의 중요도 1개 반환 (태블로 [다음 뒤 사용] > Feature 시 행별 호출).
    - Feature 리스트: 각 이름에 대응하는 중요도 리스트 반환.

    태블로 계산 필드 예시 (Compute Using > Feature 필수):
      SCRIPT_REAL("from tabpy_wholesale_cluster import get_feature_importance; return get_feature_importance(_arg1, _arg2)",
        ATTR([Feature (feature 2.csv)]), [Select ML Model])
    """
    # 태블로 파라미터가 리스트로 오는 경우 (예: [Select ML Model]) → 첫 번째 값 사용
    if hasattr(ModelName, '__len__') and not isinstance(ModelName, (str, bytes)):
        ModelName = ModelName[0] if len(ModelName) else "XGBoost"
    ModelName = str(ModelName).strip() or "XGBoost"

    name_to_imp = _get_feature_importance_dict_from_csv(ModelName)
    if name_to_imp is None:
        name_to_imp = _get_feature_importance_dict(ModelName)

    if Feature is None:
        return name_to_imp

    # 태블로에서 넘어오는 공백 방어
    def lookup(name):
        key = str(name).strip()
        if key in name_to_imp:
            return name_to_imp[key]
        # 공백/언더스코어 차이 허용 (Detergents Paper vs Detergents_Paper)
        alt = key.replace(' ', '_')
        if alt in name_to_imp:
            return name_to_imp[alt]
        return 0.0

    if hasattr(Feature, '__len__') and not isinstance(Feature, (str, bytes)):
        return [lookup(f) for f in Feature]
    return lookup(Feature)


# ---------------------------------------------------------------------------
# 자체 검사 (더블체크): TabPy 없이 로컬에서 함수·모델 동작 확인
# ---------------------------------------------------------------------------
def _run_self_check():
    """
    pkl 로드 및 주요 함수 호출을 로컬에서 검증합니다.
    실패 시 예외를 발생시키고, 성공 시 True 반환.
    """
    errors = []

    # [0] Spearman 상관계수 (pkl 불필요)
    try:
        out = calculate_spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert isinstance(out, list), "calculate_spearman 반환 타입이 list가 아님"
        assert len(out) == 5, f"calculate_spearman 길이 5 기대, 실제 {len(out)}"
        assert all(isinstance(x, (float, np.floating)) for x in out), f"calculate_spearman 값은 실수: {out}"
        assert abs(out[0] - 1.0) < 1e-5, f"완전 양의 상관이면 1.0 기대: {out[0]}"
    except Exception as e:
        errors.append(f"calculate_spearman: {e}")

    # [1] 군집 모델 (파일 있으면 검사)
    if _CLUSTER_MODEL_PATH.exists():
        try:
            out = get_cluster([1, 2, 3], [10, 20, 30], [100, 200, 300],
                              [5, 10, 15], [20, 40, 60], [2, 4, 6])
            assert isinstance(out, list), "get_cluster 반환 타입이 list가 아님"
            assert len(out) == 3, f"get_cluster 길이 3 기대, 실제 {len(out)}"
            assert all(isinstance(x, (int, np.integer)) and 0 <= x <= 4 for x in out), \
                f"get_cluster 값은 0~4 정수여야 함: {out}"

            coords = get_pca_coords([1, 2, 3], [10, 20, 30], [100, 200, 300],
                                    [5, 10, 15], [20, 40, 60], [2, 4, 6])
            assert isinstance(coords, list), "get_pca_coords 반환 타입이 list가 아님"
            assert len(coords) == 3, f"get_pca_coords 길이 3 기대, 실제 {len(coords)}"
            for i, row in enumerate(coords):
                assert isinstance(row, list), f"get_pca_coords[{i}]가 list가 아님"
                assert len(row) >= 1, f"get_pca_coords[{i}]가 비어 있음"
                assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in row), \
                    f"get_pca_coords[{i}] 값은 숫자여야 함: {row}"
        except Exception as e:
            errors.append(f"군집 get_cluster/get_pca_coords: {e}")
    else:
        errors.append(f"군집 모델 파일 없음 (건너뜀): {_CLUSTER_MODEL_PATH.name}")

    # [2] 채널 분류 (scaler + 모델 하나라도 있으면 검사)
    if _CHANNEL_SCALER_PATH.exists():
        available = [k for k, f in _CHANNEL_MODEL_FILES.items() if (_DIR / f).exists()]
        if available:
            _name_map = {"logisticregression": "LogisticRegression", "xgboost": "XGBoost",
                        "sgdclassifier": "SGDClassifier", "randomforest": "RandomForest"}
            model_name = _name_map.get(available[0], available[0].title())
            try:
                # 3행 샘플
                proba = get_channel_proba(
                    [100, 200, 300], [50, 100, 150], [200, 400, 600],
                    [20, 40, 60], [30, 60, 90], [10, 20, 30],
                    [1, 2, 3], model_name
                )
                assert isinstance(proba, list), "get_channel_proba 반환 타입이 list가 아님"
                assert len(proba) == 3, f"get_channel_proba 길이 3 기대, 실제 {len(proba)}"
                assert all(isinstance(p, (float, np.floating)) and 0 <= p <= 1 for p in proba), \
                    f"get_channel_proba 값은 0~1 실수여야 함: {proba}"

                pred = get_channel(
                    [100, 200], [50, 100], [200, 400], [20, 40], [30, 60], [10, 20],
                    [1, 2], model_name
                )
                assert isinstance(pred, list) and len(pred) == 2, f"get_channel 길이 2 기대: {pred}"
                assert all(x in (0, 1) for x in pred), f"get_channel 값은 0 또는 1: {pred}"

                imp = get_feature_importance(ModelName=model_name)
                assert isinstance(imp, dict) and len(imp) == 8, \
                    f"get_feature_importance는 8개 키 딕셔너리: {len(imp)}개"
            except Exception as e:
                errors.append(f"채널 분류 ({model_name}): {e}")
        else:
            errors.append("채널용 pkl 중 모델 파일이 하나도 없음 (wholesale_xgb.pkl 등)")
    else:
        errors.append(f"채널 scaler 없음 (건너뜀): {_CHANNEL_SCALER_PATH.name}")

    if errors:
        raise RuntimeError("자체 검사 실패:\n  " + "\n  ".join(errors))
    return True


# ---------------------------------------------------------------------------
# TabPy 서버에 엔드포인트 배포 (태블로에서 호출 가능하도록 등록)
# ---------------------------------------------------------------------------
# 사용 방법: TabPy 서버를 켠 뒤, 터미널에서 한 번 실행
#   python tabpy_wholesale_cluster.py
# 배포 없이 로컬 검사만: python tabpy_wholesale_cluster.py --check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # --check: 자체 검사만 수행 (TabPy 서버 없이 더블체크)
    if "--check" in sys.argv or "-c" in sys.argv:
        try:
            _run_self_check()
            print("자체 검사 통과: tabpy_wholesale_cluster.py 함수·모델이 정상 동작합니다.")
        except Exception as e:
            print("자체 검사 실패:", e)
            sys.exit(1)
        sys.exit(0)

    # 배포 전 로컬 자체 검사
    try:
        _run_self_check()
        print("자체 검사 통과. TabPy 서버에 배포를 진행합니다...")
    except Exception as e:
        print("자체 검사 실패로 배포를 중단합니다:", e)
        sys.exit(1)

    
    # TabPy 서버 연결 (기본 포트 9004, 서버 주소/포트는 환경에 맞게 수정)
    _TABPY_URL = "http://localhost:9004/"
    client = Client(_TABPY_URL)

    _OVERRIDE = True

    # [1] 군집 예측
    client.deploy(
        "get_cluster",
        get_cluster,
        "6개 구매 컬럼 → 군집 ID (0~4) 리스트.",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_pca_coords",
        get_pca_coords,
        "6개 구매 컬럼 → PCA 좌표 리스트 (행별 [PC1, PC2, ...]).",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_pca1",
        get_pca1,
        "행별 PC1 점수 (SCRIPT_REAL).",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_pca2",
        get_pca2,
        "행별 PC2 점수 (SCRIPT_REAL).",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_pca3",
        get_pca3,
        "행별 PC3 점수 (SCRIPT_REAL).",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_biplot_pc1_loadings",
        get_biplot_pc1_loadings,
        "Biplot 변수별 PC1 로딩 (SCRIPT_REAL).",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_biplot_pc2_loadings",
        get_biplot_pc2_loadings,
        "Biplot 변수별 PC2 로딩 (SCRIPT_REAL).",
        override=_OVERRIDE,
    )

    # [2] 채널 분류 (AUC-ROC용 확률 포함)
    client.deploy(
        "get_channel",
        get_channel,
        "6개 구매 + Region → 채널 0/1 예측 (SCRIPT_INT). ModelName 선택 가능.",
        override=_OVERRIDE,
    )
    client.deploy(
        "get_channel_proba",
        get_channel_proba,
        "6개 구매 + Region → 채널 1(양성) 확률 (SCRIPT_REAL). AUC-ROC 곡선용.",
        override=_OVERRIDE,
    )
    # EC2 배포 시 메모리 부족으로 끊기면 주석 해제하지 말고, 필요 시 feature_importance_for_tableau.csv 생성 후 배포
    client.deploy(
        "get_feature_importance",
        get_feature_importance,
        "모델별 특성 중요도 반환 (Feature, ModelName).",
        override=_OVERRIDE,
    )

    # ROC Curve & 혼동행렬 (같은 client로 한 번만 배포)
    client.deploy("Get_ROC_TPR", get_tpr_for_tableau, "Returns TPR list for ROC Curve", override=_OVERRIDE)
    client.deploy("Get_ROC_AUC", get_auc_for_tableau, "Returns AUC value list", override=_OVERRIDE)
    client.deploy("Get_Confusion_Matrix", get_confusion_metrics, "Returns 4 values (TN, FP, FN, TP) for Heatmap", override=_OVERRIDE)
    client.deploy("Get_CM_Live", get_confusion_matrix_live, "Returns [TN, FP, FN, TP] for selected model", override=_OVERRIDE)

    # Spearman 상관계수 (두 지출 항목 리스트 → 테이블 계산용 리스트)
    client.deploy(
        "calculate_spearman",
        calculate_spearman,
        "두 지출 항목 리스트 → Spearman 상관계수 (입력 길이만큼 복제된 리스트 반환). SCRIPT_REAL 사용.",
        override=_OVERRIDE,
    )

    print("TabPy 엔드포인트 배포 완료!")
    print("  Spearman: calculate_spearman")
    print("  군집: get_cluster, get_pca_coords, get_pca1, get_pca2, get_pca3, get_biplot_pc1_loadings, get_biplot_pc2_loadings")
    print("  get_pca_coords 실행 검증 완료 (자체 검사 통과)")
    print("  채널: get_channel, get_channel_proba  (get_feature_importance는 주석 처리됨)")
    print("  ROC/CM: Get_ROC_TPR, Get_ROC_AUC, Get_Confusion_Matrix, Get_CM_Live")
