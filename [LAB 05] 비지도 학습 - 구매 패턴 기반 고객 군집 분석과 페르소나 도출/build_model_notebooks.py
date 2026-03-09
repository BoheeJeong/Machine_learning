# -*- coding: utf-8 -*-
"""LAB 05 모델별 노트북 02~07 생성: 동일 전처리 + 단일 모델 학습 + pkl 저장."""

import json
from pathlib import Path

LAB05_DIR = Path(__file__).resolve().parent
NB01 = LAB05_DIR / "01-XGBoost_Channel_분류_TabPy.ipynb"

MODELS = [
    {
        "num": "02",
        "name": "LogisticRegression",
        "title": "로지스틱 회귀",
        "pkl": "wholesale_logistic.pkl",
        "imports": "from sklearn.linear_model import LogisticRegression",
        "clf_line": "clf = LogisticRegression(C=0.1, random_state=52, max_iter=1000)",
    },
    {
        "num": "03",
        "name": "SGDClassifier",
        "title": "SGD(경사하강)",
        "pkl": "wholesale_sgd.pkl",
        "imports": "from sklearn.linear_model import SGDClassifier",
        "clf_line": "clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=52)",
    },
    {
        "num": "04",
        "name": "DecisionTree",
        "title": "결정 트리",
        "pkl": "wholesale_dt.pkl",
        "imports": "from sklearn.tree import DecisionTreeClassifier",
        "clf_line": "clf = DecisionTreeClassifier(max_depth=5, random_state=52)",
    },
    {
        "num": "05",
        "name": "RandomForest",
        "title": "랜덤 포레스트",
        "pkl": "wholesale_rf.pkl",
        "imports": "from sklearn.ensemble import RandomForestClassifier",
        "clf_line": "clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=52, n_jobs=-1)",
    },
    {
        "num": "06",
        "name": "GradientBoosting",
        "title": "그래디언트 부스팅",
        "pkl": "wholesale_gb.pkl",
        "imports": "from sklearn.ensemble import GradientBoostingClassifier",
        "clf_line": "clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=52)",
    },
    {
        "num": "07",
        "name": "CatBoost",
        "title": "CatBoost",
        "pkl": "wholesale_catboost.pkl",
        "imports": "from catboost import CatBoostClassifier",
        "clf_line": "clf = CatBoostClassifier(iterations=100, depth=4, random_state=52, verbose=0)",
    },
]

# 공통 import (모델별 import 제외)
COMMON_IMPORTS = '''from hossam import load_data
import pandas as pd
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import pickle
from pathlib import Path

my_dpi = 200
RAW_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
LAB05_DIR = Path('.').resolve()
'''


def make_cell(cell_type, source, outputs=None):
    c = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if outputs:
        c["outputs"] = outputs
    return c


def run():
    with open(NB01, "r", encoding="utf-8") as f:
        nb01 = json.load(f)
    cells01 = nb01["cells"]

    # 01에서 복사할 셀 인덱스 (마크다운/코드 내용만 참고)
    # 셀 3: 데이터 로드 코드
    data_load_src = "".join(cells01[3]["source"])
    # 셀 5: 전처리 코드 (X, y, scaler, FEATURE_COLS)
    preprocess_src = "".join(cells01[5]["source"])
    # 셀 7: hs_cls_bin_scores
    hs_cls_src = "".join(cells01[7]["source"])
    # 셀 8: hs_learning_cv
    hs_lc_src = "".join(cells01[8]["source"])

    for m in MODELS:
        title_md = f"""# [LAB 05] {m['title']} Channel 분류
## 구매 패턴 기반 고객 군집 노트북과 동일 전처리 → Channel(1/2) 예측

- **전처리**: 구매액 6개 np.log1p → StandardScaler, 학습 피처는 **스케일된 log 6개 + Region** 7개
- **타겟**: Channel (1 또는 2) → 라벨 0/1로 변환하여 이진 분류
- **모델**: {m['title']} 분류기
- **저장**: scaler, {m['name']} 모델 각각 pkl 저장
"""

        import_src = COMMON_IMPORTS.strip() + "\n\n" + m["imports"].strip() + "\n"

        train_src = f"""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=52, stratify=y
)

{m['clf_line']}
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_1 = y_pred_proba[:, 1]  # 양성(Channel=2) 확률
"""

        save_src = f"""scaler_path = LAB05_DIR / 'wholesale_scaler.pkl'
model_path = LAB05_DIR / '{m['pkl']}'

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print('저장 완료:', scaler_path)
print('저장 완료:', model_path)
"""

        cells = [
            make_cell("markdown", title_md),
            make_cell("markdown", "## 1. 라이브러리 및 데이터 로드"),
            make_cell("code", import_src, []),
            make_cell("code", data_load_src, []),
            make_cell("markdown", "## 2. 전처리"),
            make_cell("code", preprocess_src, []),
            make_cell("markdown", "## 2.5 성능평가 함수 (LAB 09 공통)\n\nLAB 09 `03-로지스틱 성능평가함수개선.ipynb`에서 사용하는 **이진 분류 성능평가**·**학습 곡선(과적합 판정)** 함수를 정의합니다."),
            make_cell("code", hs_cls_src, []),
            make_cell("code", hs_lc_src, []),
            make_cell("markdown", f"## 3. train/test split 및 {m['title']} 학습"),
            make_cell("code", train_src, []),
            make_cell("markdown", "## 4. 성능 지표: Accuracy, Confusion Matrix, ROC-AUC"),
            make_cell("code", "# LAB 09 공통 함수: 이진 분류 성능평가 + ROC 곡선\nscore_df = hs_cls_bin_scores(clf, X_test, y_test)\nscore_df", []),
            make_cell("markdown", "## 4.6 학습 곡선 (Learning Curve)"),
            make_cell("code", "# LAB 09 공통 함수: 학습 곡선 + 과적합·일반화 판정 (시각화는 함수 내부에서 출력)\nresult_df_lc = hs_learning_cv(clf, X_train, y_train)\nresult_df_lc", []),
            make_cell("code", "# 학습 곡선 시각화 및 판정 결과는 위 셀의 hs_learning_cv() 내부에서 출력됨.", []),
            make_cell("markdown", "## 5. Scaler / 모델 저장"),
            make_cell("code", save_src, []),
        ]

        out_nb = {
            "cells": cells,
            "metadata": nb01.get("metadata", {}),
            "nbformat": nb01.get("nbformat", 4),
            "nbformat_minor": nb01.get("nbformat_minor", 4),
        }
        out_path = LAB05_DIR / f"{m['num']}-{m['name']}_Channel_분류.ipynb"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_nb, f, ensure_ascii=False, indent=1)
        print("Created:", out_path.name)
    print("Done. 02~07 notebooks and pkl save cells ready.")


if __name__ == "__main__":
    run()
