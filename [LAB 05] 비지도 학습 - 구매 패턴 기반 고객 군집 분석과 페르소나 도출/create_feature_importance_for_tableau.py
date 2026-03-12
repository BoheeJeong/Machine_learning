# -*- coding: utf-8 -*-
"""
feature_importance_for_tableau.csv 생성 스크립트

[LAB 05] 폴더의 채널 분류 pkl(wholesale_logistic.pkl, wholesale_xgb.pkl 등)에서
특성 중요도를 읽어 feature_importance_for_tableau.csv 를 만듭니다.
이 CSV가 있으면 tabpy_wholesale_cluster.py 의 get_feature_importance 는
모델을 로드하지 않고 CSV만 읽어서 EC2/TabPy 메모리를 절약합니다.

사용법 (이 폴더에서 실행):
  python create_feature_importance_for_tableau.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent

FEATURE_ORDER = [
    "log_Fresh",
    "log_Milk",
    "log_Grocery",
    "log_Frozen",
    "log_Detergents_Paper",
    "log_Delicassen",
    "Region_2",
    "Region_3",
]

MODEL_FILES = {
    "LogisticRegression": "wholesale_logistic.pkl",
    "XGBoost": "wholesale_xgb.pkl",
    "SGDClassifier": "wholesale_sgd.pkl",
    "RandomForest": "wholesale_rf.pkl",
}


def main():
    rows = []
    for model_name, filename in MODEL_FILES.items():
        path = _DIR / filename
        if not path.exists():
            print(f"  건너뜀 (파일 없음): {filename}")
            continue
        with open(path, "rb") as f:
            clf = pickle.load(f)
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            imp = np.zeros(len(FEATURE_ORDER))
        for feat, val in zip(FEATURE_ORDER, imp):
            rows.append({"model_name": model_name, "feature_name": feat, "importance": float(val)})
        print(f"  추가: {model_name}")

    if not rows:
        print("저장된 pkl이 없어 CSV를 생성하지 않습니다.")
        return

    out_path = _DIR / "feature_importance_for_tableau.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
