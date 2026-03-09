# -*- coding: utf-8 -*-
"""
[LAB 05] 01~07 Channel 분류 모델 종합 요약
- 동일 train/test로 저장된 pkl을 불러와 성능·학습곡선(과적합 판정)을 한 번에 계산합니다.
- 실행 전: 01~07 노트북에서 각각 저장 셀까지 실행해 두세요 (wholesale_*.pkl 생성).
- 실행 방법:
  1) [LAB 05] 폴더를 현재 디렉터리로 두고: python 00-모델_요약_성능_과적합_리포트.py
  2) 한글/이모지 오류 시: set PYTHONIOENCODING=utf-8 후 실행 또는 Jupyter에서 실행
- 결과: 00-모델_요약_성능_과적합_리포트.csv, .md 생성
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame

# 데이터 로드 (노트북과 동일)
from hossam import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

LAB05_DIR = Path(__file__).resolve().parent
RAW_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def get_data_and_split():
    """노트북과 동일한 전처리·split."""
    origin = load_data("wholesale_customers")
    df = origin[RAW_COLS + ["Region", "Channel"]].copy()
    df_log = df[RAW_COLS].copy()
    for c in RAW_COLS:
        df_log[f"log_{c}"] = np.log1p(df_log[c])
    df_log["Region"] = df["Region"].values
    log_cols = [f"log_{c}" for c in RAW_COLS]
    FEATURE_COLS = log_cols + ["Region"]

    scaler_path = LAB05_DIR / "wholesale_scaler.pkl"
    if not scaler_path.exists():
        scaler = StandardScaler()
        X = scaler.fit_transform(df_log[FEATURE_COLS])
    else:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(df_log[FEATURE_COLS])

    y = (df["Channel"] - 1).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=52, stratify=y
    )
    return X_train, X_test, y_train, y_test


def overfit_status_classification(final_train, final_cv, final_std):
    """분류용 과적합 판정 (hs_learning_cv 분류 로직)."""
    gap_ratio = final_train - final_cv
    var_ratio = final_std
    if final_train < 0.6 and final_cv < 0.6:
        return "⚠ 과소적합"
    if gap_ratio > 0.1:
        return "⚠ 과대적합"
    if gap_ratio <= 0.05 and var_ratio <= 0.05:
        return "✅ 일반화 양호"
    if var_ratio > 0.1:
        return "⚠ 데이터 부족"
    return "⚠ 판단유보"


# (노트북 번호, 표시 이름, pkl 파일명)
MODELS = [
    ("01", "XGBoost", "wholesale_xgb.pkl"),
    ("02", "LogisticRegression", "wholesale_logistic.pkl"),
    ("03", "SGDClassifier", "wholesale_sgd.pkl"),
    ("04", "DecisionTree", "wholesale_dt.pkl"),
    ("05", "RandomForest", "wholesale_rf.pkl"),
    ("06", "GradientBoosting", "wholesale_gb.pkl"),
    ("07", "CatBoost", "wholesale_catboost.pkl"),
]


def main():
    X_train, X_test, y_train, y_test = get_data_and_split()

    rows = []
    for nb, name, pkl_name in MODELS:
        path = LAB05_DIR / pkl_name
        if not path.exists():
            rows.append({
                "노트북": nb,
                "모델": name,
                "정확도": np.nan,
                "정밀도": np.nan,
                "재현율": np.nan,
                "F1": np.nan,
                "AUC": np.nan,
                "Train_ROC_AUC": np.nan,
                "CV_ROC_AUC": np.nan,
                "Train-CV_gap": np.nan,
                "CV_std": np.nan,
                "과적합_판정": "미저장",
                "비고": f"{pkl_name} 없음",
            })
            continue

        with open(path, "rb") as f:
            clf = pickle.load(f)

        # Test 성능
        y_pred = clf.predict(X_test)
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5

        # 학습 곡선 (과적합 판정)
        try:
            train_sizes, train_scores, cv_scores = learning_curve(
                clf,
                X_train,
                y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=52,
            )
            train_mean = train_scores.mean(axis=1)
            cv_mean = cv_scores.mean(axis=1)
            cv_std_arr = cv_scores.std(axis=1)
            final_train = float(train_mean[-1])
            final_cv = float(cv_mean[-1])
            final_std = float(cv_std_arr[-1])
            gap = final_train - final_cv
            status = overfit_status_classification(final_train, final_cv, final_std)
        except Exception as e:
            final_train = final_cv = final_std = gap = np.nan
            status = "계산 오류"
            rows.append({
                "노트북": nb,
                "모델": name,
                "정확도": round(acc, 4),
                "정밀도": round(prec, 4),
                "재현율": round(rec, 4),
                "F1": round(f1, 4),
                "AUC": round(auc, 4),
                "Train_ROC_AUC": np.nan,
                "CV_ROC_AUC": np.nan,
                "Train-CV_gap": np.nan,
                "CV_std": np.nan,
                "과적합_판정": status,
                "비고": str(e)[:80],
            })
            continue

        rows.append({
            "노트북": nb,
            "모델": name,
            "정확도": round(acc, 4),
            "정밀도": round(prec, 4),
            "재현율": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4),
            "Train_ROC_AUC": round(final_train, 4),
            "CV_ROC_AUC": round(final_cv, 4),
            "Train-CV_gap": round(gap, 4),
            "CV_std": round(final_std, 4),
            "과적합_판정": status,
            "비고": "",
        })

    summary = DataFrame(rows)

    # CSV 저장
    csv_path = LAB05_DIR / "00-모델_요약_성능_과적합_리포트.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("저장:", csv_path)

    # Markdown 테이블 저장 (to_markdown 미지원 시 수동 생성)
    md_path = LAB05_DIR / "00-모델_요약_성능_과적합_리포트.md"
    cols = list(summary.columns)
    lines = [
        "# [LAB 05] Channel 분류 모델 종합 요약 (01~07)",
        "",
        "동일 train/test, 동일 전처리 기준. 저장된 pkl 로드 후 Test 성능 + 학습곡선(과적합 판정) 계산.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in summary.iterrows():
        cells = [str(r[c]) if pd.notna(r[c]) else "" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 과적합 판정 기준 (분류)",
        "",
        "- `✅ 일반화 양호`: Train-CV gap ≤ 0.05 이고 CV 표준편차 ≤ 0.05",
        "- `⚠ 과대적합`: Train-CV gap > 0.1",
        "- `⚠ 과소적합`: Train·CV 둘 다 < 0.6",
        "- `⚠ 데이터 부족`: CV 표준편차 > 0.1",
        "- `⚠ 판단유보`: 위에 해당하지 않는 경우",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("저장:", md_path)
    return summary


if __name__ == "__main__":
    df = main()
    print("\n요약 테이블:")
    print(df.to_string())
