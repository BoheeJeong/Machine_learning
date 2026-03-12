# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
p = Path(r"c:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출\wholesale_for_tableau.csv")
df = pd.read_csv(p)
cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
out = Path(r"c:\Users\bohee\Documents\GitHub\Machine_learning\segment_stats_result.txt")
with open(out, 'w', encoding='utf-8') as f:
    f.write("=== 군집별 고객 수 ===\n")
    cnt = df['ClusterID'].value_counts().sort_index()
    for c in range(5):
        f.write(f"  세그먼트 {c}: {cnt.get(c, 0)}명\n")
    m = df.groupby('ClusterID')[cols].mean().round(0)
    f.write("\n=== 군집별 평균 구매액 ===\n" + m.to_string() + "\n")
    total_mean = df[cols].mean()
    ratio = m / total_mean
    f.write("\n=== 군집별 전체대비 비율 ===\n" + ratio.round(2).to_string() + "\n")
    df['TotalSpend'] = df[cols].sum(axis=1)
    avg_total = df.groupby('ClusterID')['TotalSpend'].mean()
    f.write("\n=== 1인당 평균 총 구매액 ===\n")
    for c in range(5):
        f.write(f"  세그먼트 {c}: {avg_total.get(c, 0):.0f}\n")
print("Done. Check segment_stats_result.txt")
