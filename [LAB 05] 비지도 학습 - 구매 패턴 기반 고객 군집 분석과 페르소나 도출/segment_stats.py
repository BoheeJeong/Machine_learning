# -*- coding: utf-8 -*-
"""군집별 통계 계산 (페르소나/마케팅 문서용)."""
import pandas as pd
from pathlib import Path

DIR = Path(__file__).resolve().parent
df = pd.read_csv(DIR / "wholesale_for_tableau.csv")
cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

print('=== 군집별 고객 수 ===')
cnt = df['ClusterID'].value_counts().sort_index()
for c in range(5):
    print(f"  세그먼트 {c}: {cnt.get(c, 0)}명")
print()

print('=== 군집별 평균 구매액 (원본 스케일, mu) ===')
m = df.groupby('ClusterID')[cols].mean().round(0)
print(m.to_string())
print()

print('=== 군집별 총 매출액 (합계) ===')
s = df.groupby('ClusterID')[cols].sum()
s['Total'] = s.sum(axis=1)
print(s.round(0).to_string())
print()

total_mean = df[cols].mean()
print('=== 전체 평균 (참고) ===')
print(total_mean.round(0))
print()

print('=== 군집별 - 전체대비 비율 (1이면 평균과 동일) ===')
ratio = m / total_mean
print(ratio.round(2).to_string())
print()

print('=== 군집별 1인당 평균 총 구매액 ===')
df['TotalSpend'] = df[cols].sum(axis=1)
avg_total = df.groupby('ClusterID')['TotalSpend'].mean()
for c in range(5):
    print(f"  세그먼트 {c}: {avg_total.get(c, 0):.0f} mu")
print()

# 비율 기준 세그먼트 특징 요약
print('=== 세그먼트별 상대적 강점 (비율 > 1.1인 품목) ===')
for c in range(5):
    r = ratio.loc[c]
    strong = [cols[i] for i in range(len(cols)) if r.iloc[i] > 1.1]
    weak = [cols[i] for i in range(len(cols)) if r.iloc[i] < 0.9]
    print(f"  세그먼트 {c}: 강점 {strong}, 약점 {weak}")
