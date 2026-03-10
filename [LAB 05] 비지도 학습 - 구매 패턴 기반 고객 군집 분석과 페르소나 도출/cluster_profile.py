# -*- coding: utf-8 -*-
import pandas as pd
import os
os.chdir(r"c:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출")
df = pd.read_csv('wholesale_for_tableau.csv')
cols = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
print('=== 군집별 고객 수 ===')
print(df['ClusterID'].value_counts().sort_index())
print()
print('=== 군집별 평균 구매액 (원본 스케일) ===')
m = df.groupby('ClusterID')[cols].mean()
m = m.round(0)
print(m.to_string())
print()
total_mean = df[cols].mean()
print('=== 전체 평균 (참고) ===')
print(total_mean.round(0))
print()
print('=== 군집별 - 전체대비 비율 (1이면 평균과 동일) ===')
ratio = m / total_mean
print(ratio.round(2).to_string())
