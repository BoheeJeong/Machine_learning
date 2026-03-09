# Tableau에서 PCA 산점도 쓰기

## CSV에 PC1, PC2, PC3 포함하기

### 방법 1: 스크립트로 한 번에 생성 (권장)

아래 명령 한 번이면 **원본 + PC1, PC2, PC3 + ClusterID** 가 모두 들어간 CSV가 만들어집니다.

```bash
python create_tableau_csv.py
```

생성되는 **wholesale_for_tableau.csv** 에는 다음 컬럼이 포함됩니다.  
CustomerID, Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, **PC1, PC2, PC3**, ClusterID

---

### 방법 2: 노트북에서 직접 저장할 때

노트북에서 `pca_df`, `estimator`(KMeans), `df`(또는 `origin`)까지 실행한 상태에서 아래처럼 하면 됩니다.

- **Cluster ID** 는 노트북에서는 `estimator.labels_` 또는 `clusters` / `cdf['ClusterID']` 로 씁니다. (`kmeans` 라는 변수명을 쓰지 않았다면 `estimator` 사용.)

```python
# 노트북 변수명에 맞춤 (estimator = KMeans 모델, pca_df = PCA 점수)
final_df = df.copy()
if 'Channel' in origin.columns:
    final_df.insert(0, 'Channel', origin['Channel'].values)
if 'Region' in origin.columns:
    final_df.insert(0, 'Region', origin['Region'].values)
final_df.insert(0, 'CustomerID', np.arange(1, len(final_df) + 1))

final_df['PC1'] = pca_df['PC1'].values
final_df['PC2'] = pca_df['PC2'].values
final_df['PC3'] = pca_df['PC3'].values
final_df['Cluster ID'] = estimator.labels_   # 또는 clusters

final_df.to_csv("wholesale_pca_cluster.csv", index=False)
```

- 저장 파일명을 `wholesale_for_tableau.csv` 로 맞추면 기존 Tableau 데이터 소스와 그대로 연결해 쓸 수 있습니다.

---

## Tableau에서 PCA 산점도 만들기

### A) CSV에 PC1/PC2가 있는 경우

1. **wholesale_for_tableau.csv** (또는 wholesale_pca_cluster.csv) 연결.
2. **열**: **PC1**  
   **행**: **PC2**
3. **마크** → **색상**: **Cluster ID** (또는 ClusterID)  
   **세부 정보**: **Customer ID** (또는 CustomerID) — 점 하나가 고객 한 명.
4. 필요하면 **도구 설명**에 Fresh, Milk 등 구매 컬럼 추가.

이렇게 하면 PCA 공간(PC1–PC2)에서 군집별로 색이 나뉜 산점도를 Tableau에서 볼 수 있습니다.

---

### B) TabPy로 PCA 계산 (엑셀/CSV 업데이트 없이)

데이터에 PC1/PC2 컬럼 없이 **TabPy 서버에서 pkl로 PCA만 계산**해서 쓰고 싶다면:

1. **TabPy** 실행 및 Tableau **Analytics Extension** 연결 (localhost:9004).
2. **계산된 필드** 두 개 만듦.

**PC1 (TabPy):**

```tableau
SCRIPT_REAL(
  "
  import sys
  sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
  from tabpy_wholesale_cluster import get_pca1
  return get_pca1(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6)
  ",
  [Fresh], [Milk], [Grocery], [Frozen], [Detergents Paper], [Delicassen]
)
```

**PC2 (TabPy):**

```tableau
SCRIPT_REAL(
  "
  import sys
  sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
  from tabpy_wholesale_cluster import get_pca2
  return get_pca2(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6)
  ",
  [Fresh], [Milk], [Grocery], [Frozen], [Detergents Paper], [Delicassen]
)
```

- `sys.path.append(...)` 경로는 본인 PC의 **Lab 05 폴더**로 수정.
- Tableau 필드명이 `Detergents Paper` 이면 위처럼, `Detergents_Paper` 이면 `[Detergents_Paper]` 로 맞추기.

3. 시트에서 **열**: **PC1 (TabPy)** / **행**: **PC2 (TabPy)** / **색상**: **Cluster ID** (또는 예측 군집 계산 필드) 로 PCA 산점도 완성.

---

## PCA Biplot (변수 화살표 추가)

**wholesale_pca_loadings.csv** 에는 `Feature`, `PointOrder`, `PC1`, `PC2` 가 들어 있습니다.  
각 변수(Fresh, Milk, Grocery 등)마다 **원점 (0,0) → (PC1 로딩, PC2 로딩)** 두 점으로 Line을 그리면 Biplot 화살표가 됩니다.

### 1) 로딩 CSV 생성

`python create_tableau_csv.py` 를 실행하면 **wholesale_pca_loadings.csv** 도 함께 생성됩니다.

### 2) Tableau에서 Biplot 만들기

1. **wholesale_for_tableau** 로 기존 PCA 산점도 시트 만듦 (열: PC1, 행: PC2, 색상: Cluster ID).
2. **데이터** 메뉴 → **데이터 원본 추가** → **wholesale_pca_loadings.csv** 연결.
3. **관계 설정**: 두 원본 간 공통 필드가 없으므로 **연결**하지 않고, **같은 시트에 이중 축**으로 올립니다.
4. **새 시트**에서:
   - **열**: PC1 (loadings 원본)  
   - **행**: PC2 (loadings 원본)  
   - **마크**: **선(Line)** 선택  
   - **경로**: **PointOrder** (0 → 1 순서로 선이 그려짐)  
   - **세부 정보**: **Feature** (변수별로 한 줄씩)  
   → 원점에서 각 변수 방향으로 선(화살표)이 그려짐.
5. **기존 PCA 시트**와 **이 로딩 시트**를 **대시보드**에서 겹치거나, **이중 축**으로 합치면 Biplot 완성.  
   (이중 축 사용 시: PCA 시트 복제 후 로딩 데이터를 두 번째 축에 올리고, 마크를 선으로 맞춘 뒤 축 동기화.)

### 3) 화살표 끝에 변수 이름 넣기

- 로딩 데이터에서 **PointOrder = 1** 인 행만 사용하는 시트를 만들고, **PC1**, **PC2**를 열/행에, **Feature**를 레이블에 넣으면 화살표 끝에 Fresh, Milk 등 이름이 표시됩니다. 이 시트를 Biplot 위에 겹쳐서 사용하면 됩니다.

---

### 4) TabPy로 Biplot 로딩 계산 (CSV 없이)

**tabpy_wholesale_cluster.py** 에 `get_biplot_pc1_loadings(Feature)`, `get_biplot_pc2_loadings(Feature)` 가 있습니다.  
Tableau에서 **Feature** 가 6개 값(Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicassen)인 데이터 원본(예: 6행짜리 CSV나 파라미터)을 만들고:

- **PC1 로딩 (TabPy)** 계산 필드:
  ```tableau
  SCRIPT_REAL(
    "
    import sys
    sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
    from tabpy_wholesale_cluster import get_biplot_pc1_loadings
    return get_biplot_pc1_loadings(_arg1)
    ",
    [Feature]
  )
  ```
- **PC2 로딩 (TabPy)** 계산 필드:
  ```tableau
  SCRIPT_REAL(
    "
    import sys
    sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
    from tabpy_wholesale_cluster import get_biplot_pc2_loadings
    return get_biplot_pc2_loadings(_arg1)
    ",
    [Feature]
  )
  ```
- 시트: **열** PC1 로딩, **행** PC2 로딩, **마크** 선(Line).  
  화살표를 그리려면 **원점(0,0)** 과 **끝점(PC1로딩, PC2로딩)** 두 점이 필요하므로, 데이터에 **PointOrder**(0 또는 1)를 두고 **경로**에 넣거나, 위처럼 TabPy로 끝점만 구한 뒤 별도로 원점 행을 합치는 방식으로 구성하면 됩니다.
