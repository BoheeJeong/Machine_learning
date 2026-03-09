# TabPy로 Tableau에서 구매 패턴 군집 시각화하기

LAB 05 노트북의 **구매 패턴 기반 고객 군집 분석** 결과를 pkl로 저장한 뒤, **TabPy**를 통해 Tableau에서 클러스터 ID를 계산·시각화하는 방법입니다.

---

## 1. 사전 준비

- **Python**: 노트북과 동일한 환경 권장 (sklearn, pandas, numpy)
- **TabPy** 설치 및 Tableau 연동  
  - [TabPy 문서](https://tableau.github.io/TabPy/docs/) 참고
- **데이터**: 노트북의 `df`(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen 6컬럼)에 해당하는 데이터

---

## 2. 모델 pkl 생성

### 방법 A: 노트북에서 데이터를 CSV로 저장한 뒤 스크립트 실행

1. 노트북에서 데이터프레임 `df`까지 실행한 다음, 다음을 실행해 CSV 저장:
   ```python
   df.to_csv('wholesale_customers_6cols.csv', index=False)
   ```
2. 같은 폴더에서:
   ```bash
   python save_wholesale_cluster_model.py
   ```
3. `wholesale_cluster_model.pkl`이 생성됩니다.

### 방법 B: hossam으로 wholesale_customers 사용

- `hossam` 패키지로 `load_data('wholesale_customers')`를 쓸 수 있으면, CSV 없이 그대로:
  ```bash
  python save_wholesale_cluster_model.py
  ```
- 스크립트가 자동으로 Channel/Region을 제거하고 6컬럼만 사용합니다.

---

## 3. TabPy에서 사용

- `tabpy_wholesale_cluster.py`와 `wholesale_cluster_model.pkl`은 **같은 폴더**에 두세요.
- Tableau에서 데이터 소스에 **Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen** 컬럼이 있어야 합니다.

### Tableau 계산 필드 예시 (행 단위 클러스터)

TabPy가 해당 스크립트 경로를 `sys.path`에서 찾을 수 있도록 설정한 뒤, 아래와 같이 계산 필드를 만듭니다.

- **계산 필드 이름**: 예) `Cluster ID`
- **계산식** (SCRIPT_INT 사용):

```tableau
SCRIPT_INT(
  "
  import sys
  sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
  from tabpy_wholesale_cluster import get_cluster
  return get_cluster(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6)
  ",
  SUM([Fresh]), SUM([Milk]), SUM([Grocery]), SUM([Frozen]),
  SUM([Detergents_Paper]), SUM([Delicassen])
)
```

- `sys.path.append(...)` 안의 경로를 실제 **Lab 05 폴더 경로**로 바꿔주세요.
- 행 단위로 집계되지 않도록 하려면, 차트에 **고객 ID** 등 행 식별자를 넣고, 위 인자를 해당 행의 값이 되도록 조정할 수 있습니다 (필요 시 SUM 대신 ATTR 등 사용).

### TabPy 서버에 함수 배포하는 경우

TabPy 서버에 `get_cluster`를 등록해 두었다면, Tableau에서는 해당 함수 이름과 인자만 넘기도록 설정하면 됩니다. (TabPy 배포 방법은 TabPy 문서 참고.)

---

## 4. Tableau 시각화 예시

- **Cluster ID**를 차원으로 넣고, 색상 또는 마크 유형으로 구분
- 축에 Fresh, Milk, Grocery 등 구매액을 넣어 군집별 분포 확인
- 군집별 평균/합계 등 집계로 페르소나 요약

---

## 5. Tableau에 넣을 데이터 (CSV) — 로그/PCA 아님

**Tableau(또는 엑셀)에 넣는 파일은 원본 구매 금액 그대로 사용하면 됩니다.** 로그 변환이나 PCA 처리된 값을 넣을 필요 없습니다.

- **권장**: `create_tableau_csv.py` 를 실행해 생성하는 **`wholesale_for_tableau.csv`** 사용  
  - 구성: `CustomerID`, `Channel`, `Region`, `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`, **`ClusterID`**  
  - 모두 **원본 단위(금액)** + 미리 계산된 군집 번호(0~4).  
  - 이 CSV만 연결해도 Tableau에서 ClusterID로 색/필터 바로 사용 가능.

```bash
python create_tableau_csv.py
```

- 데이터는 `hossam` 의 `wholesale_customers` 또는 `wholesale_customers_6cols.csv` 에서 읽습니다.  
- 생성된 CSV를 Tableau에서 연결한 뒤, 필요하면 같은 파일을 엑셀에서 열어 수정·저장해도 됩니다.

## 6. 파일 정리

| 파일 | 설명 |
|------|------|
| `create_tableau_csv.py` | **Tableau용 CSV 생성** (원본 금액 + ClusterID). 실행 후 `wholesale_for_tableau.csv` 생성 |
| `save_wholesale_cluster_model.py` | 노트북과 동일 파이프라인으로 학습 후 `wholesale_cluster_model.pkl` 생성 |
| `tabpy_wholesale_cluster.py` | TabPy에서 불러와 `get_cluster(...)` 로 클러스터 ID 반환 |
| `wholesale_for_tableau.csv` | **Tableau/엑셀 연결용** (원본 금액 + ClusterID) |
| `wholesale_cluster_model.pkl` | 저장된 모델. TabPy용. `save_wholesale_cluster_model.py` 실행 후 생성 |
| `wholesale_customers_6cols.csv` | (선택) 노트북에서 내보낸 6컬럼 데이터. 없으면 hossam 사용 시도 |

이 구성을 유지하면 Tableau에서 원본 데이터로 시각화하고, 동일한 군집 결과를 재현할 수 있습니다.
