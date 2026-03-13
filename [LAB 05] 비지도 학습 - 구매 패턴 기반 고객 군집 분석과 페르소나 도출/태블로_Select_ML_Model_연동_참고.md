# 태블로 [Select ML Model] ↔ Python(TabPy) 연동 참고

태블로의 **Select ML Model** 매개변수를 Python과 맞출 때 참고할 설정 요약입니다.

---

## 1. 지원 모델 이름 (Python 기준)

Python 쪽에서 **채널 예측**에 사용하는 모델은 아래 **4개만** 지원합니다.

| Python 내부 키 (소문자) | 권장 표시 이름 (태블로 목록에 넣을 값) | 사용하는 pkl 파일 |
|------------------------|----------------------------------------|---------------------|
| `logisticregression`   | **LogisticRegression**                 | wholesale_logistic.pkl |
| `xgboost`              | **XGBoost**                            | wholesale_xgb.pkl     |
| `sgdclassifier`        | **SGDClassifier**                     | wholesale_sgd.pkl     |
| `randomforest`         | **RandomForest**                      | wholesale_rf.pkl      |

- **대소문자**: Python은 `str(model_name).strip().lower()`로 비교하므로 **태블로에서는 어떤 대소문자로 보내도 인식됩니다.**  
  예: `XGBoost`, `xgboost`, `XGBOOST` 모두 동일하게 처리.
- **권장**: 태블로 매개변수 목록에는 위 표의 **권장 표시 이름** 그대로 넣으면, 리포트 문구(CASE [Select ML Model])와도 일치합니다.

---

## 2. ModelName을 받는 함수 (인자 순서)

TabPy에서 **Select ML Model**과 통신하는 함수는 아래 세 가지입니다. 모두 **마지막 인자**가 `ModelName`입니다.

### 2.1 get_channel (채널 0/1 예측) — SCRIPT_INT

- **시그니처**:  
  `get_channel(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost")`
- **인자 순서**:  
  `_arg1`=Fresh, `_arg2`=Milk, `_arg3`=Grocery, `_arg4`=Frozen, `_arg5`=Detergents_Paper, `_arg6`=Delicassen, `_arg7`=Region, **`_arg8`=ModelName**
- **기본값**: 인자가 비어 있거나 None이면 `"XGBoost"` 사용.

### 2.2 get_channel_proba (채널 1 확률) — SCRIPT_REAL

- **시그니처**:  
  `get_channel_proba(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen, Region, ModelName="XGBoost")`
- **인자 순서**:  
  `_arg1`~`_arg7` 동일, **`_arg8`=ModelName**
- **기본값**: 동일하게 `"XGBoost"`.

### 2.3 get_feature_importance (특성 중요도, 사용 시에만)

- **시그니처**:  
  `get_feature_importance(Feature=None, ModelName="XGBoost")`
- **태블로 예시**:  
  `SCRIPT_REAL("... get_feature_importance(_arg1, _arg2)", ATTR([Feature]), [Select ML Model])`  
  → `_arg1`=Feature, **`_arg2`=ModelName**

---

## 3. Python 쪽 ModelName 처리 방식

- **리스트로 오는 경우** (태블로가 집계 결과로 리스트 전달):  
  `ModelName[0]` 사용 후, 빈 리스트면 `"XGBoost"` 사용.
- **문자열로 오는 경우**: 그대로 사용.
- **None / 빈 문자열**: `"XGBoost"`로 fallback.
- **최종 비교**: `str(ModelName).strip().lower()` 한 값으로 위 4개 키와 매칭.

→ **태블로에서는 `_arg8`(또는 Feature 중요도 시 `_arg2`)에 [Select ML Model]을 그대로 넘기면 됩니다.**  
스크립트 안에서 `_arg8[0]`처럼 인덱싱하지 말 것.

---

## 4. 태블로 매개변수 설정 권장

| 항목 | 설정 |
|------|------|
| **매개변수 이름** | `Select ML Model` (다른 이름이어도 계산 필드에서 같은 필드를 8번째 인자로 넘기면 됨) |
| **데이터 유형** | 문자열 |
| **허용 가능한 값** | **목록(고정)** |
| **목록 값** | `XGBoost`, `RandomForest`, `LogisticRegression`, `SGDClassifier` (위 4개와 동일하게 권장) |
| **현재 값 / 통합 문서가 열릴 때** | 예: `XGBoost` |

---

## 5. 태블로 계산 필드에서 넘기는 방법

- **8번째 인자에 그대로 전달**:  
  `get_channel(..., _arg8)`, `get_channel_proba(..., _arg8)`  
  → 인자: `..., ATTR([Select ML Model])` 또는 `MAX([P_Region])` 다음에 `ATTR([Select ML Model])`.
- **주의**:  
  스크립트 안에서 `_arg8[0]`을 쓰면, Tableau가 문자열 하나만 보낼 때 `_arg8[0]`이 **첫 글자**만 되어 오류가 납니다. 반드시 **`_arg8`만** 넘기세요.

---

## 6. Python 쪽 파일 위치 (TabPy가 읽는 경로)

- **스크립트**: `tabpy_wholesale_cluster.py` (및 별칭 `tabpy_wholesale.py`)
- **모델/스칼라 파일** (같은 디렉터리에 있어야 함):  
  - wholesale_scaler.pkl  
  - wholesale_logistic.pkl  
  - wholesale_xgb.pkl  
  - wholesale_sgd.pkl  
  - wholesale_rf.pkl  

TabPy 서버를 **이 스크립트가 있는 폴더**에서 실행하거나, 해당 폴더가 Python 경로에 포함되어 있어야 합니다.

---

## 7. [통합 시뮬레이션 전략 리포트]의 CASE 문과 일치

리포트 하단 "현재 사용하는 Channel 예측 모델" 문구와 맞추려면, 매개변수 **목록 표시 값**을 아래와 같이 두면 됩니다.

- `LogisticRegression`
- `RandomForest`
- `SGDClassifier`
- `XGBoost`

Python은 대소문자를 구분하지 않으므로, 위 표기로 보내면 그대로 인식됩니다.
