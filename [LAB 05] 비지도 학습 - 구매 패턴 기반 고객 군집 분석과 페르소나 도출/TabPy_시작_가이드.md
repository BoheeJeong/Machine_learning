# TabPy "No module named 'tabpy_wholesale' / 'tabpy_wholesale_cluster'" 해결

Tableau에서 다음 오류가 나면, TabPy가 **이 폴더([LAB 05])**를 Python 경로로 인식하지 못하는 상태입니다.

- **ModuleNotFoundError: No module named 'tabpy_wholesale'**
- **ModuleNotFoundError: No module named 'tabpy_wholesale_cluster'**

(태블로 계산 필드에서 `from tabpy_wholesale import ...` 를 쓰면 `tabpy_wholesale`, `from tabpy_wholesale_cluster import ...` 를 쓰면 `tabpy_wholesale_cluster` 가 필요합니다. 둘 다 **이 폴더**에 있어야 합니다.)

---

## 방법 1: 이 폴더에서 TabPy 실행 (권장)

TabPy를 **반드시** `tabpy_wholesale_cluster.py` 가 있는 **이 폴더**를 작업 디렉터리로 두고 실행하세요.

### PowerShell

```powershell
cd "C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출"
tabpy
```

### 명령 프롬프트(cmd)

```cmd
cd /d "C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출"
tabpy
```

- 경로에 **대괄호 `[ ]`** 가 있으므로 반드시 **큰따옴표**로 감싸세요.
- `tabpy` 가 실행된 **그 터미널 창을 닫지 말고** 두고, Tableau에서 분석 확장 프로그램 연결 후 사용하세요.

---

## 방법 2: Tableau 계산 필드에서 경로 추가

TabPy를 다른 폴더에서 실행 중이라면, 계산 필드 **맨 앞**에 아래 두 줄을 넣어 이 폴더를 경로에 추가할 수 있습니다.

**본인 PC의 실제 경로**로 바꿔서 사용하세요. (Python은 `/` 도 인식합니다.)

```
SCRIPT_REAL("
import sys
sys.path.append(r'C:/Users/bohee/Documents/GitHub/Machine_learning/[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
from tabpy_wholesale_cluster import get_feature_importance
return get_feature_importance(_arg1, _arg2)
",
ATTR([Feature (feature 2.csv)]),
[Select ML Model]
)
```

- `sys.path.append(...)` 안의 경로만 본인 **[LAB 05] 폴더 전체 경로**로 수정하면 됩니다.

---

## 확인 사항

| 확인 항목 | 내용 |
|-----------|------|
| TabPy 실행 위치 | `tabpy` 실행 시 터미널의 현재 폴더가 **[LAB 05] 폴더**인지 확인 |
| 필요한 파일 | 같은 폴더에 `tabpy_wholesale_cluster.py`, `wholesale_scaler.pkl`, `wholesale_xgb.pkl` 등 사용하는 pkl 존재 여부 |
| Tableau 연결 | 분석 → 분석 확장 프로그램 연결 → TabPy 서버 주소(예: localhost:9004) 연결 여부 |

방법 1으로 이 폴더에서 TabPy를 실행하면 대부분 `tabpy_wholesale_cluster` 오류가 사라집니다.

---

## 01~07 노트북 실행 시 pkl 파일과 TabPy/Tableau 연동

| 단계 | 설명 |
|------|------|
| **노트북 실행** | 01~07 노트북에서 **「5. Scaler / 모델 저장」** 셀까지 실행하면, **같은 [LAB 05] 폴더**에 `wholesale_scaler.pkl`, `wholesale_xgb.pkl` 등이 **덮어쓰기** 됩니다. |
| **TabPy가 읽는 위치** | `tabpy_wholesale_cluster.py`는 **이 폴더**의 pkl을 읽습니다. 디스크에 저장된 pkl과 동일한 파일을 사용합니다. |
| **Tableau** | Tableau는 pkl을 직접 읽지 않고, **TabPy 서버**를 통해 예측을 요청합니다. 즉, TabPy가 쓰는 모델 = Tableau가 쓰는 모델입니다. |

**중요:** TabPy는 **처음 호출될 때** pkl을 메모리에 올려두고 **캐시**합니다.  
그래서 **노트북으로 pkl을 새로 저장한 뒤에는 TabPy를 한 번 종료했다가 다시 실행**해야, 새 pkl이 반영됩니다.  
TabPy를 재시작하지 않으면 예전 모델이 그대로 사용됩니다.

---

## Predicted Probability / AUC-ROC가 모든 Customer ID에서 같은 값(예: 0.83090)으로 나올 때

**원인:** Tableau가 스크립트를 **전체 시트를 한 덩어리로** 한 번만 계산해서, 모든 마크에 같은 값이 붙는 경우입니다.

**해결:** 스크립트가 **고객(Customer) 단위로** 계산되도록 뷰의 **세부 수준(Level of Detail)**을 맞춰야 합니다.

1. **Customer ID**를 뷰에 반드시 넣기  
   - **[행]** 또는 **[열]**에 `[Customer ID]`를 드래그하거나  
   - **[마크] → [세부 정보]**에 `[Customer ID]`를 드래그합니다.
2. 그러면 Tableau가 **마크(행)마다** SCRIPT_REAL/SCRIPT_INT를 한 번씩 호출하고, 그 마크에 해당하는 Fresh, Milk, Region 등만 TabPy로 전달합니다.
3. 결과적으로 **고객별로 다른** 확률값·예측값이 나옵니다.

계산 필드 예시는 그대로 두고, **시트에서 Customer ID가 어디에 들어가는지**만 위처럼 맞추면 됩니다.
