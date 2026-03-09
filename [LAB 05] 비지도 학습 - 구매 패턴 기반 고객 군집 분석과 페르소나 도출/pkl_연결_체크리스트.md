# pkl 연결 체크리스트 (Tableau에서 군집 예측이 되려면)

pkl이 “연결”된다 = **TabPy가 실행 중**이고, **Tableau가 TabPy를 통해** 우리 Python 스크립트를 부르고, 그 스크립트가 **같은 폴더의 wholesale_cluster_model.pkl**을 읽어서 예측하는 상태를 말합니다.

---

## 1. pkl 파일이 있는지 확인

- 아래 폴더에 **wholesale_cluster_model.pkl** 파일이 있어야 합니다.  
  (이 파일과 **tabpy_wholesale_cluster.py** 가 **같은 폴더**에 있어야 함.)

```
[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출
├── wholesale_cluster_model.pkl   ← 있어야 함
├── tabpy_wholesale_cluster.py
└── ...
```

- **없다면** 명령 프롬프트에서:
  ```bash
  cd "C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출"
  python save_wholesale_cluster_model.py
  ```
  실행 후 같은 폴더에 **wholesale_cluster_model.pkl** 이 생겼는지 확인.

---

## 2. TabPy 서버 실행 (매번 Tableau 쓸 때)

- **명령 프롬프트** 또는 **PowerShell**을 열고:

  ```bash
  cd "C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출"
  tabpy
  ```

- `TabPy server is listening on port 9004` 같은 메시지가 나오면 성공.
- **이 창은 닫지 말고** 둔 상태에서 Tableau 사용.

---

## 3. Tableau에서 TabPy 연결

- Tableau 메뉴: **연결** → **Analytics Extension에 연결**
- **TabPy** 선택
- **서버**: `localhost`  
- **포트**: `9004`
- **연결 테스트** 후 확인

---

## 4. 계산 필드 경로가 “Lab 05 폴더”를 가리키는지 확인

- **예측 군집** 계산 필드를 열어서 수식 확인.
- 안에 있는 `sys.path.append(r'...')` 의 `...` 부분이 **반드시** 아래 폴더 전체 경로와 같아야 합니다 (본인 PC에 맞게 수정).

  ```
  C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출
  ```

- Tableau 수식 예시 (경로만 위와 같이 맞추기):

  ```tableau
  SCRIPT_INT(
    "
    import sys
    sys.path.append(r'C:\Users\bohee\Documents\GitHub\Machine_learning\[LAB 05] 비지도 학습 - 구매 패턴 기반 고객 군집 분석과 페르소나 도출')
    from tabpy_wholesale_cluster import get_cluster
    return get_cluster(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6)
    ",
    [P_Fresh], [P_Milk], [P_Grocery], [P_Frozen], [P_Detergents_Paper], [P_Delicassen]
  )
  ```

- 이 경로가 맞아야 TabPy가 **tabpy_wholesale_cluster.py**를 찾고, 그 스크립트가 **같은 폴더의 wholesale_cluster_model.pkl**을 자동으로 읽습니다. → **이게 “pkl 연결”**입니다.

---

## 5. 연결 확인 방법

- 위 1~4를 모두 한 뒤, Tableau 시트에서 **예측 군집**을 열에 넣고, 6개 파라미터 슬라이더를 움직여 봅니다.
- **0, 1, 2, 3, 4** 중 하나의 숫자가 바뀌면 → pkl이 연결되어 예측이 되는 상태입니다.
- **빈칸**이거나 **에러**가 나면:
  - TabPy 창에 에러 메시지가 찍혀 있는지 확인.
  - `sys.path.append(...)` 경로에 `tabpy_wholesale_cluster.py` 와 `wholesale_cluster_model.pkl` 이 같이 있는지 다시 확인.

---

## 요약

| 단계 | 할 일 |
|------|--------|
| 1 | **wholesale_cluster_model.pkl** 이 Lab 05 폴더에 있는지 확인 (없으면 `save_wholesale_cluster_model.py` 실행) |
| 2 | **tabpy** 실행 (Lab 05 폴더에서, 창 유지) |
| 3 | Tableau **연결 → Analytics Extension → TabPy** (localhost:9004) |
| 4 | **예측 군집** 계산 필드 안 **sys.path.append** 경로 = Lab 05 폴더 전체 경로 |

이 네 가지가 맞으면 pkl이 연결된 상태로 군집 예측이 동작합니다.
