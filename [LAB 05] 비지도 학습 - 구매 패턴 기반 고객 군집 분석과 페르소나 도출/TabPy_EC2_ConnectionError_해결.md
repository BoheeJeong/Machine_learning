# TabPy 배포 시 "Remote end closed connection without response" 해결

EC2에서 `python tabpy_wholesale_cluster.py` 실행 시 **`get_feature_importance`** 배포 단계에서 아래 오류가 나는 경우 참고하세요.

```
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

---

## 원인

에러가 나는 위치는 **TabPy 클라이언트가 `client.deploy("get_feature_importance", ...)` 를 POST로 보낸 직후**입니다.

- **"Remote end closed connection without response"** = 서버(TabPy)가 **응답을 보내기 전에** 연결을 끊었다는 뜻입니다.
- 즉, **TabPy 서버 쪽**에서 처리 중 문제가 생긴 상황입니다.

가능한 원인은 다음과 같습니다.

| 원인 | 설명 |
|------|------|
| **1. 메모리 부족 (OOM)** | TabPy가 배포 요청을 받아 함수를 **역직렬화(언피클)**할 때, `get_feature_importance` → `_get_feature_importance_dict` → `_load_channel_model` 등으로 이어지는 **모듈 의존성**(numpy, pandas, sklearn, **XGBoost** 등)이 함께 로드됩니다. EC2 인스턴스 메모리가 작으면 이 과정에서 프로세스가 죽으면서 연결이 끊깁니다. |
| **2. 타임아웃** | 배포 처리 시간이 길어져 클라이언트 또는 서버의 타임아웃에 걸려 연결이 끊깁니다. |
| **3. 요청 본문 크기 제한** | 직렬화된 페이로드가 커서 리버스 프록시·웹 서버의 body size 제한에 걸릴 수 있습니다. |

`get_cluster`, `get_channel`, `get_channel_proba`까지는 성공하고 **`get_feature_importance`에서만** 끊긴다면, 이 엔드포인트가 참조하는 **모델 로딩 경로**(XGBoost 등) 때문에 직렬화/역직렬화 시 메모리나 처리 시간이 가장 크게 나오는 경우가 많습니다.

---

## 해결 방법

### 1. TabPy 서버 로그 확인 (EC2)

TabPy를 실행한 터미널 또는 로그에서 **배포 요청 직후** 예외/메모리 관련 메시지가 있는지 봅니다.

```bash
# TabPy를 포그라운드로 실행해 두고, 다른 터미널에서 배포 스크립트 실행
tabpy
```

- `Killed` 또는 메모리 부족 메시지가 보이면 → **2번(메모리 증설)** 적용.
- 타임아웃 비슷한 메시지가 보이면 → **3번(타임아웃)** 적용.

### 2. EC2 메모리 증설 / Swap

- 인스턴스 타입을 **메모리 더 큰 것으로** 변경하거나,
- Swap을 추가해 OOM을 완화합니다.

```bash
# Swap 2GB 추가 (예시)
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

필요하면 `/etc/fstab`에 추가해 재부팅 후에도 사용하도록 설정할 수 있습니다.

### 3. TabPy / requests 타임아웃 늘리기

배포 스크립트에서 **타임아웃을 크게** 주어서, 서버가 느리게 처리해도 연결이 끊기지 않게 합니다.  
(아래는 `tabpy_wholesale_cluster.py`에 반영할 수 있는 방식입니다.)

- TabPy `Client`가 내부적으로 쓰는 `requests`의 타임아웃을 늘리려면, TabPy 쪽 설정이나 환경 변수를 확인하거나,
- **한 번에 하나씩** 엔드포인트만 배포해 보면서 `get_feature_importance` 직전/직후에서 끊기는지 확인합니다.

### 4. Feature Importance를 CSV로 두고 배포 부담 줄이기 (권장)

`get_feature_importance`가 **실행 시** 모델을 로드하지 않고 **CSV만 읽도록** 하면, TabPy 서버가 이 엔드포인트를 **호출할 때** 메모리 사용이 줄어듭니다.  
(배포 시 직렬화 크기는 비슷할 수 있지만, 서버가 나중에 이 엔드포인트를 로드/실행할 때 가벼워집니다.)

- `feature_importance_for_tableau.csv`를 한 번 생성해 두고,
- `tabpy_wholesale_cluster.py`의 `get_feature_importance`는 **해당 CSV가 있으면 CSV만 사용**하도록 되어 있습니다 (아래 구현 참고).

CSV 형식 예시:

- 파일명: `feature_importance_for_tableau.csv`
- 컬럼: `model_name`, `feature_name`, `importance`
- 모델별 8개 feature 한 행씩 (예: XGBoost, LogisticRegression 등).  
  `feature_name`은 `log_Fresh`, `log_Milk`, `log_Grocery`, `log_Frozen`, `log_Detergents_Paper`, `log_Delicassen`, `Region_2`, `Region_3` 순서/이름과 맞추면 됩니다.

이 CSV가 있으면 `get_feature_importance`는 **모델(pkl)을 로드하지 않고** CSV만 읽습니다.  
08번 노트북 등에서 `feature_importances_` / `coef_`를 DataFrame으로 만든 뒤 위 컬럼으로 저장하면 됩니다.

### 5. 배포 순서 조정

`get_feature_importance`를 **맨 마지막**에 배포하거나, 일시적으로 주석 처리한 뒤 나머지만 배포해 보세요.  
다른 엔드포인트만 쓰는 경우에는 TabPy를 먼저 안정적으로 띄우고, 메모리/타임아웃을 조정한 다음 `get_feature_importance`만 나중에 추가할 수 있습니다.

---

## 요약

| 조치 | 내용 |
|------|------|
| 원인 | TabPy 서버가 `get_feature_importance` 배포 처리 중 메모리 부족 또는 타임아웃으로 연결 종료 |
| 확인 | TabPy 로그에서 OOM/타임아웃 메시지 확인 |
| 대응 | EC2 메모리 증설 또는 Swap, 타임아웃 증가, Feature Importance CSV 사용, 배포 순서 조정 |

위 순서대로 적용해 보시고, 로그에 나오는 구체적인 메시지가 있으면 그에 맞춰 조정하면 됩니다.
