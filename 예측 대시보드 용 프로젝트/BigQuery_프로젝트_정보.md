# BigQuery 프로젝트 정보 (Cursor AI 참고용)

BigQuery 연동 시 Cursor가 더 정확히 도와주도록, **비밀번호/키 값은 적지 말고** 아래만 채워 두세요.

## GCP / BigQuery

| 항목 | 값 (예시만 넣고, 실제 값으로 수정) |
|------|-----------------------------------|
| GCP 프로젝트 ID | `my-gcp-project` |
| BigQuery 데이터셋 이름 | `olist_dashboard` |

## 인증 방식 (참고)

- **로컬**: `gcloud auth application-default login` 사용 중이면 별도 설정 없이 코드에서 기본 인증 사용 가능.
- **서비스 계정 키 파일** 사용 시: 키 파일 경로를 코드에 직접 넣지 말고, 환경 변수(예: `GOOGLE_APPLICATION_CREDENTIALS`)로 지정하는 방식 권장.

---

이 파일을 저장해 두면 BigQuery 관련 코드 제안 시 위 프로젝트 ID·데이터셋을 반영해 줄 수 있습니다.
