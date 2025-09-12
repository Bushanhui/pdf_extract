# PDF 텍스트 추출 및 병렬 코퍼스 구축 시스템

## 1. 개요

이 시스템은 PDF 파일에서 한국어 또는 영어 텍스트를 추출하여 병렬 코퍼스(Parallel Corpus)를 구축하기 위한 도구입니다. Google의 Generative AI (Gemini 2.5 Flash) 모델을 사용하여 PDF 내의 텍스트를 문장 단위로 정확하게 추출하고, 이를 SQLite 데이터베이스에 저장합니다.

대용량 PDF 파일을 효율적으로 처리하기 위한 배치 처리 기능과 세션 관리, 재시도 로직, 상세한 로깅 및 통계 기능을 포함하고 있습니다.

## 2. 시스템 구성

이 시스템은 다음과 같은 모듈로 구성되어 있습니다.

-   `main.py`: 핵심 로직을 포함하는 메인 모듈입니다. PDF 처리, AI 모델 호출, 데이터베이스 저장을 총괄합니다.
-   `cli.py`: 사용자가 시스템과 상호작용할 수 있는 커맨드 라인 인터페이스(CLI)를 제공합니다.
-   `database.py`: SQLite 데이터베이스의 스키마 정의와 데이터 CRUD(생성, 읽기, 업데이트, 삭제) 작업을 관리합니다.
-   `prompt.py`: Gemini AI 모델에 전송할 프롬프트를 관리합니다. 한국어와 영어 텍스트 추출을 위한 규칙이 정의되어 있습니다.
-   `session_manager.py`: 대용량 PDF를 위한 배치 처리 세션을 생성하고 관리합니다. PDF 분할, 진행 상황 추적, 재시도 로직을 담당합니다.
-   `utils.py`: PDF 분할, JSON 백업, 파일 크기 포맷팅 등 시스템 전반에서 사용되는 헬퍼 함수들을 포함합니다.

## 3. 주요 기능

-   **PDF 텍스트 추출**: 한국어 또는 영어로 작성된 PDF 파일에서 텍스트를 추출합니다.
-   **AI 기반 문장 분리**: Gemini 모델을 사용하여 텍스트를 문법적으로 정확한 문장 단위로 분리하고 JSON 형식으로 반환합니다.
-   **텍스트 출처 유형 분류**: 추출된 각 문장의 출처를 `table`(표), `image`(이미지), `text`(일반 텍스트)로 자동 분류하여 기록합니다.
-   **병렬 코퍼스 DB 구축**: 추출된 한국어와 영어 문장을 출처 유형과 함께 별도의 테이블에 저장하여 병렬 코퍼스를 구축합니다.
-   **배치 처리**: 대용량 PDF 파일을 지정된 페이지 크기(배치)로 나누어 순차적으로 처리하여 메모리 문제를 방지하고 안정성을 높입니다.
-   **세션 관리 및 복구**: 각 배치 처리 작업을 세션으로 관리하여 중단 시에도 해당 세션부터 이어서 처리하거나 실패한 배치만 재시도할 수 있습니다.
-   **스마트 재시도 시스템**: 실패한 배치들을 자동으로 감지하고 언어별로 구분하여 재처리합니다. 전체 재시도, 세션별 재시도, 개별 배치 재시도 등 다양한 재시도 옵션을 제공합니다.
-   **고급 API 키 관리**: 메인 키와 여러 백업 키를 자동 순환하며, 할당량 초과나 파일 접근 오류 시 즉시 다음 키로 전환합니다. 구체적인 에러 분류와 상세한 로깅을 통해 디버깅을 지원합니다.
-   **지능형 파일 재업로드**: API 키 전환 시 Google AI에 업로드된 파일을 새 키로 자동 재업로드하여 파일 접근 권한 문제를 해결합니다.
-   **상세 로깅 및 통계**: `Langfuse`를 연동하여 AI 모델의 성능을 추적하고, 각 배치 및 페이지별 문장 추출 통계를 데이터베이스에 기록합니다.
-   **유연한 CLI**: 단일 파일 처리, 폴더 일괄 처리, 실패 배치 재시도, 데이터베이스 상태 조회 등 다양한 기능을 CLI 명령어로 제공합니다.

## 4. API 키 관리 및 에러 처리

### 4.1 다중 API 키 설정
시스템은 메인 키와 여러 백업 키를 지원하여 안정적인 대용량 처리를 보장합니다:

```bash
# 환경변수 설정 예시
# 최소 요구사항: 메인 키 1개만 있으면 동작
GOOGLE_API_KEY="your_main_key"

# 권장 설정: 백업 키들 추가 (할당량 초과 시 자동 전환)
GOOGLE_API_KEY_BACKUP_1="your_backup_key_1"  # 선택사항
GOOGLE_API_KEY_BACKUP_2="your_backup_key_2"  # 선택사항
GOOGLE_API_KEY_BACKUP_3="your_backup_key_3"  # 선택사항
```

### 4.2 지능형 에러 처리
시스템은 다음과 같은 에러를 자동으로 분류하고 적절히 대응합니다:

-   **할당량 초과 에러**: 일별 API 사용량 초과 시 다음 키로 자동 전환
-   **파일 접근 권한 에러**: API 키 변경으로 인한 파일 접근 불가 시 파일 재업로드
-   **API 키 인증 에러**: 잘못된 키 감지 시 다음 키로 즉시 전환
-   **기타 API 오류**: 일시적 네트워크 오류 등에 대한 재시도

### 4.3 파일 재업로드 메커니즘
Google AI에서 업로드된 파일은 특정 API 키에 바인딩됩니다. API 키 전환 시:
1. 기존 업로드 파일 삭제 시도
2. 새 API 키로 파일 재업로드
3. 재업로드 실패 시 다음 키로 계속 시도
4. 모든 과정이 자동화되어 사용자 개입 불필요

## 5. 처리 프로세스

1.  **사용자 입력**: 사용자가 `cli.py`를 통해 처리할 PDF 파일(또는 폴더)과 언어 등 옵션을 지정합니다.
2.  **세션 생성 (배치 모드)**: `SessionManager`가 PDF의 페이지 수를 확인하고, 지정된 배치 크기에 따라 처리 세션과 하위 배치 정보를 `database.py`를 통해 DB에 생성합니다.
3.  **PDF 분할 (배치 모드)**: `utils.py`의 `PDFSplitter`가 원본 PDF를 여러 개의 작은 임시 PDF 파일로 분할합니다.
4.  **AI 모델 호출**: `main.py`가 각 PDF(또는 배치 PDF)를 `prompt.py`에 정의된 프롬프트와 함께 Gemini AI 모델에 전송합니다.
5.  **텍스트 추출 및 파싱**: AI 모델이 PDF 내용을 분석하여 문장 단위로 추출된 텍스트와 각 문장의 출처 유형(`table`, `image`, `text`)을 JSON 형식으로 반환합니다. `main.py`는 이 JSON 응답을 파싱하여 문장 리스트와 출처 정보를 얻습니다.
6.  **데이터베이스 저장**: 파싱된 문장 리스트를 출처 유형 정보와 함께 `database.py`를 통해 SQLite 데이터베이스의 해당 언어 테이블에 저장합니다. 배치 처리 시 세션 및 배치 정보도 함께 기록됩니다.
7.  **결과 출력**: 처리 완료 후, 추출된 문장 수와 저장된 데이터베이스 정보를 사용자에게 보여줍니다.

## 6. 안전 설정 및 모델 구성

### 6.1 Safety Settings
규제 및 기술 문서 처리를 위해 모든 안전 필터를 비활성화:
```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}
```

### 6.2 모델 구성
-   **모델**: Gemini 2.5 Flash (빠른 처리 속도와 정확성)
-   **Temperature**: 한국어 0.5, 영어 0.6 (일관성과 자연스러움 균형)
-   **출력 형식**: JSON Schema를 통한 구조화된 응답 보장
-   **배치 크기**: 10페이지씩 분할 (메모리 효율성과 처리 속도 최적화)

## 7. 데이터베이스 스키마 (`corpus.db`)

-   **`korean_sentences`**: 추출된 한국어 문장을 저장합니다.
    -   `id`: Primary Key
    -   `sentence`: 추출된 문장
    -   `sentence_index`: 문장 순서
    -   `batch_id`: 배치 세션 ID
    -   `batch_number`: 배치 번호
    -   `pdf_file_path`: PDF 파일 경로
    -   `original_filename`: 원본 PDF 파일명
    -   `source_type`: 텍스트 출처 유형 (`table`, `image`, `text`)
    -   `page_number`: 페이지 번호
    -   `created_at`: 생성 시각
-   **`english_sentences`**: 추출된 영어 문장을 저장합니다. (스키마는 `korean_sentences`와 동일)
-   **`processing_sessions`**: 배치 처리 세션 정보를 관리합니다.
    -   `id`: 세션 ID (Primary Key)
    -   `source_file`: 원본 PDF 파일 경로
    -   `language`: 처리 언어
    -   `total_pages`, `batch_size`, `total_batches`: 페이징 정보
    -   `status`: 세션 상태 (`in_progress`, `completed`, `failed`)
-   **`batch_progress`**: 각 배치의 진행 상황을 추적하고 재시도 기능을 지원합니다.
    -   `session_id`: 세션 ID
    -   `batch_number`: 배치 번호
    -   `start_page`, `end_page`: 배치의 페이지 범위
    -   `status`: 배치 상태 (`pending`, `ready`, `in_progress`, `completed`, `failed`)
    -   `pdf_file_path`: 분할된 임시 PDF 파일 경로
    -   `json_backup_path`: AI 응답 백업 파일 경로
    -   `error_message`: 실패 시 오류 메시지 (재시도 참고용)
    -   `sentences_extracted`: 추출된 문장 수
-   **`extraction_stats`**: 페이지별 추출 통계를 저장합니다.
-   **`batch_summary_stats`**: 배치별 요약 통계를 저장합니다.

## 8. 설치 및 환경 설정

### 8.1 Conda 환경 생성
```bash
# Python 3.11 conda 환경 생성 및 활성화
conda create -n hanhwa python=3.11 -y
conda activate hanhwa
```

### 8.2 의존성 설치
```bash
# requirements.txt를 이용한 패키지 설치
pip install -r requirements.txt
```

### 8.3 환경 변수 설정
```bash
# .env 파일 생성 (프로젝트 루트에 생성)
GOOGLE_API_KEY="your_main_key"
GOOGLE_API_KEY_BACKUP_1="your_backup_key_1"  # 선택사항
GOOGLE_API_KEY_BACKUP_2="your_backup_key_2"  # 선택사항

# Langfuse 설정 (선택사항)
LANGFUSE_SECRET_KEY="your_secret_key"
LANGFUSE_PUBLIC_KEY="your_public_key"
LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

### 8.4 실행 방법
Conda 환경을 활성화한 후 Python으로 직접 실행:
```bash
conda activate hanhwa
python cli.py --help
```

## 9. 주요 CLI 명령어

-   **단일 PDF 파일 처리**:
    ```bash
    python cli.py <PDF_경로> --language [korean|english]
    ```
-   **폴더 내 모든 PDF 일괄 처리**:
    -   파일명에 `_kr.pdf` 또는 `_en.pdf`가 포함되어 있어야 언어가 자동 감지됩니다.
    ```bash
    python cli.py --folder <폴더_경로>
    ```
-   **대용량 PDF 배치 처리 모드**:
    ```bash
    python cli.py <PDF_경로> --language [korean|english] --batch-processing
    ```
-   **데이터베이스 문장 수 확인**:
    ```bash
    python cli.py --count
    ```
-   **실패한 배치 재시도**:
    ```bash
    # 모든 실패한 배치 자동 재시도 (언어별 자동 구분)
    python cli.py --retry-failed-all
    
    # 특정 언어의 실패한 배치만 재시도
    python cli.py --retry-failed-all --language korean
    
    # 특정 세션의 실패한 배치들만 재시도
    python cli.py --retry-session <세션_ID> --failed-only
    
    # 특정 배치 번호만 재시도 (휴먼 평가 후 재처리)
    python cli.py --retry-session <세션_ID> --retry-batch <배치_번호>
    ```
-   **상세 정보 출력**:
    -   모든 명령어에 `--verbose` 또는 `-v` 옵션을 추가하여 더 자세한 처리 과정을 볼 수 있습니다.
    ```bash
    python cli.py <PDF_경로> --language korean --verbose
    ```

## 10. 시스템 요구사항 및 제한사항

### 10.1 파일 크기 제한
-   **개별 배치**: 최대 20MB (Google AI 업로드 제한)
-   **전체 PDF**: 제한 없음 (배치 처리로 자동 분할)
-   **권장 크기**: 배치당 1-3MB (최적 성능)

### 10.2 언어 감지
-   **파일명 기반**: `_kr.pdf` (한국어), `_en.pdf` (영어)
-   **수동 지정**: `--language` 옵션으로 명시적 지정 가능
-   **폴더 처리**: 파일명 패턴에 따라 자동 언어 구분

### 10.3 성능 특성
-   **처리 속도**: 페이지당 약 2-5초 (배치 크기와 내용 복잡도에 따라 변동)
-   **메모리 사용량**: 배치 처리로 대용량 파일도 저메모리 처리
-   **동시 처리**: API 키별 순차 처리 (안정성 우선)

## 11. 문제 해결 가이드

### 11.1 일반적인 오류
-   **403 파일 접근 오류**: API 키 전환 시 자동 해결 (파일 재업로드)
-   **할당량 초과**: 백업 키로 자동 전환, 모든 키 소진 시 24시간 대기
-   **JSON 파싱 오류**: 백업 파일에서 수동 복구 가능

### 11.2 로그 확인
-   **실시간 로그**: 터미널 출력으로 진행 상황 모니터링
-   **파일 로그**: `pdf_processing_YYYYMMDD.log` 파일에 상세 기록
-   **Langfuse**: 웹 인터페이스를 통한 AI 모델 성능 추적

---

**마지막 업데이트**: 2025년 8월 20일  
**버전**: 스마트 API 키 관리 및 에러 처리 시스템 통합
