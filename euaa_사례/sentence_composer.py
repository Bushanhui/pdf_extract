"""
EUAA 텍스트 조각을 의미적으로 완전한 문장으로 재구성하는 시스템

JSON 형식의 텍스트 조각들을 LLM을 통해 완전한 문장으로 구성합니다.
배치 처리 및 중복 제거를 통해 효율적으로 대량 데이터를 처리합니다.
"""

import os
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langfuse.decorators import observe

# 환경 변수 로드
load_dotenv()


class Logger:
    """로깅 시스템 관리 클래스"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 세션별 로그 파일
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = self.log_dir / f"sentence_compose_{timestamp}.log"

        # 로거 설정
        self.logger = logging.getLogger('SentenceComposer')
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러
        file_handler = logging.FileHandler(self.session_log, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 포맷터
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        """정보 메시지 로깅"""
        self.logger.info(message)

    def warning(self, message: str):
        """경고 메시지 로깅"""
        self.logger.warning(f"⚠️  {message}")

    def error(self, message: str):
        """오류 메시지 로깅"""
        self.logger.error(f"❌ {message}")

    def success(self, message: str):
        """성공 메시지 로깅"""
        self.logger.info(f"✓ {message}")


class FileProcessor:
    """sentences 폴더의 JSON 파일들을 스캔하고 관리하는 클래스"""

    def __init__(self, sentences_dir: str = "sentences"):
        self.sentences_dir = Path(sentences_dir)
        if not self.sentences_dir.exists():
            raise FileNotFoundError(f"sentences 폴더를 찾을 수 없습니다: {sentences_dir}")

    def scan_input_files(self) -> List[Dict[str, Any]]:
        """llm_input_*.json 파일들을 스캔하여 정보를 반환합니다."""
        pattern = "llm_input_*부_*.json"
        input_files = list(self.sentences_dir.glob(pattern))

        file_info = []
        for file_path in sorted(input_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                info = {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "object_count": len(data),
                    "file_size": file_path.stat().st_size,
                    "part_number": self._extract_part_number(file_path.name),
                    "language": self._extract_language(file_path.name),
                    "output_path": self._get_output_path(file_path.name)
                }
                file_info.append(info)

            except Exception as e:
                print(f"파일 스캔 실패: {file_path.name} - {e}")

        return file_info

    def _extract_part_number(self, filename: str) -> str:
        """파일명에서 부 번호 추출 (예: '1부', '2부')"""
        import re
        match = re.search(r'(\d+부)', filename)
        return match.group(1) if match else "알수없음"

    def _extract_language(self, filename: str) -> str:
        """파일명에서 언어 추출 (en 또는 kr)"""
        if '_en.json' in filename:
            return 'en'
        elif '_kr.json' in filename:
            return 'kr'
        return 'unknown'

    def _get_output_path(self, input_filename: str) -> str:
        """입력 파일명에서 출력 파일명 생성"""
        output_filename = input_filename.replace('llm_input_', 'llm_output_')
        return str(self.sentences_dir / output_filename)


class BatchProcessor:
    """JSON 배열을 겹침을 포함한 배치로 분할하는 클래스"""

    def __init__(self, batch_size: int = 250, overlap_size: int = 10):
        self.batch_size = batch_size
        self.overlap_size = overlap_size

    def split_into_batches(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        JSON 배열을 겹침을 포함한 배치로 분할합니다.

        Args:
            data: JSON 객체 배열

        Returns:
            배치 정보 리스트
        """
        if not data:
            return []

        batches = []
        total_objects = len(data)
        batch_num = 1
        start_idx = 0

        while start_idx < total_objects:
            # 배치 끝 인덱스 계산
            end_idx = min(start_idx + self.batch_size, total_objects)

            # 배치 데이터 추출
            batch_data = data[start_idx:end_idx]

            batch_info = {
                "batch_number": batch_num,
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "object_count": len(batch_data),
                "data": batch_data,
                "is_last_batch": end_idx >= total_objects
            }

            batches.append(batch_info)

            # 다음 배치 시작 인덱스 계산 (겹침 고려)
            if end_idx >= total_objects:
                break

            start_idx = end_idx - self.overlap_size
            batch_num += 1

        return batches


class APIKeyManager:
    """API 키 관리 클래스 (일일 할당량 추적 포함)"""

    def __init__(self):
        # API 키들 로드 (BACKUP1, BACKUP2, ... 형식)
        self.api_keys = []

        # 메인 키 추가
        primary_key = os.getenv("GOOGLE_API_KEY")
        if primary_key:
            self.api_keys.append(primary_key)

        # 백업 키들 추가 (GOOGLE_API_KEY_BACKUP_1, GOOGLE_API_KEY_BACKUP_2, ...)
        backup_index = 1
        while True:
            backup_key = os.getenv(f"GOOGLE_API_KEY_BACKUP_{backup_index}")
            if backup_key:
                self.api_keys.append(backup_key)
                backup_index += 1
            else:
                break

        if not self.api_keys:
            raise ValueError("API 키가 설정되지 않았습니다.")

        # API 키 상태 관리
        self.usage_file = "api_key_usage.json"
        self._load_key_usage()  # JSON에서 키 상태 로드
        self.current_key_index = self._get_available_key_index()
        self.current_api_key = self.api_keys[self.current_key_index]

        # genai 초기화
        genai.configure(api_key=self.current_api_key)

        # API 키 상태 출력
        print(f"✅ 설정된 API 키 개수: {len(self.api_keys)}개")
        for i, key in enumerate(self.api_keys):
            key_type = "메인" if i == 0 else f"백업{i}"
            key_id = f"***{key[-4:]}"
            key_info = self.key_usage.get(key_id, {'status': 'available'})
            status_emoji = "🟢" if key_info['status'] == 'available' else "🔴"
            current_mark = " ←현재선택" if i == self.current_key_index else ""
            print(f"  {key_type} API 키: {key_id} {status_emoji}{key_info['status']}{current_mark}")

    def _load_key_usage(self) -> None:
        """JSON 파일에서 API 키 사용 상태를 로드하고 24시간 경과 키는 자동으로 리셋"""
        try:
            if not Path(self.usage_file).exists():
                self.key_usage = self._initialize_key_usage()
                self._save_key_usage()
                print(f"📝 API 키 사용량 추적 파일 생성: {self.usage_file}")
                return

            with open(self.usage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.key_usage = data.get('keys', {})

            # 24시간 경과 키들을 자동으로 available로 리셋
            current_time = datetime.now()
            reset_count = 0

            for key_id, key_info in self.key_usage.items():
                if key_info.get('reset_time'):
                    reset_time = datetime.fromisoformat(key_info['reset_time'])
                    if current_time >= reset_time and key_info['status'] == 'exhausted':
                        key_info['status'] = 'available'
                        key_info['first_used'] = None
                        key_info['reset_time'] = None
                        reset_count += 1

            if reset_count > 0:
                print(f"🔄 {reset_count}개 API 키가 24시간 경과로 사용 가능 상태로 리셋됨")
                self._save_key_usage()

            print(f"✅ API 키 사용량 상태 로드 완료: {len(self.key_usage)}개 키")

        except Exception as e:
            print(f"⚠️ API 키 사용량 로드 실패, 초기화합니다: {e}")
            self.key_usage = self._initialize_key_usage()
            self._save_key_usage()

    def _initialize_key_usage(self) -> Dict[str, Dict[str, Any]]:
        """모든 API 키를 사용 가능 상태로 초기화"""
        usage = {}
        for api_key in self.api_keys:
            key_id = f"***{api_key[-4:]}"
            usage[key_id] = {
                'first_used': None,
                'status': 'available',
                'reset_time': None
            }
        return usage

    def _save_key_usage(self) -> None:
        """현재 API 키 사용 상태를 JSON 파일에 저장"""
        try:
            data = {
                'keys': self.key_usage,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ API 키 사용량 저장 실패: {e}")

    def _get_available_key_index(self) -> int:
        """사용 가능한 첫 번째 API 키의 인덱스를 반환"""
        for i, api_key in enumerate(self.api_keys):
            key_id = f"***{api_key[-4:]}"
            key_info = self.key_usage.get(key_id, {'status': 'available'})

            if key_info['status'] == 'available':
                print(f"🔑 사용 가능한 API 키 선택: {key_id} (인덱스 {i})")
                return i

        # 사용 가능한 키가 없으면 첫 번째 키 사용
        print(f"⚠️ 사용 가능한 키가 없어 메인 키부터 시작합니다")
        return 0

    def _mark_key_exhausted(self, key_index: int) -> None:
        """지정된 키를 할당량 소진 상태로 마킹하고 24시간 후 리셋 시간을 설정"""
        if key_index >= len(self.api_keys):
            return

        api_key = self.api_keys[key_index]
        key_id = f"***{api_key[-4:]}"
        current_time = datetime.now()

        if key_id not in self.key_usage:
            self.key_usage[key_id] = {'first_used': None, 'status': 'available', 'reset_time': None}

        # 첫 사용인 경우 시작 시간 기록
        if not self.key_usage[key_id]['first_used']:
            self.key_usage[key_id]['first_used'] = current_time.isoformat()

        # 상태를 exhausted로 변경하고 24시간 후 리셋 시간 설정
        self.key_usage[key_id]['status'] = 'exhausted'
        self.key_usage[key_id]['reset_time'] = (current_time + timedelta(hours=24)).isoformat()

        self._save_key_usage()

        reset_time_str = (current_time + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏰ API 키 {key_id} 할당량 소진으로 마킹, 리셋 시간: {reset_time_str}")

    def switch_to_next_key(self) -> bool:
        """다음 API 키로 전환 (현재 키를 exhausted로 마킹)"""
        # 현재 키를 할당량 소진 상태로 마킹
        self._mark_key_exhausted(self.current_key_index)

        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.current_api_key = self.api_keys[self.current_key_index]

            key_type = "메인" if self.current_key_index == 0 else f"백업 키 {self.current_key_index}"
            print(f"🔄 다음 API 키로 전환: {key_type} (***...{self.current_api_key[-4:]})")

            genai.configure(api_key=self.current_api_key)
            return True

        print(f"❌ 모든 API 키 ({len(self.api_keys)}개) 사용 완료")
        return False


class SentenceComposer:
    """LLM을 사용하여 텍스트 조각들을 완전한 문장으로 구성하는 클래스"""

    def __init__(self, logger: Logger):
        self.api_manager = APIKeyManager()
        self.logger = logger
        self.session_id = f"sentence_compose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 문장 구성 전용 프롬프트 (P 태그만 처리)
        self.compose_prompt = """당신은 기계 번역 학습 데이터셋을 만들기 위해, 줄바꿈으로 나뉜 텍스트 조각들을 의미론적으로 완전한 문장 단위로 재구성하는 AI 전문가입니다.

**# 입력 형식**
줄바꿈(`\n`)으로 구분된 텍스트 조각들이 주어집니다. 각 줄은 본문(P) 텍스트의 일부입니다.

**# 최종 목표**
입력된 텍스트 조각들을 합쳐서 의미상 완전한 문장들로 구성된 JSON 배열을 출력하는 것이 목표입니다.

**# 처리 규칙**

1. **문장 병합 규칙**:
   - 줄바꿈으로 나뉜 텍스트를 순서대로 읽으면서, 완전한 문장이 될 때까지 이어 붙입니다.
   - 완전한 문장 기준:
     * 마침표(.), 물음표(?), 느낌표(!) 등으로 끝나고 의미가 완결된 경우
     * 괄호 짝이 맞는 경우: 여는 괄호(`[`, `(`)가 있으면 닫는 괄호(`]`, `)`)까지 이어 붙여야 함
   - 이미 완전한 문장인 경우 그대로 출력 배열에 추가합니다.

2. **문장 분리 규칙**:
   - 하나의 줄에 여러 개의 완전한 문장이 있으면, 각각을 별개의 JSON 객체로 분리합니다.
   - 각 JSON 객체는 반드시 하나의 완전한 문장만 포함해야 합니다.

3. **텍스트 정리**:
   - 불필요한 공백 제거 (여러 개의 공백은 하나로)
   - 괄호 앞뒤 공백 정리: `( text )` → `(text)`

**# 출력 형식**
JSON 배열로 출력하며, 각 객체는 다음 스키마를 따릅니다:
```
{"text": "완전한 문장", "source_type": "P"}
```
- `text`: 의미적으로 완전한 하나의 문장
- `source_type`: 항상 "P" (본문)

**# 예시**

입력 텍스트:
```
보호자 미동반 아동
룩셈부르크 행정법원은 A 와 S 에 대한 유럽연합사법재판소 (CJEU) 판결 ( 제 C-550/16 호 ) 을 고려하여 미성년자가
보호자 미동반 아동으로 간주되는 조건을 분석하였다 .
```

출력 JSON:
```json
[
  {"text": "보호자 미동반 아동", "source_type": "P"},
  {"text": "룩셈부르크 행정법원은 A와 S에 대한 유럽연합사법재판소(CJEU) 판결(제C-550/16호)을 고려하여 미성년자가 보호자 미동반 아동으로 간주되는 조건을 분석하였다.", "source_type": "P"}
]
```

**반드시 JSON 배열만 출력하고, 다른 설명이나 텍스트는 포함하지 마세요.**"""

    def split_into_sections(self, input_data: List[Dict]) -> List[Dict[str, Any]]:
        """입력 데이터를 H1/H2/H3 기준으로 섹션으로 분할

        Args:
            input_data: JSON 객체 배열 (id, text, source_type 포함)

        Returns:
            섹션 리스트. 각 섹션은 {'headers': [H1/H2/H3 항목들], 'p_items': [P 항목들]} 형태
        """
        if not input_data:
            return []

        sections = []
        current_section = {'headers': [], 'p_items': []}

        for item in input_data:
            if item.get('source_type') in ['H1', 'H2', 'H3']:
                # 이전 섹션이 있으면 저장
                if current_section['p_items'] or current_section['headers']:
                    sections.append(current_section)
                # 새 섹션 시작
                current_section = {
                    'headers': [item],
                    'p_items': []
                }
            else:  # P
                current_section['p_items'].append(item)

        # 마지막 섹션 저장
        if current_section['p_items'] or current_section['headers']:
            sections.append(current_section)

        return sections

    def process_section_p_batch(self, p_items: List[Dict], section_number: int, part_info: str, batch_size: int = 100) -> List[Dict]:
        """섹션 내 P 항목들을 배치 단위로 LLM 처리

        Args:
            p_items: 처리할 P 항목 리스트
            section_number: 섹션 번호 (로깅용)
            part_info: 파트 정보 (예: "1부_kr")
            batch_size: 배치 크기 (기본값 100)

        Returns:
            LLM 처리된 P 항목 리스트
        """
        if not p_items:
            return []

        all_composed_p = []
        total_p_count = len(p_items)
        batch_num = 1
        start_idx = 0

        self.logger.info(f"  섹션 {section_number}: {total_p_count}개 P 항목을 {batch_size}개씩 배치 처리")

        while start_idx < total_p_count:
            # 배치 끝 인덱스 계산
            end_idx = min(start_idx + batch_size, total_p_count)
            batch_data = p_items[start_idx:end_idx]

            self.logger.info(f"    배치 {batch_num} 처리 중... ({len(batch_data)}개 P 항목)")

            # LLM 호출
            result = self.process_batch(batch_data, batch_num, part_info)

            if result['success']:
                all_composed_p.extend(result['composed_data'])
                self.logger.info(f"    ✓ 배치 {batch_num} 완료: {len(batch_data)}개 → {len(result['composed_data'])}개")
            else:
                self.logger.error(f"    ✗ 배치 {batch_num} 실패: {result.get('error', '알 수 없는 오류')}")
                # 실패한 경우 원본 데이터를 그대로 추가
                all_composed_p.extend(batch_data)

            start_idx = end_idx
            batch_num += 1

        return all_composed_p

    def process_file(self, input_data: List[Dict], part_info: str, batch_size: int = 100) -> Tuple[List[Dict], Dict[str, Any], List[Dict]]:
        """파일 전체를 섹션별로 처리

        Args:
            input_data: 입력 JSON 데이터 (전체 파일)
            part_info: 파트 정보 (예: "1부_kr")
            batch_size: P 배치 처리 크기 (기본값 100)

        Returns:
            (최종 결과 리스트, 메타데이터 딕셔너리, H1/H2/H3 헤더 리스트)
        """
        # 1. 섹션으로 분할
        self.logger.info(f"섹션 분할 중...")
        sections = self.split_into_sections(input_data)
        self.logger.info(f"총 {len(sections)}개 섹션으로 분할됨")

        # 2. 각 섹션 처리
        final_results = []
        all_headers = []  # H1/H2/H3 헤더만 별도로 수집
        total_h1_count = 0
        total_h2_count = 0
        total_h3_count = 0
        total_input_p_count = 0
        total_output_p_count = 0

        for idx, section in enumerate(sections, 1):
            self.logger.info(f"\n섹션 {idx}/{len(sections)} 처리 중...")

            # H1/H2/H3 직접 추가 (LLM 처리 없음)
            headers = section.get('headers', [])
            for header in headers:
                final_results.append(header)
                all_headers.append(header)  # anchor용 별도 수집
                if header.get('source_type') == 'H1':
                    total_h1_count += 1
                elif header.get('source_type') == 'H2':
                    total_h2_count += 1
                elif header.get('source_type') == 'H3':
                    total_h3_count += 1

            if headers:
                self.logger.info(f"  H1/H2/H3 헤더: {len(headers)}개 직접 추가 (LLM 처리 안 함)")

            # P 항목 처리
            p_items = section.get('p_items', [])
            if p_items:
                total_input_p_count += len(p_items)

                # P가 1개만 있으면 LLM 처리 없이 직접 추가
                if len(p_items) == 1:
                    final_results.extend(p_items)
                    total_output_p_count += 1
                    self.logger.info(f"  P 항목: 1개 직접 추가 (LLM 처리 안 함)")
                else:
                    # P가 2개 이상이면 배치 처리
                    composed_p = self.process_section_p_batch(p_items, idx, part_info, batch_size)
                    final_results.extend(composed_p)
                    total_output_p_count += len(composed_p)

        # 3. 메타데이터 생성
        metadata = {
            "total_input_count": len(input_data),
            "total_sections": len(sections),
            "h1_count": total_h1_count,
            "h2_count": total_h2_count,
            "h3_count": total_h3_count,
            "input_p_count": total_input_p_count,
            "output_p_count": total_output_p_count,
            "overall_compression_ratio": total_input_p_count / total_output_p_count if total_output_p_count > 0 else 0
        }

        self.logger.info(f"\n처리 완료!")
        self.logger.info(f"  전체 입력: {len(input_data):,}개")
        self.logger.info(f"  전체 출력: {len(final_results):,}개")
        self.logger.info(f"  H1 헤더: {total_h1_count}개 (100% 보존)")
        self.logger.info(f"  H2 헤더: {total_h2_count}개 (100% 보존)")
        self.logger.info(f"  H3 헤더: {total_h3_count}개 (100% 보존)")
        self.logger.info(f"  P 압축률: {metadata['overall_compression_ratio']:.1f}배 ({total_input_p_count} → {total_output_p_count})")

        return final_results, metadata, all_headers

    def process_batch(self, batch_data: List[Dict], batch_number: int, part_info: str) -> Dict[str, Any]:
        """
        배치 데이터를 LLM으로 처리하여 문장을 구성합니다.

        Args:
            batch_data: 처리할 배치 데이터 (P 항목들)
            batch_number: 배치 번호
            part_info: 파트 정보 (예: "1부_kr")

        Returns:
            처리 결과
        """
        try:
            self.logger.info(f"배치 {batch_number} 처리 시작 ({len(batch_data)}개 P 항목)")

            # P 항목들의 text를 줄바꿈으로 이어붙이기
            input_text = "\n".join([item['text'] for item in batch_data])

            # LLM 호출
            start_time = time.time()
            response = self._call_llm_simple(input_text)

            if not response:
                self.logger.error(f"배치 {batch_number}: LLM 응답을 받지 못했습니다")
                return {
                    "success": False,
                    "error": "LLM 응답을 받지 못했습니다",
                    "batch_number": batch_number
                }

            # 응답에서 JSON 추출 시도
            composed_data = self._extract_json_from_response(response)

            if not composed_data:
                self.logger.error(f"배치 {batch_number}: 유효한 JSON을 추출할 수 없습니다")
                return {
                    "success": False,
                    "error": "LLM 응답에서 유효한 JSON을 추출할 수 없습니다",
                    "batch_number": batch_number,
                    "raw_response": response
                }

            processing_time = time.time() - start_time

            # 압축률 계산
            compression_ratio = len(batch_data) / len(composed_data) if len(composed_data) > 0 else 0

            # 로그 출력
            self.logger.info(f"├─ 압축률: {compression_ratio:.1f}배 ({len(batch_data)}개 P → {len(composed_data)}개 문장)")
            self.logger.info(f"└─ 처리시간: {processing_time:.1f}초")

            return {
                "success": True,
                "batch_number": batch_number,
                "input_count": len(batch_data),
                "output_count": len(composed_data),
                "compression_ratio": compression_ratio,
                "composed_data": composed_data,
                "processing_time": processing_time
            }

        except Exception as e:
            self.logger.error(f"배치 {batch_number} 처리 중 예외 발생: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "batch_number": batch_number
            }

    @observe(name="gemini_sentence_composition")
    def _call_llm_simple(self, input_text: str) -> Optional[str]:
        """간단한 LLM 호출 (텍스트 입력받아 JSON 응답 반환)"""
        from langfuse.decorators import langfuse_context

        max_retries = len(self.api_manager.api_keys)

        # JSON Schema 정의
        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "구성된 완전한 문장"
                    },
                    "source_type": {
                        "type": "string",
                        "description": "원본 출처 유형: 항상 P"
                    }
                },
                "required": ["text", "source_type"]
            }
        }

        # 전체 프롬프트 구성
        full_prompt = f"{self.compose_prompt}\n\n입력 텍스트:\n{input_text}"

        # Langfuse에 프롬프트와 입력 메타데이터 기록
        langfuse_context.update_current_observation(
            input=full_prompt,
            metadata={
                "model": "gemini-2.5-flash",
                "temperature": 0.7,
                "max_output_tokens": 65536,
                "input_text_length": len(input_text),
                "input_line_count": input_text.count('\n') + 1
            }
        )

        for attempt in range(max_retries):
            try:
                llm_start_time = time.time()

                # 모델 초기화 (structured output 및 temperature 0.7 설정)
                model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": response_schema,
                        "temperature": 0.7,
                        "max_output_tokens": 65536
                    }
                )

                # LLM 호출
                response = model.generate_content(full_prompt)

                llm_duration = time.time() - llm_start_time

                if response and response.text:
                    # Langfuse에 응답과 소요시간 기록
                    langfuse_context.update_current_observation(
                        output=response.text,
                        metadata={
                            "llm_duration_seconds": round(llm_duration, 2),
                            "output_length": len(response.text),
                            "attempt": attempt + 1,
                            "api_key_index": self.api_manager.current_key_index
                        }
                    )
                    return response.text

                return None

            except Exception as e:
                error_message = str(e)
                llm_duration = time.time() - llm_start_time if 'llm_start_time' in locals() else 0

                # Langfuse에 에러 기록
                langfuse_context.update_current_observation(
                    metadata={
                        "error": error_message,
                        "attempt": attempt + 1,
                        "llm_duration_seconds": round(llm_duration, 2)
                    }
                )

                self.logger.warning(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {error_message}")

                # 할당량 초과 감지
                if self._is_daily_quota_exceeded(e):
                    self.logger.warning(f"📊 일별 할당량 초과 감지")
                    # 다음 API 키로 전환
                    if attempt < max_retries - 1 and self.api_manager.switch_to_next_key():
                        self.logger.info(f"다음 API 키로 전환하여 재시도...")
                        continue
                else:
                    # 다음 API 키로 전환 (일반 에러)
                    if attempt < max_retries - 1 and self.api_manager.switch_to_next_key():
                        self.logger.info(f"다음 API 키로 전환하여 재시도...")
                        continue

                # 모든 키 시도 완료
                if attempt == max_retries - 1:
                    self.logger.error(f"모든 API 키 시도 완료. 마지막 오류: {error_message}")
                    return None

        return None

    def _is_daily_quota_exceeded(self, error: Exception) -> bool:
        """일별 할당량 초과 에러인지 확인"""
        error_str = str(error).lower()
        quota_keywords = [
            "quota exceeded",
            "daily limit exceeded",
            "daily quota exceeded",
            "exceeded your current quota",
            "daily usage limit exceeded",
            "requests per day exceeded",
            "current quota"
        ]
        return any(keyword in error_str for keyword in quota_keywords)

    def _extract_json_from_response(self, response: str) -> Optional[List[Dict]]:
        """LLM 응답에서 JSON 배열을 추출합니다."""
        try:
            # 응답이 이미 JSON 배열인 경우
            if response.strip().startswith('['):
                return json.loads(response.strip())

            # 마크다운 코드 블록에서 JSON 추출
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # JSON 배열 패턴 직접 추출
            json_match = re.search(r'(\[[\s\S]*\])', response)
            if json_match:
                return json.loads(json_match.group(1))

            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON 파싱 오류: {e}")
            return None


def save_results_with_metadata(output_path: str, data: List[Dict], metadata: Dict[str, Any]) -> None:
    """결과 데이터를 메타데이터와 함께 JSON 파일로 저장합니다."""
    result = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_input_count": metadata.get("total_input_count", 0),
            "total_output_count": len(data),
            "overall_compression_ratio": metadata.get("overall_compression_ratio", 0),
            "successful_batches": metadata.get("successful_batches", 0),
            "total_batches": metadata.get("total_batches", 0),
            "average_processing_time": metadata.get("average_processing_time", 0),
            "processing_summary": metadata.get("processing_summary", {})
        },
        "data": data
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    """메인 실행 함수"""
    try:
        # 로거 초기화
        logger = Logger()
        logger.info("=== EUAA 문장 구성 시스템 시작 ===")

        # 파일 스캐너 초기화
        file_processor = FileProcessor()
        sentence_composer = SentenceComposer(logger)

        # 입력 파일들 스캔
        logger.info("입력 파일 스캔 중...")
        input_files = file_processor.scan_input_files()

        if not input_files:
            logger.error("처리할 입력 파일이 없습니다.")
            return

        logger.info(f"발견된 파일: {len(input_files)}개")
        for file_info in input_files:
            logger.info(f"  - {file_info['filename']}: {file_info['object_count']:,}개 객체")

        # 각 파일별로 처리
        for file_info in input_files:
            logger.info(f"\n처리 시작: {file_info['filename']}")

            # 입력 데이터 로드
            with open(file_info['file_path'], 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            part_info = f"{file_info['part_number']}_{file_info['language']}"

            # 배치 크기 설정 (H1/H2/H3가 많아서 섹션당 P가 200개 넘지 않음)
            batch_size = 200
            logger.info(f"언어: {file_info['language']}, 배치 크기: {batch_size}")

            # 섹션 기반 처리 (H1/H2/H3는 직접 추가, P만 LLM 처리)
            final_results, metadata, all_headers = sentence_composer.process_file(
                input_data,
                part_info,
                batch_size=batch_size
            )

            # 메타데이터 확장 (save_results_with_metadata에서 필요한 필드 추가)
            extended_metadata = {
                "total_input_count": metadata["total_input_count"],
                "overall_compression_ratio": metadata["overall_compression_ratio"],
                "successful_batches": 0,  # 섹션 기반 처리에서는 의미 없음
                "total_batches": 0,  # 섹션 기반 처리에서는 의미 없음
                "average_processing_time": 0,  # 섹션 기반 처리에서는 의미 없음
                "processing_summary": {
                    "total_sections": metadata["total_sections"],
                    "h1_count": metadata["h1_count"],
                    "h2_count": metadata["h2_count"],
                    "h3_count": metadata["h3_count"],
                    "input_p_count": metadata["input_p_count"],
                    "output_p_count": metadata["output_p_count"]
                }
            }

            # 결과 저장 (메타데이터 포함)
            output_path = file_info['output_path']
            save_results_with_metadata(output_path, final_results, extended_metadata)

            logger.success(f"저장 완료: {output_path}")
            logger.info(f"   원본: {len(input_data):,}개 → 결과: {len(final_results):,}개")
            logger.info(f"   P 압축률: {metadata['overall_compression_ratio']:.1f}배")
            logger.info(f"   H1 보존: {metadata['h1_count']}개 (100%)")
            logger.info(f"   H2 보존: {metadata['h2_count']}개 (100%)")
            logger.info(f"   H3 보존: {metadata['h3_count']}개 (100%)")

            # H1/H2/H3 anchor 파일 저장
            anchor_filename = file_info['filename'].replace('llm_input_', 'anchors_')
            anchor_path = str(file_processor.sentences_dir / anchor_filename)
            with open(anchor_path, 'w', encoding='utf-8') as f:
                json.dump(all_headers, f, ensure_ascii=False, indent=2)

            logger.success(f"Anchor 파일 저장: {anchor_filename}")
            logger.info(f"   총 {len(all_headers)}개 헤더 (H1: {metadata['h1_count']}, H2: {metadata['h2_count']}, H3: {metadata['h3_count']})")

        logger.success("모든 파일 처리 완료")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"오류 발생: {e}")
        else:
            print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()