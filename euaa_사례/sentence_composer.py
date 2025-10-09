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
from datetime import datetime
import time
import google.generativeai as genai
from dotenv import load_dotenv

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
    """API 키 관리 클래스 (main.py에서 핵심 로직만 추출)"""

    def __init__(self):
        # API 키들 로드
        self.api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_BACKUP1"),
            os.getenv("GOOGLE_API_KEY_BACKUP2"),
            os.getenv("GOOGLE_API_KEY_BACKUP3")
        ]

        # None 값 제거
        self.api_keys = [key for key in self.api_keys if key]

        if not self.api_keys:
            raise ValueError("API 키가 설정되지 않았습니다.")

        self.current_key_index = 0
        self.current_api_key = self.api_keys[0]

        # genai 초기화
        genai.configure(api_key=self.current_api_key)

    def switch_to_next_key(self) -> bool:
        """다음 API 키로 전환"""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.current_api_key = self.api_keys[self.current_key_index]
            genai.configure(api_key=self.current_api_key)
            return True
        return False


class SentenceComposer:
    """LLM을 사용하여 텍스트 조각들을 완전한 문장으로 구성하는 클래스"""

    def __init__(self, logger: Logger):
        self.api_manager = APIKeyManager()
        self.logger = logger
        self.session_id = f"sentence_compose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 문장 구성 전용 프롬프트
        self.compose_prompt = """당신은 기계 번역 학습 데이터셋을 만들기 위해, 텍스트 덩어리를 의미론적으로 완전한 문장 또는 구문 단위로 재구성하는 AI 전문가입니다.

DB에 한 문장(또는 구문)씩 저장하기 위해 주어진 JSON 형식에 따라 출력해주세요.

입력 데이터는 'source_type'과 'text' 키를 가진 JSON 객체들의 배열입니다.

**# 최종 목표**
입력된 텍스트 조각들을 합쳐서 의미상 완전한 문장들로 구성된 새로운 배열을 최종 출력물로 만드는 것이 목표입니다.

**# 처리 규칙**

1. **H1, H2 태그 처리**:
   - 'source_type'이 'H1' 또는 'H2'인 객체는 **내용을 변경하지 말고 그대로** 최종 출력 배열에 추가합니다.

2. **P (본문) 태그 처리 (가장 중요)**:
(1) 작업 순서:
   - 'source_type'이 'P'인 객체들을 순서대로 읽습니다.
   - 완전한 하나의 문장 또는 구문 단위를 완성할 때까지 **다음 'P' 객체의 텍스트를 계속 이어 붙여서** 하나의 완전한 문장(또는 구문)을 완성합니다.
   - 만약 이미 완전한 문장 또는 구문인 경우, 불필요한 공백을 제거하고 출력 배열에 추가합니다.
   - 완성된 문장은 불필요한 공백을 제거하고 하나의 JSON 객체로 만들어 출력 배열에 추가합니다.
(2) 작업 힌트:
   - 완전한 문장이 완료되는 지점은 다음과 같은 규칙에서 힌트를 얻을 수 있습니다.
    규칙1: 마침표(.), 물음표(?), 느낌표(!) 등으로 끝나는 경우 완전한 문장이 완료되었다고 판단합니다.
    규칙2: 괄호 처리. 여는 괄호(`[`, `(`)로 시작한 텍스트는 짝이 맞는 닫는 괄호(`]`, `)`)가 나올 때까지 다음 객체의 텍스트를 계속 이어 붙여야 합니다. 괄호 안의 내용이 여러 줄에 걸쳐 나뉘어 있어도 하나의 의미 단위로 합쳐야 합니다.
(3) 주의 사항:
   - 주의 1: 하나의 입력 객체 안에 여러 개의 완전한 문장이 있을 경우, 각각을 별개의 출력 객체로 나눠야 합니다. 즉, 각 `{}` 객체는 반드시 하나의 문장만을 가져야 합니다. 여러 문장을 하나의 객체에 합치지 마세요.
   - 주의 2: 하나의 입력 객체는 반드시 한 번만 처리되어야 합니다. 여러 번 처리하지 마세요.
   - 주의 3: 각 객체의 순서는 원본 파일의 순서를 유지하며 처리해주세요.

3. **출력 형식**:
   - 최종 출력물은 `{"text": "...", "source_type": "H1|H2|P"}` 스키마를 따르는 JSON 배열이어야 합니다.
   - 출력 배열의 순서는 원본 파일의 순서를 유지하며 처리해주세요.

   **예시(H1, H2)**:
   - input:
    {
        "text": "더블린 규정에 따라 이탈리아로 이송할 수 있는 요건의 완화",
        "source_type": "H2"
    },
    - output:
    {
        "text": "더블린 규정에 따라 이탈리아로 이송할 수 있는 요건의 완화",
        "source_type": "H2"
    }

   **예시(P)**:
   - input:
    {
        "text": "보호자 미동반 아동",
        "source_type": "P"
    },
    {
        "text": "룩셈부르크 행정법원은 A 와 S 에 대한 유럽연합사법재판소 (CJEU) 판결 ( 제 C-550/16 호 ) 을 고려하여 미성년자가",
        "source_type": "P"
    },
    {
        "text": "보호자 미동반 아동으로 간주되는 조건을 분석하였다 .",
        "source_type": "P"
    }
    - output:
    {
        "text": "보호자 미동반 아동",
        "source_type": "P"
    },
    {
        "text": "룩셈부르크 행정법원은 A와 S에 대한 유럽연합사법재판소(CJEU) 판결(제C-550/16호)을 고려하여 미성년자가 보호자 미동반 아동으로 간주되는 조건을 분석하였다.",
        "source_type": "P"
    }

**반드시 JSON 배열만 출력하고, 다른 설명이나 텍스트는 포함하지 마세요.**"""

    def process_batch(self, batch_data: List[Dict], batch_number: int, part_info: str) -> Dict[str, Any]:
        """
        배치 데이터를 LLM으로 처리하여 문장을 구성합니다.

        Args:
            batch_data: 처리할 배치 데이터
            batch_number: 배치 번호
            part_info: 파트 정보 (예: "1부_kr")

        Returns:
            처리 결과
        """
        try:
            self.logger.info(f"배치 {batch_number} 처리 시작 ({len(batch_data)}개 객체)")

            # 입력 데이터 분석
            input_h1_count = len([item for item in batch_data if item.get('source_type') == 'H1'])
            input_h2_count = len([item for item in batch_data if item.get('source_type') == 'H2'])

            # 입력 데이터를 JSON 문자열로 변환
            input_json = json.dumps(batch_data, ensure_ascii=False, indent=2)

            # LLM에 전송할 메시지 구성
            messages = [
                {"role": "user", "content": f"{self.compose_prompt}\n\n입력 데이터:\n{input_json}"}
            ]

            # LLM 호출
            start_time = time.time()
            response = self._call_llm_simple(input_json)

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

            # 출력 데이터 분석
            output_h1_count = len([item for item in composed_data if item.get('source_type') == 'H1'])
            output_h2_count = len([item for item in composed_data if item.get('source_type') == 'H2'])

            # 보존율 계산
            h1_preservation = (output_h1_count / input_h1_count) * 100 if input_h1_count > 0 else 100
            h2_preservation = (output_h2_count / input_h2_count) * 100 if input_h2_count > 0 else 100

            # 압축률 계산
            compression_ratio = len(batch_data) / len(composed_data) if len(composed_data) > 0 else 0

            # 로그 출력
            self.logger.info(f"├─ 압축률: {compression_ratio:.1f}배 ({len(batch_data)}개 → {len(composed_data)}개)")

            if input_h1_count > 0:
                status = "✓" if h1_preservation == 100 else "⚠️"
                self.logger.info(f"├─ H1 보존율: {h1_preservation:.1f}% ({output_h1_count}/{input_h1_count}개) {status}")
                if h1_preservation < 100:
                    self.logger.warning(f"H1 보존율이 100% 미만입니다. 확인 필요.")

            if input_h2_count > 0:
                status = "✓" if h2_preservation == 100 else "⚠️"
                self.logger.info(f"├─ H2 보존율: {h2_preservation:.1f}% ({output_h2_count}/{input_h2_count}개) {status}")
                if h2_preservation < 100:
                    self.logger.warning(f"H2 보존율이 100% 미만입니다. 확인 필요.")

            self.logger.info(f"└─ 처리시간: {processing_time:.1f}초")

            return {
                "success": True,
                "batch_number": batch_number,
                "input_count": len(batch_data),
                "output_count": len(composed_data),
                "compression_ratio": compression_ratio,
                "h1_preservation_rate": h1_preservation,
                "h2_preservation_rate": h2_preservation,
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

    def _call_llm_simple(self, input_json: str) -> Optional[str]:
        """간단한 LLM 호출 (JSON 입력받아 JSON 응답 반환)"""
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
                        "description": "원본 출처 유형: H1|H2|P"
                    }
                },
                "required": ["text", "source_type"]
            }
        }

        for attempt in range(max_retries):
            try:
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

                # 전체 프롬프트 구성
                full_prompt = f"{self.compose_prompt}\n\n입력 데이터:\n{input_json}"

                # LLM 호출
                response = model.generate_content(full_prompt)

                if response and response.text:
                    return response.text

                return None

            except Exception as e:
                error_message = str(e)
                self.logger.warning(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {error_message}")

                # 다음 API 키로 전환
                if attempt < max_retries - 1 and self.api_manager.switch_to_next_key():
                    self.logger.info(f"다음 API 키로 전환하여 재시도...")
                    continue

                # 모든 키 시도 완료
                if attempt == max_retries - 1:
                    self.logger.error(f"모든 API 키 시도 완료. 마지막 오류: {error_message}")
                    return None

        return None

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
        batch_processor = BatchProcessor(batch_size=200, overlap_size=10)
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

            # 배치로 분할
            batches = batch_processor.split_into_batches(input_data)
            part_info = f"{file_info['part_number']}_{file_info['language']}"

            logger.info(f"배치 분할 완료: 총 {len(batches)}개 배치 생성 (겹침: 10개)")

            # 각 배치 처리
            all_results = []
            successful_batches = 0
            total_processing_time = 0
            batch_summaries = []

            for batch in batches:
                result = sentence_composer.process_batch(
                    batch['data'],
                    batch['batch_number'],
                    part_info
                )

                if result['success']:
                    all_results.extend(result['composed_data'])
                    successful_batches += 1
                    total_processing_time += result['processing_time']

                    batch_summaries.append({
                        "batch_number": result['batch_number'],
                        "compression_ratio": result['compression_ratio'],
                        "h1_preservation": result['h1_preservation_rate'],
                        "h2_preservation": result['h2_preservation_rate']
                    })
                else:
                    logger.error(f"배치 {batch['batch_number']} 실패: {result['error']}")

            # 중복 제거 (text 기준) - 주석 처리됨
            # logger.info(f"\n결과 통합 및 중복 제거...")
            # seen_texts = set()
            # unique_results = []
            #
            # for item in all_results:
            #     text = item.get('text', '')
            #     if text and text not in seen_texts:
            #         seen_texts.add(text)
            #         unique_results.append(item)

            # 중복 제거 없이 전체 결과 사용
            logger.info(f"\n결과 통합...")
            unique_results = all_results

            # 메타데이터 준비
            metadata = {
                "total_input_count": len(input_data),
                "overall_compression_ratio": len(input_data) / len(unique_results) if len(unique_results) > 0 else 0,
                "successful_batches": successful_batches,
                "total_batches": len(batches),
                "average_processing_time": total_processing_time / successful_batches if successful_batches > 0 else 0,
                "processing_summary": {
                    "total_duplicates_removed": 0,  # 중복 제거 비활성화
                    "batch_details": batch_summaries
                }
            }

            # 결과 저장 (메타데이터 포함)
            output_path = file_info['output_path']
            save_results_with_metadata(output_path, unique_results, metadata)

            logger.success(f"저장 완료: {output_path}")
            logger.info(f"   원본: {len(input_data):,}개 → 결과: {len(unique_results):,}개")
            logger.info(f"   전체 압축률: {metadata['overall_compression_ratio']:.1f}배")
            logger.info(f"   성공한 배치: {successful_batches}/{len(batches)}")
            # logger.info(f"   중복 제거: {len(all_results) - len(unique_results):,}개")  # 중복 제거 비활성화

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