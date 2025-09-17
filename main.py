"""
PDF 파일에서 한국어 또는 영어 문장을 추출하여 SQLite 데이터베이스로 저장하는 모듈

Google Generative AI를 사용하여 PDF 파일에서 텍스트를 추출하고 
병렬 코퍼스 데이터베이스에 저장합니다.
"""

import os
import sqlite3
import re
import json
import time
import logging
import glob
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from langfuse import Langfuse
from prompt import get_korean_extraction_prompt, get_english_extraction_prompt
from database import DatabaseManager
from session_manager import SessionManager
from utils import PDFSplitter, JSONBackupManager, format_duration

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pdf_processing_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Langfuse 초기화
try:
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )
    logger.info("✅ Langfuse 초기화 완료")
except Exception as e:
    logger.warning(f"⚠️ Langfuse 초기화 실패: {e}")
    langfuse = None


class PDFToCorpusConverter:
    """PDF에서 한국어 또는 영어 문장을 추출하여 병렬 코퍼스 데이터베이스에 저장하는 클래스"""

    def __init__(self, api_key: Optional[str] = None, db_path: str = "corpus.db"):
        """
        PDFToCorpusConverter 초기화
        
        Args:
            api_key (str, optional): Google AI API 키. 
                                   None인 경우 GOOGLE_API_KEY 환경변수에서 가져옵니다.
            db_path (str): SQLite 데이터베이스 파일 경로
        """
        # API 키 리스트 설정
        self.api_keys = []
        
        # 메인 키 추가
        primary_key = api_key or os.getenv("GOOGLE_API_KEY")
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
        
        # 기존 단일 백업키도 지원 (하위 호환성)
        legacy_backup = os.getenv("GOOGLE_API_KEY_BACKUP")
        if legacy_backup and legacy_backup not in self.api_keys:
            self.api_keys.append(legacy_backup)
        
        # API 키 상태 관리
        self.usage_file = "api_key_usage.json"
        self._load_key_usage()  # JSON에서 키 상태 로드
        self.current_key_index = self._get_available_key_index()
        self.current_api_key = self.api_keys[self.current_key_index] if self.api_keys else None
        
        if not self.api_keys:
            raise ValueError(
                "API 키가 필요합니다. api_key 매개변수를 제공하거나 "
                "GOOGLE_API_KEY 환경변수를 설정해주세요."
            )
        
        self.db_path = db_path
        
        # Google AI 설정
        genai.configure(api_key=self.current_api_key)
        
        # Safety Settings 정의 - 규제/기술 문서 처리를 위해 모든 카테고리 비활성화
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 배치 처리 관련 초기화
        self.session_manager = SessionManager(db_path)
        self.json_backup = JSONBackupManager()
        
        # API 키 상태 로깅
        logger.info(f"✅ 설정된 API 키 개수: {len(self.api_keys)}개")
        for i, key in enumerate(self.api_keys):
            key_type = "메인" if i == 0 else f"백업{i}"
            key_id = f"***{key[-4:]}"
            key_info = self.key_usage.get(key_id, {'status': 'available'})
            status_emoji = "🟢" if key_info['status'] == 'available' else "🔴" if key_info['status'] == 'exhausted' else "🟡"
            current_mark = " ←현재선택" if i == self.current_key_index else ""
            logger.info(f"  {key_type} API 키: {key_id} {status_emoji}{key_info['status']}{current_mark}")
        
        if len(self.api_keys) == 1:
            logger.warning(f"⚠️  백업 API 키가 없습니다. GOOGLE_API_KEY_BACKUP_1, GOOGLE_API_KEY_BACKUP_2 등 환경변수 설정을 권장합니다.")

    def _init_database(self) -> None:
        """SQLite 데이터베이스 연결 가능성을 확인합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 데이터베이스 연결 테스트 (실제 스키마 초기화는 DatabaseManager에서 수행)
                cursor.execute("SELECT 1")
                conn.commit()
                
        except Exception as e:
            raise Exception(f"데이터베이스 연결 실패: {e}")


    def _get_original_filename(self, temp_file_path: str) -> str:
        """
        temp 파일 경로에서 원본 파일명을 추출합니다.
        
        Args:
            temp_file_path (str): temp PDF 파일 경로
            
        Returns:
            str: 원본 파일명 (예: "9_40_kr.pdf")
        """
        file_name = Path(temp_file_path).name
        
        # UUID와 배치 정보를 제거하여 원본 파일명 추출
        # 예: "336b0b16-98b5-43fd-a159-b8703e499ccf_01_02_kr_batch_001_pages_1-30.pdf"
        # → "01_02_kr.pdf"
        
        # UUID 패턴 제거 (8-4-4-4-12 형식)과 뒤따르는 '_'를 제거
        base_name = re.sub(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_", "", file_name)
        
        # 배치 정보 제거 (batch_XXX_pages_XX-XX 패턴)
        base_name = re.sub(r"_batch_\d+_pages_\d+-\d+", "", base_name)
        
        return base_name

    def extract_sentences_from_pdf(
        self, 
        file_path: str, 
        language: str  # "korean", "english"
    ) -> str:
        """
        PDF 파일에서 문장을 추출합니다.

        Args:
            file_path (str): 분석할 PDF 파일의 경로
            language (str): 추출할 언어 ("korean", "english")
        
        Returns:
            str: 추출된 텍스트 응답
            
        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
            Exception: API 호출 중 오류가 발생한 경우
        """
        # 전체 처리 시작 시간
        total_start_time = time.time()
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 파일 정보 로깅
        file_size_mb = Path(file_path).stat().st_size / 1024 / 1024
        logger.info(f"📄 PDF 처리 시작: {Path(file_path).name} ({file_size_mb:.2f} MB)")

        # Langfuse 트레이스 시작
        trace = None
        if langfuse:
            trace = langfuse.trace(
                name="pdf_sentence_extraction",
                metadata={
                    "file_path": Path(file_path).name,
                    "file_size_mb": file_size_mb,
                    "language": language,
                    "model_name": "gemini-2.5-flash"
                }
            )

        # 기본 프롬프트 설정
        if language == "korean":
            prompt = get_korean_extraction_prompt()
        elif language == "english":
            prompt = get_english_extraction_prompt()
        else:
            raise ValueError(f"지원하지 않는 언어입니다: {language}. 'korean' 또는 'english'만 지원합니다.")
        
        logger.info(f"📝 프롬프트 길이: {len(prompt)} 문자")

        try:
            # 1. PDF 업로드 단계
            upload_start = time.time()
            upload_span = trace.span(name="pdf_upload", input={"file_path": Path(file_path).name, "file_size_mb": file_size_mb}) if trace else None
            
            logger.info(f"📤 PDF 업로드 시작: {Path(file_path).name}")
            print(f"'{file_path}' 파일을 업로드하는 중...")
            
            uploaded_file = genai.upload_file(path=file_path)
            upload_time = time.time() - upload_start
            
            logger.info(f"✅ PDF 업로드 완료 ({upload_time:.2f}초)")
            if upload_span:
                upload_span.end(output={"upload_success": True, "upload_time_seconds": upload_time})
            
            # 2. LLM 호출 단계
            llm_start = time.time()
            llm_span = trace.span(
                name="llm_generation",
                input={
                    "model": "gemini-2.5-flash",
                    "prompt_length": len(prompt),
                    "language": language
                }
            ) if trace else None
            
            logger.info(f"🤖 LLM 요청 시작 - 모델: gemini-2.5-flash, 언어: {language}")
            print(f"모델에 요청하여 {language} 문장을 추출하는 중...")
            
            # JSON Schema 정의 (출처 유형 정보 포함)
            response_schema = {
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string", 
                                    "description": "추출된 문장"
                                },
                                "source_type": {
                                    "type": "string", 
                                    "description": "출처 유형: table|image|text"
                                }
                            },
                            "required": ["text", "source_type"]
                        }
                    }
                },
                "required": ["sentences"]
            }
            
            # 모델 초기화 (structured output 설정)
            # 모델 temperature 설정 - 한국어는 0.5, 영어는 0.6
            temperature = 0.5 if language == "korean" else 0.6
            model = genai.GenerativeModel(
                "gemini-2.5-flash",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                    "temperature": temperature,
                    "max_output_tokens": 65536
                }
            )
            
            # 컨텐츠 생성 (에러 발생 시 다음 키로 자동 전환 및 파일 재업로드)
            max_retries = len(self.api_keys)  # 모든 키를 시도
            current_uploaded_file = uploaded_file  # 현재 업로드된 파일 추적
            
            for attempt in range(max_retries):
                try:
                    response = model.generate_content([current_uploaded_file, prompt], safety_settings=self.safety_settings)
                    llm_time = time.time() - llm_start
                    break  # 성공하면 루프 종료
                except Exception as e:
                    error_message = str(e)
                    error_code = getattr(e, 'code', None)
                    
                    # 에러 타입 분류 및 상세 로깅
                    if self._is_daily_quota_exceeded(e):
                        logger.warning(f"📊 일별 할당량 초과 감지: {error_message}")
                        self._mark_key_quota_exceeded(self.current_key_index)
                        error_type = "일별 할당량 초과"
                    elif self._is_file_access_error(e):
                        logger.warning(f"🔒 파일 접근 권한 에러 감지: {error_message}")
                        error_type = "파일 접근 권한 오류"
                    elif self._is_api_key_invalid(e):
                        logger.warning(f"🔑 API 키 인증 에러 감지: {error_message}")
                        error_type = "API 키 인증 오류"
                    else:
                        logger.warning(f"❓ 기타 API 에러 감지: {error_message}")
                        error_type = "기타 API 오류"
                    
                    # API 키 전환 시도
                    if self._switch_to_next_key():
                        current_key_info = f"백업 키 {self.current_key_index}" if self.current_key_index > 0 else "메인 키"
                        logger.warning(f"🔄 API 키 전환 - 원인: {error_type}")
                        logger.warning(f"🔄 {current_key_info}로 전환하여 재시도... (***...{self.current_api_key[-4:]})")
                        
                        # 파일 재업로드 (새 API 키로는 이전 파일에 접근 불가)
                        try:
                            # 기존 업로드 파일 삭제 시도
                            if current_uploaded_file != uploaded_file:  # 이미 재업로드된 파일인 경우
                                try:
                                    genai.delete_file(current_uploaded_file.name)
                                    logger.info(f"🗑️ 이전 API 키의 업로드 파일 삭제: {current_uploaded_file.name}")
                                except Exception as delete_e:
                                    logger.warning(f"⚠️ 이전 업로드 파일 삭제 실패 (무시함): {delete_e}")
                            
                            # 새 API 키로 파일 재업로드
                            logger.info(f"📤 새 API 키로 파일 재업로드 중: {Path(file_path).name}")
                            current_uploaded_file = genai.upload_file(path=file_path)
                            logger.info(f"✅ 파일 재업로드 완료: {current_uploaded_file.name}")
                            
                        except Exception as upload_e:
                            logger.error(f"❌ 파일 재업로드 실패: {upload_e}")
                            # 재업로드 실패 시 다음 키로 계속 시도
                            continue
                        
                        # 새 키로 모델 재설정
                        model = genai.GenerativeModel(
                            "gemini-2.5-flash",
                            generation_config={
                                "response_mime_type": "application/json",
                                "response_schema": response_schema,
                                "temperature": temperature,
                                "max_output_tokens": 65536
                            }
                        )
                        continue  # 재시도
                    else:
                        # 모든 키 사용 완료
                        logger.error(f"❌ 모든 API 키 ({len(self.api_keys)}개) 사용 완료")
                        if self._is_daily_quota_exceeded(e):
                            raise Exception("❌ 모든 API 키의 일별 할당량이 초과되었습니다. 24시간 후에 다시 시도해주세요.")
                        else:
                            raise Exception(f"❌ 모든 API 키로 시도했지만 API 호출이 실패했습니다. 마지막 에러: {error_type} - {error_message}")
            else:
                # 모든 재시도 실패
                raise Exception("❌ 모든 API 키로 시도했지만 API 호출이 실패했습니다.")
            
            # 3. 응답 분석
            response_length = len(response.text) if response.text else 0
            response_preview = response.text[:200] if response.text else "None"
            
            logger.info(f"📥 LLM 응답 수신 ({llm_time:.2f}초)")
            logger.info(f"📊 응답 길이: {response_length} 문자")
            logger.info(f"🔍 응답 미리보기: {response_preview}")
            
            print(f"🔍 LLM 응답 (첫 200자): {response_preview}")
            
            # JSON 완성도 체크
            is_complete_json = self._validate_json_completeness(response.text)
            logger.info(f"✅ JSON 완성도: {'완료' if is_complete_json else '⚠️ 불완전'}")
            
            if llm_span:
                llm_span.end(
                    output={
                        "response_length": response_length,
                        "response_preview": response_preview,
                        "processing_time_seconds": llm_time,
                        "json_complete": is_complete_json
                    }
                )
            
            print("✅ Structured Output으로 JSON 형식 보장됨")
            
            # 성공한 경우 Google AI 업로드 파일 즉시 삭제 (로컬 temp 파일은 보존)
            try:
                # 재업로드된 파일이 있으면 그것을 삭제, 아니면 원본 삭제
                file_to_delete = current_uploaded_file if current_uploaded_file != uploaded_file else uploaded_file
                genai.delete_file(file_to_delete.name)
                logger.info(f"🗑️ Google AI 업로드 파일 삭제 완료: {file_to_delete.name}")
            except Exception as e:
                logger.warning(f"⚠️ Google AI 업로드 파일 삭제 실패: {e}")
            
            # 전체 처리 시간
            total_time = time.time() - total_start_time
            logger.info(f"🎯 PDF 처리 완료 ({total_time:.2f}초 총 소요)")
            print("작업 완료 및 업로드된 파일 삭제.")
            
            # 트레이스 완료
            if trace:
                trace.update(
                    output={
                        "total_processing_time": total_time,
                        "response_length": response_length,
                        "json_complete": is_complete_json,
                        "success": True
                    }
                )
            
            return response.text

        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ PDF 처리 실패: {error_message}")
            logger.error(f"📊 처리 시간: {time.time() - total_start_time:.2f}초")
            
            # 트레이스에 에러 기록
            if trace:
                trace.update(
                    output={
                        "success": False,
                        "error": error_message,
                        "processing_time": time.time() - total_start_time
                    }
                )
            
            print(f"오류가 발생했습니다: {e}")
            # 실패한 경우 Google AI 업로드 파일 보존 (재시도를 위해)
            if 'current_uploaded_file' in locals():
                logger.warning(f"💾 실패한 작업의 Google AI 파일 보존: {current_uploaded_file.name} (재시도 가능)")
            elif 'uploaded_file' in locals():
                logger.warning(f"💾 실패한 작업의 Google AI 파일 보존: {uploaded_file.name} (재시도 가능)")
            raise

    def _validate_json_completeness(self, json_text: str) -> bool:
        """JSON 응답의 완성도를 검증 (수리하지 않고 로깅만)"""
        if not json_text:
            logger.warning("⚠️ JSON 응답이 비어있음")
            return False
        
        try:
            # 기본 JSON 파싱 시도
            json.loads(json_text)
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON 파싱 오류: {e}")
            
            # 불완전한 JSON 패턴 로깅
            if "Unterminated string" in str(e):
                logger.warning("🔧 문자열 미완료 감지")
            elif "Expecting" in str(e):
                logger.warning("🔧 구조 미완료 감지")
            
            return False

    def _parse_json_response(self, json_text: str, language: str) -> List[Dict[str, Any]]:
        """
        JSON 형식의 응답을 파싱하여 문장 정보 리스트로 반환합니다.
        기존 형식과 새로운 형식 모두 지원합니다.
        
        Args:
            json_text (str): JSON 형식의 텍스트
            language (str): 추출된 언어 ("korean", "english")
            
        Returns:
            List[Dict[str, Any]]: 문장 정보 리스트
            [
                {
                    "text": "문장 텍스트",
                    "source_type": "table|image|text"
                }
            ]
        """
        import json
        
        # JSON 내용에서 불필요한 마크다운 문법 제거
        cleaned_content = json_text.strip()
        if cleaned_content.startswith('```'):
            lines = cleaned_content.split('\n')
            # 첫 번째와 마지막 ```줄 제거
            cleaned_content = '\n'.join(line for line in lines if not line.strip().startswith('```') and not line.strip() == 'json')
        
        sentences = []
        
        try:
            # JSON 파싱
            data = json.loads(cleaned_content)
            
            # 새로운 형식: {"sentences": [{"text": "...", "source_type": "...", ...}]}
            if 'sentences' in data and isinstance(data['sentences'], list):
                for i, sentence_data in enumerate(data['sentences']):
                    if isinstance(sentence_data, dict):
                        # 새로운 형식: 딕셔너리 객체
                        text = sentence_data.get('text', '').strip()
                        if text:
                            sentences.append({
                                'text': text,
                                'source_type': sentence_data.get('source_type', 'text')
                            })
                    elif isinstance(sentence_data, str) and sentence_data.strip():
                        # 기존 형식: 문자열 배열 (호환성 유지)
                        sentences.append({
                            'text': sentence_data.strip(),
                            'source_type': 'text'  # 기본값
                        })
            
            # 기존 형식: 직접 문자열 배열
            elif isinstance(data, list):
                for i, sentence in enumerate(data):
                    if isinstance(sentence, str) and sentence.strip():
                        sentences.append({
                            'text': sentence.strip(),
                            'source_type': 'text'  # 기본값
                        })
            
            else:
                print("  ⚠️  JSON 형식을 인식할 수 없습니다.")
            
            print(f"파싱 완료: {len(sentences)}개 문장 추출")
            return sentences
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 중 오류: {e}")
            print(f"응답 내용 (첫 500자): {cleaned_content[:500]}")
            # JSON 파싱 실패 시 빈 리스트 반환 (failed 처리)
            return []
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            return []

    def save_to_database(self, sentences: List[Dict[str, Any]], language: str, pdf_file_path: str = None) -> int:
        """
        문장들을 언어별 테이블에 저장합니다.
        
        Args:
            sentences (List[Dict[str, Any]]): 문장 정보 리스트
            language (str): 처리된 언어 타입
            pdf_file_path (str): PDF 파일 경로
            
        Returns:
            int: 저장된 레코드 수
        """
        if not sentences:
            print("저장할 데이터가 없습니다.")
            return 0
        
        # 원본 파일명 추출 (pdf_file_path가 있는 경우)
        original_filename = self._get_original_filename(pdf_file_path) if pdf_file_path else None
        
        # 언어에 따른 테이블 선택
        table_name = "korean_sentences" if language == "korean" else "english_sentences"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                insert_query = f"""
                    INSERT INTO {table_name} 
                    (sentence, sentence_index, pdf_file_path, original_filename, source_type)
                    VALUES (?, ?, ?, ?, ?)
                """
                
                inserted_count = 0
                for idx, sentence_data in enumerate(sentences, 1):
                    try:
                        # 호환성을 위해 문자열과 딕셔너리 모두 지원
                        if isinstance(sentence_data, str):
                            sentence_text = sentence_data
                            source_type = 'text'
                        else:
                            sentence_text = sentence_data.get('text', '')
                            source_type = sentence_data.get('source_type', 'text')
                        
                        cursor.execute(insert_query, (
                            sentence_text, 
                            idx, 
                            pdf_file_path, 
                            original_filename,
                            source_type
                        ))
                        inserted_count += 1
                    except Exception as e:
                        sentence_display = sentence_text[:50] + "..." if len(sentence_text) > 50 else sentence_text
                        print(f"  ⚠️  데이터 삽입 실패: '{sentence_display}': {e}")
                
                conn.commit()
                print(f"데이터베이스 저장 완료: {inserted_count}개 {language} 문장 저장")
                
                return inserted_count
                
        except Exception as e:
            raise Exception(f"데이터베이스 저장 중 오류: {e}")

    def process_pdf_to_corpus(
        self, 
        pdf_path: str, 
        language: str
    ) -> Dict[str, Any]:
        """
        PDF 파일을 처리하여 병렬 코퍼스 데이터베이스에 저장하는 전체 과정을 수행합니다.
        
        Args:
            pdf_path (str): 입력 PDF 파일 경로
            language (str): 추출할 언어 ("korean", "english")
            
        Returns:
            Dict[str, Any]: 처리 결과 정보
        """
        try:
            # PDF에서 텍스트 추출
            extracted_text = self.extract_sentences_from_pdf(pdf_path, language)
            
            # 응답 파싱
            sentences = self._parse_json_response(extracted_text, language)
            
            # 데이터베이스에 저장
            saved_count = self.save_to_database(sentences, language, pdf_path)
            
            message = f"성공적으로 {saved_count}개의 {language} 문장을 병렬 코퍼스 데이터베이스에 저장했습니다."
            
            return {
                "status": "success",
                "input_file": pdf_path,
                "database": self.db_path,
                "language": language,
                "extracted_sentences": saved_count,
                "message": message
            }
            
        except Exception as e:
            return {
                "status": "error",
                "input_file": pdf_path,
                "database": self.db_path,
                "language": language,
                "error": str(e),
                "message": f"처리 중 오류가 발생했습니다: {e}"
            }

    def get_corpus_count(self, count_type: str = "total") -> int:
        """데이터베이스에 저장된 문장 수를 반환합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if count_type == "korean":
                    cursor.execute("SELECT COUNT(*) FROM korean_sentences")
                elif count_type == "english":
                    cursor.execute("SELECT COUNT(*) FROM english_sentences")
                else:  # total
                    cursor.execute("""
                        SELECT 
                            (SELECT COUNT(*) FROM korean_sentences) + 
                            (SELECT COUNT(*) FROM english_sentences) as total
                    """)
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"문장 수 조회 중 오류: {e}")
            return 0

    # 재시도 메서드들
    def retry_all_failed_batches(self, language_filter: Optional[str] = None) -> Dict[str, Any]:
        """모든 실패한 배치들을 재시도합니다."""
        try:
            start_time = time.time()
            
            # 실패한 배치들 조회
            failed_batches = self.session_manager.db_manager.get_all_failed_batches(language_filter)
            
            if not failed_batches:
                return {
                    "success": True,
                    "retried_batches": 0,
                    "successful_batches": 0,
                    "still_failed_batches": 0,
                    "total_sentences": 0,
                    "duration": "0초",
                    "message": "재시도할 실패한 배치가 없습니다."
                }
            
            print(f"🔄 {len(failed_batches)}개의 실패한 배치를 재시도합니다...")
            
            # 세션별로 그룹핑
            sessions = {}
            for batch in failed_batches:
                session_id = batch["session_id"]
                if session_id not in sessions:
                    sessions[session_id] = {
                        "language": batch["language"],
                        "batches": []
                    }
                sessions[session_id]["batches"].append(batch["batch_number"])
            
            total_sentences = 0
            successful_batches = 0
            still_failed_batches = 0
            
            # 세션별로 재시도
            for session_id, session_data in sessions.items():
                print(f"\n📁 세션 {session_id[:8]}... ({session_data['language']}) 재시도 중...")
                
                for batch_number in session_data["batches"]:
                    try:
                        result = self._process_single_batch_internal(
                            session_id=session_id,
                            batch_number=batch_number,
                            language=session_data["language"]
                        )
                        
                        if result["success"]:
                            sentences_count = result.get("sentences_count", 0)
                            total_sentences += sentences_count
                            successful_batches += 1
                            print(f"✅ 배치 {batch_number} 재시도 성공: {sentences_count}개 문장")
                        else:
                            still_failed_batches += 1
                            print(f"❌ 배치 {batch_number} 재시도 실패: {result.get('error', '알 수 없는 오류')}")
                    
                    except Exception as e:
                        still_failed_batches += 1
                        print(f"❌ 배치 {batch_number} 재시도 중 예외: {e}")
            
            end_time = time.time()
            duration = format_duration(end_time - start_time)
            
            # 재시도 결과 요약 출력
            print("\n" + "="*60)
            print("🔄 실패한 배치 재시도 결과 요약")
            print("="*60)
            print(f"🔄 재시도한 배치: {len(failed_batches)}개")
            print(f"✅ 성공한 배치: {successful_batches}개")
            print(f"❌ 여전히 실패한 배치: {still_failed_batches}개")
            if len(failed_batches) > 0:
                success_rate = (successful_batches / len(failed_batches)) * 100
                print(f"📈 재시도 성공률: {success_rate:.1f}%")
            print(f"📝 추출된 문장: {total_sentences:,}개")
            print(f"⏱️ 총 소요시간: {duration}")
            
            if still_failed_batches > 0:
                print(f"\n⚠️ {still_failed_batches}개 배치가 여전히 실패 상태입니다.")
                print("💡 다시 재시도하거나 수동 확인이 필요합니다.")
            
            print("="*60)
            
            return {
                "success": True,
                "retried_batches": len(failed_batches),
                "successful_batches": successful_batches,
                "still_failed_batches": still_failed_batches,
                "total_sentences": total_sentences,
                "duration": duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"전체 재시도 실패: {e}"
            }
    
    def retry_session_batches(self, session_id: str, failed_only: bool = True) -> Dict[str, Any]:
        """특정 세션의 배치들을 재시도합니다."""
        try:
            start_time = time.time()
            
            # 세션 정보 조회
            session_progress = self.session_manager.db_manager.get_session_progress(session_id)
            if "error" in session_progress:
                return {"success": False, "error": session_progress["error"]}
            
            language = session_progress["language"]
            
            # 재시도할 배치들 결정
            if failed_only:
                batch_numbers = self.session_manager.retry_failed_batches(session_id)
                print(f"🔄 세션 {session_id[:8]}...의 실패한 배치 {len(batch_numbers)}개를 재시도합니다...")
            else:
                batch_numbers = self.session_manager.get_incomplete_batches(session_id)
                print(f"🔄 세션 {session_id[:8]}...의 미완료 배치 {len(batch_numbers)}개를 재시도합니다...")
            
            if not batch_numbers:
                return {
                    "success": True,
                    "retried_batches": 0,
                    "successful_batches": 0,
                    "total_sentences": 0,
                    "duration": "0초",
                    "message": "재시도할 배치가 없습니다."
                }
            
            total_sentences = 0
            successful_batches = 0
            
            for batch_number in batch_numbers:
                try:
                    result = self._process_single_batch_internal(
                        session_id=session_id,
                        batch_number=batch_number,
                        language=language
                    )
                    
                    if result["success"]:
                        sentences_count = result.get("sentences_count", 0)
                        total_sentences += sentences_count
                        successful_batches += 1
                        print(f"✅ 배치 {batch_number} 재시도 성공: {sentences_count}개 문장")
                    else:
                        print(f"❌ 배치 {batch_number} 재시도 실패: {result.get('error', '알 수 없는 오류')}")
                
                except Exception as e:
                    print(f"❌ 배치 {batch_number} 재시도 중 예외: {e}")
            
            end_time = time.time()
            duration = format_duration(end_time - start_time)
            
            failed_batches = len(batch_numbers) - successful_batches
            
            # 세션 재시도 결과 요약 출력
            print("\n" + "="*60)
            print(f"🔄 세션 {session_id[:8]}... 재시도 결과 요약")
            print("="*60)
            print(f"🔄 재시도한 배치: {len(batch_numbers)}개")
            print(f"✅ 성공한 배치: {successful_batches}개")
            print(f"❌ 실패한 배치: {failed_batches}개")
            if len(batch_numbers) > 0:
                success_rate = (successful_batches / len(batch_numbers)) * 100
                print(f"📈 재시도 성공률: {success_rate:.1f}%")
            print(f"📝 추출된 문장: {total_sentences:,}개")
            print(f"⏱️ 총 소요시간: {duration}")
            
            if failed_batches > 0:
                print(f"\n⚠️ {failed_batches}개 배치가 여전히 실패 상태입니다.")
                print("💡 실패한 배치만 재시도하려면:")
                print(f"   python cli.py --retry-session {session_id[:8]} --failed-only")
            
            print("="*60)
            
            return {
                "success": True,
                "retried_batches": len(batch_numbers),
                "successful_batches": successful_batches,
                "total_sentences": total_sentences,
                "duration": duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"세션 재시도 실패: {e}"
            }
    
    def retry_specific_batch(self, session_id: str, batch_number: int) -> Dict[str, Any]:
        """특정 배치를 재시도합니다."""
        try:
            start_time = time.time()
            
            # 세션 정보 조회
            session_progress = self.session_manager.db_manager.get_session_progress(session_id)
            if "error" in session_progress:
                return {"success": False, "error": session_progress["error"]}
            
            language = session_progress["language"]
            
            print(f"🔄 세션 {session_id[:8]}...의 배치 {batch_number}를 재시도합니다...")
            
            result = self._process_single_batch_internal(
                session_id=session_id,
                batch_number=batch_number,
                language=language
            )
            
            end_time = time.time()
            duration = format_duration(end_time - start_time)
            
            if result["success"]:
                sentences_count = result.get("sentences_count", 0)
                return {
                    "success": True,
                    "retried_batches": 1,
                    "successful_batches": 1,
                    "total_sentences": sentences_count,
                    "duration": duration
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "배치 재시도 실패")
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"특정 배치 재시도 실패: {e}"
            }

    # 배치 처리 메서드들
    def create_batch_session(
        self, 
        pdf_path: str, 
        language: str
    ) -> Dict[str, Any]:
        """배치 처리 세션을 생성합니다."""
        return self.session_manager.create_batch_session(pdf_path, language, 10)
    
    def process_pdf_batch(
        self,
        pdf_path: str,
        language: str,
        resume_session_id: Optional[str] = None,
        retry_failed_only: bool = False
    ) -> Dict[str, Any]:
        """
        PDF 파일을 배치 단위로 처리합니다.
        
        Args:
            pdf_path (str): 처리할 PDF 파일 경로
            language (str): 추출할 언어
            resume_session_id (str, optional): 재개할 세션 ID
            retry_failed_only (bool): 실패한 배치만 재시도할지 여부
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            start_time = time.time()
            
            # 기존 세션 재개 또는 새 세션 생성
            if resume_session_id:
                session_id = resume_session_id
                print(f"세션 재개: {session_id}")
                
                # 세션 상태 확인
                session_status = self.session_manager.get_session_status(session_id)
                if "error" in session_status:
                    return {"success": False, "error": session_status["error"]}
                
                print(f"진행 상황: {session_status['progress_percentage']:.1f}% 완료")
                
                # 재처리할 배치들 결정
                if retry_failed_only:
                    # 실패한 배치만 재시도
                    batches_to_process = self.session_manager.retry_failed_batches(session_id)
                    print(f"실패한 배치만 재시도: {len(batches_to_process)}개")
                else:
                    # 완료되지 않은 모든 배치들 재시도 (실패 + 대기)
                    batches_to_process = self.session_manager.get_incomplete_batches(session_id)
                    failed_count = len(self.session_manager.retry_failed_batches(session_id))
                    pending_count = len(self.session_manager.get_pending_batches(session_id))
                    print(f"재처리할 배치: {len(batches_to_process)}개 (실패: {failed_count}개, 대기: {pending_count}개)")
                
            else:
                # 새 세션 생성
                session_result = self.create_batch_session(pdf_path, language)
                if not session_result["success"]:
                    return session_result
                
                session_id = session_result["session_id"]
                print(f"새 세션 생성: {session_id}")
                print(f"총 {session_result['total_batches']}개 배치, 예상 시간: {session_result['estimated_time']}")
                
                # PDF 분할
                split_result = self.session_manager.start_batch_processing(session_id)
                if not split_result["success"]:
                    return split_result
                
                batches_to_process = list(range(1, session_result['total_batches'] + 1))
            
            # 배치별 처리
            total_sentences = 0
            processed_batches = 0
            failed_batches = 0
            
            print(f"\n배치 처리 시작 ({len(batches_to_process)}개 배치)...")
            
            for batch_number in batches_to_process:
                print(f"\n--- 배치 {batch_number} 처리 중 ---")
                
                try:
                    # 단일 배치 처리
                    batch_result = self._process_single_batch_internal(
                        session_id=session_id,
                        batch_number=batch_number,
                        language=language
                    )
                    
                    if batch_result["success"]:
                        sentences_count = batch_result.get("sentences_count", 0)
                        total_sentences += sentences_count
                        processed_batches += 1
                        print(f"✅ 배치 {batch_number} 완료: {sentences_count}개 문장 추출")
                    else:
                        failed_batches += 1
                        print(f"❌ 배치 {batch_number} 실패: {batch_result.get('error', '알 수 없는 오류')}")
                    
                    # 진행 상황 출력
                    current_progress = (processed_batches / len(batches_to_process)) * 100
                    elapsed_time = time.time() - start_time
                    print(f"📊 진행률: {current_progress:.1f}% | 경과 시간: {format_duration(elapsed_time)}")
                    
                except KeyboardInterrupt:
                    print("\n사용자에 의해 중단되었습니다.")
                    break
                except Exception as e:
                    failed_batches += 1
                    print(f"❌ 배치 {batch_number} 예외 발생: {e}")
                    continue
            
            # 세션 완료
            completion_result = self.session_manager.complete_session(session_id)
            
            end_time = time.time()
            total_duration = format_duration(end_time - start_time)
            
            # 처리 결과 요약 출력
            print("\n" + "="*60)
            print("📊 배치 처리 결과 요약")
            print("="*60)
            print(f"🆔 세션 ID: {session_id[:8]}...")
            print(f"📁 파일: {Path(pdf_path).name}")
            print(f"🌐 언어: {language}")
            print(f"📄 총 배치 수: {len(batches_to_process)}개")
            print(f"✅ 성공한 배치: {processed_batches}개")
            print(f"❌ 실패한 배치: {failed_batches}개")
            if failed_batches > 0:
                success_rate = (processed_batches / len(batches_to_process)) * 100
                print(f"📈 성공률: {success_rate:.1f}%")
            print(f"📝 추출된 문장: {total_sentences:,}개")
            print(f"⏱️ 총 소요시간: {total_duration}")
            
            if failed_batches > 0:
                print(f"\n⚠️ {failed_batches}개 배치가 실패했습니다.")
                print("💡 실패한 배치만 재시도하려면 다음 명령어를 사용하세요:")
                print(f"   python cli.py --retry-session {session_id[:8]} --failed-only")
            
            print("="*60)
            
            return {
                "success": True,
                "session_id": session_id,
                "processed_batches": processed_batches,
                "failed_batches": failed_batches,
                "total_sentences": total_sentences,
                "duration": total_duration,
                "message": f"배치 처리 완료: {processed_batches}개 배치 처리, {total_sentences:,}개 문장 추출 ({total_duration})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"배치 처리 실패: {e}"
            }
    
    def _process_single_batch_internal(
        self,
        session_id: str,
        batch_number: int,
        language: str
    ) -> Dict[str, Any]:
        """
        내부용 단일 배치 처리 메서드 (통계 수집 포함)
        """
        start_time = time.time()
        
        # Langfuse 배치 트레이스 시작
        batch_trace = None
        if langfuse:
            batch_trace = langfuse.trace(
                name="batch_processing",
                metadata={
                    "session_id": session_id,
                    "batch_number": batch_number,
                    "language": language,
                    "model_name": "gemini-2.5-flash"
                }
            )
        
        logger.info(f"🚀 배치 {batch_number} 처리 시작 (세션: {session_id[:8]}...)")
        
        # 배치 상태를 in_progress로 업데이트
        self.session_manager.db_manager.update_batch_status(
            session_id=session_id,
            batch_number=batch_number,
            status="in_progress"
        )
        
        try:
            # 데이터베이스에서 배치 PDF 파일 경로 조회
            batch_pdf_path = self.session_manager.db_manager.get_batch_pdf_path(session_id, batch_number)
            
            if not batch_pdf_path:
                error_msg = f"배치 {batch_number} PDF 파일 경로를 찾을 수 없습니다"
                logger.error(f"❌ {error_msg}")
                if batch_trace:
                    batch_trace.update(output={"success": False, "error": error_msg})
                return {"success": False, "error": error_msg}
            
            # 파일 존재 확인 및 정보 로깅
            if not Path(batch_pdf_path).exists():
                error_msg = f"배치 {batch_number} PDF 파일이 존재하지 않습니다: {batch_pdf_path}"
                logger.error(f"❌ {error_msg}")
                if batch_trace:
                    batch_trace.update(output={"success": False, "error": error_msg})
                return {"success": False, "error": error_msg}
            
            # PDF 파일 정보 로깅
            pdf_size_mb = Path(batch_pdf_path).stat().st_size / 1024 / 1024
            logger.info(f"📄 PDF 정보: {Path(batch_pdf_path).name} ({pdf_size_mb:.2f} MB)")
            
            if batch_trace:
                batch_trace.update(
                    input={
                        "pdf_path": Path(batch_pdf_path).name,
                        "pdf_size_mb": pdf_size_mb,
                        "language": language
                    }
                )
            
            # LLM으로 문장 추출
            llm_start_time = time.time()
            logger.info(f"🤖 배치 {batch_number} LLM 처리 시작")
            
            extracted_text = self.extract_sentences_from_pdf(
                file_path=batch_pdf_path,
                language=language
            )
            llm_processing_time = time.time() - llm_start_time
            logger.info(f"✅ 배치 {batch_number} LLM 처리 완료 ({llm_processing_time:.2f}초)")
            
            # JSON 백업 저장
            logger.info(f"💾 배치 {batch_number} JSON 백업 저장 중")
            backup_path = self.json_backup.save_llm_response(
                session_id=session_id,
                batch_number=batch_number,
                response_text=extracted_text,
                metadata={
                    "pdf_path": batch_pdf_path,
                    "language": language,
                    "model_name": "gemini-2.5-flash",
                    "processing_time": llm_processing_time
                }
            )
            logger.info(f"✅ JSON 백업 저장 완료: {Path(backup_path).name}")
            
            # 응답 파싱
            parse_start = time.time()
            logger.info(f"🔍 배치 {batch_number} JSON 파싱 시작")
            sentences = self._parse_json_response(extracted_text, language)
            parse_time = time.time() - parse_start
            
            logger.info(f"📝 파싱 결과: {len(sentences)} 문장 추출 ({parse_time:.2f}초)")
            
            if not sentences:
                error_msg = "추출된 문장이 없습니다"
                logger.error(f"❌ 배치 {batch_number}: {error_msg}")
                
                # 배치 상태를 failed로 업데이트
                self.session_manager.db_manager.update_batch_status(
                    session_id=session_id,
                    batch_number=batch_number,
                    status="failed",
                    error_message=error_msg
                )
                
                if batch_trace:
                    batch_trace.update(
                        output={
                            "success": False,
                            "error": error_msg,
                            "backup_path": backup_path,
                            "processing_time": time.time() - start_time
                        }
                    )
                
                # 예외 발생시켜 except 블록에서 통합 처리
                raise Exception(error_msg)
            
            # 데이터베이스에 저장 (배치 정보 추가) 및 페이지별 통계 수집
            db_start = time.time()
            logger.info(f"💾 배치 {batch_number} 데이터베이스 저장 중")
            saved_count, page_stats = self._save_batch_to_database(
                sentences=sentences,
                source_file=batch_pdf_path,
                language=language,
                session_id=session_id,
                batch_number=batch_number
            )
            db_time = time.time() - db_start
            logger.info(f"✅ 데이터베이스 저장 완료: {saved_count}개 문장 ({db_time:.2f}초)")
            
            # 전체 처리 시간 계산
            total_processing_time = time.time() - start_time
            
            # 배치 페이지 범위 조회
            page_start, page_end = self._get_batch_page_range(session_id, batch_number)
            total_pages = page_end - page_start + 1
            
            # 통계 저장
            self._save_batch_statistics(
                session_id=session_id,
                batch_number=batch_number,
                language=language,
                saved_count=saved_count,
                total_pages=total_pages,
                processing_time=total_processing_time
            )
            
            # 배치 상태를 completed로 업데이트
            self.session_manager.db_manager.update_batch_status(
                session_id=session_id,
                batch_number=batch_number,
                status="completed",
                sentences_extracted=saved_count
            )
            
            # 성공 로깅
            logger.info(f"🎯 배치 {batch_number} 처리 완료: {saved_count}개 문장, {total_processing_time:.2f}초 총 소요")
            
            # Langfuse 트레이스 완료
            if batch_trace:
                batch_trace.update(
                    output={
                        "success": True,
                        "sentences_count": saved_count,
                        "total_pages": total_pages,
                        "processing_time": total_processing_time,
                        "llm_time": llm_processing_time,
                        "parse_time": parse_time,
                        "db_time": db_time
                    }
                )
            
            return {
                "success": True,
                "sentences_count": saved_count,
                "backup_path": backup_path,
                "processing_time": total_processing_time,
                "pages_processed": total_pages
            }
            
        except Exception as e:
            error_message = str(e)
            total_time = time.time() - start_time
            logger.error(f"❌ 배치 {batch_number} 처리 실패: {error_message}")
            logger.error(f"📊 처리 시간: {total_time:.2f}초")
            
            # 배치 상태를 failed로 업데이트
            self.session_manager.db_manager.update_batch_status(
                session_id=session_id,
                batch_number=batch_number,
                status="failed",
                error_message=error_message
            )
            
            # Langfuse 트레이스에 에러 기록
            if batch_trace:
                batch_trace.update(
                    output={
                        "success": False,
                        "error": error_message,
                        "processing_time": total_time
                    }
                )
            
            return {
                "success": False,
                "error": error_message
            }
    
    def _save_batch_to_database(
        self,
        sentences: List[Dict[str, Any]],
        source_file: str,
        language: str,
        session_id: str,
        batch_number: int
    ) -> Tuple[int, Dict[int, Dict[str, int]]]:
        """배치 정보와 함께 문장들을 언어별 테이블에 저장하고 페이지별 통계를 반환합니다."""
        if not sentences:
            return 0, {}
        
        # 원본 파일명 추출
        original_filename = self._get_original_filename(source_file)
        
        # 언어에 따른 테이블 선택
        table_name = "korean_sentences" if language == "korean" else "english_sentences"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                saved_count = 0
                
                for idx, sentence_data in enumerate(sentences, 1):
                    try:
                        # 호환성을 위해 문자열과 딕셔너리 모두 지원
                        if isinstance(sentence_data, str):
                            sentence_text = sentence_data
                            source_type = 'text'
                        else:
                            sentence_text = sentence_data.get('text', '')
                            source_type = sentence_data.get('source_type', 'text')
                        
                        cursor.execute(f"""
                            INSERT INTO {table_name} 
                            (sentence, sentence_index, batch_id, batch_number, pdf_file_path, original_filename, source_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (sentence_text, idx, session_id, batch_number, source_file, original_filename, source_type))
                        
                        saved_count += 1
                        
                    except Exception as e:
                        print(f"문장 저장 실패: {sentence_text[:50]}... - {e}")
                
                conn.commit()
                return saved_count, {}
                
        except Exception as e:
            raise Exception(f"배치 데이터베이스 저장 실패: {e}")
    
    def _get_batch_page_range(self, session_id: str, batch_number: int) -> Tuple[int, int]:
        """배치의 페이지 범위를 조회합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT start_page, end_page
                    FROM batch_progress 
                    WHERE session_id = ? AND batch_number = ?
                """, (session_id, batch_number))
                
                result = cursor.fetchone()
                if result:
                    return result[0], result[1]
                else:
                    # 기본값 반환
                    return 1, 1
                    
        except Exception as e:
            print(f"배치 페이지 범위 조회 실패: {e}")
            return 1, 1
    
    def _save_batch_statistics(
        self,
        session_id: str,
        batch_number: int,
        language: str,
        saved_count: int,
        total_pages: int,
        processing_time: float
    ) -> None:
        """배치 처리 통계를 저장합니다."""
        try:
            # 배치 요약 통계 저장
            korean_count = saved_count if language == "korean" else 0
            english_count = saved_count if language == "english" else 0
            
            self.session_manager.db_manager.save_batch_summary_stats(
                session_id=session_id,
                batch_number=batch_number,
                korean_count=korean_count,
                english_count=english_count,
                total_pages=total_pages,
                processing_duration=processing_time
            )
            
        except Exception as e:
            print(f"통계 저장 실패: {e}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태를 조회합니다."""
        return self.session_manager.get_session_status(session_id)
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 세션 목록을 조회합니다."""
        return self.session_manager.list_sessions(limit)
    
    def get_extraction_stats(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 페이지별 추출 통계를 조회합니다."""
        return self.session_manager.db_manager.get_extraction_stats(session_id)
    
    def get_batch_summary_stats(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 배치별 요약 통계를 조회합니다."""
        return self.session_manager.db_manager.get_batch_summary_stats(session_id)
    
    def _detect_language_from_filename(self, filename: str) -> Optional[str]:
        """
        파일명 패턴으로 언어를 자동 감지합니다.
        
        Args:
            filename (str): 파일명
            
        Returns:
            Optional[str]: "korean" 또는 "english", 감지 실패 시 None
        """
        filename_lower = filename.lower()
        if filename_lower.endswith('_kr.pdf'):
            return "korean"
        elif filename_lower.endswith('_en.pdf'):
            return "english"
        else:
            return None
    
    def _is_daily_quota_exceeded(self, error: Exception) -> bool:
        """
        일별 할당량 초과 에러인지 확인합니다.
        
        Args:
            error (Exception): 발생한 예외
            
        Returns:
            bool: 일별 할당량 초과 에러인 경우 True
        """
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
    
    def _is_file_access_error(self, error: Exception) -> bool:
        """
        파일 접근 권한 에러인지 확인합니다.
        
        Args:
            error (Exception): 발생한 예외
            
        Returns:
            bool: 파일 접근 권한 에러인 경우 True
        """
        error_str = str(error).lower()
        file_access_keywords = [
            "you do not have permission to access the file",
            "file not found",
            "file does not exist",
            "403",
            "forbidden",
            "access denied",
            "permission denied"
        ]
        return any(keyword in error_str for keyword in file_access_keywords)
    
    def _is_api_key_invalid(self, error: Exception) -> bool:
        """
        API 키 인증 에러인지 확인합니다.
        
        Args:
            error (Exception): 발생한 예외
            
        Returns:
            bool: API 키 인증 에러인 경우 True
        """
        error_str = str(error).lower()
        auth_keywords = [
            "invalid api key",
            "unauthorized",
            "401",
            "authentication failed",
            "invalid credentials",
            "api key not valid",
            "please provide a valid api key"
        ]
        return any(keyword in error_str for keyword in auth_keywords)
    
    def _load_key_usage(self) -> None:
        """
        JSON 파일에서 API 키 사용 상태를 로드하고 24시간 경과 키는 자동으로 리셋합니다.
        """
        try:
            if not Path(self.usage_file).exists():
                # 파일이 없으면 모든 키를 available 상태로 초기화
                self.key_usage = self._initialize_key_usage()
                self._save_key_usage()
                logger.info(f"📝 API 키 사용량 추적 파일 생성: {self.usage_file}")
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
                logger.info(f"🔄 {reset_count}개 API 키가 24시간 경과로 사용 가능 상태로 리셋됨")
                self._save_key_usage()
            
            logger.info(f"✅ API 키 사용량 상태 로드 완료: {len(self.key_usage)}개 키")
            
        except Exception as e:
            logger.warning(f"⚠️ API 키 사용량 로드 실패, 초기화합니다: {e}")
            self.key_usage = self._initialize_key_usage()
            self._save_key_usage()
    
    def _initialize_key_usage(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 API 키를 사용 가능 상태로 초기화합니다.
        """
        usage = {}
        for api_key in self.api_keys:
            key_id = f"***{api_key[-4:]}"  # 마지막 4자리로 식별
            usage[key_id] = {
                'first_used': None,
                'status': 'available',  # available, active, exhausted
                'reset_time': None
            }
        return usage
    
    def _save_key_usage(self) -> None:
        """
        현재 API 키 사용 상태를 JSON 파일에 저장합니다.
        """
        try:
            data = {
                'keys': self.key_usage,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"⚠️ API 키 사용량 저장 실패: {e}")
    
    def _get_available_key_index(self) -> int:
        """
        사용 가능한 첫 번째 API 키의 인덱스를 반환합니다.
        
        우선순위: available 상태 키 > 24시간 경과 키 > 첫 번째 키 (기본값)
        """
        for i, api_key in enumerate(self.api_keys):
            key_id = f"***{api_key[-4:]}"
            key_info = self.key_usage.get(key_id, {'status': 'available'})
            
            if key_info['status'] == 'available':
                logger.info(f"🔑 사용 가능한 API 키 선택: {key_id} (인덱스 {i})")
                return i
        
        # 사용 가능한 키가 없으면 첫 번째 키 사용 (기본 동작)
        logger.warning(f"⚠️ 사용 가능한 키가 없어 메인 키부터 시작합니다")
        return 0
    
    def _mark_key_exhausted(self, key_index: int) -> None:
        """
        지정된 키를 할당량 소진 상태로 마킹하고 24시간 후 리셋 시간을 설정합니다.
        """
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
        logger.warning(f"⏰ API 키 {key_id} 할당량 소진으로 마킹, 리셋 시간: {reset_time_str}")
    
    def _switch_to_next_key(self) -> bool:
        """
        다음 API 키로 전환합니다.
        
        Returns:
            bool: 전환 성공 시 True, 더 이상 사용할 키가 없으면 False
        """
        # 현재 키를 할당량 소진 상태로 마킹
        self._mark_key_exhausted(self.current_key_index)
        
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.current_api_key = self.api_keys[self.current_key_index]
            
            key_type = "메인" if self.current_key_index == 0 else f"백업 키 {self.current_key_index}"
            logger.warning(f"🔄 다음 API 키로 전환: {key_type} (***...{self.current_api_key[-4:]})")
            
            # genai 클라이언트 재설정
            genai.configure(api_key=self.current_api_key)
            
            return True
        
        logger.error(f"❌ 모든 API 키 ({len(self.api_keys)}개) 사용 완료")
        return False
    
    def _validate_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """
        PDF 파일의 유효성을 검사합니다.
        
        Args:
            file_path (str): PDF 파일 경로
            
        Returns:
            Dict[str, Any]: 검증 결과
        """
        result = {
            "valid": False,
            "error": None,
            "file_size_mb": 0
        }
        
        try:
            file_path_obj = Path(file_path)
            
            # 파일 존재 확인
            if not file_path_obj.exists():
                result["error"] = "파일이 존재하지 않습니다"
                return result
            
            # 읽기 권한 확인
            if not os.access(file_path, os.R_OK):
                result["error"] = "파일 읽기 권한이 없습니다"
                return result
            
            # 파일 크기 확인
            file_size = file_path_obj.stat().st_size
            result["file_size_mb"] = file_size / 1024 / 1024
            
            if file_size == 0:
                result["error"] = "빈 파일입니다"
                return result
            
            # PDF 헤더 확인
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    result["error"] = "유효한 PDF 파일이 아닙니다"
                    return result
            
            result["valid"] = True
            return result
            
        except Exception as e:
            result["error"] = f"파일 검증 중 오류: {str(e)}"
            return result
    
    def process_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        폴더 내 모든 PDF 파일을 순차적으로 처리합니다.
        
        Args:
            folder_path (str): 처리할 폴더 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            start_time = time.time()
            folder_path_obj = Path(folder_path)
            
            # 폴더 존재 확인
            if not folder_path_obj.exists():
                return {
                    "success": False,
                    "error": f"폴더가 존재하지 않습니다: {folder_path}"
                }
            
            if not folder_path_obj.is_dir():
                return {
                    "success": False,
                    "error": f"지정된 경로가 폴더가 아닙니다: {folder_path}"
                }
            
            print(f"📁 폴더 처리 시작: {folder_path}")
            
            # PDF 파일 검색 및 정렬
            pdf_files = sorted(folder_path_obj.glob("*.pdf"))
            
            if not pdf_files:
                return {
                    "success": False,
                    "error": f"폴더에 PDF 파일이 없습니다: {folder_path}"
                }
            
            print(f"📄 발견된 PDF 파일: {len(pdf_files)}개")
            
            # 처리 결과 추적
            total_files = len(pdf_files)
            processed_files = 0
            failed_files = 0
            total_sentences = 0
            processing_results = []
            
            # 각 파일 순차 처리
            for i, pdf_file in enumerate(pdf_files, 1):
                file_name = pdf_file.name
                print(f"\n--- [{i}/{total_files}] 처리 중: {file_name} ---")
                
                try:
                    # 언어 자동 감지
                    language = self._detect_language_from_filename(file_name)
                    if not language:
                        print(f"⚠️  언어를 감지할 수 없습니다. 파일명이 '_kr.pdf' 또는 '_en.pdf'로 끝나야 합니다: {file_name}")
                        failed_files += 1
                        processing_results.append({
                            "file": file_name,
                            "status": "failed",
                            "error": "언어 감지 실패",
                            "sentences": 0
                        })
                        continue
                    
                    print(f"🔍 감지된 언어: {language}")
                    
                    # 파일 검증
                    validation_result = self._validate_pdf_file(str(pdf_file))
                    if not validation_result["valid"]:
                        print(f"❌ 파일 검증 실패: {validation_result['error']}")
                        failed_files += 1
                        processing_results.append({
                            "file": file_name,
                            "status": "failed",
                            "error": validation_result["error"],
                            "sentences": 0
                        })
                        continue
                    
                    print(f"✅ 파일 검증 완료 ({validation_result['file_size_mb']:.2f} MB)")
                    
                    # PDF 배치 처리
                    print(f"🚀 {language} 문장 배치 추출 시작...")
                    result = self.process_pdf_batch(str(pdf_file), language)
                    
                    if result["success"]:
                        sentences_count = result["total_sentences"]
                        total_sentences += sentences_count
                        processed_files += 1
                        print(f"✅ 완료: {sentences_count}개 문장 추출 ({result['processed_batches']}개 배치)")
                        
                        processing_results.append({
                            "file": file_name,
                            "status": "success",
                            "language": language,
                            "sentences": sentences_count,
                            "batches": result['processed_batches'],
                            "session_id": result.get('session_id', '')
                        })
                    else:
                        failed_files += 1
                        print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
                        
                        processing_results.append({
                            "file": file_name,
                            "status": "failed",
                            "error": result.get("error", "배치 처리 실패"),
                            "sentences": 0
                        })
                    
                    # 진행률 표시
                    progress = (i / total_files) * 100
                    elapsed_seconds = time.time() - start_time
                    estimated_total_seconds = elapsed_seconds * total_files / i if i > 0 else elapsed_seconds
                    remaining_seconds = estimated_total_seconds - elapsed_seconds
                    
                    print(f"📊 진행률: {progress:.1f}% | 성공: {processed_files} | 실패: {failed_files}")
                    print(f"⏱️  경과 시간: {format_duration(elapsed_seconds)} | 예상 남은 시간: {format_duration(remaining_seconds)}")
                    
                except KeyboardInterrupt:
                    print("\n🛑 사용자에 의해 중단되었습니다.")
                    break
                except Exception as e:
                    failed_files += 1
                    error_msg = str(e)
                    print(f"❌ 예외 발생: {error_msg}")
                    
                    processing_results.append({
                        "file": file_name,
                        "status": "failed", 
                        "error": error_msg,
                        "sentences": 0
                    })
            
            # 최종 결과
            end_time = time.time()
            total_duration = format_duration(end_time - start_time)
            
            print(f"\n🎯 폴더 처리 완료!")
            print(f"📁 폴더: {folder_path}")
            print(f"📄 총 파일: {total_files}개")
            print(f"✅ 성공: {processed_files}개")
            print(f"❌ 실패: {failed_files}개") 
            print(f"📝 총 추출 문장: {total_sentences:,}개")
            print(f"⏱️  총 소요 시간: {total_duration}")
            
            return {
                "success": True,
                "folder": folder_path,
                "total_files": total_files,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "total_sentences": total_sentences,
                "duration": total_duration,
                "results": processing_results,
                "message": f"폴더 처리 완료: {processed_files}/{total_files}개 파일 성공, {total_sentences:,}개 문장 추출"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"폴더 처리 중 오류 발생: {str(e)}"
            }


if __name__ == "__main__":
    # 예시 실행 코드
    import sys
    
    # 기본값 설정
    default_pdf = "document.pdf"
    default_language = "korean"
    default_db = "corpus.db"
    
    # 커맨드라인 인자에서 파일 경로 가져오기
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    language = sys.argv[2] if len(sys.argv) > 2 else default_language
    db_file = sys.argv[3] if len(sys.argv) > 3 else default_db
    
    try:
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("오류: GOOGLE_API_KEY 환경변수를 설정해주세요.")
            sys.exit(1)
        
        print("=== PDF to 병렬 코퍼스 변환기 ===")
        print(f"입력 파일: {pdf_file}")
        print(f"언어: {language}")
        print(f"데이터베이스: {db_file}")
        print()
        
        # 변환기 생성 및 실행
        converter = PDFToCorpusConverter(api_key, db_file)
        result = converter.process_pdf_to_corpus(pdf_file, language)
        
        # 결과 출력
        print("\n=== 처리 결과 ===")
        print(f"상태: {result['status']}")
        print(f"메시지: {result['message']}")
        
        if result['status'] == 'success':
            print(f"추출된 문장 수: {result['extracted_sentences']}개")
            print(f"데이터베이스: {result['database']}")
            
            # 총 문장 수 출력
            total_count = converter.get_corpus_count("total")
            korean_count = converter.get_corpus_count("korean")
            english_count = converter.get_corpus_count("english")
            print(f"데이터베이스 총 문장 수: {total_count}개 (한국어: {korean_count}개, 영어: {english_count}개)")
            
            print(f"\n✅ {language} 문장 추출 완료")
        
    except KeyboardInterrupt:
        print("\n작업이 취소되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)
