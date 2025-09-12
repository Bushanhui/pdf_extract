#!/usr/bin/env python3
"""
PDF to 코퍼스 변환기 - 커맨드라인 인터페이스

사용법:
    python cli.py input.pdf --language korean
    python cli.py input.pdf --language english --prompt-type technical
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from main import PDFToCorpusConverter

# 환경 변수 로드
load_dotenv()


def create_parser() -> argparse.ArgumentParser:
    """커맨드라인 인자 파서를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="PDF 파일에서 한국어 또는 영어 문장을 추출하여 병렬 코퍼스 데이터베이스에 저장합니다.",
        epilog="""
사용 예시:
  단일 파일 처리:
    %(prog)s korean_doc.pdf --language korean         # 한국어 문장 추출
    %(prog)s english_doc.pdf --language english       # 영어 문장 추출
    %(prog)s input.pdf --language korean --db-path output.db
  
  폴더 일괄 처리:
    %(prog)s --folder data_kr                         # data_kr 폴더 내 모든 *_kr.pdf 파일 처리
    %(prog)s --folder data_en                         # data_en 폴더 내 모든 *_en.pdf 파일 처리
    %(prog)s --folder /path/to/pdfs --verbose         # 상세 진행 정보 표시

배치 처리 예시:
  %(prog)s large.pdf --language korean --batch-processing

환경 변수:
  GOOGLE_API_KEY           Google AI API 키 (필수)
  GOOGLE_API_KEY_BACKUP    백업 API 키 (하위 호환성)
  GOOGLE_API_KEY_BACKUP_1  백업 API 키 #1 (권장)
  GOOGLE_API_KEY_BACKUP_2  백업 API 키 #2 (권장)
  GOOGLE_API_KEY_BACKUP_3  백업 API 키 #3 (선택)
  ...                      (필요에 따라 추가)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_pdf",
        nargs='?',
        help="분석할 PDF 파일 경로"
    )
    
    parser.add_argument(
        "--folder",
        help="처리할 PDF 폴더 경로 (폴더 내 모든 PDF 파일을 순차 처리)"
    )
    
    parser.add_argument(
        "--language",
        choices=['korean', 'english'],
        required=False,
        help="PDF 언어 지정 (korean: 한국어 문장 추출, english: 영어 문장 추출)"
    )
    
    parser.add_argument(
        "--db-path",
        default="corpus.db",
        help="SQLite 데이터베이스 파일 경로 (기본값: corpus.db)"
    )
    
    parser.add_argument(
        "--api-key",
        help="Google AI API 키 (환경변수 GOOGLE_API_KEY보다 우선)"
    )
    
    
    
    
    
    
    parser.add_argument(
        "--count",
        action="store_true",
        help="데이터베이스에 저장된 총 문장 수 표시"
    )
    
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세한 출력 표시"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    # 배치 처리 관련 옵션들
    parser.add_argument(
        "--batch-processing",
        action="store_true",
        help="배치 처리 모드 활성화 (대용량 PDF 파일용)"
    )
    
    # 재시도 관련 옵션들
    parser.add_argument(
        "--retry-failed-all",
        action="store_true",
        help="모든 실패한 배치들을 재시도"
    )
    
    parser.add_argument(
        "--retry-session",
        help="특정 세션의 배치들을 재시도 (세션 ID 필요)"
    )
    
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="실패한 배치들만 재시도 (--retry-session과 함께 사용)"
    )
    
    parser.add_argument(
        "--retry-batch",
        type=int,
        help="특정 배치 번호를 재시도 (--retry-session과 함께 사용)"
    )
    
    return parser




def main():
    """메인 CLI 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    
    # API 키 확인 (재시도 명령어에는 필요)
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key and not args.count and not (args.retry_failed_all or args.retry_session):
        print("오류: Google AI API 키가 필요합니다.")
        print("다음 중 하나의 방법으로 API 키를 제공해주세요:")
        print("  1. --api-key 옵션 사용")
        print("  2. GOOGLE_API_KEY 환경변수 설정")
        print("\nAPI 키 발급: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    
    try:
        # 변환기 생성 (카운트 모드에서는 API 키 없이도 가능)
        converter = PDFToCorpusConverter(api_key, args.db_path)
        
        
        # 재시도 모드들 처리
        if args.retry_failed_all:
            # 모든 실패한 배치 재시도
            result = converter.retry_all_failed_batches(args.language)
            if result["success"]:
                print(f"🎉 전체 실패 배치 재시도 완료!")
                print(f"📊 재시도한 배치: {result['retried_batches']}개")
                print(f"📊 성공한 배치: {result['successful_batches']}개")
                print(f"📊 여전히 실패한 배치: {result['still_failed_batches']}개")
                print(f"📝 추출된 문장: {result['total_sentences']:,}개")
                print(f"⏱️  총 소요 시간: {result['duration']}")
            else:
                print(f"❌ 전체 실패 배치 재시도 실패: {result['error']}")
                sys.exit(1)
            return
        
        if args.retry_session:
            # 특정 세션 재시도
            if args.retry_batch:
                # 특정 배치만 재시도
                result = converter.retry_specific_batch(args.retry_session, args.retry_batch)
            else:
                # 세션의 실패한 배치들 재시도
                result = converter.retry_session_batches(args.retry_session, args.failed_only)
            
            if result["success"]:
                print(f"🎉 세션 재시도 완료!")
                print(f"📁 세션 ID: {args.retry_session}")
                print(f"📊 재시도한 배치: {result['retried_batches']}개")
                print(f"📊 성공한 배치: {result['successful_batches']}개")
                print(f"📝 추출된 문장: {result['total_sentences']:,}개")
                print(f"⏱️  총 소요 시간: {result['duration']}")
            else:
                print(f"❌ 세션 재시도 실패: {result['error']}")
                sys.exit(1)
            return

        # 문장 수 표시 모드
        if args.count:
            total_count = converter.get_corpus_count("total")
            korean_count = converter.get_corpus_count("korean")
            english_count = converter.get_corpus_count("english")
            
            print(f"데이터베이스에 저장된 총 문장 수: {total_count}개")
            print(f"  - 한국어: {korean_count}개")
            print(f"  - 영어: {english_count}개")
            print(f"데이터베이스 경로: {args.db_path}")
            return
        
        
        # 폴더 처리 모드 vs 단일 파일 처리 모드 구분
        if args.folder:
            # 폴더 처리 모드
            if not Path(args.folder).exists():
                print(f"오류: 지정된 폴더를 찾을 수 없습니다: {args.folder}")
                sys.exit(1)
            
            if args.verbose:
                print("=== PDF 폴더 일괄 처리 모드 ===")
                print(f"처리 폴더: {args.folder}")
                print(f"데이터베이스: {args.db_path}")
                print(f"사용 모델: gemini-2.5-flash")
                print(f"배치 크기: 10페이지 (PDF 분할 단위)")
                print("언어: 파일명 패턴으로 자동 감지 (_kr.pdf → korean, _en.pdf → english)")
                
                # API 키 상태 표시
                api_key_count = 1  # 메인 키
                backup_index = 1
                while os.getenv(f"GOOGLE_API_KEY_BACKUP_{backup_index}"):
                    api_key_count += 1
                    backup_index += 1
                
                # 기존 단일 백업키도 확인
                if os.getenv("GOOGLE_API_KEY_BACKUP"):
                    api_key_count += 1
                
                if api_key_count > 1:
                    print(f"API 키: {api_key_count}개 설정됨 (일별 할당량 자동 전환)")
                else:
                    print(f"API 키: 1개만 설정됨 (백업 키 권장: GOOGLE_API_KEY_BACKUP_1, GOOGLE_API_KEY_BACKUP_2, ...)")
                print()
            
            # 폴더 처리 실행
            print("🚀 폴더 배치 처리 모드로 실행합니다...")
            print("📦 각 파일을 10페이지씩 배치 분할하여 처리합니다.")
            
            result = converter.process_folder(args.folder)
            
            if result["success"]:
                print(f"\n🎉 폴더 처리 완료!")
                print(f"📁 처리된 폴더: {result['folder']}")
                print(f"📄 총 파일: {result['total_files']}개")
                print(f"✅ 성공: {result['processed_files']}개")
                print(f"❌ 실패: {result['failed_files']}개")
                print(f"📦 총 처리 배치: {sum(r.get('batches', 0) for r in result['results'] if r['status'] == 'success')}개")
                print(f"📝 총 추출 문장: {result['total_sentences']:,}개")
                print(f"⏱️  총 소요 시간: {result['duration']}")
                
                # 실패한 파일이 있는 경우 상세 정보 표시
                if result['failed_files'] > 0 and args.verbose:
                    print(f"\n⚠️  실패한 파일 목록:")
                    for file_result in result['results']:
                        if file_result['status'] == 'failed':
                            print(f"  - {file_result['file']}: {file_result['error']}")
            else:
                print(f"❌ 폴더 처리 실패: {result['error']}")
                sys.exit(1)
            
            return
        
        # 단일 PDF 변환 모드
        if not args.input_pdf:
            print("오류: 입력 PDF 파일 또는 폴더를 지정해야 합니다.")
            print("사용법: ")
            print("  단일 파일: python cli.py <PDF파일> --language <언어> [옵션]")
            print("  폴더 처리: python cli.py --folder <폴더경로> [옵션]")
            print("자세한 도움말: python cli.py --help")
            sys.exit(1)
        
        # 일반 PDF 처리 모드
        if not args.language:
            print("오류: 언어를 지정해야 합니다.")
            print("--language korean 또는 --language english를 사용하세요.")
            sys.exit(1)
        
        # 입력 파일 존재 확인
        if not Path(args.input_pdf).exists():
            print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input_pdf}")
            sys.exit(1)
        
        
        if args.verbose:
            print("=== PDF to 병렬 코퍼스 변환기 ===")
            print(f"입력 파일: {args.input_pdf}")
            print(f"언어: {args.language}")
            print(f"데이터베이스: {args.db_path}")
            print(f"사용 모델: gemini-2.5-flash")
            print(f"프롬프트: 기본 프롬프트")
            if args.batch_processing:
                print(f"배치 처리: 활성화 (배치 크기: 10페이지)")
            
            # API 키 상태 표시
            api_key_count = 1  # 메인 키
            backup_index = 1
            while os.getenv(f"GOOGLE_API_KEY_BACKUP_{backup_index}"):
                api_key_count += 1
                backup_index += 1
            
            # 기존 단일 백업키도 확인
            if os.getenv("GOOGLE_API_KEY_BACKUP"):
                api_key_count += 1
            
            if api_key_count > 1:
                print(f"API 키: {api_key_count}개 설정됨 (일별 할당량 자동 전환)")
            else:
                print(f"API 키: 1개만 설정됨 (백업 키 권장: GOOGLE_API_KEY_BACKUP_1, GOOGLE_API_KEY_BACKUP_2, ...)")
            print()
        
        # 배치 처리 모드
        if args.batch_processing:
            print("🚀 배치 처리 모드로 실행합니다...")
            
            result = converter.process_pdf_batch(
                pdf_path=args.input_pdf,
                language=args.language
            )
            
            if result["success"]:
                print(f"\n🎉 배치 처리 완료!")
                print(f"📁 세션 ID: {result['session_id']}")
                print(f"📊 처리된 배치: {result['processed_batches']}개")
                print(f"📊 실패한 배치: {result['failed_batches']}개")
                print(f"📊 추출된 문장: {result['total_sentences']:,}개")
                print(f"⏱️  총 소요 시간: {result['duration']}")
                
                if result['failed_batches'] > 0:
                    print(f"\n⚠️  실패한 배치가 있습니다. 다음 명령으로 재시도할 수 있습니다:")
                    print(f"python cli.py --retry-failed {result['session_id']}")
            else:
                print(f"❌ 배치 처리 실패: {result['error']}")
                sys.exit(1)
            
            return
        
        # 일반 변환 실행
        result = converter.process_pdf_to_corpus(
            pdf_path=args.input_pdf,
            language=args.language
        )
        
        # 결과 출력
        if args.verbose:
            print("\n=== 처리 결과 ===")
            print(f"상태: {result['status']}")
            print(f"메시지: {result['message']}")
            
            if result['status'] == 'success':
                print(f"추출된 {result['language']} 문장 수: {result['extracted_sentences']}개")
                print(f"데이터베이스: {result['database']}")
                    
                # 총 문장 수 출력
                total_count = converter.get_corpus_count("total")
                korean_count = converter.get_corpus_count("korean")
                english_count = converter.get_corpus_count("english")
                print(f"데이터베이스 총 문장 수: {total_count}개 (한국어: {korean_count}개, 영어: {english_count}개)")
                
                print(f"\n✅ {result['language']} 문장 추출 완료")
        else:
            if result['status'] == 'success':
                print(f"✅ 완료: {result['extracted_sentences']}개 {result['language']} 문장이 {args.db_path}에 저장되었습니다.")
                
            else:
                print(f"❌ 실패: {result['message']}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n작업이 취소되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
