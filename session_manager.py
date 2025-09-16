"""
배치 처리 세션 관리 시스템

PDF 파일의 배치 처리 세션을 관리하고 진행 상황을 추적합니다.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from database import DatabaseManager
from utils import PDFSplitter, JSONBackupManager, format_duration, estimate_processing_time


class SessionManager:
    """배치 처리 세션 관리 클래스"""
    
    def __init__(self, db_path: str = "corpus.db"):
        self.db_manager = DatabaseManager(db_path)
        self.pdf_splitter = PDFSplitter()
        self.json_backup = JSONBackupManager()
    
    def create_batch_session(
        self,
        pdf_path: str,
        language: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        새로운 배치 처리 세션을 생성합니다.
        
        Args:
            pdf_path (str): 처리할 PDF 파일 경로
            language (str): 추출할 언어
            batch_size (int): 배치 크기
            
        Returns:
            Dict[str, Any]: 세션 생성 결과
        """
        try:
            # PDF 파일 존재 확인
            if not Path(pdf_path).exists():
                return {
                    "success": False,
                    "error": f"PDF 파일을 찾을 수 없습니다: {pdf_path}"
                }
            
            # PDF 페이지 수 확인
            total_pages = self.pdf_splitter.get_pdf_page_count(pdf_path)
            if total_pages == 0:
                return {
                    "success": False,
                    "error": "PDF 파일에 페이지가 없습니다"
                }
            
            # 처리 시간 예상
            total_batches, estimated_time = estimate_processing_time(total_pages, batch_size)
            
            # 세션 생성
            session_id = self.db_manager.create_processing_session(
                source_file=pdf_path,
                language=language,
                total_pages=total_pages,
                batch_size=batch_size
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "pdf_path": pdf_path,
                "language": language,
                "total_pages": total_pages,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "estimated_time": estimated_time,
                "message": f"배치 처리 세션 생성 완료 ({total_batches}개 배치, {estimated_time})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"세션 생성 실패: {e}"
            }
    
    def start_batch_processing(self, session_id: str) -> Dict[str, Any]:
        """
        배치 처리를 시작합니다.
        
        Args:
            session_id (str): 세션 ID
            
        Returns:
            Dict[str, Any]: 처리 시작 결과
        """
        try:
            # 세션 정보 조회
            progress = self.db_manager.get_session_progress(session_id)
            if "error" in progress:
                return {
                    "success": False,
                    "error": progress["error"]
                }
            
            pdf_path = progress["source_file"]
            batch_size = progress["batch_size"]
            
            print(f"배치 처리 시작: {Path(pdf_path).name}")
            print(f"  세션 ID: {session_id}")
            print(f"  총 페이지: {progress['total_pages']:,}페이지")
            print(f"  배치 크기: {batch_size}페이지")
            print(f"  총 배치: {progress['total_batches']}개")
            print()
            
            # PDF 분할
            print("PDF 파일 분할 중...")
            batch_info_list = self.pdf_splitter.split_pdf_into_batches(
                pdf_path=pdf_path,
                batch_size=batch_size,
                session_id=session_id
            )
            
            # 각 배치의 PDF 파일 경로를 데이터베이스에 저장
            for batch_info in batch_info_list:
                self.db_manager.update_batch_status(
                    session_id=session_id,
                    batch_number=batch_info["batch_number"],
                    status="ready",
                    pdf_file_path=batch_info["file_path"]
                )
            
            return {
                "success": True,
                "session_id": session_id,
                "batch_files_created": len(batch_info_list),
                "batch_info": batch_info_list,
                "message": f"PDF 분할 완료: {len(batch_info_list)}개 배치 파일 생성"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"배치 처리 시작 실패: {e}"
            }
    
    def process_single_batch(
        self, 
        session_id: str, 
        batch_number: int,
        extractor_func,  # PDF 처리 함수
        **extractor_kwargs
    ) -> Dict[str, Any]:
        """
        단일 배치를 처리합니다.
        
        Args:
            session_id (str): 세션 ID
            batch_number (int): 배치 번호
            extractor_func: PDF 추출 함수
            **extractor_kwargs: 추출 함수에 전달할 추가 인수들
            
        Returns:
            Dict[str, Any]: 배치 처리 결과
        """
        try:
            # 배치 상태를 진행 중으로 업데이트
            self.db_manager.update_batch_status(
                session_id=session_id,
                batch_number=batch_number,
                status="in_progress"
            )
            
            # 세션 정보 조회
            progress = self.db_manager.get_session_progress(session_id)
            if "error" in progress:
                return {"success": False, "error": progress["error"]}
            
            # 배치 PDF 파일 경로 조회
            # 실제로는 batch_progress 테이블에서 pdf_file_path를 조회해야 함
            # 여기서는 간단히 구현
            language = progress["language"]
            
            # PDF 처리 함수 호출 (실제 LLM 처리)
            result = extractor_func(
                session_id=session_id,
                batch_number=batch_number,
                language=language,
                **extractor_kwargs
            )
            
            if result["success"]:
                # 성공 시 배치 상태 업데이트
                self.db_manager.update_batch_status(
                    session_id=session_id,
                    batch_number=batch_number,
                    status="completed",
                    sentences_extracted=result.get("sentences_count", 0),
                    json_backup_path=result.get("backup_path", "")
                )
                
                return {
                    "success": True,
                    "batch_number": batch_number,
                    "sentences_extracted": result.get("sentences_count", 0),
                    "message": f"배치 {batch_number} 처리 완료"
                }
            else:
                # 실패 시 배치 상태 업데이트
                self.db_manager.update_batch_status(
                    session_id=session_id,
                    batch_number=batch_number,
                    status="failed",
                    error_message=result.get("error", "알 수 없는 오류")
                )
                
                return {
                    "success": False,
                    "batch_number": batch_number,
                    "error": result.get("error", "배치 처리 실패"),
                    "message": f"배치 {batch_number} 처리 실패"
                }
                
        except Exception as e:
            # 예외 발생 시 배치 상태 업데이트
            self.db_manager.update_batch_status(
                session_id=session_id,
                batch_number=batch_number,
                status="failed",
                error_message=str(e)
            )
            
            return {
                "success": False,
                "batch_number": batch_number,
                "error": str(e),
                "message": f"배치 {batch_number} 처리 중 예외 발생"
            }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션의 현재 상태를 조회합니다."""
        progress = self.db_manager.get_session_progress(session_id)
        
        if "error" in progress:
            return progress
        
        # 진행률 계산
        total_batches = progress["total_batches"]
        completed_batches = progress["completed_batches"]
        failed_batches = progress["failed_batches_count"]
        pending_batches = progress["pending_batches"]
        
        progress_percentage = (completed_batches / total_batches) * 100 if total_batches > 0 else 0
        
        # 상태 요약
        status_summary = {
            "session_id": session_id,
            "pdf_file": Path(progress["source_file"]).name,
            "language": progress["language"],
            "total_pages": progress["total_pages"],
            "total_batches": total_batches,
            "completed_batches": completed_batches,
            "failed_batches": failed_batches,
            "pending_batches": pending_batches,
            "progress_percentage": round(progress_percentage, 1),
            "session_status": progress["session_status"],
            "started_at": progress["started_at"],
            "completed_at": progress["completed_at"],
            "created_sentences": progress["created_sentences"],
            "failed_batch_details": progress["failed_batches"]
        }
        
        return status_summary
    
    def retry_failed_batches(self, session_id: str) -> List[int]:
        """실패한 배치들의 번호를 반환합니다."""
        failed_batches = self.db_manager.get_failed_batches(session_id)
        return [batch["batch_number"] for batch in failed_batches]
    
    def get_pending_batches(self, session_id: str) -> List[int]:
        """대기 중인 배치들의 번호를 반환합니다."""
        return self.db_manager.get_pending_batches(session_id)
    
    def get_incomplete_batches(self, session_id: str) -> List[int]:
        """완료되지 않은 모든 배치들의 번호를 반환합니다."""
        return self.db_manager.get_incomplete_batches(session_id)
    
    def complete_session(self, session_id: str) -> Dict[str, Any]:
        """세션을 완료 처리합니다."""
        try:
            # 세션 진행 상황 조회
            progress = self.db_manager.get_session_progress(session_id)
            if "error" in progress:
                return {"success": False, "error": progress["error"]}
            
            # 총 추출된 문장 수 계산 (실제로는 corpus 테이블에서 조회해야 함)
            total_sentences = progress.get("created_sentences", 0)
            
            # 세션 완료
            self.db_manager.complete_session(session_id, total_sentences)
            
            # 임시 파일 정리 (재시도를 위해 보존)
            # self.pdf_splitter.cleanup_temp_files()
            
            return {
                "success": True,
                "session_id": session_id,
                "total_sentences": total_sentences,
                "message": f"세션 완료: {total_sentences}개 문장 추출"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"세션 완료 처리 실패: {e}"
            }
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 세션들의 목록을 반환합니다."""
        return self.db_manager.list_recent_sessions(limit)
    
    def cleanup_session_files(self, session_id: str, cleanup_json: bool = False) -> Dict[str, Any]:
        """세션과 관련된 임시 파일들을 정리합니다."""
        try:
            cleanup_count = 0
            
            # PDF 임시 파일 정리 (재시도를 위해 보존)
            # self.pdf_splitter.cleanup_temp_files()
            # cleanup_count += len(self.pdf_splitter.created_files)
            
            # JSON 백업 파일 정리 (선택적)
            if cleanup_json:
                backup_files = self.json_backup.find_backup_files(session_id)
                for backup_file in backup_files:
                    try:
                        os.remove(backup_file)
                        cleanup_count += 1
                    except Exception as e:
                        print(f"백업 파일 삭제 실패: {backup_file} - {e}")
            
            return {
                "success": True,
                "cleaned_files": cleanup_count,
                "message": f"세션 파일 정리 완료: {cleanup_count}개 파일 삭제"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"파일 정리 실패: {e}"
            }


def print_session_progress(session_status: Dict[str, Any]) -> None:
    """세션 진행 상황을 보기 좋게 출력합니다."""
    print(f"\n=== 세션 진행 상황 ===")
    print(f"세션 ID: {session_status['session_id']}")
    print(f"파일: {session_status['pdf_file']}")
    print(f"언어: {session_status['language']}")
    print(f"총 페이지: {session_status['total_pages']:,}페이지")
    print(f"진행률: {session_status['progress_percentage']:.1f}% "
          f"({session_status['completed_batches']}/{session_status['total_batches']} 배치)")
    
    if session_status['failed_batches'] > 0:
        print(f"실패한 배치: {session_status['failed_batches']}개")
    
    if session_status['created_sentences'] > 0:
        print(f"추출된 문장: {session_status['created_sentences']:,}개")
    
    print(f"상태: {session_status['session_status']}")
    print(f"시작 시간: {session_status['started_at']}")
    
    if session_status['completed_at']:
        print(f"완료 시간: {session_status['completed_at']}")
    
    print("=" * 50)