"""
PDF 처리 및 배치 작업을 위한 유틸리티 함수들

PDF 파일 분할, 임시 파일 관리, JSON 백업 등의 기능을 제공합니다.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        raise ImportError("pypdf 또는 PyPDF2 패키지가 필요합니다. pip install pypdf 또는 pip install PyPDF2로 설치해주세요.")


class PDFSplitter:
    """PDF 파일을 배치 단위로 분할하는 클래스"""
    
    def __init__(self, temp_dir: str = "temp_pdfs"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.created_files = []
    
    def get_pdf_page_count(self, pdf_path: str) -> int:
        """PDF 파일의 총 페이지 수를 반환합니다."""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as e:
            raise Exception(f"PDF 페이지 수 조회 실패: {e}")
    
    def split_pdf_into_batches(
        self, 
        pdf_path: str, 
        batch_size: int = 50,
        session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        PDF 파일을 배치 단위로 분할합니다.
        
        Args:
            pdf_path (str): 분할할 PDF 파일 경로
            batch_size (int): 배치 크기 (페이지 수)
            session_id (str): 세션 ID (파일명에 포함)
            
        Returns:
            List[Dict[str, Any]]: 배치 정보 리스트
        """
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            if total_pages == 0:
                raise Exception("PDF 파일에 페이지가 없습니다.")
            
            batch_info = []
            source_name = Path(pdf_path).stem
            session_prefix = f"{session_id}_" if session_id else ""
            
            print(f"PDF 분할 시작: {total_pages}페이지 → {batch_size}페이지씩 분할")
            
            for batch_num in range(1, (total_pages + batch_size - 1) // batch_size + 1):
                start_page = (batch_num - 1) * batch_size
                end_page = min(start_page + batch_size, total_pages)
                
                # 배치 PDF 파일 생성
                writer = PdfWriter()
                for page_idx in range(start_page, end_page):
                    writer.add_page(reader.pages[page_idx])
                
                # 배치 파일명 생성
                batch_filename = f"{session_prefix}{source_name}_batch_{batch_num:03d}_pages_{start_page+1}-{end_page}.pdf"
                batch_filepath = self.temp_dir / batch_filename
                
                # 배치 PDF 저장
                with open(batch_filepath, 'wb') as output_file:
                    writer.write(output_file)
                
                self.created_files.append(str(batch_filepath))
                
                batch_info.append({
                    "batch_number": batch_num,
                    "start_page": start_page + 1,
                    "end_page": end_page,
                    "page_count": end_page - start_page,
                    "file_path": str(batch_filepath),
                    "file_size": batch_filepath.stat().st_size
                })
                
                print(f"  배치 {batch_num}: {start_page+1}-{end_page}페이지 → {batch_filename}")
            
            print(f"PDF 분할 완료: {len(batch_info)}개 배치 생성")
            return batch_info
            
        except Exception as e:
            # 오류 발생 시 생성된 파일들 정리 (재시도를 위해 보존)
            # self.cleanup_temp_files()
            raise Exception(f"PDF 분할 실패: {e}")
    
    def cleanup_temp_files(self) -> None:
        """생성된 임시 PDF 파일들을 삭제합니다."""
        deleted_count = 0
        for file_path in self.created_files:
            try:
                if Path(file_path).exists():
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                print(f"파일 삭제 실패: {file_path} - {e}")
        
        self.created_files.clear()
        
        if deleted_count > 0:
            print(f"임시 파일 정리 완료: {deleted_count}개 파일 삭제")
    
    def cleanup_all_temp_files(self) -> None:
        """임시 디렉토리의 모든 파일을 삭제합니다."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
                print(f"임시 디렉토리 전체 정리 완료: {self.temp_dir}")
        except Exception as e:
            print(f"임시 디렉토리 정리 실패: {e}")


class JSONBackupManager:
    """JSON 백업 파일 관리 클래스"""
    
    def __init__(self, backup_dir: str = "json_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def save_llm_response(
        self, 
        session_id: str, 
        batch_number: int, 
        response_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        LLM 응답을 JSON 파일로 백업합니다.
        
        Args:
            session_id (str): 세션 ID
            batch_number (int): 배치 번호
            response_text (str): LLM 응답 텍스트
            metadata (Dict): 추가 메타데이터
            
        Returns:
            str: 백업 파일 경로
        """
        try:
            # 백업 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{session_id}_batch_{batch_number:03d}_{timestamp}.json"
            backup_filepath = self.backup_dir / backup_filename
            
            # 백업 데이터 구성
            backup_data = {
                "session_id": session_id,
                "batch_number": batch_number,
                "timestamp": timestamp,
                "response_text": response_text,
                "metadata": metadata or {},
                "backup_created_at": datetime.now().isoformat()
            }
            
            # JSON 파일로 저장
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            print(f"JSON 백업 저장: {backup_filename}")
            return str(backup_filepath)
            
        except Exception as e:
            print(f"JSON 백업 저장 실패: {e}")
            return ""
    
    def load_backup_response(self, backup_filepath: str) -> Optional[Dict[str, Any]]:
        """백업된 JSON 파일을 로드합니다."""
        try:
            with open(backup_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 백업 로드 실패: {e}")
            return None
    
    def find_backup_files(self, session_id: str) -> List[str]:
        """특정 세션의 백업 파일들을 찾습니다."""
        try:
            pattern = f"{session_id}_batch_*.json"
            backup_files = list(self.backup_dir.glob(pattern))
            return [str(f) for f in sorted(backup_files)]
        except Exception as e:
            print(f"백업 파일 검색 실패: {e}")
            return []
    
    def cleanup_old_backups(self, days: int = 30) -> None:
        """오래된 백업 파일들을 정리합니다."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            deleted_count = 0
            for backup_file in self.backup_dir.glob("*.json"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"오래된 백업 파일 정리: {deleted_count}개 파일 삭제")
                
        except Exception as e:
            print(f"백업 파일 정리 실패: {e}")


def format_file_size(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 형태로 포맷합니다."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def format_duration(duration_seconds: float) -> str:
    """작업 시간을 읽기 쉬운 형태로 포맷합니다."""
    if duration_seconds < 1.0:
        return f"{duration_seconds:.2f}초"
    
    total_seconds = int(duration_seconds)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"


def safe_filename(filename: str) -> str:
    """파일명에서 안전하지 않은 문자들을 제거합니다."""
    import re
    # 윈도우와 유닉스에서 안전하지 않은 문자들 제거
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 연속된 언더스코어 제거
    safe_name = re.sub(r'_+', '_', safe_name)
    # 앞뒤 공백과 점 제거
    safe_name = safe_name.strip(' .')
    return safe_name


def estimate_processing_time(total_pages: int, batch_size: int, avg_batch_time: float = 30.0) -> Tuple[int, str]:
    """예상 처리 시간을 계산합니다."""
    total_batches = (total_pages + batch_size - 1) // batch_size
    estimated_seconds = total_batches * avg_batch_time
    
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    
    if hours > 0:
        time_str = f"약 {hours}시간 {minutes}분"
    elif minutes > 0:
        time_str = f"약 {minutes}분"
    else:
        time_str = "1분 이내"
    
    return total_batches, time_str