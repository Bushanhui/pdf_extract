"""
데이터베이스 스키마 및 관리 모듈

배치 처리 세션 관리와 진행 상황 추적을 위한 데이터베이스 스키마를 정의합니다.
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


class DatabaseManager:
    """데이터베이스 스키마 관리 클래스"""
    
    def __init__(self, db_path: str = "corpus.db"):
        self.db_path = db_path
        self._init_all_tables()
    
    def _init_all_tables(self) -> None:
        """모든 테이블을 초기화합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 한국어 문장 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS korean_sentences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sentence TEXT NOT NULL,
                        source_type TEXT DEFAULT 'text',
                        sentence_index INTEGER NOT NULL,
                        batch_number INTEGER,
                        batch_id TEXT,
                        original_filename TEXT,
                        pdf_file_path TEXT,
                        created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours'))
                    )
                """)
                
                # 영어 문장 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS english_sentences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sentence TEXT NOT NULL,
                        source_type TEXT DEFAULT 'text',
                        sentence_index INTEGER NOT NULL,
                        batch_number INTEGER,
                        batch_id TEXT,
                        original_filename TEXT,
                        pdf_file_path TEXT,
                        created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours'))
                    )
                """)
                
                
                # 배치 처리 세션 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_sessions (
                        id TEXT PRIMARY KEY,
                        source_file TEXT NOT NULL,
                        language TEXT NOT NULL,
                        total_pages INTEGER NOT NULL,
                        batch_size INTEGER NOT NULL,
                        total_batches INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'in_progress',
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        created_sentences INTEGER DEFAULT 0
                    )
                """)
                
                # 배치 진행 상황 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS batch_progress (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        batch_number INTEGER NOT NULL,
                        start_page INTEGER NOT NULL,
                        end_page INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        pdf_file_path TEXT,
                        json_backup_path TEXT,
                        sentences_extracted INTEGER DEFAULT 0,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        error_message TEXT,
                        FOREIGN KEY (session_id) REFERENCES processing_sessions (id),
                        UNIQUE(session_id, batch_number)
                    )
                """)
                
                # 페이지/배치별 추출 통계 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS extraction_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        batch_number INTEGER NOT NULL,
                        page_number INTEGER NOT NULL,
                        language TEXT NOT NULL,
                        sentences_count INTEGER NOT NULL DEFAULT 0,
                        processing_time REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours')),
                        FOREIGN KEY (session_id) REFERENCES processing_sessions (id),
                        UNIQUE(session_id, batch_number, page_number, language)
                    )
                """)
                
                # 배치별 요약 통계 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS batch_summary_stats (
                        session_id TEXT NOT NULL,
                        batch_number INTEGER NOT NULL,
                        korean_sentences_count INTEGER DEFAULT 0,
                        english_sentences_count INTEGER DEFAULT 0,
                        total_pages INTEGER NOT NULL,
                        avg_sentences_per_page REAL DEFAULT 0.0,
                        processing_duration REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours')),
                        PRIMARY KEY (session_id, batch_number),
                        FOREIGN KEY (session_id) REFERENCES processing_sessions (id)
                    )
                """)
                
                # 인덱스 생성
                self._create_indexes(cursor)
                
                conn.commit()
                print(f"데이터베이스 스키마 초기화 완료: {self.db_path}")
                
        except Exception as e:
            raise Exception(f"데이터베이스 초기화 실패: {e}")
    
    def _create_indexes(self, cursor) -> None:
        """인덱스를 생성합니다."""
        indexes = [
            # korean_sentences 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_korean_sentence ON korean_sentences(sentence)",
            "CREATE INDEX IF NOT EXISTS idx_korean_batch_id ON korean_sentences(batch_id)",
            "CREATE INDEX IF NOT EXISTS idx_korean_batch_number ON korean_sentences(batch_number)",
            "CREATE INDEX IF NOT EXISTS idx_korean_source_type ON korean_sentences(source_type)",
            
            # english_sentences 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_english_sentence ON english_sentences(sentence)",
            "CREATE INDEX IF NOT EXISTS idx_english_batch_id ON english_sentences(batch_id)",
            "CREATE INDEX IF NOT EXISTS idx_english_batch_number ON english_sentences(batch_number)",
            "CREATE INDEX IF NOT EXISTS idx_english_source_type ON english_sentences(source_type)",
            
            # processing_sessions 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_session_source_file ON processing_sessions(source_file)",
            "CREATE INDEX IF NOT EXISTS idx_session_language ON processing_sessions(language)",
            "CREATE INDEX IF NOT EXISTS idx_session_status ON processing_sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_session_started_at ON processing_sessions(started_at)",
            
            # batch_progress 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_batch_session_id ON batch_progress(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_progress(status)",
            "CREATE INDEX IF NOT EXISTS idx_batch_number_idx ON batch_progress(batch_number)",
            
            # extraction_stats 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_extraction_session_id ON extraction_stats(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_extraction_batch_number ON extraction_stats(batch_number)",
            "CREATE INDEX IF NOT EXISTS idx_extraction_page_number ON extraction_stats(page_number)",
            "CREATE INDEX IF NOT EXISTS idx_extraction_language ON extraction_stats(language)",
            "CREATE INDEX IF NOT EXISTS idx_extraction_session_batch ON extraction_stats(session_id, batch_number)",
            
            # batch_summary_stats 테이블 인덱스
            "CREATE INDEX IF NOT EXISTS idx_batch_summary_session_id ON batch_summary_stats(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_batch_summary_created_at ON batch_summary_stats(created_at)",
        ]
        
        for index_query in indexes:
            cursor.execute(index_query)
    
    def create_processing_session(
        self, 
        source_file: str, 
        language: str, 
        total_pages: int, 
        batch_size: int
    ) -> str:
        """새로운 처리 세션을 생성합니다."""
        session_id = str(uuid.uuid4())
        total_batches = (total_pages + batch_size - 1) // batch_size
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO processing_sessions 
                    (id, source_file, language, total_pages, batch_size, total_batches, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'in_progress')
                """, (session_id, source_file, language, total_pages, batch_size, total_batches))
                
                # 배치 진행 상황 초기화
                for batch_num in range(1, total_batches + 1):
                    start_page = (batch_num - 1) * batch_size + 1
                    end_page = min(batch_num * batch_size, total_pages)
                    
                    cursor.execute("""
                        INSERT INTO batch_progress 
                        (session_id, batch_number, start_page, end_page, status)
                        VALUES (?, ?, ?, ?, 'pending')
                    """, (session_id, batch_num, start_page, end_page))
                
                conn.commit()
                print(f"처리 세션 생성: {session_id} ({total_batches}개 배치)")
                return session_id
                
        except Exception as e:
            raise Exception(f"세션 생성 실패: {e}")
    
    def update_batch_status(
        self, 
        session_id: str, 
        batch_number: int, 
        status: str,
        **kwargs
    ) -> None:
        """배치 상태를 업데이트합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 동적으로 업데이트할 컬럼들 구성
                update_fields = ["status = ?"]
                values = [status]
                
                if status == 'in_progress':
                    update_fields.append("started_at = CURRENT_TIMESTAMP")
                elif status == 'completed':
                    update_fields.append("completed_at = CURRENT_TIMESTAMP")
                
                # 추가 매개변수들 처리
                for key, value in kwargs.items():
                    if key in ['pdf_file_path', 'json_backup_path', 'sentences_extracted', 'error_message']:
                        update_fields.append(f"{key} = ?")
                        values.append(value)
                
                values.extend([session_id, batch_number])
                
                query = f"""
                    UPDATE batch_progress 
                    SET {', '.join(update_fields)}
                    WHERE session_id = ? AND batch_number = ?
                """
                
                cursor.execute(query, values)
                conn.commit()
                
        except Exception as e:
            raise Exception(f"배치 상태 업데이트 실패: {e}")
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """세션의 진행 상황을 조회합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 세션 정보 조회
                cursor.execute("""
                    SELECT * FROM processing_sessions WHERE id = ?
                """, (session_id,))
                session_row = cursor.fetchone()
                
                if not session_row:
                    return {"error": "세션을 찾을 수 없습니다"}
                
                # 배치 진행 상황 조회
                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM batch_progress 
                    WHERE session_id = ? 
                    GROUP BY status
                """, (session_id,))
                
                status_counts = dict(cursor.fetchall())
                
                # 실패한 배치들 조회
                cursor.execute("""
                    SELECT batch_number, start_page, end_page, error_message
                    FROM batch_progress 
                    WHERE session_id = ? AND status = 'failed'
                    ORDER BY batch_number
                """, (session_id,))
                
                failed_batches = [
                    {
                        "batch_number": row[0],
                        "start_page": row[1], 
                        "end_page": row[2],
                        "error_message": row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                return {
                    "session_id": session_id,
                    "source_file": session_row[1],
                    "language": session_row[2],
                    "total_pages": session_row[3],
                    "batch_size": session_row[4],
                    "total_batches": session_row[5],
                    "session_status": session_row[6],
                    "started_at": session_row[7],
                    "completed_at": session_row[8],
                    "created_sentences": session_row[10],
                    "batch_status_counts": status_counts,
                    "failed_batches": failed_batches,
                    "completed_batches": status_counts.get('completed', 0),
                    "pending_batches": status_counts.get('pending', 0),
                    "failed_batches_count": status_counts.get('failed', 0)
                }
                
        except Exception as e:
            return {"error": f"진행 상황 조회 실패: {e}"}
    
    def get_failed_batches(self, session_id: str) -> List[Dict[str, Any]]:
        """실패한 배치들의 목록을 반환합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT batch_number, start_page, end_page, error_message, pdf_file_path
                    FROM batch_progress 
                    WHERE session_id = ? AND status = 'failed'
                    ORDER BY batch_number
                """, (session_id,))
                
                return [
                    {
                        "batch_number": row[0],
                        "start_page": row[1],
                        "end_page": row[2],
                        "error_message": row[3],
                        "pdf_file_path": row[4]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"실패한 배치 조회 실패: {e}")
            return []
    
    def get_all_failed_batches(self, language_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """모든 세션의 실패한 배치들을 조회합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if language_filter:
                    cursor.execute("""
                        SELECT bp.session_id, bp.batch_number, bp.start_page, bp.end_page, 
                               bp.error_message, bp.pdf_file_path, ps.language, ps.source_file
                        FROM batch_progress bp
                        JOIN processing_sessions ps ON bp.session_id = ps.id
                        WHERE bp.status = 'failed' AND ps.language = ?
                        ORDER BY ps.language, bp.session_id, bp.batch_number
                    """, (language_filter,))
                else:
                    cursor.execute("""
                        SELECT bp.session_id, bp.batch_number, bp.start_page, bp.end_page, 
                               bp.error_message, bp.pdf_file_path, ps.language, ps.source_file
                        FROM batch_progress bp
                        JOIN processing_sessions ps ON bp.session_id = ps.id
                        WHERE bp.status = 'failed'
                        ORDER BY ps.language, bp.session_id, bp.batch_number
                    """)
                
                return [
                    {
                        "session_id": row[0],
                        "batch_number": row[1],
                        "start_page": row[2],
                        "end_page": row[3],
                        "error_message": row[4],
                        "pdf_file_path": row[5],
                        "language": row[6],
                        "source_file": row[7]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"전체 실패한 배치 조회 실패: {e}")
            return []
    
    def get_pending_batches(self, session_id: str) -> List[int]:
        """대기 중인 배치들의 번호 목록을 반환합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT batch_number
                    FROM batch_progress 
                    WHERE session_id = ? AND status IN ('pending', 'ready')
                    ORDER BY batch_number
                """, (session_id,))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"대기 중인 배치 조회 실패: {e}")
            return []
    
    def get_incomplete_batches(self, session_id: str) -> List[int]:
        """완료되지 않은 모든 배치들의 번호 목록을 반환합니다 (실패 + 대기)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT batch_number
                    FROM batch_progress 
                    WHERE session_id = ? AND status NOT IN ('completed')
                    ORDER BY batch_number
                """, (session_id,))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"미완료 배치 조회 실패: {e}")
            return []
    
    def get_batch_pdf_path(self, session_id: str, batch_number: int) -> Optional[str]:
        """특정 배치의 PDF 파일 경로를 반환합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT pdf_file_path
                    FROM batch_progress 
                    WHERE session_id = ? AND batch_number = ?
                """, (session_id, batch_number))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            print(f"배치 PDF 경로 조회 실패: {e}")
            return None
    
    def complete_session(self, session_id: str, total_sentences: int) -> None:
        """세션을 완료 상태로 업데이트합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE processing_sessions 
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP, created_sentences = ?
                    WHERE id = ?
                """, (total_sentences, session_id))
                
                conn.commit()
                print(f"세션 완료: {session_id} ({total_sentences}개 문장 생성)")
                
        except Exception as e:
            raise Exception(f"세션 완료 업데이트 실패: {e}")
    
    def list_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 세션들의 목록을 반환합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, source_file, language, status, started_at, completed_at, 
                           total_batches, created_sentences
                    FROM processing_sessions 
                    ORDER BY started_at DESC 
                    LIMIT ?
                """, (limit,))
                
                return [
                    {
                        "session_id": row[0],
                        "source_file": Path(row[1]).name,
                        "language": row[2],
                        "status": row[3],
                        "started_at": row[4],
                        "completed_at": row[5],
                        "total_batches": row[6],
                        "created_sentences": row[7]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"세션 목록 조회 실패: {e}")
            return []
    
    def save_page_extraction_stats(
        self,
        session_id: str,
        batch_number: int,
        page_stats: Dict[int, Dict[str, int]],
        processing_time: float = 0.0
    ) -> None:
        """페이지별 추출 통계를 저장합니다.
        
        Args:
            session_id: 세션 ID
            batch_number: 배치 번호
            page_stats: {page_number: {'korean': count, 'english': count}} 형태
            processing_time: 처리 시간(초)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for page_number, stats in page_stats.items():
                    for language, count in stats.items():
                        if count > 0:  # 추출된 문장이 있는 경우만 저장
                            cursor.execute("""
                                INSERT INTO extraction_stats 
                                (session_id, batch_number, page_number, language, sentences_count, processing_time)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ON CONFLICT(session_id, batch_number, page_number, language)
                                DO UPDATE SET 
                                    sentences_count = excluded.sentences_count,
                                    processing_time = excluded.processing_time
                            """, (session_id, batch_number, page_number, language, count, processing_time))
                
                conn.commit()
                
        except Exception as e:
            print(f"페이지 통계 저장 실패: {e}")
    
    def save_batch_summary_stats(
        self,
        session_id: str,
        batch_number: int,
        korean_count: int,
        english_count: int,
        total_pages: int,
        processing_duration: float
    ) -> None:
        """배치별 요약 통계를 저장합니다."""
        try:
            avg_sentences_per_page = (korean_count + english_count) / total_pages if total_pages > 0 else 0.0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO batch_summary_stats 
                    (session_id, batch_number, korean_sentences_count, english_sentences_count, 
                     total_pages, avg_sentences_per_page, processing_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, batch_number)
                    DO UPDATE SET 
                        korean_sentences_count = excluded.korean_sentences_count,
                        english_sentences_count = excluded.english_sentences_count,
                        total_pages = excluded.total_pages,
                        avg_sentences_per_page = excluded.avg_sentences_per_page,
                        processing_duration = excluded.processing_duration
                """, (session_id, batch_number, korean_count, english_count, 
                       total_pages, avg_sentences_per_page, processing_duration))
                
                conn.commit()
                
        except Exception as e:
            print(f"배치 요약 통계 저장 실패: {e}")
    
    def get_extraction_stats(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 페이지별 추출 통계를 조회합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT batch_number, page_number, language, sentences_count, processing_time, created_at
                    FROM extraction_stats 
                    WHERE session_id = ?
                    ORDER BY batch_number, page_number, language
                """, (session_id,))
                
                return [
                    {
                        "batch_number": row[0],
                        "page_number": row[1],
                        "language": row[2],
                        "sentences_count": row[3],
                        "processing_time": row[4],
                        "created_at": row[5]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"추출 통계 조회 실패: {e}")
            return []
    
    def get_batch_summary_stats(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 배치별 요약 통계를 조회합니다."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT batch_number, korean_sentences_count, english_sentences_count,
                           total_pages, avg_sentences_per_page, processing_duration, created_at
                    FROM batch_summary_stats 
                    WHERE session_id = ?
                    ORDER BY batch_number
                """, (session_id,))
                
                return [
                    {
                        "batch_number": row[0],
                        "korean_sentences_count": row[1],
                        "english_sentences_count": row[2],
                        "total_pages": row[3],
                        "avg_sentences_per_page": row[4],
                        "processing_duration": row[5],
                        "created_at": row[6]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            print(f"배치 요약 통계 조회 실패: {e}")
            return []
    
