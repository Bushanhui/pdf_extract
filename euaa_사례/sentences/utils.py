"""
공통 유틸리티 함수들
PDF 추출 및 정렬 프로젝트의 공통 기능들을 모아둔 모듈
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict


def get_default_config_path(config_name: str = "config_1-10_규칙.json") -> str:
    """
    기본 config 파일 경로를 반환합니다.
    
    Args:
        config_name (str): config 파일명. 기본값은 "config_1-10_규칙.json"
        
    Returns:
        str: config 파일의 절대 경로
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'configs', 
        config_name
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """
    JSON config 파일을 로드합니다.
    
    Args:
        config_path (str, optional): config 파일 경로. None이면 기본 config 사용
        
    Returns:
        dict: 로드된 config 데이터
        
    Raises:
        Exception: config 파일 로딩 실패 시
    """
    if config_path is None:
        config_path = get_default_config_path()
        print(f"기본 config 파일을 사용합니다: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # pdf_info가 있으면 출력, 없으면 기본 메시지
            if 'pdf_info' in config and 'description' in config['pdf_info']:
                print(f"Config 로딩 성공: {config['pdf_info']['description']}")
            else:
                print(f"Config 로딩 성공: {config_path}")
            return config
    except FileNotFoundError:
        raise Exception(f"Config 파일을 찾을 수 없습니다: {config_path}")
    except json.JSONDecodeError as e:
        raise Exception(f"Config 파일 형식이 올바르지 않습니다: {e}")
    except Exception as e:
        raise Exception(f"Config 파일 로딩 실패: {e}")


def load_models() -> Tuple[Any, Any]:
    """
    무거운 언어 모델들을 로드합니다.
    
    Returns:
        Tuple[SentenceTransformer, SentenceTransformer]: (labse_model, distiluse_model)
        실패 시 (None, None) 반환
    """
    print("언어 모델을 로드합니다... (이 작업은 한 번만 실행됩니다)")
    try:
        from sentence_transformers import SentenceTransformer
        labse_model = SentenceTransformer('sentence-transformers/LaBSE')
        distiluse_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        print("✅ 모델 로딩 완료.")
        return labse_model, distiluse_model
    except ImportError:
        print("🚨 'sentence-transformers' 라이브러리가 설치되지 않았습니다.")
        print("pip install sentence-transformers 로 설치해주세요.")
        return None, None
    except Exception as e:
        print(f"🚨 모델 로딩 중 오류 발생: {e}")
        return None, None


def validate_file_pair(input_dir: str, filename: str) -> Tuple[bool, Path, Path]:
    """
    한국어/영어 파일 쌍이 존재하는지 확인합니다.
    
    Args:
        input_dir (str): 입력 디렉토리 경로
        filename (str): 파일명 (확장자 제외)
        
    Returns:
        Tuple[bool, Path, Path]: (존재 여부, 한국어 파일 경로, 영어 파일 경로)
    """
    input_path = Path(input_dir)
    kr_file_path = input_path / f'{filename}_kr.xlsx'
    en_file_path = input_path / f'{filename}_en.xlsx'
    
    exists = kr_file_path.exists() and en_file_path.exists()
    return exists, kr_file_path, en_file_path


def ensure_output_directory(output_dir: str) -> Path:
    """
    출력 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    
    Args:
        output_dir (str): 출력 디렉토리 경로
        
    Returns:
        Path: 출력 디렉토리 Path 객체
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# 공통 상수들
DEFAULT_MODEL_NAMES = {
    'labse': 'sentence-transformers/LaBSE',
    'distiluse': 'distiluse-base-multilingual-cased-v1'
}

SUPPORTED_CONFIG_TYPES = {
    '1-10_규칙': 'config_1-10_규칙.json',
    '13편': 'config_13편.json'
}


def generate_database_path_with_suffix(suffix: str) -> str:
    """
    suffix에 맞는 데이터베이스 파일 경로를 생성합니다.

    Args:
        suffix (str): 데이터베이스 파일명에 추가할 suffix

    Returns:
        str: 생성된 데이터베이스 파일 경로

    Examples:
        >>> generate_database_path_with_suffix("호주")
        "../pdf_extract/corpus_호주.db"
        >>> generate_database_path_with_suffix("")
        "../pdf_extract/corpus.db"
    """
    if suffix:
        return f"../pdf_extract/corpus_{suffix}.db"
    else:
        return "../pdf_extract/corpus.db"


def update_config_with_suffix(config: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """
    config 딕셔너리의 database_path와 workflow 폴더들을 suffix 기반으로 동적 수정합니다.

    Args:
        config (Dict[str, Any]): 원본 config 딕셔너리
        suffix (str): 데이터베이스 및 폴더 suffix

    Returns:
        Dict[str, Any]: database_path와 workflow 폴더들이 업데이트된 config 딕셔너리 (복사본)
    """
    updated_config = config.copy()

    # workflow.database_path 업데이트
    if 'workflow' in updated_config:
        updated_config['workflow'] = updated_config['workflow'].copy()
        updated_config['workflow']['database_path'] = generate_database_path_with_suffix(suffix)

        # workflow 폴더들을 suffix 서브폴더로 업데이트
        if suffix and 'folders' in updated_config['workflow']:
            updated_config['workflow']['folders'] = updated_config['workflow']['folders'].copy()
            folders = updated_config['workflow']['folders']

            # 각 폴더에 suffix 서브폴더 추가
            if 'excel_results' in folders:
                folders['excel_results'] = f"excel_results/{suffix}"
            if 'final_aligned_results' in folders:
                folders['final_aligned_results'] = f"final_aligned_results/{suffix}"
            if 'filtering_metadata' in folders:
                folders['filtering_metadata'] = f"filtering_metadata/{suffix}"
            if 'aligned_anchor_context' in folders:
                folders['aligned_anchor_context'] = f"aligned_anchor_context/{suffix}"
            if 'table_aligned_results' in folders:
                folders['table_aligned_results'] = f"table_aligned_results/{suffix}"
            if 'experiment_logs' in folders:
                folders['experiment_logs'] = f"experiment_logs/{suffix}"

    return updated_config


def ensure_suffix_subdirectories(config: Dict[str, Any]) -> None:
    """
    config에 정의된 모든 workflow 폴더를 자동 생성합니다.

    Args:
        config (Dict[str, Any]): workflow 폴더 정보가 포함된 config 딕셔너리
    """
    if 'workflow' not in config or 'folders' not in config['workflow']:
        return

    folders = config['workflow']['folders']

    # 각 workflow 폴더 생성
    for folder_key, folder_path in folders.items():
        if folder_path:  # 폴더 경로가 빈 문자열이 아닌 경우
            folder_path_obj = Path(folder_path)
            folder_path_obj.mkdir(parents=True, exist_ok=True)
            print(f"📁 폴더 생성: {folder_path}")


# 공통 상수들