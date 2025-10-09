# config.py

"""
PDF 텍스트 추출 및 분석을 위한 설정값
이 파일의 값을 조정하여 추출 로직을 미세 조정할 수 있습니다.
"""

# --- 파일 경로 설정 ---
# 분석할 PDF 파일의 경로
INPUT_PDF_PATH = "33. EU 분기별 주요 난민소송 판례 1부_en.pdf"

# PDF 파일명에서 뒷 부분 5글자를 추출하여 동적 파일명 생성
def get_file_suffix():
    """PDF 파일명에서 .pdf를 제외한 뒷 부분 5글자를 추출"""
    filename_without_ext = INPUT_PDF_PATH.replace('.pdf', '')
    return filename_without_ext[-5:]  # 뒷 부분 5글자

# 추출 결과를 저장할 JSON 파일의 경로 (동적 생성)
OUTPUT_JSON_PATH = f"raw_output_{get_file_suffix()}.json"


# --- 레이아웃 분석 설정 ---
# 페이지 상단에서 헤더로 간주할 영역의 비율 (예: 0.1 = 상위 10%)
HEADER_MARGIN_RATIO = 0.1

# 페이지 하단에서 푸터로 간주할 영역의 비율 (예: 0.1 = 하위 10%)
FOOTER_MARGIN_RATIO = 0.1


# --- 스타일 분석 및 태깅 설정 ---
# H1 제목으로 판단할 기준: 본문 텍스트 크기보다 몇 배 이상 큰가
H1_SIZE_MULTIPLIER = 1.5
# H1 제목의 특정 색상값 (JSON 결과에서 확인)
H1_COLOR = 16760832

# H2 제목으로 판단할 기준: 본문 텍스트 크기보다 몇 배 이상 큰가 (값을 약간 낮춤)
H2_SIZE_MULTIPLIER = 1.05
# H2 제목의 특정 색상값 (JSON 결과에서 확인)
H2_COLOR = 2302539

# 텍스트의 굵기(bold) 속성을 나타내는 PyMuPDF의 비트 플래그 값
BOLD_FLAG = 16