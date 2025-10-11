# layout_analyzer.py

"""
PDF 페이지의 레이아웃과 스타일을 분석하는 핵심 모듈.
- 헤더/푸터 필터링
- 다단(multi-column) 레이아웃 감지 및 텍스트 순서 정렬
- 스타일 분석을 통한 제목(H1, H2), 본문(P) 태깅
"""

import fitz  # PyMuPDF
from collections import Counter
import config  # 우리가 만든 설정 파일 import
import re

def process_page(page: fitz.Page) -> list:
    """
    하나의 PDF 페이지를 처리하여 구조화된 텍스트 블록 리스트를 반환합니다.
    """
    
    # 1. 페이지의 모든 텍스트 블록을 원본 그대로 추출
    all_blocks = page.get_text("blocks")

    # 2. 헤더/푸터 필터링
    content_blocks = _filter_blocks_by_area(all_blocks, page)

    # 3. 레이아웃에 따라 텍스트 블록 정렬
    sorted_blocks = _sort_blocks_by_layout(content_blocks, page)

    # 4. 정렬된 블록을 span 단위로 풀어서 스타일 분석 및 태깅
    tagged_spans = _analyze_and_tag_spans(sorted_blocks, page)

    return tagged_spans

def _filter_blocks_by_area(blocks: list, page: fitz.Page) -> list:
    """config에 정의된 마진을 기준으로 헤더/푸터를 제외한 본문 블록만 필터링"""
    
    page_height = page.rect.height
    header_boundary = page_height * config.HEADER_MARGIN_RATIO
    footer_boundary = page_height * (1 - config.FOOTER_MARGIN_RATIO)
    
    content_blocks = []
    for block in blocks:
        # 블록 타입이 텍스트가 아니면(예: 이미지) 무시
        if block[6] != 0: 
            continue

        block_top = block[1]    # bbox y0
        block_bottom = block[3] # bbox y1

        if block_bottom < header_boundary or block_top > footer_boundary:
            continue  # 헤더 또는 푸터 영역에 속하면 건너뛰기
        
        content_blocks.append(block)
        
    return content_blocks

def _sort_blocks_by_layout(blocks: list, page: fitz.Page) -> list:
    """페이지 중앙선을 기준으로 2단 구성을 감지하고 읽기 순서에 맞게 정렬"""

    midpoint = page.rect.width / 2
    
    left_column = sorted([b for b in blocks if b[0] < midpoint], key=lambda b: b[1])
    right_column = sorted([b for b in blocks if b[0] >= midpoint], key=lambda b: b[1])
    
    # 왼쪽 열과 오른쪽 열을 순서대로 합쳐서 반환
    return left_column + right_column

def _analyze_and_tag_spans(blocks: list, page: fitz.Page) -> list:
    """정렬된 블록 리스트를 받아, line별로 텍스트를 합치고 스타일 분석 후 H1, H2, H3, P 태그를 달아 반환"""

    all_lines = []
    # 블록 리스트를 순회하며 line별로 텍스트를 합쳐서 처리
    for block in blocks:
        try:
            # 블록의 좌표 정보를 사용하여 해당 영역의 텍스트를 dict 형태로 추출
            block_rect = fitz.Rect(block[0], block[1], block[2], block[3])

            # 해당 영역의 텍스트를 dict 형태로 추출하여 line 정보를 얻습니다
            text_dict = page.get_text("dict", clip=block_rect)

            # dict 형태의 텍스트에서 line별로 처리
            for block_dict in text_dict.get("blocks", []):
                if "lines" in block_dict:
                    for line in block_dict["lines"]:
                        if line.get("spans"):
                            # 같은 줄의 모든 span 텍스트를 합치기
                            raw_text = " ".join(span.get("text", "") for span in line["spans"])
                            line_text = re.sub(r'\s+', ' ', raw_text).strip()

                            if line_text:  # 빈 텍스트가 아닌 경우만 처리
                                # 첫 번째 span의 스타일 정보를 사용 (같은 줄이므로 동일한 스타일 가정)
                                first_span = line["spans"][0]
                                all_lines.append({
                                    'text': line_text,
                                    'size': first_span.get('size', 12),
                                    'flags': first_span.get('flags', 0),
                                    'font': first_span.get('font', 'unknown'),
                                    'color': first_span.get('color', 0),
                                    'bbox': first_span.get('bbox', [0,0,0,0])
                                })
        except Exception as e:
            print(f"블록 처리 중 오류 발생: {e}")
            # 오류 발생 시 블록의 텍스트를 기본 line으로 처리
            text = block[4]  # 블록의 텍스트 내용
            if text.strip():  # 빈 텍스트가 아닌 경우만 처리
                all_lines.append({
                    'text': text,
                    'size': 12,  # 기본 폰트 크기
                    'flags': 0,  # 기본 플래그 (일반 텍스트)
                    'font': 'helv',  # 기본 폰트
                    'color': 0,  # 기본 색상
                    'bbox': [block[0], block[1], block[2], block[3]]
                })
            continue

    if not all_lines:
        return []

    # 본문 텍스트의 기준 크기를 통계적으로 계산
    line_sizes = [round(line['size']) for line in all_lines]
    if not line_sizes:
        return []
    body_size = Counter(line_sizes).most_common(1)[0][0]

    # 최종 결과를 담을 리스트
    tagged_results = []

    for line in all_lines:
        size = line['size']
        is_bold = line['flags'] & config.BOLD_FLAG
        color = line['color']
        tag = 'P' # 기본 태그는 본문(Paragraph)

        # H1: 크기가 매우 크고(1.5배 이상) bold인 경우
        if size >= body_size * config.H1_SIZE_MULTIPLIER and is_bold:
            tag = 'H1'
        # H3: bold이면서 특정 색상들(3101846, 3036053, 352961) 중 하나인 경우
        elif is_bold and color in config.H3_COLORS:
            tag = 'H3'
        # H2: H1, H3가 아닌데 bold가 있으면 H2 (크기 무관)
        elif is_bold:
            tag = 'H2'
        # P: bold가 없으면 본문

        tagged_results.append({
            "tag": tag,
            "text": line['text'].strip(),
            "size": round(line.get('size', 12), 1),
            "font_name": line.get('font', 'unknown'),
            "is_bold": bool(line.get('flags', 0) & config.BOLD_FLAG),
            "is_italic": bool(line.get('flags', 0) & 2),  # ITALIC_FLAG
            "color": line.get('color', 0),
            "bbox": {
                "x0": round(line.get('bbox', [0,0,0,0])[0], 1),
                "y0": round(line.get('bbox', [0,0,0,0])[1], 1),
                "x1": round(line.get('bbox', [0,0,0,0])[2], 1),
                "y1": round(line.get('bbox', [0,0,0,0])[3], 1)
            }
        })

    return tagged_results

# def _analyze_and_tag_spans(blocks: list, page: fitz.Page) -> list:
#     """정렬된 블록 리스트를 받아, 블록 단위로 텍스트를 처리하고 스타일 분석 후 H1, H2, P 태그를 달아 반환"""
#
#     all_blocks = []
#
#     for block in blocks:
#         try:
#             # 블록의 좌표 정보를 사용하여 해당 영역의 텍스트를 dict 형태로 추출
#             block_rect = fitz.Rect(block[0], block[1], block[2], block[3])
#
#             # 해당 영역의 텍스트를 dict 형태로 추출
#             text_dict = page.get_text("dict", clip=block_rect)
#
#             # 블록 전체의 텍스트를 합치기
#             block_text_parts = []
#             block_size = 12  # 기본값
#             block_flags = 0
#             block_font = 'unknown'
#             block_color = 0
#             block_bbox = [0,0,0,0]
#
#             for block_dict in text_dict.get("blocks", []):
#                 if "lines" in block_dict:
#                     for line in block_dict["lines"]:
#                         if line.get("spans"):
#                             for span in line.get("spans", []):
#                                 if span.get("text", "").strip():
#                                     block_text_parts.append(span.get("text", ""))
#                                     # 첫 번째 span의 스타일 정보를 사용
#                                     if not block_text_parts or len(block_text_parts) == 1:
#                                         block_size = span.get('size', 12)
#                                         block_flags = span.get('flags', 0)
#                                         block_font = span.get('font', 'unknown')
#                                         block_color = span.get('color', 0)
#                                         block_bbox = span.get('bbox', [0,0,0,0])
#
#             # 블록의 모든 텍스트를 합치기
#             if block_text_parts:
#                 raw_text = " ".join(block_text_parts)
#                 block_text = re.sub(r'\s+', ' ', raw_text).strip()
#
#                 if block_text:
#                     all_blocks.append({
#                         'text': block_text,
#                         'size': block_size,
#                         'flags': block_flags,
#                         'font': block_font,
#                         'color': block_color,
#                         'bbox': block_bbox
#                     })
#
#         except Exception as e:
#             print(f"블록 처리 중 오류 발생: {e}")
#             # 오류 발생 시 블록의 텍스트를 기본으로 처리
#             text = block[4]  # 블록의 텍스트 내용
#             if text.strip():
#                 all_blocks.append({
#                     'text': text,
#                     'size': 12,
#                     'flags': 0,
#                     'font': 'helv',
#                     'color': 0,
#                     'bbox': [block[0], block[1], block[2], block[3]]
#                 })
#             continue
#
#     if not all_blocks:
#         return []
#
#     # 본문 텍스트의 기준 크기를 통계적으로 계산
#     block_sizes = [round(block['size']) for block in all_blocks]
#     if not block_sizes:
#         return []
#     body_size = Counter(block_sizes).most_common(1)[0][0]
#
#     # 최종 결과를 담을 리스트
#     tagged_results = []
#
#     for block in all_blocks:
#         size = block['size']
#         is_bold = block['flags'] & config.BOLD_FLAG
#         color = block['color']
#         tag = 'P' # 기본 태그는 본문(Paragraph)
#
#         # H1: 크기가 매우 크고(1.5배 이상) bold인 경우
#         if size >= body_size * config.H1_SIZE_MULTIPLIER and is_bold:
#             tag = 'H1'
#         # H3: bold이면서 특정 색상(3101846)인 경우
#         elif is_bold and color == config.H3_COLOR:
#             tag = 'H3'
#         # H2: H1, H3가 아닌데 bold가 있으면 H2 (크기 무관)
#         elif is_bold:
#             tag = 'H2'
#         # P: bold가 없으면 본문
#
#         tagged_results.append({
#             "tag": tag,
#             "text": block['text'].strip(),
#             "size": round(block.get('size', 12), 1),
#             "font_name": block.get('font', 'unknown'),
#             "is_bold": bool(block.get('flags', 0) & config.BOLD_FLAG),
#             "is_italic": bool(block.get('flags', 0) & 2),  # ITALIC_FLAG
#             "color": block.get('color', 0),
#             "bbox": {
#                 "x0": round(block.get('bbox', [0,0,0,0])[0], 1),
#                 "y0": round(block.get('bbox', [0,0,0,0])[1], 1),
#                 "x1": round(block.get('bbox', [0,0,0,0])[2], 1),
#                 "y1": round(block.get('bbox', [0,0,0,0])[3], 1)
#             }
#         })
#
#     return tagged_results