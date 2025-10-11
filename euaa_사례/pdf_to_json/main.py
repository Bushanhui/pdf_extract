# main.py

"""
PDF 텍스트 추출 파이프라인의 메인 실행 파일.
1. config.py에서 설정값을 불러옵니다.
2. layout_analyzer.py의 함수를 사용하여 PDF의 각 페이지를 처리합니다.
3. 모든 페이지의 결과를 종합하고, 문서 전체에 걸쳐 고유한 순차 ID를 부여합니다.
4. 최종 결과를 JSON 파일로 저장합니다.
"""

import fitz  # PyMuPDF
import json
import config
from layout_analyzer import process_page
import re

# main.py에 추가할 함수
def merge_consecutive_tags(tagged_texts):
    """H1, H2, H3는 연속된 같은 태그를 병합하고, P는 그대로 유지하되 점 반복 제거"""
    if not tagged_texts:
        return []

    merged = []
    current_tag = None
    current_texts = []

    for item in tagged_texts:
        # P 태그는 별도 병합 없이 그대로 처리
        if item["tag"] == "P":
            # 이전에 누적된 H1/H2/H3 텍스트가 있으면 먼저 병합
            if current_texts:
                merged_text = " ".join(current_texts)
                merged_text = re.sub(r'\.{5,}', '', merged_text)
                merged.append({
                    "tag": current_tag,
                    "text": merged_text
                })
                current_texts = []

            # P 태그는 점 반복만 제거하고 그대로 추가
            cleaned_text = re.sub(r'\.{5,}', '', item["text"])
            merged.append({
                "tag": item["tag"],
                "text": cleaned_text
            })
            current_tag = None  # P 태그 후에는 누적 상태 초기화

        # H1, H2, H3 태그는 연속된 같은 태그를 병합
        elif item["tag"] in ["H1", "H2", "H3"]:
            if item["tag"] == current_tag:
                # 같은 태그면 텍스트를 누적
                current_texts.append(item["text"])
            else:
                # 다른 태그면 이전 텍스트들을 병합하고 새로 시작
                if current_texts:
                    merged_text = " ".join(current_texts)
                    merged_text = re.sub(r'\.{5,}', '', merged_text)
                    merged.append({
                        "tag": current_tag,
                        "text": merged_text
                    })
                current_tag = item["tag"]
                current_texts = [item["text"]]

    # 마지막에 누적된 H1/H2/H3 텍스트들 처리
    if current_texts:
        merged_text = " ".join(current_texts)
        merged_text = re.sub(r'\.{5,}', '', merged_text)
        merged.append({
            "tag": current_tag,
            "text": merged_text
        })

    return merged

def main():
    """메인 처리 함수"""
    print(f"'{config.INPUT_PDF_PATH}' 파일 처리를 시작합니다...")
    
    try:
        doc = fitz.open(config.INPUT_PDF_PATH)
    except Exception as e:
        print(f"오류: PDF 파일을 열 수 없습니다. 경로를 확인하세요. -> {e}")
        return

    all_tagged_texts = []
    
    # PDF의 모든 페이지를 순차적으로 처리
    for i, page in enumerate(doc):
        print(f" - 페이지 {i + 1}/{len(doc)} 처리 중...")
        # layout_analyzer를 통해 현재 페이지의 텍스트를 분석 및 태깅
        page_results = process_page(page)
        all_tagged_texts.extend(page_results)

    # 모든 결과를 종합한 후, 문서 전체에 걸쳐 순차적인 ID 부여
    final_results = []
    for i, item in enumerate(all_tagged_texts):
        if item["text"]:
            # 'size'와 'color' 정보를 최종 결과에 포함시킵니다.
            final_results.append({
                "id": i + 1,
                "tag": item["tag"],
                "text": item["text"],
                "size": item["size"],
                "color": item["color"]
            })

    # 최종 결과를 JSON 파일로 저장
    try:
        with open(config.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 처리 완료! 결과가 '{config.OUTPUT_JSON_PATH}' 파일에 저장되었습니다.")
        print(f"총 {len(final_results)}개의 텍스트 조각을 추출했습니다.")
    except Exception as e:
        print(f"오류: 결과를 파일로 저장하는 데 실패했습니다. -> {e}")

    # [수정됨] LLM에 보내기 전, Python에서 연속 태그를 미리 병합
    print("\n연속 태그 병합 중...")
    merged_texts = merge_consecutive_tags(final_results)  # final_results 사용

    # 병합된 결과를 LLM에 전달하기 위한 중간 파일로 저장 (동적 파일명 사용)
    # (이 파일을 복사해서 LLM 프롬프트에 붙여넣으면 됩니다.)
    llm_input_path = f"llm_input_{config.get_file_suffix()}.json"
    with open(llm_input_path, 'w', encoding='utf-8') as f:
        # LLM에게 전달할 데이터에 순차 ID 부여
        llm_input_data = [
            {
                "id": idx + 1,
                "text": item["text"],
                "source_type": item["tag"]
            }
            for idx, item in enumerate(merged_texts)
        ]
        json.dump(llm_input_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ LLM 입력 파일 생성 완료! '{llm_input_path}'의 내용을 복사하여 다음 단계의 LLM 프롬프트에 사용하세요.")
    print(f"총 {len(merged_texts)}개의 텍스트 덩어리로 병합되었습니다.")

if __name__ == "__main__":
    main()