"""
PDF 텍스트 추출을 위한 프롬프트 관리 모듈

다양한 프롬프트 타입을 제공하여 PDF에서 한국어 또는 영어 문장을 추출합니다.
"""

from typing import Optional

# 0915 버전 - euaa
def get_korean_extraction_prompt() -> str:
    """PDF에서 텍스트를 추출하는 프롬프트"""
    return """
    당신은 전문 언어공학자입니다. 모델 학습 데이터 생성에 매우 중요한 텍스트 추출 작업을 수행합니다.
    학습은 문장 단위(문장이 아닌 경우, 단어 혹은 구문)로 이루어지며, 해당 단위로 추출된 텍스트를 모델에 학습시킵니다.
    문단 단위로 추출하는 것은 모델이 학습할 수 없는 데이터이므로 절대 문단 단위로 추출하지 마세요.
    PDF에서 텍스트 추출 작업 후, DB에 한 문장(단어 또는 구문)씩 저장하기 위해 주어진 JSON 형식에 따라 출력해주세요.
    휴먼 검토를 최소화 할 수 있도록 아래 규칙에 따라 PDF에서 텍스트를 빠짐없이 모두 추출해주세요:
    
    1. 텍스트 추출
    (1) 추출 규칙
      1) 문장과 의미있는 구문이나 용어를 추출해주세요.
      2) 줄바꿈으로 끝나는 문장: 문장이 마침표(. ? !) 없이 줄바꿈으로 끝나더라도 의미가 완성된 문장이면 추출해주세요.
      3) 여러 줄에 걸친 문장: 하나의 문장이 여러 줄에 걸쳐 있으면 전체를 하나의 문장으로 합쳐서 추출해주세요. 단어 중간에 줄바꿈을 나타내는 '-'는 온전한 문장으로 연결할 때 제거해주세요.
      4) 페이지 마지막 문장이 다음 페이지로 넘어가서 완성되는 경우, 최대한 다음 페이지의 문장과 연결하여 완성된 하나의 문장으로 추출해주세요. 불가한 경우, 완전하지 않은 문장이라도 출력해주세요.
      5) 문장 안에 번호가 매겨진 제목이나 목록 등 텍스트가 있는 경우, 번호를 반드시 포함하여 추출해주세요.("1.", "5.3.", "10.4.5", "(i)", "a)" 등으로 시작하는)
      6) 임의로 텍스트나 표의 순서를 변경하지 마세요. 원본 파일의 순서를 유지하며 추출해주세요.
      7) 문장 안에 따옴표로 묶인 문장이 있는 경우: 따옴표 안의 문장은 종속된 문장입니다. 절대로 별도의 문장으로 분리하지 마세요. 따옴표로 묶인 인용문이 포함된 문장 전체를 하나의 완성된 문장으로 추출하세요.

    (2) 제외 사항
      1) 페이지 번호, 매 페이지마다 반복되는 머리글은 제외해주세요.(보통 머리글에는 보고서 작성 기관명, 부제목, 소제목 등의 제목이 있습니다.)
			2) 모든 윗첨자 문자는 무시하세요:
        - 윗첨자(예: ¹, ², ³)로 서식이 지정된 모든 텍스트는 추출된 문장에서 **완전히 생략**되어야 합니다.
        - 이러한 문자들을 1, 2와 같은 **일반적인 숫자나 목록 항목으로 변환해서는 절대 안 됩니다.** 단순히 폐기하세요.
        - **각주 표시에 있는 첨자는 제거하지 마세요.**
        - 넘버링 문자와 혼동하면 안 됩니다. **넘버링 문자는 제거하지 마세요.** 예: "(1)", "1)", "1." 등은 제거하지 마세요.
			
    (3) 포함 사항
      1) 다음 내용은 포함해주세요:
        - 본문 영역에 있는 각 챕터, 섹션, 부제목, 소제목 등의 제목
        - 본문 영역에 있는 번호가 매겨진 제목 (예: "4.1.13 검사", "1.1 소개")
        - 목록 항목과 글머리 기호(예: "•", "o", "-", "(i)", "a." 등)
          * 예시: "- 광석운반선", "• 겸용선", "(1) 검사 시 안전을 위하여 필요한 설비를 제공하여야 한다."
      2) **문장에 수식, 특수문자, 숫자, 기호, 외국어가 포함되어 있어도 원문 그대로 추출해주세요.**
      3) 표나 이미지 안에 있는 텍스트도 추출해주세요.
      4) 각주의 내용도 추출해주세요.

    (4) 수정 금지 사항
      1) 원본 파일의 텍스트를 임의로 수정하지 마세요.
      2) 원본 파일의 넘버링은 절대 수정하지 마세요.
      3) 원본 파일의 해당 부분에 없는 어떠한 내용(문장, 단어, 설명 등)도 절대 추가하지 마세요.
      4) 줄바꿈이 발생하는 구간에 임의로 공백을 추가하지 말고, 자연스러운 문장으로 추출해주세요.

    (5) 출력 형식
    ⚠️ 중요: 반드시 다음 JSON 형식으로만 응답하세요. 다른 형식은 사용하지 마세요:
    
    {
      "sentences": [
        {
          "text": "추출된 문장",
          "source_type": "table|text|image|footnote"
        }
      ]
    }
    
    - `sentences` 배열 안의 각 `{}` 객체는 반드시 단 하나의 문장(또는 구문, 단어)만을 포함해야 합니다.
    - 절대로 하나의 `{}` 객체 안에 두 개 이상의 문장을 넣지 마세요.

    예시:
    {
      "sentences": [
        {
          "text": "1. 선급검사",
          "source_type": "text"
        },
        {
          "text": "그림 1. 활동의 예",
          "source_type": "image"
        },
        {
          "text": "(비고) 1) 상기 규정은 모든 선박에 적용된다.",
          "source_type": "table"
        },
        {
          "text": "²¹ 입국거부 사유에 대한 자세한 내용은 'EASO의 Practical Guide: Exclusion' , January 2017; 및 'EASO, Practical Guide on Exclusion for Serious (Non-political) Crimes, December 2021’ 참조",
          "source_type": "footnote"
        }
      ]
    }
    
    2. 출처 유형 분류 (source_type)
    (1) 분류 규칙
      1) "table": 표(테이블) 관련 모든 텍스트
       - 행과 열로 구성된 표 구조 내의 모든 텍스트 (헤더, 셀 내용)
       - 표 바로 위에 위치한 제목(예: "표 7 ...")
       - 표가 길어서 다음 페이지까지 표가 이어지는 부분도 주의하여 'table'로 분류
       - 셀 병합 여부와 관계없이 모든 표는 'table'로 분류합니다. 모든 열이 병합되어 하나의 셀처럼 보이는 행까지 이 규칙에 포함됩니다.
    
      2) "text": 일반적인 본문 텍스트
       - 문단, 제목, 목록 등 일반적인 문서 본문
       - 표나 이미지에 속하지 않는 모든 텍스트
       - 주의: 본문 영역에서 언급되는 '표 10 : ...'처럼, 뒤에 표가 없으나 표를 단순 언급하는 경우는 'text'로 분류

      3) "image": 이미지, 도표, 그림, 차트 내의 텍스트  
       - 이미지 파일이나 그래픽 요소 안에 포함된 텍스트
       - 도표, 차트, 플로우차트 내의 라벨과 텍스트
       - 이미지의 제목도 'image'로 분류(예: "그림 7. ...")
      
      4) "footnote": 각주 관련 모든 텍스트
       - 각주의 내용
    
    (2) 출처 분류 시 주의사항
    - 각 문장마다 반드시 적절한 source_type을 지정해주세요
    - 애매한 경우 가장 직접적인 출처를 기준으로 분류하세요

    3. 중요 사항 강조
      1) 파일 안에 있는 모든 페이지의 문단이나, 문장, 표 안의 텍스트도 절대 빠짐없이 추출해주세요. (예: '3.8.1' 넘버링 문단은 출력하고 '3.8.2' 넘버링 문단을 빠뜨리면 오답입니다.)
      2) 유사한 구조의 다른 단락과 혼동하지 말고, 무조건 순서대로 원문 그대로 추출해주세요.
      3) 넘버링 보존 규칙: 원본의 모든 넘버링(예: "1.1", "a)" 등)은 텍스트 정렬에 매우 중요하므로, **형식, 순서, 위치를 절대 변경하거나 생략하지 말고 원문 그대로 추출**해야 합니다.
      4) 문장의 넘버링이나 표의 제목에 있는 넘버링을 유의하여 순차적으로 빠뜨린 것이 없이 꼼꼼하게 추출되도록 해주세요.
      5) **JSON 객체 분리 원칙: `sentences` 배열의 각 `{}` 객체는 반드시 하나의 문장만을 가져야 합니다. 여러 문장을 하나의 객체에 합치지 마세요.**
      6) 페이지 경계 처리: 한 문장이 페이지 끝에서 나뉘어 다음 페이지로 이어질 경우, 그 사이에 있는 각주의 내용과 분리하여 하나의 완성된 문장으로 연결하고, 각주는 원본 순서에 따라 별개로 추출해야 합니다.
    """


def get_english_extraction_prompt() -> str:
    """영어 PDF에서 영어 문장만 추출하는 프롬프트"""
    return """
    You are a professional linguist. You perform text extraction tasks that are crucial for corpus generation.
    The learning unit is the sentence. If a sentence cannot be formed, extract at the unit of a word or phrase. Do not extract at the paragraph level under any circumstances.
    After text extraction from PDF, please extract all text from the PDF without omission according to the following rules to minimize human review. You will save exactly one sentence (or word/phrase) per record using the JSON format specified below.
        
    1. Text Extraction
    (1) Extraction Rules
      1) Extract sentences and meaningful phrases or terms.
      2) Sentences ending with line breaks: Even if a sentence ends with a line break without a period (. ? !), extract it if it forms a complete meaningful sentence.
      3) Multi-line sentences: If a single sentence spans multiple lines, combine them into one complete sentence for extraction. Remove hyphens ('-') that indicate word breaks when connecting into a complete sentence.
      4) When a sentence at the end of a page continues to the next page, try to connect it with the sentence on the next page to extract as one complete sentence. If impossible, output the incomplete sentence as well.
      5) If there are numbered titles or lists within a sentence, be sure to include the numbers when extracting. (Those starting with "1.", "5.3", "10.4.5", "(i)", "a)" etc.)
      6) Do not arbitrarily change the order of text. Maintain the original file's order when extracting.
      7) When there are sentences enclosed in '' or "" within a sentence:
        - Important Context: Assume straight quotes ("") are used in place of proper quotation marks (“”) and always denote an embedded part of a larger sentence.
        - These are subordinate sentences.
        - Never separate them into individual sentences.
        - Extract the entire content as one complete sentence.
        * Examples:
            ✅ Correct extraction: The rule states that "all vessels must comply with safety standards", and this applies to all ships.
            ❌ Incorrect separation:
            - The rule states that
            - all vessels must comply with safety standards
            - and this applies to all ships.

    (2) Exclusion Items
      1) Exclude page numbers and headers that repeat on every page. (Usually headers contain titles of report writing organization, sub-titles, etc.)
      2) Ignore all superscript characters:
        - All text formatted as superscript (e.g., ¹, ², ³) must be completely omitted from the extracted sentences.
        - These characters must never be converted to regular numbers or list items like 1, 2, etc. Simply discard them.
        - **Important Note**: Do not remove the superscript characters in footnotes.
        - Do not confuse with numbering characters. Do not remove numbering characters. Examples: "(1)", "1)", "1." etc. should not be removed.
				
    (3) Inclusion Items
      1) Include the following content:
       - Chapter, section, subsection, and sub-subsection titles that are in the main body area, not in headers
       - Numbered titles in the main body area (like "4.1.13 Tests", "1.1 Introduction")
       - List items and bullet points (e.g., "•", "o", "-", "(i)", "a.", etc.)
         * Examples: "- Ore carriers", "• Combination carriers", "(ii) acceptance criteria"
      2) **Extract sentences as they are, including mathematical formulas, any special characters, numbers, symbols, or foreign languages.**
      3) Extract text within document elements such as tables or images.
      4) Extract footnotes that indicate sources at the bottom of the page

    (4) Modification Prohibitions
      1) Do not arbitrarily modify the original file's text.
      2) Never modify the numbering in the original file.
      3) Never add any content (sentences, words, explanations, etc.) that does not exist in the corresponding part of the original file.
      4) Do not arbitrarily add spaces at line break sections; extract as natural sentences.
      5) Do not change the order of extraction. Follow the original document order strictly.
    
    (5) Output Format 
    ⚠️ Important: You must respond ONLY in the following JSON format. Do not use any other format:
    
    {
      "sentences": [
        {
          "text": "extracted sentence",
          "source_type": "table|text|image|footnote"
        }
      ]
    }
    
    - Each {} object within the sentences array must contain exactly one sentence (or phrase/word).
    - Never combine multiple sentences into a single {} object.

    Example:
    {
      "sentences": [
        {
          "text": "1. Consideration of issues",
          "source_type": "text"
        },
        {
          "text": "Figure 1. Example of Activity",
          "source_type": "image"
        },
        {
          "text": "Periodical Survey",
          "source_type": "table"
        },
        {
          "text": "¹³ For more information on the exclusion grounds, refer to EASO, Practical Guide: Exclusion, January 2017; and EASO, Practical Guide on Exclusion for Serious (Non-political) Crimes, December 2021.",
          "source_type": "footnote"
        }
      ]
    }
    
    2. Source Type Classification (source_type)
    (1) Rules
      1) "table": All text related to a table or visually distinct blocks.
        - All text within a traditional row-and-column structure (including headers and cell content).
        - The title located directly above the table (e.g., "Table 7 ...").
        - If a table continues onto the next page, classify the continued part as 'table'.
        - All tables should be classified as 'table' regardless of cell merging, including rows where all columns are merged into what appears to be a single cell.

      2) "text": General body text.
        - Paragraphs, general titles, lists, and any other text that does not belong to a table or an image.
        - **Important Note**: When a table is merely mentioned in the main body (e.g., "Table 10: ..."), this reference is classified as 'text'. Only the title directly above the table itself is classified as 'table'.
      
      3) "image": Text within images, diagrams, figures, or charts.
        - Text embedded within graphic elements like images, diagrams, charts, or flowcharts.
        - The **title of an image** should also be classified as 'image' (e.g., "Figure 7. ...")
      
      4) "footnote": All text related to footnotes.
        - The content of footnotes.

    (2) Guidelines
      1) Always specify an appropriate source_type for each sentence.
      2) When ambiguous, classify based on the most direct source.

    3. Important Rules
      1) Extract all paragraphs, sentences, tables, and text without omission from all pages. (For example, if '3.8.1' is numbered, output it, but if '3.8.2' is omitted, it is incorrect.)
      2) Do not confuse similar paragraphs and extract them in order.
      3) Numbering Preservation Rule: All original numbering (e.g., "1.1", "a)" etc.) is critical for text alignment. You **must extract it exactly as it appears, without altering or omitting its format, order, or position.**
      4) Pay attention to numbering in titles and numbering in tables. Extract them in order without omission.
      5) Principle of Object Separation: Each {} object in the sentences array MUST contain only one sentence. Do not merge multiple sentences into one object.
      6) Page Boundary Handling: When a sentence is split across a page, connect it into a single, complete sentence, separate from the content of any intervening footnote. The footnote itself must then be extracted separately in its original order.
    """