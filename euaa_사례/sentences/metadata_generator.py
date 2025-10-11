import re
import pandas as pd
import unicodedata
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Set, Dict, List, Any
import warnings
import numpy as np
import utils

# Pandas 경고 메시지 비활성화 (SettingWithCopyWarning 등)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# ==============================================================================
# 제공된 모든 헬퍼 함수 및 클래스를 여기에 포함시킵니다.
# (이 파일 하나만으로 모든 기능이 동작하도록 self-contained하게 만듭니다)
# ==============================================================================

# Global variables for numbering patterns - will be initialized by load_config
COMMON_NUMBERING_PATTERNS = []
TEXT_ONLY_NUMBERING_PATTERNS = []

def load_config(config_path=None):
    """Config 파일을 로드하고 전역 패턴 변수들을 초기화합니다."""
    global COMMON_NUMBERING_PATTERNS, TEXT_ONLY_NUMBERING_PATTERNS
    
    config = utils.load_config(config_path)
    
    # 공통 패턴 초기화
    COMMON_NUMBERING_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in config['numbering_patterns']['common']]
    
    # 텍스트 전용 패턴 초기화  
    TEXT_ONLY_NUMBERING_PATTERNS = [re.compile(pattern) for pattern in config['numbering_patterns']['text_only']]
    
    return config

# 기본 config 로드 (필요시 명시적으로 호출)
# load_config()


# [수정됨] 문맥(context)에 따라 넘버링 제거를 다르게 수행하는 함수
def remove_numbering(sentence, context='text', config_path=None):
    """
    문맥(context)에 따라 다른 넘버링 패턴 세트를 적용하여 문장을 정리합니다.
    :param sentence: 처리할 문장
    :param context: 'text' 또는 'table'. 기본값은 'text'.
    :param config_path: config 파일 경로 (선택사항)
    """
    global COMMON_NUMBERING_PATTERNS, TEXT_ONLY_NUMBERING_PATTERNS
    
    # config_path가 제공되면 새로 로드
    if config_path is not None:
        load_config(config_path)
    
    if not sentence or not isinstance(sentence, str):
        return sentence

    # 문맥에 따라 적용할 패턴 리스트를 결정
    patterns_to_apply = COMMON_NUMBERING_PATTERNS
    if context == 'text':
        # 텍스트 문맥에서는 모든 패턴을 순서대로 적용 (위험한 패턴은 나중에)
        patterns_to_apply = patterns_to_apply + TEXT_ONLY_NUMBERING_PATTERNS

    cleaned_sentence = sentence
    for pattern in patterns_to_apply:
        match = pattern.match(cleaned_sentence)
        if match:
            # 패턴이 일치하면 해당 부분만 제거하고 다음 패턴으로 넘어가지 않음
            cleaned_sentence = pattern.sub(' ', cleaned_sentence, count=1)
            break
            
    return cleaned_sentence.lstrip()

def normalize_quotes_simple(text):
    if not text or not isinstance(text, str): return text
    result = []
    for char in text:
        if ord(char) in {0x201C, 0x201D, 0x201E, 0x201F}: result.append('"')
        elif ord(char) in {0x2018, 0x2019, 0x0060}: result.append("'")
        else: result.append(char)
    return ''.join(result)

def normalize_quotes_in_dataframe(df, columns=['kor_sentence_cleaned', 'eng_sentence_cleaned']):
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: normalize_quotes_simple(x) if pd.notna(x) else x)
    return df_copy

def extract_punctuation(text):
    if not text or not isinstance(text, str): return []
    end_periods = re.findall(r'\.$', text)
    other_punct = re.findall(r'[!?:;%\"`~…]', text)
    return end_periods + other_punct

def compare_punctuation(kor_text, eng_text):
    kor_punct = extract_punctuation(kor_text) if kor_text else []
    eng_punct = extract_punctuation(eng_text) if eng_text else []
    kor_counter, eng_counter = Counter(kor_punct), Counter(eng_punct)
    differences = {}
    for punct in set(kor_punct + eng_punct):
        kor_count, eng_count = kor_counter.get(punct, 0), eng_counter.get(punct, 0)
        if kor_count != eng_count:
            differences[punct] = {'korean': kor_count, 'english': eng_count, 'diff': kor_count - eng_count}
    return {
        'match_type': 'match' if kor_punct == eng_punct else 'no_match',
        'kor_punct_str': ''.join(kor_punct),
        'eng_punct_str': ''.join(eng_punct),
        'differences': differences,
    }

def analyze_punctuation_dataframe(df, kor_col='kor_sentence_cleaned', eng_col='eng_sentence_cleaned'):
    df_result = df.copy()
    results = [compare_punctuation(row.get(kor_col), row.get(eng_col)) for _, row in df.iterrows()]
    df_result['kor_punct'] = [r['kor_punct_str'] for r in results]
    df_result['eng_punct'] = [r['eng_punct_str'] for r in results]
    df_result['punct_match_type'] = [r['match_type'] for r in results]
    df_result['punct_differences'] = [r['differences'] for r in results]
    return df_result

class IntegratedNumberComparer:
    def __init__(self):
        self._NUMERIC_PATTERNS = [
            re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?'), 
            re.compile(r'\d+\.\d+'), 
            re.compile(r'\d+-\d+'), 
            re.compile(r'\b(\d+)(?:st|nd|rd|th)\b', re.IGNORECASE), 
            re.compile(r'\d+')
        ]
        eng_mappings = {'1':['one','first','primary','January','Jan','single','uni','once','mono'],'2':['two','second','secondary','February','Feb','dual','double','bi','twin','pair','couple','twice','di'],'3':['three','third','March','Mar','triple','tri','trio','thrice'],'4':['four','fourth','April','Apr','quad','tetra','quartet','quarter'],'5':['five','fifth','May','penta','quintet'],'6':['six','sixth','June','Jun','hexa','sextet'],'7':['seven','seventh','July','Jul','septet','hepta'],'8':['eight','eighth','August','Aug','octet','octa'],'9':['nine','ninth','September','Sep','Sept','nona'],'10':['ten','tenth','October','Oct','deca','decade'],'11':['eleven','eleventh','November','Nov'],'12':['twelve','twelfth','December','Dec','dozen'],'13':['thirteen','thirteenth'],'14':['fourteen','fourteenth'],'15':['fifteen','fifteenth'],'16':['sixteen','sixteenth'],'17':['seventeen','seventeenth'],'18':['eighteen','eighteenth'],'19':['nineteen','nineteenth'],'20':['twenty','twentieth'],'30':['thirty','thirtieth'],'40':['forty','fortieth'],'50':['fifty','fiftieth'],'60':['sixty','sixtieth'],'70':['seventy','seventieth'],'80':['eighty','eightieth'],'90':['ninety','ninetieth'],'100':['hundred'],'1000':['thousand'],'0':['zero','oh']}
        self.text_to_num_map = {text: num for num, texts in eng_mappings.items() for text in (texts + [t.capitalize() for t in texts if t and not t[0].isupper()])}
        kor_mappings = {'2': ['이중', '더블']}
        self.kor_text_to_num_map = {text: num for num, texts in kor_mappings.items() for text in texts}
        kor_keywords = self.kor_text_to_num_map.keys()
        if kor_keywords:
            self._KOR_NUMERIC_TEXT_PATTERN = re.compile('|'.join(re.escape(k) for k in kor_keywords))
        else:
            self._KOR_NUMERIC_TEXT_PATTERN = None

    # [수정됨] 반환 타입을 List[str]로 변경하고, set 대신 list를 사용합니다.
    def _extract_korean_numeric_tokens(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text: return []
        nums = [] # ✅ set() -> []
        for pat in self._NUMERIC_PATTERNS: 
            # ✅ nums.add -> nums.append
            text = pat.sub(lambda m: nums.append(m.group(0).replace(',','')) or ' ', text)
        return nums

    # [수정됨] 반환 타입을 List[str]로 변경하고, set 대신 list를 사용합니다.
    def _extract_english_numeric_tokens(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text: return []
        # ✅ set() -> []
        nums = [digit for sup, digit in {'⁰':'0','¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9'}.items() if sup in text]
        for pat in self._NUMERIC_PATTERNS: 
            # ✅ nums.add -> nums.append
            text = pat.sub(lambda m: nums.append(m.group(0).replace(',','')) or ' ', text)
        return nums

    # [수정됨] 반환 타입을 List[str]로 변경하고, set 대신 list를 사용합니다.
    def _map_textual_numbers(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text: return []
        found_nums = []
        tokens = re.findall(r'[\w-]+', text.lower())  # 소문자로 변환
        
        # 매핑도 소문자로 변환해서 비교
        lower_text_to_num_map = {k.lower(): v for k, v in self.text_to_num_map.items()}
        
        for token in tokens:
            # 1. 단독 단어 매칭 (소문자 매핑 사용)
            if token in lower_text_to_num_map:
                found_nums.append(lower_text_to_num_map[token])
                continue
            
            # 2. 하이픈이 있는 토큰 처리
            if '-' in token:
                token_parts = token.split('-')
                for part in token_parts:
                    for num_word, num_val in lower_text_to_num_map.items():
                        if part.startswith(num_word) and len(num_word) >= 2:
                            found_nums.append(num_val)
                            break
                    else:
                        continue
                    break
                continue
            
            # 3. 하이픈 없는 단어 - 단어 시작으로만 매칭
            for num_word, num_val in lower_text_to_num_map.items():
                if token.startswith(num_word) and len(num_word) >= 2:
                    found_nums.append(num_val)
                    break
                        
        return found_nums

    # [수정됨] 반환 타입을 List[str]로 변경하고, set 대신 list를 사용합니다.
    def _map_korean_textual_numbers(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text or not self._KOR_NUMERIC_TEXT_PATTERN:
            return []
        found_tokens = self._KOR_NUMERIC_TEXT_PATTERN.findall(text)
        # ✅ set comprehension -> list comprehension
        return [self.kor_text_to_num_map[token] for token in found_tokens]            

    # [수정됨] Counter 객체를 받아 빈도수 차이를 계산하는 로직으로 전면 수정
    def _calculate_number_differences(self, kor_counter: Counter, eng_counter: Counter) -> Dict[str, Dict[str, int]]:
        differences = {}
        all_nums = sorted(list(kor_counter.keys() | eng_counter.keys()))

        for num in all_nums:
            kor_count = kor_counter.get(num, 0)
            eng_count = eng_counter.get(num, 0)
            if kor_count != eng_count:
                differences[num] = {
                    'korean': kor_count, 
                    'english': eng_count, 
                    'diff': kor_count - eng_count
                }
        return differences

    # [수정됨] 리스트와 Counter를 사용하여 전체 비교 로직을 재구성
    def compare(self, kor_text: str, eng_text: str) -> Dict[str, Any]:
        # 1. 실제 숫자만 먼저 추출 (텍스트 매핑 없이)
        kor_nums = self._extract_korean_numeric_tokens(kor_text)
        eng_nums = self._extract_english_numeric_tokens(eng_text)

        # 2. 초기 Counter 생성
        kor_counter = Counter(kor_nums)
        eng_counter = Counter(eng_nums)

        # 3. 상태 정의 함수 (Counter 비교 기반)
        def get_status(k_counter, e_counter):
            if not k_counter and not e_counter:
                return "no_numbers"
            if k_counter == e_counter:
                return "all_match"
            # 교집합(Counter의 & 연산)이 있으면 partial_match
            if bool(k_counter & e_counter):
                return "partial_match"
            return "no_match"

        # 4. 초기 상태 확인
        initial_status = get_status(kor_counter, eng_counter)

        # 5. 매핑 시도 조건: no_numbers나 all_match가 아닐 때만
        if initial_status != "no_numbers" and initial_status != "all_match":
            # 조건 A: 한쪽만 숫자 있음
            if bool(kor_counter) != bool(eng_counter):
                if kor_counter and not eng_counter:
                    # 한국어만 숫자 있음 → 영어에서 텍스트→숫자 매핑
                    eng_nums.extend(self._map_textual_numbers(eng_text))
                else:
                    # 영어만 숫자 있음 → 한국어에서 텍스트→숫자 매핑
                    kor_nums.extend(self._map_korean_textual_numbers(kor_text))
            # 조건 B: 둘 다 숫자 있지만 불일치
            elif kor_counter and eng_counter:
                # 양쪽 모두 텍스트→숫자 매핑 시도
                kor_nums.extend(self._map_korean_textual_numbers(kor_text))
                eng_nums.extend(self._map_textual_numbers(eng_text))

            # 재계산
            kor_counter = Counter(kor_nums)
            eng_counter = Counter(eng_nums)

        # 6. 결과 반환
        return {
            "kor_numbers": sorted(kor_nums),
            "eng_numbers_after_mapping": sorted(eng_nums),
            "number_match_status": get_status(kor_counter, eng_counter),
            "num_differences": self._calculate_number_differences(kor_counter, eng_counter)
        }

def analyze_dataframe_with_comparer(df: pd.DataFrame, comparer: IntegratedNumberComparer, kor_col: str = "kor_sentence_cleaned", eng_col: str = "eng_sentence_cleaned") -> pd.DataFrame:
    results = [comparer.compare(str(row.get(kor_col, "")), str(row.get(eng_col, ""))) for _, row in df.iterrows()]
    out = df.copy()
    out["number_match_status"] = [r["number_match_status"] for r in results]
    out["kor_numbers"] = [", ".join(r["kor_numbers"]) for r in results]
    out["eng_numbers_after_mapping"] = [", ".join(r["eng_numbers_after_mapping"]) for r in results]
    out["num_differences"] = [r["num_differences"] for r in results]
    return out

# [수정됨] 단어 빈도수까지 비교하도록 로직 전면 수정
def compare_english_words(kor_text: str, eng_text: str) -> Dict[str, Any]:
    word_pattern = r'[a-zA-Z]+(?:[.-][a-zA-Z]+)*'
    normalize = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
    roman_map = {'i':'1','ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10'}

    # 1. [수정] 한국어 문장에서 영어 단어를 찾아 Counter로 빈도를 계산합니다.
    kor_eng_words_orig = re.findall(word_pattern, kor_text or "")
    if not kor_eng_words_orig:
        return {"kor_eng_words": [], "status": "no_eng_in_kor", "differences": {}}

    # 소문자로 변환하여 Counter 생성
    kor_word_counter = Counter(w.lower() for w in kor_eng_words_orig)

    # 2. [수정] 영어 문장에서도 모든 단어의 빈도를 Counter로 계산합니다.
    #    - 정규 표현식으로 찾은 단어와, '1st' -> 'st' 같은 접미사도 포함합니다.
    eng_lower = (eng_text or "").lower()
    eng_words = re.findall(word_pattern, eng_lower) + re.findall(r'\d+([a-zA-Z]+)', eng_lower)
    eng_word_counter = Counter(eng_words)
    eng_nums = set(re.findall(r'\d+', (eng_text or "")))

    # 3. [수정] 영어 Counter에 단어의 변형(복수형, 하이픈 등)을 추가로 고려합니다.
    #    - 예를 들어 'axes'가 있으면 'axis'도 카운트 되도록 합니다.
    #    - 주의: 원본 Counter를 복사해서 사용해야 루프에 영향을 주지 않습니다.
    for word, count in list(eng_word_counter.items()):
        # 복수형 처리 ('ies' -> 'y', 's' -> '')
        if word.endswith('ies'):
            eng_word_counter[word[:-3] + 'y'] += count
        elif len(word) > 2 and word.endswith('s'):
            eng_word_counter[word[:-1]] += count
        # 하이픈 처리 ('state-of-the-art' -> 'state', 'of', 'the', 'art')
        if '-' in word:
            for part in word.split('-'):
                if part:
                    eng_word_counter[part] += count

    # 4. [수정] Counter 뺄셈으로 부족한 단어와 개수를 계산합니다.
    missing_counter = Counter()
    for word, kor_count in kor_word_counter.items():
        # 로마 숫자 특별 처리 (e.g., 'i', 'ii')
        is_roman_match = word in roman_map and roman_map[word] in eng_nums
        if is_roman_match:
            continue

        # 영어 문장에서 해당 단어의 가용 개수 확인
        eng_count = eng_word_counter.get(word, 0)
        
        # 필요한 개수보다 부족하다면, 부족한 만큼 missing_counter에 추가
        if kor_count > eng_count:
            missing_counter[word] = kor_count - eng_count

    # 5. [수정] 결과를 새로운 형식에 맞게 반환합니다.
    missing_dict = dict(missing_counter)
    status = "all_match" if not missing_dict else "no_match" if len(missing_dict) == len(kor_word_counter) else "partial_match"
    
    return {
        "kor_eng_words": sorted(list(kor_word_counter.keys())), 
        "status": status, 
        "differences": missing_dict # ✅ List 대신 Dict(단어: 부족한 개수) 반환
    }

# [수정됨] differences 컬럼에 Dict가 저장되도록 기존 구조 유지
def analyze_english_words_in_korean(df: pd.DataFrame, kor_col: str = "kor_sentence_cleaned", eng_col: str = "eng_sentence_cleaned") -> pd.DataFrame:
    results = [compare_english_words(row.get(kor_col, ""), row.get(eng_col, "")) for _, row in df.iterrows()]
    df_out = df.copy()
    df_out['kor_eng_words'] = [r['kor_eng_words'] for r in results]
    df_out['eng_word_match_status'] = [r['status'] for r in results]
    # ✅ 반환 형식이 List에서 Dict로 변경되었지만, 그대로 컬럼에 저장하면 됩니다.
    df_out['eng_word_differences'] = [r['differences'] for r in results]
    return df_out

ALLOWED_CHARS_PATTERN = re.compile(r'[a-zA-Z0-9_ㄱ-ㅎㅏ-ㅣ가-힣\s.,!?:;\'\"`~%()’“”·/&-]')
def extract_special_symbols(text: str) -> List[str]:
    if not isinstance(text, str): return []
    return list(ALLOWED_CHARS_PATTERN.sub('', text))

def compare_special_symbols(kor_text: str, eng_text: str) -> Dict[str, Any]:
    kor_text, eng_text = unicodedata.normalize("NFKC", kor_text or ""), unicodedata.normalize("NFKC", eng_text or "")
    special_map = {'¹':'1','²':'2','³':'3','⁴':'4','⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9'}
    k_sym, e_sym, k_num, e_num = Counter(extract_special_symbols(kor_text)), Counter(extract_special_symbols(eng_text)), Counter(re.findall(r'\d',kor_text)), Counter(re.findall(r'\d',eng_text))
    k_rem, e_rem = k_sym.copy(), e_sym.copy()
    common = k_rem & e_rem; k_rem -= common; e_rem -= common
    for sym, count in k_rem.copy().items():
        if (d:=special_map.get(sym)) and e_num[d]>0: m=min(count,e_num[d]); k_rem[sym]-=m; e_num[d]-=m
    for sym, count in e_rem.copy().items():
        if (d:=special_map.get(sym)) and k_num[d]>0: m=min(count,k_num[d]); e_rem[sym]-=m; k_num[d]-=m
    k_rem += Counter(); e_rem += Counter()
    total_initial, total_rem = sum(k_sym.values())+sum(e_sym.values()), sum(k_rem.values())+sum(e_rem.values())
    status = "no_special_symbols" if total_initial == 0 else "all_match" if total_rem == 0 else "partial_match" if total_initial > total_rem else "no_match"
    diffs = {s:{'korean':k_rem[s],'english':e_rem[s]} for s in k_rem.keys()|e_rem.keys() if k_rem[s]>0 or e_rem[s]>0}
    return {'kor_special_symbols': "".join(sorted(k_sym.elements())), 'eng_special_symbols': "".join(sorted(e_sym.elements())), 'symbol_match_status': status, 'symbol_differences': diffs}

def analyze_special_symbols_dataframe(df: pd.DataFrame, kor_col: str = "kor_sentence_cleaned", eng_col: str = "eng_sentence_cleaned") -> pd.DataFrame:
    results = [compare_special_symbols(str(row.get(kor_col, "")), str(row.get(eng_col, ""))) for _, row in df.iterrows()]
    df_out = df.copy()
    df_out['kor_special_symbols'] = [r['kor_special_symbols'] for r in results]
    df_out['eng_special_symbols'] = [r['eng_special_symbols'] for r in results]
    df_out['symbol_match_status'] = [r['symbol_match_status'] for r in results]
    df_out['symbol_differences'] = [r['symbol_differences'] for r in results]
    return df_out

def check_only_eng_korean_sentence(kor_text: str, eng_text: str) -> str:
    """
    한-영 문장 쌍을 분석하여 상태를 반환합니다.

    Returns:
        - "invalid_input": 입력값이 문자열이 아닌 경우
        - "empty": 입력 문장이 모두 비어있는 경우
        - "no_korean_in_kor": 한국어 문장에 한글이 없는 경우
        - "all_match": 대소문자 무시하고 영어와 한국어 문장이 완전히 같은 경우
        - "partial_match": 공백 제거 후 영어와 한국어 문장이 같은 경우
        - "valid_candidate": 위의 모든 조건에 해당하지 않는, 유효한 번역 코퍼스 후보
    """
    # 1. 타입 유효성 검사
    if not isinstance(kor_text, str) or not isinstance(eng_text, str):
        return "invalid_input"

    kor_lower, eng_lower = kor_text.lower().strip(), eng_text.lower().strip()

    # 2. 빈 문자열 처리
    if not kor_lower and not eng_lower:
        return "empty"
    
    # 4. 완전 일치 검사
    if kor_lower == eng_lower:
        return "all_match"
    
    # 5. 공백 제거 후 부분 일치 검사
    kor_no_space = re.sub(r'\s+', '', kor_lower)
    eng_no_space = re.sub(r'\s+', '', eng_lower)
    if kor_no_space == eng_no_space:
        return "partial_match"
    
    # 3. (핵심 추가) 한국어 문장에 한글이 없는 경우 "no_korean_in_kor"로 분류
    # 정규식: 자음/모음(ㄱ-ㅎ, ㅏ-ㅣ) 또는 완성형 한글(가-힣)
    if not re.search(r'[\u3131-\u318E\uAC00-\uD7A3]', kor_lower):
        return "no_korean_in_kor"
        
    # 6. 모든 필터링을 통과한 유효한 후보군
    return "valid_candidate"    

def analyze_only_eng_korean_sentence(df: pd.DataFrame, kor_col: str, eng_col: str) -> pd.DataFrame:
    df_out = df.copy()
    if 'only_eng_korean_sentence' in df_out.columns:
        df_out = df_out.drop(columns=['only_eng_korean_sentence'])
    df_out['only_eng_korean_sentence'] = [check_only_eng_korean_sentence(str(row.get(kor_col, "")), str(row.get(eng_col, ""))) for _, row in df.iterrows()]
    return df_out

def is_numbers_only(text: str) -> bool:
    if not isinstance(text, str) or not text.strip(): return False
    return not bool(re.sub(r'[\d\s,.]', '', text))

def is_symbols_or_single_alphabet_only(text: str) -> bool:
    """
    문장이 아래 조건 중 하나에 해당하면 True를 반환합니다.
    1. 알파벳, 한글, 숫자 없이 '단 하나의 기호'로만 구성된 경우
    2. 대소문자 상관없이 '단 하나의 알파벳'으로만 구성된 경우
    """
    if not isinstance(text, str):
        return False
    
    # 1. 양쪽 공백을 제거하고 'nan' 문자열 처리
    cleaned_text = text.strip().replace('nan', '')
    
    # 2. 정리된 문자열의 길이가 정확히 1인지 확인
    if len(cleaned_text) == 1:
        char = cleaned_text[0]
        
        # 3. 아래 두 조건 중 하나라도 만족하면 True (제거 대상)
        # 조건 A: 해당 문자가 기호일 경우 (숫자, 한글, 알파벳이 아님)
        is_symbol = not re.match(r'^[a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ가-힣]$', char)
        
        # 조건 B: 해당 문자가 알파벳일 경우
        is_alphabet = re.match(r'^[a-zA-Z]$', char)
        
        if is_symbol or is_alphabet:
            return True
            
    # 길이가 1이 아니거나, 위 조건에 해당하지 않으면 False (제거 대상 아님)
    return False

# ==============================================================================
# 2. 메인 자동화 함수 (요구사항에 맞춰 재작성)
# ==============================================================================
def process_and_filter_files(input_dir: str, metadata_output_dir: str, config_path: str = None, config_obj: dict = None):
    # config_obj가 제공되면 해당 config 사용, 그렇지 않으면 config_path로 로드
    if config_obj is not None:
        # config 객체가 직접 제공된 경우 전역 패턴 변수 업데이트
        global COMMON_NUMBERING_PATTERNS, TEXT_ONLY_NUMBERING_PATTERNS
        COMMON_NUMBERING_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in config_obj['numbering_patterns']['common']]
        TEXT_ONLY_NUMBERING_PATTERNS = [re.compile(pattern) for pattern in config_obj['numbering_patterns']['text_only']]
    elif config_path is not None:
        load_config(config_path)

    input_path = Path(input_dir)
    metadata_path = Path(metadata_output_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 입력 폴더: '{input_path.resolve()}'")
    print(f"📂 메타데이터 폴더: '{metadata_path.resolve()}'")

    files_to_process = list(input_path.glob('*_final_alignment_results.xlsx'))
    if not files_to_process:
        tqdm.write(f"⚠️ 입력 폴더 '{input_dir}'에서 '_final_alignment_results.xlsx' 파일을 찾을 수 없습니다.")
        return

    filenames = sorted([p.stem.replace('_final_alignment_results', '') for p in files_to_process])
    tqdm.write(f"🔍 총 {len(filenames)}개의 파일을 처리합니다.")

    # --- 내부 헬퍼 함수 정의 ---
    def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
        display_cols = ['kor_sentence_cleaned', 'eng_sentence_cleaned', 'kor_sentence_normalized', 'eng_sentence_normalized']
        status_cols  = ['punct_match_type', 'number_match_status', 'eng_word_match_status', 'symbol_match_status', 'only_eng_korean_sentence']
        def is_analysis_aux(c: str) -> bool: return any(s in c for s in ['_cleaned', '_normalized', '_match_', '_differences', '_numbers', '_punct', '_symbols', '_words'])
        existing_display = [c for c in display_cols if c in df.columns]
        existing_status  = [c for c in status_cols  if c in df.columns]
        original_cols = [c for c in df.columns if not is_analysis_aux(c) and c not in existing_display and c not in existing_status]
        remaining_analysis_cols = [c for c in df.columns if c not in original_cols + existing_display + existing_status]
        final_order = original_cols + existing_display + existing_status + remaining_analysis_cols
        final_order_unique = []
        seen = set()
        for c in final_order:
            if c not in seen:
                final_order_unique.append(c)
                seen.add(c)
        return df[final_order_unique]

    # [수정됨] 문맥(context) 파라미터를 추가하여 분석을 수행하는 내부 함수
    def _run_full_analysis(df: pd.DataFrame, kor_col: str, eng_col: str, context: str, config_path: str = None) -> pd.DataFrame:
        """
        데이터프레임에 대한 전체 분석을 수행합니다.
        :param df: 분석할 데이터프레임
        :param kor_col: 한국어 원본 컬럼명
        :param eng_col: 영어 원본 컬럼명
        :param context: 넘버링 제거를 위한 문맥 ('text' 또는 'table')
        :param config_path: config 파일 경로 (선택사항)
        """
        df_copy = df.copy()
        df_copy['kor_sentence_cleaned'] = df_copy[kor_col].astype(str)
        df_copy['eng_sentence_cleaned'] = df_copy[eng_col].astype(str)
        
        for col in ['kor_sentence_cleaned', 'eng_sentence_cleaned']:
            df_copy[col] = (df_copy[col].str.replace(r'^[-\u2010\u2013\u2014\u2212·•○:.]\s*', '', regex=True)
                                      .str.replace(r'\s*\*\s*', ' ', regex=True)
                                      .str.strip())
        df_copy['eng_sentence_cleaned'] = df_copy['eng_sentence_cleaned'].str.replace(r'\s*\((IGC|IBC) Code \d+\.\d+\)', '', regex=True)
        
        # ✅ [핵심 변경] context를 전달하여 넘버링 제거 함수 호출
        for col in ['kor_sentence_cleaned', 'eng_sentence_cleaned']:
            df_copy[col] = df_copy[col].apply(lambda x: remove_numbering(x, context=context, config_path=config_path))

        df_copy = normalize_quotes_in_dataframe(df_copy, columns=['kor_sentence_cleaned', 'eng_sentence_cleaned'])

        for col in ['kor_sentence_cleaned', 'eng_sentence_cleaned']:
            df_copy[col] = df_copy[col].str.replace(r'-{2,}', '', regex=True).str.strip()

        strip_chars = ':;* '
        for col in ['kor_sentence_cleaned', 'eng_sentence_cleaned']:
            df_copy[col] = df_copy[col].str.strip(strip_chars)

        df_copy['kor_sentence_normalized'] = df_copy['kor_sentence_cleaned'].apply(lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x)
        df_copy['eng_sentence_normalized'] = df_copy['eng_sentence_cleaned'].apply(lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x)
        
        number_comparer = IntegratedNumberComparer()
        df_copy = analyze_punctuation_dataframe(df_copy, kor_col='kor_sentence_normalized', eng_col='eng_sentence_normalized')
        df_copy = analyze_dataframe_with_comparer(df_copy, number_comparer, kor_col='kor_sentence_normalized', eng_col='eng_sentence_normalized')
        df_copy = analyze_english_words_in_korean(df_copy, kor_col='kor_sentence_normalized', eng_col='eng_sentence_normalized')
        df_copy = analyze_special_symbols_dataframe(df_copy, kor_col='kor_sentence_normalized', eng_col='eng_sentence_normalized')
        df_copy = analyze_only_eng_korean_sentence(df_copy, kor_col='kor_sentence_normalized', eng_col='eng_sentence_normalized')

        return _reorder_columns(df_copy)

    def _apply_content_filters(df: pd.DataFrame) -> pd.DataFrame:
        kor_col, eng_col = 'kor_sentence_normalized', 'eng_sentence_normalized'
        masks = [
            df.apply(lambda r: is_numbers_only(r[kor_col]) and is_numbers_only(r[eng_col]), axis=1),
            df.apply(lambda r: is_symbols_or_single_alphabet_only(r[kor_col]) or is_symbols_or_single_alphabet_only(r[eng_col]), axis=1)
        ]
        is_to_be_filtered_out = pd.concat(masks, axis=1).any(axis=1)
        return df[~is_to_be_filtered_out]

    def _create_empty_table_dataframe() -> pd.DataFrame:
        """표준화된 빈 Table DataFrame을 생성합니다."""
        # Text 시트와 동일한 컬럼 구조를 가진 빈 DataFrame
        standard_columns = [
            'korean_sentence', 'english_sentence',
            'kor_sentence_cleaned', 'eng_sentence_cleaned',
            'kor_sentence_normalized', 'eng_sentence_normalized',
            'punct_match_type', 'number_match_status',
            'eng_word_match_status', 'symbol_match_status',
            'only_eng_korean_sentence',
            'kor_punct', 'eng_punct', 'punct_differences',
            'kor_numbers', 'eng_numbers_after_mapping', 'num_differences',
            'kor_eng_words', 'eng_word_differences',
            'kor_special_symbols', 'eng_special_symbols', 'symbol_differences'
        ]
        return pd.DataFrame(columns=standard_columns)

    def _process_table_sheet(df_table_original: pd.DataFrame) -> pd.DataFrame:
        """Table 시트를 처리하여 최종 DataFrame을 반환합니다."""

        # 1. 기본 검증
        if len(df_table_original) == 0:
            tqdm.write(f"  - ⚠️ Table 시트가 비어있습니다.")
            return _create_empty_table_dataframe()

        tqdm.write(f"  - 📋 Table 시트 컬럼: {list(df_table_original.columns)}")

        # 2. match_source 필터링
        if 'match_source' in df_table_original.columns:
            df_filtered = df_table_original[df_table_original['match_source'].notna()].copy()
            tqdm.write(f"  - 📝 match_source 필터링 후: {len(df_filtered)}행")
        else:
            tqdm.write(f"  - ⚠️ match_source 컬럼 없음, 전체 데이터 사용")
            df_filtered = df_table_original.copy()

        # 3. 필수 컬럼 검증
        required_columns = ['korean_sentence', 'english_sentence']
        if not all(col in df_filtered.columns for col in required_columns):
            tqdm.write(f"  - ⚠️ 필수 컬럼 누락: {required_columns}")
            tqdm.write(f"  - 📋 사용 가능한 컬럼: {list(df_filtered.columns)}")
            return _create_empty_table_dataframe()

        # 4. 데이터 처리 파이프라인
        if len(df_filtered) == 0:
            tqdm.write(f"  - ⚠️ 필터링 후 데이터 없음")
            return _create_empty_table_dataframe()

        try:
            # 분석 → 필터링 → 중복제거
            analyzed_df = _run_full_analysis(df_filtered, 'korean_sentence', 'english_sentence', context='table', config_path=config_path)
            tqdm.write(f"  - 🔬 Table 시트 분석 완료: {len(analyzed_df)}행")

            filtered_df = _apply_content_filters(analyzed_df)
            tqdm.write(f"  - 🧹 Table 시트 콘텐츠 필터링 후: {len(filtered_df)}행")

            final_df = filtered_df.drop_duplicates(
                subset=['kor_sentence_normalized', 'eng_sentence_normalized'],
                keep='first'
            )
            tqdm.write(f"  - 🔄 Table 시트 중복 제거 후: {len(final_df)}행")
            return final_df

        except Exception as e:
            tqdm.write(f"  - ❌ Table 처리 중 오류: {e}")
            return _create_empty_table_dataframe()

    # --- 메인 파일 처리 루프 ---
    for filename in tqdm(filenames, desc="전체 파일 처리 중"):
        input_file = input_path / f'{filename}_final_alignment_results.xlsx'
        tqdm.write(f"\n--- 📄 '{filename}' 파일 처리 시작 ---")

        try:
            tqdm.write(f"  - 📖 Excel 파일 읽기 시작...")
            df_text_original = pd.read_excel(input_file, sheet_name='Text', engine='openpyxl')
            tqdm.write(f"  - 📊 Text 시트 읽기 완료: {len(df_text_original)}행")
            df_table_original = pd.read_excel(input_file, sheet_name='Table', engine='openpyxl')
            tqdm.write(f"  - 📊 Table 시트 읽기 완료: {len(df_table_original)}행")

            # --- Text 시트 처리 ---
            tqdm.write(f"  - 🔍 Text 시트 처리 시작...")
            df_text_filtered_initial = df_text_original[df_text_original['type'] != 'subordinate_unmatched'].copy()
            tqdm.write(f"  - 📝 Text 시트 필터링 후: {len(df_text_filtered_initial)}행")

            # ✅ context='text'로 분석 함수 호출
            analyzed_text_df = _run_full_analysis(df_text_filtered_initial, 'kor_sentence', 'eng_sentence', context='text', config_path=config_path)
            tqdm.write(f"  - 🔬 Text 시트 분석 완료: {len(analyzed_text_df)}행")

            final_text_df = _apply_content_filters(analyzed_text_df)
            tqdm.write(f"  - 🧹 Text 시트 콘텐츠 필터링 후: {len(final_text_df)}행")

            final_text_df = final_text_df.drop_duplicates(
                subset=['kor_sentence_normalized', 'eng_sentence_normalized'],
                keep='first'
            )
            tqdm.write(f"  - 🔄 Text 시트 중복 제거 후: {len(final_text_df)}행")

            # --- Table 시트 처리 ---
            tqdm.write(f"  - 🔍 Table 시트 처리 시작...")
            final_table_df = _process_table_sheet(df_table_original)

            # 최종 결과가 비어있는지 확인
            if len(final_text_df) == 0 and len(final_table_df) == 0:
                tqdm.write(f"  - ⚠️ 경고: 모든 데이터가 필터링되어 저장할 내용이 없습니다.")
                continue

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_filename = metadata_path / f"{filename}_filtering_metadata_{timestamp}.xlsx"
            tqdm.write(f"  - 💾 Excel 파일 저장 시작: {metadata_filename.name}")

            with pd.ExcelWriter(metadata_filename, engine='openpyxl') as writer:
                for df in [final_text_df, final_table_df]:
                    for col in [c for c in df.columns if 'score' in c.lower() or 'similarity' in c.lower()]:
                        if col in df.columns:
                            df[col] = df[col].round(3)

                final_text_df.to_excel(writer, sheet_name='Text_Analyzed_Filtered', index=False)
                final_table_df.to_excel(writer, sheet_name='Table_Filtered', index=False)

            tqdm.write(f"  - ✅ 메타데이터 파일 저장 완료 -> {metadata_filename.name}")

        except Exception as e:
            import traceback
            tqdm.write(f"  - ❌ '{filename}' 처리 중 오류 발생: {e}")
            tqdm.write(f"  - 📋 상세 오류 정보:")
            for line in traceback.format_exc().splitlines():
                tqdm.write(f"     {line}")
            continue

if __name__ == '__main__':
    # 사용 예시:
    INPUT_FOLDER = 'final_aligned_results'  # 실제 입력 폴더 경로로 변경하세요.
    OUTPUT_FOLDER = 'filtering_metadata'     # 실제 출력 폴더 경로로 변경하세요.
    process_and_filter_files(INPUT_FOLDER, OUTPUT_FOLDER)