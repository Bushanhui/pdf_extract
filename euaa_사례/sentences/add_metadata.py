"""
merged_outer.csv에 메타데이터를 추가하는 스크립트

metadata_generator.py의 함수들을 활용하여:
- 텍스트 정규화 (cleaned, normalized)
- 구두점 매칭 분석
- 숫자 매칭 분석
- 영어 단어 매칭 분석
- 특수 기호 매칭 분석
- 한영 문장 품질 분석
- 통계 계산 (word_count, word_ratio, chr_len_ratio, potential_split)
"""

import pandas as pd
import unicodedata
from pathlib import Path

from metadata_generator import (
    load_config,
    remove_numbering,
    normalize_quotes_in_dataframe,
    analyze_punctuation_dataframe,
    IntegratedNumberComparer,
    analyze_dataframe_with_comparer,
    analyze_english_words_in_korean,
    analyze_special_symbols_dataframe,
    analyze_only_eng_korean_sentence
)


def add_metadata_to_csv(
    input_csv: str = "merged_1부_outer.csv",
    output_csv: str = "merged_1부_outer_with_metadata.csv",
    config_path: str = "config.json"
):
    """
    merged_outer.csv에 메타데이터를 추가합니다.

    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        config_path: config 파일 경로
    """
    print(f"=== 메타데이터 추가 시작 ===")
    print(f"입력 파일: {input_csv}")
    print(f"출력 파일: {output_csv}")

    # 1. config 로드
    print("\n1. Config 로드 중...")
    load_config(config_path)
    print("   ✓ Config 로드 완료")

    # 2. CSV 로드
    print("\n2. CSV 파일 로드 중...")
    df = pd.read_csv(input_csv)
    print(f"   ✓ 총 {len(df)}개 행 로드")
    print(f"   ✓ 컬럼: {list(df.columns)}")

    # 3. 텍스트 정규화
    print("\n3. 텍스트 정규화 중...")

    # 3-1. NaN 처리 및 문자열 변환
    df['kr_text'] = df['kr_text'].fillna('').astype(str)
    df['en_text'] = df['en_text'].fillna('').astype(str)

    # 3-2. cleaned 버전 생성 (기본 정리)
    df['kr_text_cleaned'] = df['kr_text'].copy()
    df['en_text_cleaned'] = df['en_text'].copy()

    # 선행/후행 기호 제거
    for col in ['kr_text_cleaned', 'en_text_cleaned']:
        df[col] = (df[col]
                   .str.replace(r'^[-\u2010\u2013\u2014\u2212·•○:.]\s*', '', regex=True)
                   .str.replace(r'\s*\*\s*', ' ', regex=True)
                   .str.strip())

    # 영어 특정 패턴 제거
    df['en_text_cleaned'] = df['en_text_cleaned'].str.replace(
        r'\s*\((IGC|IBC) Code \d+\.\d+\)', '', regex=True
    )

    # numbering 제거 (빈 패턴이므로 실제로는 아무것도 제거 안 됨)
    for col in ['kr_text_cleaned', 'en_text_cleaned']:
        df[col] = df[col].apply(lambda x: remove_numbering(x, context='text', config_path=config_path))

    # 따옴표 정규화
    df = normalize_quotes_in_dataframe(df, columns=['kr_text_cleaned', 'en_text_cleaned'])

    # 연속 하이픈 제거
    for col in ['kr_text_cleaned', 'en_text_cleaned']:
        df[col] = df[col].str.replace(r'-{2,}', '', regex=True).str.strip()

    # 특정 문자 제거
    strip_chars = ':;* '
    for col in ['kr_text_cleaned', 'en_text_cleaned']:
        df[col] = df[col].str.strip(strip_chars)

    # 3-3. normalized 버전 생성 (NFKC 정규화)
    df['kr_text_normalized'] = df['kr_text_cleaned'].apply(
        lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x
    )
    df['en_text_normalized'] = df['en_text_cleaned'].apply(
        lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x
    )

    print("   ✓ 텍스트 정규화 완료")

    # 4. 메타데이터 분석
    print("\n4. 메타데이터 분석 중...")

    # 4-1. 구두점 분석
    print("   - 구두점 매칭 분석...")
    df = analyze_punctuation_dataframe(
        df,
        kor_col='kr_text_normalized',
        eng_col='en_text_normalized'
    )

    # 4-2. 숫자 분석
    print("   - 숫자 매칭 분석...")
    number_comparer = IntegratedNumberComparer()
    df = analyze_dataframe_with_comparer(
        df,
        number_comparer,
        kor_col='kr_text_normalized',
        eng_col='en_text_normalized'
    )

    # 4-3. 영어 단어 분석
    print("   - 영어 단어 매칭 분석...")
    df = analyze_english_words_in_korean(
        df,
        kor_col='kr_text_normalized',
        eng_col='en_text_normalized'
    )

    # 4-4. 특수 기호 분석
    print("   - 특수 기호 매칭 분석...")
    df = analyze_special_symbols_dataframe(
        df,
        kor_col='kr_text_normalized',
        eng_col='en_text_normalized'
    )

    # 4-5. 한영 문장 품질 분석
    print("   - 한영 문장 품질 분석...")
    df = analyze_only_eng_korean_sentence(
        df,
        kor_col='kr_text_normalized',
        eng_col='en_text_normalized'
    )

    print("   ✓ 메타데이터 분석 완료")

    # 5. 추가 통계 계산
    print("\n5. 추가 통계 계산 중...")

    # 5-1. 단어 수 및 비율
    df['word_count_kr'] = df['kr_text_normalized'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    df['word_count_en'] = df['en_text_normalized'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    df['word_ratio'] = df.apply(
        lambda row: round(row['word_count_en'] / row['word_count_kr'], 2)
        if row['word_count_kr'] > 0 else 0,
        axis=1
    )

    # 5-2. 문자 길이 및 비율
    df['chr_len_kr'] = df['kr_text_normalized'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    df['chr_len_en'] = df['en_text_normalized'].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    df['chr_len_ratio'] = df.apply(
        lambda row: round(row['chr_len_en'] / row['chr_len_kr'], 2)
        if row['chr_len_kr'] > 0 else 0,
        axis=1
    )

    # 5-3. 잠재적 분할 가능성
    # matched 항목 중 한국어는 길지만 영어가 짧은 경우 (1:多 매칭 가능성)
    df['potential_split'] = (
        (df['match_type'] != 'unmatched') &
        (df['word_count_kr'] > 4) &
        (df['word_count_en'] > 0) &
        (df['word_ratio'] < 0.7) &
        (df['chr_len_ratio'] < 1.2)
    )

    print("   ✓ 추가 통계 계산 완료")

    # 6. 컬럼 순서 재정렬
    print("\n6. 컬럼 순서 재정렬 중...")

    # 원본 컬럼 (1-10)
    original_cols = [
        'kr_idx', 'kr_id', 'kr_text', 'kr_source_type',
        'en_idx', 'en_id', 'en_text', 'en_source_type',
        'similarity', 'match_type'
    ]

    # 통계 컬럼 (11-17) - match_type 바로 뒤
    stat_cols = [
        'potential_split',
        'word_ratio', 'word_count_kr', 'word_count_en',
        'chr_len_ratio', 'chr_len_kr', 'chr_len_en'
    ]

    # 매칭 상태 (18-22)
    status_cols = [
        'punct_match_type', 'number_match_status',
        'eng_word_match_status', 'symbol_match_status',
        'only_eng_korean_sentence'
    ]

    # 상세 정보 (23-33) - Dict 타입
    detail_cols = [
        'kor_punct', 'eng_punct', 'punct_differences',
        'kor_numbers', 'eng_numbers_after_mapping', 'num_differences',
        'kor_eng_words', 'eng_word_differences',
        'kor_special_symbols', 'eng_special_symbols', 'symbol_differences'
    ]

    # 정규화된 텍스트 (34-37) - 가장 뒤
    text_cols = [
        'kr_text_cleaned', 'en_text_cleaned',
        'kr_text_normalized', 'en_text_normalized'
    ]

    # 최종 컬럼 순서: original → stat → status → detail → text
    final_cols = []
    for col_list in [original_cols, stat_cols, status_cols, detail_cols, text_cols]:
        final_cols.extend([c for c in col_list if c in df.columns])

    # 누락된 컬럼 추가
    remaining_cols = [c for c in df.columns if c not in final_cols]
    final_cols.extend(remaining_cols)

    df = df[final_cols]
    print("   ✓ 컬럼 순서 재정렬 완료")

    # 7. 결과 저장
    print("\n7. 결과 저장 중...")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"   ✓ 저장 완료: {output_csv}")

    # 8. 통계 출력
    print("\n=== 처리 결과 통계 ===")
    print(f"총 행 수: {len(df)}")
    print(f"총 컬럼 수: {len(df.columns)}")
    print()

    # 매칭 상태 통계
    print("[ 매칭 상태 ]")
    print(df['match_type'].value_counts())
    print()

    # 품질 상태 통계
    print("[ 구두점 매칭 ]")
    print(df['punct_match_type'].value_counts())
    print()

    print("[ 숫자 매칭 ]")
    print(df['number_match_status'].value_counts())
    print()

    print("[ 영어 단어 매칭 ]")
    print(df['eng_word_match_status'].value_counts())
    print()

    print("[ 특수 기호 매칭 ]")
    print(df['symbol_match_status'].value_counts())
    print()

    print("[ 한영 문장 품질 ]")
    print(df['only_eng_korean_sentence'].value_counts())
    print()

    print("[ 잠재적 분할 가능성 ]")
    print(f"True: {df['potential_split'].sum()}")
    print(f"False: {(~df['potential_split']).sum()}")
    print()

    print("=== 메타데이터 추가 완료 ===")
    return df


if __name__ == "__main__":
    # 실행
    df_result = add_metadata_to_csv(
        input_csv="merged_1부_outer.csv",
        output_csv="merged_1부_outer_with_metadata.csv",
        config_path="config.json"
    )
