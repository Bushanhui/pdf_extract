"""
aligned 결과를 CSV 파일로 변환하는 스크립트

두 가지 CSV 파일 생성:
1. aligned_1부.csv: aligned_1부.json의 매칭 쌍을 직접 CSV로 변환
2. merged_1부.csv: llm_output_kr 전체 데이터에 영어 매칭 결과를 병합
"""

import json
import csv
import pandas as pd
from pathlib import Path


def create_aligned_csv(aligned_json_path: str, output_csv_path: str):
    """aligned JSON을 CSV로 변환"""
    print(f"\n=== {output_csv_path} 생성 중 ===")

    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    pairs = aligned_data['aligned_pairs']

    # CSV 데이터 준비
    rows = []
    for pair in pairs:
        row = {
            'kr_id': pair['kr'].get('id'),
            'kr_text': pair['kr'].get('text', ''),
            'kr_source_type': pair['kr'].get('source_type', ''),
            'en_id': pair['en'].get('id'),
            'en_text': pair['en'].get('text', ''),
            'en_source_type': pair['en'].get('source_type', ''),
            'similarity': pair.get('similarity', ''),
            'match_type': pair.get('type', '')
        }
        rows.append(row)

    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"✓ {len(rows)}개 매칭 쌍을 CSV로 저장했습니다.")
    return df


def create_merged_csv(kr_json_path: str, aligned_json_path: str, output_csv_path: str):
    """llm_output_kr 전체에 영어 매칭 결과를 병합한 CSV 생성 (인덱스 기반)"""
    print(f"\n=== {output_csv_path} 생성 중 ===")

    # 한국어 원본 데이터 로드
    with open(kr_json_path, 'r', encoding='utf-8') as f:
        kr_json = json.load(f)
        kr_data = kr_json['data'] if 'data' in kr_json else kr_json

    # aligned 결과 로드
    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    pairs = aligned_data['aligned_pairs']

    # 한국어 원본 인덱스 → 영어 매칭 정보 매핑
    idx_to_pair = {}
    for pair in pairs:
        kr_original_idx = pair.get('kr_original_idx')
        if kr_original_idx is not None:
            idx_to_pair[kr_original_idx] = {
                'en_id': pair['en'].get('id'),
                'en_text': pair['en'].get('text', ''),
                'en_source_type': pair['en'].get('source_type', ''),
                'en_original_idx': pair.get('en_original_idx'),
                'similarity': pair.get('similarity', ''),
                'match_type': pair.get('type', '')
            }

    # 병합 데이터 생성 (enumerate로 인덱스 확보)
    rows = []
    for idx, kr_item in enumerate(kr_data):
        kr_id = kr_item.get('id')
        kr_text = kr_item.get('text', '')

        # 기본 한국어 데이터
        row = {
            'kr_idx': idx,  # 명시적 인덱스 컬럼 추가
            'kr_id': kr_id,
            'kr_text': kr_text,
            'kr_source_type': kr_item.get('source_type', ''),
        }

        # 매칭된 영어 데이터 추가 (인덱스 기준 매칭, 없으면 빈 값)
        if idx in idx_to_pair:
            en_info = idx_to_pair[idx]
            row.update({
                'en_idx': en_info['en_original_idx'],
                'en_id': en_info['en_id'],
                'en_text': en_info['en_text'],
                'en_source_type': en_info['en_source_type'],
                'similarity': en_info['similarity'],
                'match_type': en_info['match_type']
            })
        else:
            row.update({
                'en_idx': '',
                'en_id': '',
                'en_text': '',
                'en_source_type': '',
                'similarity': '',
                'match_type': 'unmatched'
            })

        rows.append(row)

    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    matched_count = len([r for r in rows if r['match_type'] != 'unmatched'])
    unmatched_count = len([r for r in rows if r['match_type'] == 'unmatched'])

    print(f"✓ 전체 {len(rows)}개 항목 (매칭={matched_count}, 미매칭={unmatched_count})을 CSV로 저장했습니다.")
    return df


def create_merged_csv_en(en_json_path: str, aligned_json_path: str, output_csv_path: str):
    """llm_output_en 전체에 한국어 매칭 결과를 병합한 CSV 생성 (영어 기준 LEFT JOIN)"""
    print(f"\n=== {output_csv_path} 생성 중 ===")

    # 영어 원본 데이터 로드
    with open(en_json_path, 'r', encoding='utf-8') as f:
        en_json = json.load(f)
        en_data = en_json['data'] if 'data' in en_json else en_json

    # aligned 결과 로드
    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    pairs = aligned_data['aligned_pairs']

    # 영어 원본 인덱스 → 한국어 매칭 정보 매핑
    idx_to_pair = {}
    for pair in pairs:
        en_original_idx = pair.get('en_original_idx')
        if en_original_idx is not None:
            idx_to_pair[en_original_idx] = {
                'kr_id': pair['kr'].get('id'),
                'kr_text': pair['kr'].get('text', ''),
                'kr_source_type': pair['kr'].get('source_type', ''),
                'kr_original_idx': pair.get('kr_original_idx'),
                'similarity': pair.get('similarity', ''),
                'match_type': pair.get('type', '')
            }

    # 병합 데이터 생성 (영어 기준)
    rows = []
    for idx, en_item in enumerate(en_data):
        en_id = en_item.get('id')
        en_text = en_item.get('text', '')

        # 매칭된 한국어 데이터 먼저 (있으면)
        if idx in idx_to_pair:
            kr_info = idx_to_pair[idx]
            row = {
                'kr_idx': kr_info['kr_original_idx'],
                'kr_id': kr_info['kr_id'],
                'kr_text': kr_info['kr_text'],
                'kr_source_type': kr_info['kr_source_type'],
                'en_idx': idx,
                'en_id': en_id,
                'en_text': en_text,
                'en_source_type': en_item.get('source_type', ''),
                'similarity': kr_info['similarity'],
                'match_type': kr_info['match_type']
            }
        else:
            # 매칭 안 된 영어
            row = {
                'kr_idx': '',
                'kr_id': '',
                'kr_text': '',
                'kr_source_type': '',
                'en_idx': idx,
                'en_id': en_id,
                'en_text': en_text,
                'en_source_type': en_item.get('source_type', ''),
                'similarity': '',
                'match_type': 'unmatched'
            }

        rows.append(row)

    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    matched_count = len([r for r in rows if r['match_type'] != 'unmatched'])
    unmatched_count = len([r for r in rows if r['match_type'] == 'unmatched'])

    print(f"✓ 전체 {len(rows)}개 항목 (매칭={matched_count}, 미매칭={unmatched_count})을 CSV로 저장했습니다.")
    return df


def create_outer_join_csv(kr_json_path: str, en_json_path: str, aligned_json_path: str, output_csv_path: str):
    """한국어와 영어 전체를 포함한 FULL OUTER JOIN CSV 생성"""
    print(f"\n=== {output_csv_path} 생성 중 ===")

    # 한국어 원본 데이터 로드
    with open(kr_json_path, 'r', encoding='utf-8') as f:
        kr_json = json.load(f)
        kr_data = kr_json['data'] if 'data' in kr_json else kr_json

    # 영어 원본 데이터 로드
    with open(en_json_path, 'r', encoding='utf-8') as f:
        en_json = json.load(f)
        en_data = en_json['data'] if 'data' in en_json else en_json

    # aligned 결과 로드
    with open(aligned_json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    pairs = aligned_data['aligned_pairs']

    # 매칭 정보를 양방향으로 저장
    kr_idx_to_pair = {}
    en_idx_to_pair = {}
    matched_en_indices = set()

    for pair in pairs:
        kr_original_idx = pair.get('kr_original_idx')
        en_original_idx = pair.get('en_original_idx')

        if kr_original_idx is not None:
            kr_idx_to_pair[kr_original_idx] = {
                'en_id': pair['en'].get('id'),
                'en_text': pair['en'].get('text', ''),
                'en_source_type': pair['en'].get('source_type', ''),
                'en_original_idx': en_original_idx,
                'similarity': pair.get('similarity', ''),
                'match_type': pair.get('type', '')
            }

        if en_original_idx is not None:
            matched_en_indices.add(en_original_idx)
            en_idx_to_pair[en_original_idx] = {
                'kr_id': pair['kr'].get('id'),
                'kr_text': pair['kr'].get('text', ''),
                'kr_source_type': pair['kr'].get('source_type', ''),
                'kr_original_idx': kr_original_idx
            }

    rows = []

    # 1. 한국어 전체 순회 (매칭된 영어 포함)
    for idx, kr_item in enumerate(kr_data):
        kr_id = kr_item.get('id')
        kr_text = kr_item.get('text', '')

        row = {
            'kr_idx': idx,
            'kr_id': kr_id,
            'kr_text': kr_text,
            'kr_source_type': kr_item.get('source_type', ''),
        }

        # 매칭된 영어 추가
        if idx in kr_idx_to_pair:
            en_info = kr_idx_to_pair[idx]
            row.update({
                'en_idx': en_info['en_original_idx'],
                'en_id': en_info['en_id'],
                'en_text': en_info['en_text'],
                'en_source_type': en_info['en_source_type'],
                'similarity': en_info['similarity'],
                'match_type': en_info['match_type']
            })
        else:
            row.update({
                'en_idx': '',
                'en_id': '',
                'en_text': '',
                'en_source_type': '',
                'similarity': '',
                'match_type': 'unmatched'
            })

        rows.append(row)

    # 2. 매칭 안 된 영어 항목 추가
    for idx, en_item in enumerate(en_data):
        if idx not in matched_en_indices:
            en_id = en_item.get('id')
            en_text = en_item.get('text', '')

            row = {
                'kr_idx': '',
                'kr_id': '',
                'kr_text': '',
                'kr_source_type': '',
                'en_idx': idx,
                'en_id': en_id,
                'en_text': en_text,
                'en_source_type': en_item.get('source_type', ''),
                'similarity': '',
                'match_type': 'unmatched'
            }
            rows.append(row)

    # DataFrame으로 변환 후 CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    matched_count = len([r for r in rows if r['match_type'] != 'unmatched'])
    unmatched_kr_count = len([r for r in rows if r['match_type'] == 'unmatched' and r['kr_idx'] != ''])
    unmatched_en_count = len([r for r in rows if r['match_type'] == 'unmatched' and r['en_idx'] != ''])

    print(f"✓ 전체 {len(rows)}개 항목을 CSV로 저장했습니다.")
    print(f"  - 매칭: {matched_count}개")
    print(f"  - 미매칭 한국어: {unmatched_kr_count}개")
    print(f"  - 미매칭 영어: {unmatched_en_count}개")
    return df


def main():
    """메인 실행 함수"""
    print("=== CSV 파일 생성 시작 ===")

    # 파일 경로 설정
    kr_json_path = "llm_output_1부_kr.json"
    en_json_path = "llm_output_1부_en.json"
    aligned_json_path = "aligned_1부.json"
    aligned_csv_path = "aligned_1부.csv"
    merged_csv_kr_path = "merged_1부_kr.csv"
    merged_csv_en_path = "merged_1부_en.csv"
    merged_csv_outer_path = "merged_1부_outer.csv"

    # 1. aligned JSON을 직접 CSV로 변환
    aligned_df = create_aligned_csv(aligned_json_path, aligned_csv_path)

    # 2. llm_output_kr에 영어 매칭 결과 병합 (한국어 기준 LEFT JOIN)
    merged_kr_df = create_merged_csv(kr_json_path, aligned_json_path, merged_csv_kr_path)

    # 3. llm_output_en에 한국어 매칭 결과 병합 (영어 기준 LEFT JOIN)
    merged_en_df = create_merged_csv_en(en_json_path, aligned_json_path, merged_csv_en_path)

    # 4. FULL OUTER JOIN (한국어 + 영어 전체)
    merged_outer_df = create_outer_join_csv(kr_json_path, en_json_path, aligned_json_path, merged_csv_outer_path)

    print("\n=== CSV 파일 생성 완료 ===")
    print(f"  1. {aligned_csv_path}: 매칭된 쌍만 포함")
    print(f"  2. {merged_csv_kr_path}: 한국어 기준 LEFT JOIN")
    print(f"  3. {merged_csv_en_path}: 영어 기준 LEFT JOIN")
    print(f"  4. {merged_csv_outer_path}: FULL OUTER JOIN (모든 항목)")

    # 미리보기 출력
    print(f"\n[{aligned_csv_path} 미리보기]")
    print(aligned_df.head(3).to_string(index=False))

    print(f"\n[{merged_csv_kr_path} 미리보기]")
    print(merged_kr_df.head(3).to_string(index=False))

    print(f"\n[{merged_csv_en_path} 미리보기]")
    print(merged_en_df.head(3).to_string(index=False))

    print(f"\n[{merged_csv_outer_path} 미리보기]")
    print(merged_outer_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
