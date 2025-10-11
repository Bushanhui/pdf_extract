"""
한-영 문장 Alignment 시스템

llm_output_kr.json과 llm_output_en.json (또는 llm_input_en.json)을 비교하여
한-영 문장 쌍을 생성합니다.

전략:
1. LaBSE 기반 다국어 임베딩
2. 헤더(H1/H2/H3) 타입별 헝가리안 알고리즘 최적 매칭
   - Cost = (1 - similarity) + order_weight * |i - j| / max_len
   - 타입 내 단조 증가 제약 적용
3. 섹션 기반 P 항목 매칭
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime
from scipy.optimize import linear_sum_assignment


class SimilarityCalculator:
    """LaBSE 기반 유사도 계산기"""

    def __init__(self):
        print("LaBSE 모델 로딩 중...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        print("✓ LaBSE 모델 로딩 완료")

    def encode(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩 벡터로 변환"""
        return self.model.encode(texts, convert_to_numpy=True)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class HeaderMatcher:
    """헤더(H1/H2/H3) 매칭 클래스"""

    # 앵커 패턴: 첫 매칭 찾기용
    ANCHOR_PATTERNS = [
        {"kr": ["목차", "차례"], "en": ["contents", "table of contents"]},
        {"kr": ["약어", "생략"], "en": ["abbreviation", "acronym"]},
        {"kr": ["주요", "하이라이트"], "en": ["main", "highlights"]},
        {"kr": ["일러두기", "참고", "주석"], "en": ["note", "reference"]},
    ]

    def __init__(self, similarity_calc: SimilarityCalculator,
                 first_match_distance: int = 5,
                 first_match_threshold: float = 0.8,
                 rest_match_threshold: float = 0.7,
                 rest_distance_ratio: float = 0.1,
                 order_weight: float = 0.5):
        self.similarity_calc = similarity_calc
        self.first_match_distance = first_match_distance
        self.first_match_threshold = first_match_threshold
        self.rest_match_threshold = rest_match_threshold
        self.rest_distance_ratio = rest_distance_ratio
        self.order_weight = order_weight

    def extract_headers_by_type(self, data: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
        """데이터에서 타입별로 헤더 추출

        Returns:
            {
                'H1': [(global_idx, item), ...],
                'H2': [(global_idx, item), ...],
                'H3': [(global_idx, item), ...]
            }
        """
        headers = {'H1': [], 'H2': [], 'H3': []}

        for idx, item in enumerate(data):
            if item.get('source_type') in ['H1', 'H2', 'H3']:
                headers[item['source_type']].append((idx, item))

        return headers

    def find_pattern_anchors(self, kr_headers: Dict[str, List], en_headers: Dict[str, List]) -> Optional[Dict]:
        """패턴 기반으로 첫 매칭 앵커 찾기

        Returns:
            첫 번째 앵커 매칭 정보 또는 None
        """
        anchors = []

        for pattern in self.ANCHOR_PATTERNS:
            # 모든 타입의 헤더에서 패턴 검색
            for header_type in ['H1', 'H2', 'H3']:
                kr_list = kr_headers[header_type]
                en_list = en_headers[header_type]

                if not kr_list or not en_list:
                    continue

                # 한국어에서 패턴 찾기
                kr_match = None
                for kr_idx_in_type, (kr_global_idx, kr_item) in enumerate(kr_list):
                    kr_text_lower = kr_item['text'].lower()
                    if any(keyword in kr_text_lower for keyword in pattern['kr']):
                        kr_match = (kr_idx_in_type, kr_global_idx, kr_item)
                        break

                # 영어에서 패턴 찾기
                en_match = None
                for en_idx_in_type, (en_global_idx, en_item) in enumerate(en_list):
                    en_text_lower = en_item['text'].lower()
                    if any(keyword in en_text_lower for keyword in pattern['en']):
                        en_match = (en_idx_in_type, en_global_idx, en_item)
                        break

                if kr_match and en_match:
                    anchors.append({
                        'kr_global_idx': kr_match[1],
                        'en_global_idx': en_match[1],
                        'kr_text': kr_match[2]['text'],
                        'en_text': en_match[2]['text'],
                        'type': header_type,
                        'similarity': 1.0,  # 패턴 매칭은 최고 신뢰도
                        'kr_type_idx': kr_match[0],
                        'en_type_idx': en_match[0],
                        'distance': abs(kr_match[0] - en_match[0]),
                        'method': 'pattern'
                    })

        # 가장 앞에 있는 앵커 반환
        if anchors:
            return min(anchors, key=lambda x: x['kr_global_idx'])

        return None

    def _build_cost_matrix(self, kr_embeddings: np.ndarray, en_embeddings: np.ndarray) -> np.ndarray:
        """헝가리안 알고리즘을 위한 Cost Matrix 생성
        
        Args:
            kr_embeddings: 한국어 임베딩 (N_kr, embedding_dim)
            en_embeddings: 영어 임베딩 (N_en, embedding_dim)
        
        Returns:
            cost_matrix (N_kr, N_en): cost[i][j] = base_cost + order_penalty
        """
        n_kr = len(kr_embeddings)
        n_en = len(en_embeddings)
        max_len = max(n_kr, n_en)
        
        cost_matrix = np.zeros((n_kr, n_en))
        
        for i in range(n_kr):
            for j in range(n_en):
                # 1. Base cost: 1 - similarity (유사도가 높을수록 cost 낮음)
                similarity = self.similarity_calc.cosine_similarity(kr_embeddings[i], en_embeddings[j])
                base_cost = 1.0 - similarity
                
                # 2. Order penalty: 순서 차이 (정규화)
                order_penalty = self.order_weight * abs(i - j) / max_len
                
                # 3. Total cost
                cost_matrix[i][j] = base_cost + order_penalty
        
        return cost_matrix

    def _enforce_monotonic_within_type(self, matches: List[Tuple[int, int]], 
                                       kr_list: List[Tuple[int, Dict]], 
                                       en_list: List[Tuple[int, Dict]]) -> List[Dict]:
        """타입 내 단조 증가 제약을 적용하여 최종 매칭 생성
        
        Args:
            matches: [(kr_type_idx, en_type_idx), ...] 헝가리안 결과
            kr_list: [(global_idx, item), ...] 한국어 헤더 리스트
            en_list: [(global_idx, item), ...] 영어 헤더 리스트
        
        Returns:
            List of match dictionaries (단조 증가 위배 쌍 제거됨)
        """
        # kr_type_idx 순으로 정렬
        sorted_matches = sorted(matches, key=lambda x: x[0])
        
        result = []
        last_en_type_idx = -1
        
        for kr_type_idx, en_type_idx in sorted_matches:
            # 단조 증가 체크
            if en_type_idx > last_en_type_idx:
                kr_global_idx, kr_item = kr_list[kr_type_idx]
                en_global_idx, en_item = en_list[en_type_idx]
                
                # 유사도 재계산
                kr_embedding = self.similarity_calc.encode([kr_item['text']])[0]
                en_embedding = self.similarity_calc.encode([en_item['text']])[0]
                similarity = self.similarity_calc.cosine_similarity(kr_embedding, en_embedding)
                
                result.append({
                    'kr_global_idx': kr_global_idx,
                    'en_global_idx': en_global_idx,
                    'kr_text': kr_item['text'],
                    'en_text': en_item['text'],
                    'type': kr_item['source_type'],
                    'similarity': similarity,
                    'kr_type_idx': kr_type_idx,
                    'en_type_idx': en_type_idx,
                    'distance': abs(kr_type_idx - en_type_idx),
                    'method': 'hungarian'
                })
                
                last_en_type_idx = en_type_idx
        
        return result

    def _match_type_with_hungarian(self, kr_list: List[Tuple[int, Dict]], 
                                   en_list: List[Tuple[int, Dict]], 
                                   header_type: str) -> List[Dict]:
        """특정 타입에 대해 헝가리안 알고리즘으로 최적 매칭
        
        Args:
            kr_list: [(global_idx, item), ...] 한국어 헤더 리스트
            en_list: [(global_idx, item), ...] 영어 헤더 리스트
            header_type: 'H1', 'H2', or 'H3'
        
        Returns:
            List of match dictionaries
        """
        if not kr_list or not en_list:
            return []
        
        # 1. 임베딩 생성
        kr_texts = [item['text'] for _, item in kr_list]
        en_texts = [item['text'] for _, item in en_list]
        kr_embeddings = self.similarity_calc.encode(kr_texts)
        en_embeddings = self.similarity_calc.encode(en_texts)
        
        # 2. Cost Matrix 생성
        cost_matrix = self._build_cost_matrix(kr_embeddings, en_embeddings)
        
        # 3. 헝가리안 알고리즘 적용
        kr_indices, en_indices = linear_sum_assignment(cost_matrix)
        
        # 4. 결과를 (kr_type_idx, en_type_idx) 튜플 리스트로 변환
        matches = list(zip(kr_indices, en_indices))
        
        # 5. 타입 내 단조 증가 제약 적용
        result = self._enforce_monotonic_within_type(matches, kr_list, en_list)
        
        return result

    def find_header_matches(self, kr_data: List[Dict], en_data: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """헤더 매칭 수행 (하이브리드 방식)

        Returns:
            (first_match, all_matches)
        """
        # 1. 타입별로 헤더 분리
        kr_headers = self.extract_headers_by_type(kr_data)
        en_headers = self.extract_headers_by_type(en_data)

        print(f"\n=== 헤더 개수 ===")
        print(f"한국어: H1={len(kr_headers['H1'])}, H2={len(kr_headers['H2'])}, H3={len(kr_headers['H3'])}")
        print(f"영어:   H1={len(en_headers['H1'])}, H2={len(en_headers['H2'])}, H3={len(en_headers['H3'])}")

        # 2. 첫 매칭 찾기 (하이브리드)
        print(f"\n=== 첫 매칭 찾기 (하이브리드) ===")

        # 2-1. 패턴 매칭 시도
        first_match = self.find_pattern_anchors(kr_headers, en_headers)

        if first_match:
            print(f"✓ 패턴 매칭 성공!")
            print(f"  한국어[{first_match['kr_global_idx']}]: {first_match['kr_text'][:50]}...")
            print(f"  영어[{first_match['en_global_idx']}]: {first_match['en_text'][:50]}...")
        else:
            # 2-2. 유사도 기반 첫 매칭 (유사도 ≥ 0.8, 거리 ≤ 5)
            print("패턴 매칭 실패, 유사도 기반 첫 매칭 시도...")
            first_match = self._find_first_by_similarity(kr_headers, en_headers)

            if first_match:
                print(f"✓ 유사도 기반 첫 매칭 성공!")
                print(f"  한국어[{first_match['kr_global_idx']}]: {first_match['kr_text'][:50]}...")
                print(f"  영어[{first_match['en_global_idx']}]: {first_match['en_text'][:50]}...")
                print(f"  유사도: {first_match['similarity']:.3f}")

        if not first_match:
            print("❌ 첫 매칭을 찾을 수 없습니다.")
            return None, []

        # 3. 타입별 헝가리안 알고리즘 매칭
        print(f"\n=== 타입별 헝가리안 알고리즘 매칭 (order_weight={self.order_weight}) ===")
        
        matched_pairs = []
        
        for header_type in ['H1', 'H2', 'H3']:
            kr_list = kr_headers[header_type]
            en_list = en_headers[header_type]

            if not kr_list or not en_list:
                continue

            print(f"\n{header_type} 매칭 중... (kr={len(kr_list)}, en={len(en_list)})")
            
            # 헝가리안 알고리즘으로 최적 매칭
            type_matches = self._match_type_with_hungarian(kr_list, en_list, header_type)
            
            # 첫 매칭과 중복 제거
            for match in type_matches:
                if not (match['kr_global_idx'] == first_match['kr_global_idx'] and
                        match['en_global_idx'] == first_match['en_global_idx']):
                    matched_pairs.append(match)
            
            print(f"{header_type}: {len(type_matches)}개 매칭 (단조 증가 제약 후)")

        # 첫 매칭 추가
        matched_pairs.append(first_match)
        
        # 최종 결과를 global_idx 순으로 정렬
        matched_pairs.sort(key=lambda x: x['kr_global_idx'])

        print(f"\n최종 매칭된 헤더: {len(matched_pairs)}개")
        print(f"  - 패턴 매칭: {len([m for m in matched_pairs if m.get('method') == 'pattern'])}개")
        print(f"  - 헝가리안 매칭: {len([m for m in matched_pairs if m.get('method') == 'hungarian'])}개")

        return first_match, matched_pairs

    def _find_first_by_similarity(self, kr_headers: Dict[str, List], en_headers: Dict[str, List]) -> Optional[Dict]:
        """유사도 기반으로 첫 매칭 찾기 (유사도 ≥ 0.8, 거리 ≤ 5)"""
        candidates = []

        for header_type in ['H1', 'H2', 'H3']:
            kr_list = kr_headers[header_type]
            en_list = en_headers[header_type]

            if not kr_list or not en_list:
                continue

            # 임베딩 생성
            kr_texts = [item['text'] for _, item in kr_list]
            en_texts = [item['text'] for _, item in en_list]
            kr_embeddings = self.similarity_calc.encode(kr_texts)
            en_embeddings = self.similarity_calc.encode(en_texts)

            for kr_idx_in_type, (kr_global_idx, kr_item) in enumerate(kr_list):
                for en_idx_in_type, (en_global_idx, en_item) in enumerate(en_list):
                    distance = abs(kr_idx_in_type - en_idx_in_type)
                    if distance > self.first_match_distance:
                        continue

                    similarity = self.similarity_calc.cosine_similarity(
                        kr_embeddings[kr_idx_in_type],
                        en_embeddings[en_idx_in_type]
                    )

                    if similarity >= self.first_match_threshold:
                        candidates.append({
                            'kr_global_idx': kr_global_idx,
                            'en_global_idx': en_global_idx,
                            'kr_text': kr_item['text'],
                            'en_text': en_item['text'],
                            'type': header_type,
                            'similarity': similarity,
                            'kr_type_idx': kr_idx_in_type,
                            'en_type_idx': en_idx_in_type,
                            'distance': distance,
                            'method': 'similarity'
                        })

        # 가장 앞에 있는 후보 반환
        if candidates:
            return min(candidates, key=lambda x: x['kr_global_idx'])

        return None


class SectionAligner:
    """섹션 기반 P 항목 매칭 클래스"""

    def __init__(self, similarity_calc: SimilarityCalculator):
        self.similarity_calc = similarity_calc

    def split_into_sections(self, kr_data: List[Dict], en_data: List[Dict], header_matches: List[Dict]) -> List[Dict]:
        """헤더 매칭 기준으로 섹션 분할

        Returns:
            [
                {
                    'kr_header': {...},
                    'en_header': {...},
                    'kr_paragraphs': [...],
                    'en_paragraphs': [...]
                },
                ...
            ]
        """
        sections = []

        # 헤더 매칭을 한국어 인덱스 순으로 정렬
        sorted_matches = sorted(header_matches, key=lambda x: x['kr_global_idx'])

        for i, match in enumerate(sorted_matches):
            # 다음 헤더까지의 범위 계산
            kr_start = match['kr_global_idx'] + 1
            en_start = match['en_global_idx'] + 1

            if i < len(sorted_matches) - 1:
                kr_end = sorted_matches[i + 1]['kr_global_idx']
                en_end = sorted_matches[i + 1]['en_global_idx']
            else:
                kr_end = len(kr_data)  # 마지막 섹션은 끝까지
                en_end = len(en_data)

            # P 항목만 추출 (원본 인덱스 포함)
            kr_paragraphs = [
                (idx, item) for idx in range(kr_start, kr_end)
                for item in [kr_data[idx]]
                if item.get('source_type') == 'P'
            ]
            en_paragraphs = [
                (idx, item) for idx in range(en_start, en_end)
                for item in [en_data[idx]]
                if item.get('source_type') == 'P'
            ]

            sections.append({
                'kr_header': match,
                'en_header': match,
                'kr_paragraphs': kr_paragraphs,  # [(idx, item), ...]
                'en_paragraphs': en_paragraphs   # [(idx, item), ...]
            })

        return sections

    def align_paragraphs_simple(self, kr_paragraphs: List[Dict], en_paragraphs: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """섹션 내 P 항목 단순 순차 매칭"""
        pairs = []
        min_len = min(len(kr_paragraphs), len(en_paragraphs))

        for i in range(min_len):
            pairs.append((kr_paragraphs[i], en_paragraphs[i]))

        return pairs

    def align_paragraphs_similarity(self, kr_paragraphs: List[Tuple[int, Dict]],
                                   en_paragraphs: List[Tuple[int, Dict]],
                                   threshold: float = 0.7) -> List[Tuple[int, Dict, int, Dict, float]]:
        """섹션 내 P 항목 유사도 기반 Greedy Matching

        Args:
            kr_paragraphs: [(original_idx, item), ...] 한국어 P 항목 리스트
            en_paragraphs: [(original_idx, item), ...] 영어 P 항목 리스트
            threshold: 최소 유사도 임계값

        Returns:
            List of (kr_original_idx, kr_item, en_original_idx, en_item, similarity) tuples
        """
        if not kr_paragraphs or not en_paragraphs:
            return []

        # 1. 임베딩 생성
        kr_texts = [item.get('text', '') for _, item in kr_paragraphs]
        en_texts = [item.get('text', '') for _, item in en_paragraphs]

        kr_embeddings = self.similarity_calc.encode(kr_texts)
        en_embeddings = self.similarity_calc.encode(en_texts)

        # 2. 모든 쌍의 유사도 계산 및 후보 생성
        candidates = []
        for kr_list_idx, (kr_original_idx, kr_item) in enumerate(kr_paragraphs):
            for en_list_idx, (en_original_idx, en_item) in enumerate(en_paragraphs):
                similarity = self.similarity_calc.cosine_similarity(
                    kr_embeddings[kr_list_idx],
                    en_embeddings[en_list_idx]
                )

                if similarity >= threshold:
                    candidates.append({
                        'kr_list_idx': kr_list_idx,
                        'en_list_idx': en_list_idx,
                        'kr_original_idx': kr_original_idx,
                        'en_original_idx': en_original_idx,
                        'kr_item': kr_item,
                        'en_item': en_item,
                        'similarity': similarity
                    })

        # 3. Greedy Matching (높은 유사도부터)
        candidates_sorted = sorted(candidates, key=lambda x: x['similarity'], reverse=True)

        pairs = []
        used_kr = set()
        used_en = set()

        for candidate in candidates_sorted:
            kr_list_idx = candidate['kr_list_idx']
            en_list_idx = candidate['en_list_idx']

            if kr_list_idx not in used_kr and en_list_idx not in used_en:
                pairs.append((
                    candidate['kr_original_idx'],
                    candidate['kr_item'],
                    candidate['en_original_idx'],
                    candidate['en_item'],
                    candidate['similarity']
                ))
                used_kr.add(kr_list_idx)
                used_en.add(en_list_idx)

        return pairs


class Aligner:
    """전체 Alignment 시스템"""

    def __init__(self,
                 first_match_distance: int = 5,
                 first_match_threshold: float = 0.8,
                 rest_match_threshold: float = 0.7,
                 rest_distance_ratio: float = 0.2,
                 order_weight: float = 0.5):
        self.similarity_calc = SimilarityCalculator()
        self.header_matcher = HeaderMatcher(
            self.similarity_calc,
            first_match_distance=first_match_distance,
            first_match_threshold=first_match_threshold,
            rest_match_threshold=rest_match_threshold,
            rest_distance_ratio=rest_distance_ratio,
            order_weight=order_weight
        )
        self.section_aligner = SectionAligner(self.similarity_calc)

    def load_data(self, kr_path: str, en_path: str) -> Tuple[List[Dict], List[Dict]]:
        """JSON 파일 로드"""
        with open(kr_path, 'r', encoding='utf-8') as f:
            kr_json = json.load(f)
            # llm_output 형식 (metadata + data) vs llm_input 형식 (배열) 구분
            kr_data = kr_json['data'] if 'data' in kr_json else kr_json

        with open(en_path, 'r', encoding='utf-8') as f:
            en_json = json.load(f)
            en_data = en_json['data'] if 'data' in en_json else en_json

        print(f"한국어 데이터: {len(kr_data)}개 항목")
        print(f"영어 데이터: {len(en_data)}개 항목")

        return kr_data, en_data

    def align(self, kr_path: str, en_path: str, output_path: str):
        """전체 alignment 실행"""
        print("=== 한-영 문장 Alignment 시작 ===\n")

        # 1. 데이터 로드
        kr_data, en_data = self.load_data(kr_path, en_path)

        # 2. 헤더 매칭
        first_match, header_matches = self.header_matcher.find_header_matches(kr_data, en_data)

        if not header_matches:
            print("\n❌ 매칭된 헤더가 없습니다. 임계값을 낮추거나 데이터를 확인하세요.")
            return

        # 3. 섹션 분할 및 P 매칭
        print(f"\n=== 섹션 분할 및 P 매칭 ===")
        sections = self.section_aligner.split_into_sections(kr_data, en_data, header_matches)

        aligned_pairs = []
        total_p_pairs = 0

        for idx, section in enumerate(sections, 1):
            # 헤더 추가 (원본 데이터에서 가져오기)
            kr_header_match = section['kr_header']
            en_header_match = section['en_header']

            # 원본 데이터에서 실제 항목 가져오기
            kr_header_item = kr_data[kr_header_match['kr_global_idx']]
            en_header_item = en_data[en_header_match['en_global_idx']]

            aligned_pairs.append({
                'kr': kr_header_item,
                'en': en_header_item,
                'kr_original_idx': kr_header_match['kr_global_idx'],
                'en_original_idx': en_header_match['en_global_idx'],
                'type': 'header',
                'similarity': kr_header_match['similarity']
            })

            # P 항목 매칭 (유사도 기반 greedy matching)
            p_pairs = self.section_aligner.align_paragraphs_similarity(
                section['kr_paragraphs'],
                section['en_paragraphs'],
                threshold=0.5
            )

            for kr_original_idx, kr_item, en_original_idx, en_item, similarity in p_pairs:
                aligned_pairs.append({
                    'kr': kr_item,
                    'en': en_item,
                    'kr_original_idx': kr_original_idx,
                    'en_original_idx': en_original_idx,
                    'type': 'paragraph',
                    'similarity': similarity
                })

            total_p_pairs += len(p_pairs)
            print(f"섹션 {idx}: H={section['kr_header']['type']}, P={len(p_pairs)}쌍 (유사도≥0.5)")

        # 4. 결과 저장
        result = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source_kr': Path(kr_path).name,
                'source_en': Path(en_path).name,
                'total_pairs': len(aligned_pairs),
                'header_pairs': len(header_matches),
                'paragraph_pairs': total_p_pairs,
                'first_match_kr_idx': first_match['kr_global_idx'] if first_match else None,
                'first_match_en_idx': first_match['en_global_idx'] if first_match else None,
                'first_match_method': first_match.get('method', 'unknown') if first_match else None,
                'first_match_distance': self.header_matcher.first_match_distance,
                'first_match_threshold': self.header_matcher.first_match_threshold,
                'rest_match_threshold': self.header_matcher.rest_match_threshold,
                'rest_distance_ratio': self.header_matcher.rest_distance_ratio
            },
            'aligned_pairs': aligned_pairs
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Alignment 완료!")
        print(f"  총 {len(aligned_pairs)}쌍 생성 (헤더={len(header_matches)}, P={total_p_pairs})")
        print(f"  결과 저장: {output_path}")


def main():
    """메인 실행 함수"""
    # 파일 경로 설정
    kr_path = "llm_output_1부_kr.json"
    en_path = "llm_output_1부_en.json"
    output_path = "aligned_1부.json"

    # Alignment 실행 (기본값: first_match_distance=5, first_match_threshold=0.8,
    #                      rest_match_threshold=0.7, rest_distance_ratio=0.1)
    aligner = Aligner()
    aligner.align(kr_path, en_path, output_path)


if __name__ == "__main__":
    main()
