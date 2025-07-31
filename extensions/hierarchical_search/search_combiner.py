# 검색 결과 통합 및 순위 조정
from typing import List, Dict, Any, Tuple
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
import numpy as np

class SearchCombiner:
    """부모-자식 검색 결과 통합 및 순위 조정"""
    
    def __init__(self):
        # 가중치 설정
        self.parent_weight = 0.4  # 부모 문서 가중치
        self.child_weight = 0.6   # 자식 문서 가중치
        self.relevance_threshold = 0.7  # 관련성 임계값
    
    def combine_search_results(self, 
                             parent_results: List[Document], 
                             child_results: List[Document],
                             query: str) -> List[Document]:
        """부모와 자식 검색 결과를 통합하고 순위 조정"""
        
        # 결과가 없는 경우
        if not parent_results and not child_results:
            return []
        
        # 부모 결과만 있는 경우
        if not child_results:
            return self._rank_parent_results(parent_results, query)
        
        # 자식 결과만 있는 경우
        if not parent_results:
            return self._rank_child_results(child_results, query)
        
        # 부모와 자식 결과 모두 있는 경우
        combined_results = []
        
        # 부모 결과 처리 (개요/개념 정보)
        for doc in parent_results:
            doc.metadata['source_type'] = 'parent'
            doc.metadata['weight'] = self.parent_weight
            combined_results.append(doc)
        
        # 자식 결과 처리 (상세/실무 정보)
        for doc in child_results:
            doc.metadata['source_type'] = 'child'
            doc.metadata['weight'] = self.child_weight
            combined_results.append(doc)
        
        # 중복 제거 및 순위 조정
        deduplicated_results = self._remove_duplicates(combined_results)
        ranked_results = self._rank_combined_results(deduplicated_results, query)
        
        return ranked_results
    
    def _rank_parent_results(self, parent_results: List[Document], query: str) -> List[Document]:
        """부모 결과만 있는 경우 순위 조정"""
        for doc in parent_results:
            doc.metadata['source_type'] = 'parent'
            doc.metadata['weight'] = self.parent_weight
            doc.metadata['relevance_score'] = self._calculate_relevance(doc, query)
        
        # 관련성 점수로 정렬
        return sorted(parent_results, 
                     key=lambda x: x.metadata.get('relevance_score', 0), 
                     reverse=True)
    
    def _rank_child_results(self, child_results: List[Document], query: str) -> List[Document]:
        """자식 결과만 있는 경우 순위 조정"""
        for doc in child_results:
            doc.metadata['source_type'] = 'child'
            doc.metadata['weight'] = self.child_weight
            doc.metadata['relevance_score'] = self._calculate_relevance(doc, query)
        
        # 관련성 점수로 정렬
        return sorted(child_results, 
                     key=lambda x: x.metadata.get('relevance_score', 0), 
                     reverse=True)
    
    def _rank_combined_results(self, combined_results: List[Document], query: str) -> List[Document]:
        """통합된 결과의 순위 조정"""
        for doc in combined_results:
            # 관련성 점수 계산
            relevance_score = self._calculate_relevance(doc, query)
            
            # 가중치 적용
            weighted_score = relevance_score * doc.metadata.get('weight', 1.0)
            
            doc.metadata['relevance_score'] = relevance_score
            doc.metadata['weighted_score'] = weighted_score
        
        # 가중 점수로 정렬
        ranked_results = sorted(combined_results, 
                              key=lambda x: x.metadata.get('weighted_score', 0), 
                              reverse=True)
        
        # 관련성 임계값 필터링
        filtered_results = [
            doc for doc in ranked_results 
            if doc.metadata.get('relevance_score', 0) >= self.relevance_threshold
        ]
        
        return filtered_results
    
    def _calculate_relevance(self, doc: Document, query: str) -> float:
        """문서와 쿼리 간의 관련성 점수 계산"""
        # 간단한 키워드 매칭 기반 점수 계산
        query_words = set(query.lower().split())
        content_words = set(doc.page_content.lower().split())
        
        # 공통 단어 수
        common_words = query_words.intersection(content_words)
        
        # Jaccard 유사도
        if len(query_words) + len(content_words) - len(common_words) == 0:
            return 0.0
        
        jaccard_similarity = len(common_words) / (len(query_words) + len(content_words) - len(common_words))
        
        # 파일명 매칭 보너스
        file_name = doc.metadata.get('file_name', '').lower()
        file_name_bonus = 0.0
        for word in query_words:
            if word in file_name:
                file_name_bonus += 0.1
        
        # 최종 점수 (0.0 ~ 1.0)
        final_score = min(1.0, jaccard_similarity + file_name_bonus)
        
        return final_score
    
    def _remove_duplicates(self, results: List[Document]) -> List[Document]:
        """중복 문서 제거"""
        seen_content = set()
        unique_results = []
        
        for doc in results:
            # 내용의 해시값으로 중복 체크
            content_hash = hash(doc.page_content[:100])  # 처음 100자만 사용
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        return unique_results
    
    def get_result_summary(self, results: List[Document]) -> Dict[str, Any]:
        """검색 결과 요약 정보 반환"""
        if not results:
            return {
                'total_count': 0,
                'parent_count': 0,
                'child_count': 0,
                'topics': [],
                'avg_relevance': 0.0
            }
        
        parent_count = sum(1 for doc in results if doc.metadata.get('source_type') == 'parent')
        child_count = sum(1 for doc in results if doc.metadata.get('source_type') == 'child')
        
        topics = list(set(doc.metadata.get('topic', 'unknown') for doc in results))
        relevance_scores = [doc.metadata.get('relevance_score', 0) for doc in results]
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        return {
            'total_count': len(results),
            'parent_count': parent_count,
            'child_count': child_count,
            'topics': topics,
            'avg_relevance': avg_relevance
        } 