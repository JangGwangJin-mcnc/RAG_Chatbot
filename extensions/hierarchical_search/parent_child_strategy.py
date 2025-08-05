# 부모-자식 전략 메인 클래스
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
import streamlit as st
import os

from .metadata_manager import MetadataManager
from .vector_db_hierarchy import VectorDBHierarchy, SafeSentenceTransformerEmbeddings
from .search_combiner import SearchCombiner

class ParentChildSearchStrategy:
    """부모-자식 전략을 통한 계층적 검색"""
    
    def __init__(self, embedding_model: SafeSentenceTransformerEmbeddings):
        self.embedding_model = embedding_model
        self.metadata_manager = MetadataManager()
        self.vector_db_hierarchy = VectorDBHierarchy(embedding_model)
        self.search_combiner = SearchCombiner()
        
        # 검색 설정
        self.parent_search_k = 2  # 부모 DB에서 검색할 문서 수
        self.child_search_k = 3   # 자식 DB에서 검색할 문서 수
    
    def setup_hierarchy(self, documents: List[Document]) -> bool:
        """문서들을 부모-자식으로 분류하고 계층적 벡터DB 생성"""
        try:
            st.info("📊 문서를 부모-자식으로 분류하는 중...")
            
            # 메타데이터 업데이트 및 분류
            classified_docs = []
            parent_count = 0
            child_count = 0
            
            for doc in documents:
                updated_metadata = self.metadata_manager.classify_document(doc)
                doc.metadata = updated_metadata
                classified_docs.append(doc)
                
                if updated_metadata.get('doc_type') == 'parent':
                    parent_count += 1
                else:
                    child_count += 1
            
            st.success(f"✅ 문서 분류 완료: 부모 {parent_count}개, 자식 {child_count}개")
            
            # 계층적 벡터DB 생성
            st.info("🗂️ 계층적 벡터DB를 생성하는 중...")
            self.vector_db_hierarchy.setup_hierarchy(classified_docs, self.metadata_manager)
            
            st.success("✅ 계층적 벡터DB 생성 완료!")
            return True
            
        except Exception as e:
            st.error(f"❌ 계층적 벡터DB 생성 실패: {str(e)}")
            return False
    
    def load_hierarchy(self) -> bool:
        """저장된 계층적 벡터DB 로드"""
        try:
            if self.vector_db_hierarchy.load_hierarchy():
                st.success("✅ 계층적 벡터DB 로드 완료")
                return True
            else:
                st.warning("⚠️ 저장된 계층적 벡터DB가 없습니다")
                return False
        except Exception as e:
            st.error(f"❌ 계층적 벡터DB 로드 실패: {str(e)}")
            return False
    
    def search(self, query: str) -> Tuple[List[Document], Dict[str, Any]]:
        """계층적 검색 수행"""
        try:
            # 1. 부모 DB에서 관련 주제 검색
            parent_results = self._search_parent(query)
            
            # 2. 부모 결과에서 관련 자식 주제들 찾기
            related_child_topics = self._extract_related_child_topics(parent_results, query)
            
            # 3. 관련 자식 DB들에서 검색
            child_results = self._search_children(query, related_child_topics)
            
            # 4. 결과 통합 및 순위 조정
            combined_results = self.search_combiner.combine_search_results(
                parent_results, child_results, query
            )
            
            # 5. 검색 결과 요약
            summary = self.search_combiner.get_result_summary(combined_results)
            
            return combined_results, summary
            
        except Exception as e:
            st.error(f"❌ 계층적 검색 실패: {str(e)}")
            return [], {}
    
    def _search_parent(self, query: str) -> List[Document]:
        """부모 벡터DB에서 검색"""
        parent_db = self.vector_db_hierarchy.get_parent_db()
        if not parent_db:
            return []
        
        try:
            retriever = parent_db.as_retriever(search_kwargs={"k": self.parent_search_k})
            results = retriever.invoke(query)
            return results
        except Exception as e:
            st.warning(f"⚠️ 부모 DB 검색 실패: {str(e)}")
            return []
    
    def _extract_related_child_topics(self, parent_results: List[Document], query: str) -> List[str]:
        """부모 결과에서 관련 자식 주제들 추출"""
        related_topics = set()
        
        # 부모 결과에서 주제 추출
        for doc in parent_results:
            topic = doc.metadata.get('topic', '')
            if topic:
                child_topics = self.vector_db_hierarchy.get_related_child_topics(topic)
                related_topics.update(child_topics)
        
        # 쿼리에서 직접 주제 추출
        query_lower = query.lower()
        for parent_topic in self.metadata_manager.parent_child_mapping.keys():
            if parent_topic.lower() in query_lower:
                child_topics = self.metadata_manager.get_related_child_topics(parent_topic)
                related_topics.update(child_topics)
        
        return list(related_topics)
    
    def _search_children(self, query: str, child_topics: List[str]) -> List[Document]:
        """관련 자식 벡터DB들에서 검색"""
        all_child_results = []
        
        for topic in child_topics:
            child_db = self.vector_db_hierarchy.get_child_db(topic)
            if child_db:
                try:
                    retriever = child_db.as_retriever(search_kwargs={"k": self.child_search_k})
                    results = retriever.invoke(query)
                    all_child_results.extend(results)
                except Exception as e:
                    st.warning(f"⚠️ 자식 DB 검색 실패 ({topic}): {str(e)}")
        
        return all_child_results
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """계층 구조 정보 반환"""
        return {
            'parent_child_mapping': self.metadata_manager.parent_child_mapping,
            'child_dbs': list(self.vector_db_hierarchy.child_dbs.keys()),
            'base_path': self.vector_db_hierarchy.base_path
        }
    
    def exists(self) -> bool:
        """계층적 벡터DB가 존재하는지 확인"""
        return self.vector_db_hierarchy.exists()
    
    def get_statistics(self) -> Dict[str, Any]:
        """계층적 벡터DB 통계 정보"""
        if not self.exists():
            return {'error': '계층적 벡터DB가 존재하지 않습니다'}
        
        parent_db = self.vector_db_hierarchy.get_parent_db()
        child_dbs = self.vector_db_hierarchy.get_all_child_dbs()
        
        stats = {
            'parent_documents': len(parent_db.docstore._dict) if parent_db else 0,
            'child_databases': len(child_dbs),
            'total_child_documents': sum(len(db.docstore._dict) for db in child_dbs.values()),
            'topics': list(child_dbs.keys())
        }
        
        return stats 