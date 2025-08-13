"""
벡터 DB 관리 모듈
"""

import streamlit as st
import os
import re
import pandas as pd
import json
import logging
from typing import List
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 로깅 설정
logger = logging.getLogger(__name__)


class VectorDBManager:
    """벡터 DB 관리 클래스"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def get_vector_db_path(self) -> str:
        """현재 선택된 AI 모델에 맞는 벡터DB 경로 반환"""
        ai_model = st.session_state.get('selected_model', 'hyperclovax')
        # 파일명에 사용할 수 없는 문자는 언더스코어로 대체
        safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
        return f"bizmob_faiss_index_{safe_model}"
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할"""
        return self.text_splitter.split_documents(documents)
    
    def save_to_vector_store(self, documents: List[Document]) -> bool:
        """문서를 벡터 스토어에 저장"""
        try:
            # 임베딩 모델 로드 - 메타 텐서 오류 해결
            selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
            
            # 메타 텐서 오류 해결을 위한 설정
            import torch
            torch.set_num_threads(1)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"임베딩 모델 로딩 중: {selected_embedding}")
            
            # 벡터 스토어 생성 및 저장
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local(self.get_vector_db_path())
            
            logger.info("벡터 데이터베이스 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"벡터 데이터베이스 저장 실패: {str(e)}")
            return False
    
    def initialize_vector_db(self, documents: List[Document]) -> bool:
        """벡터 DB 초기화"""
        try:
            if not documents:
                logger.error("로드할 문서가 없습니다.")
                return False
            
            # 문서 청킹
            logger.info("문서를 청크로 분할하는 중...")
            chunked_documents = self.chunk_documents(documents)
            logger.info(f"{len(chunked_documents)}개의 청크로 분할 완료")
            
            # 벡터DB 저장
            logger.info("벡터 데이터베이스에 저장하는 중...")
            success = self.save_to_vector_store(chunked_documents)
            
            if success:
                # 성공적으로 초기화된 모델 정보를 파일에 저장
                try:
                    model_info = {
                        'ai_model': st.session_state.get('selected_model', 'hyperclovax'),
                        'embedding_model': st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'),
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    with open('vector_db_model_info.json', 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, ensure_ascii=False, indent=2)
                    
                    logger.info("모델 정보가 저장되었습니다.")
                except Exception as e:
                    logger.warning(f"모델 정보 저장 중 오류: {str(e)}")
            
            return success
            
        except Exception as e:
            logger.error(f"벡터 DB 초기화 실패: {str(e)}")
            return False
    
    def save_model_info(self):
        """모델 정보를 파일에 저장"""
        try:
            model_info = {
                'ai_model': st.session_state.get('selected_model', 'hyperclovax'),
                'embedding_model': st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open('vector_db_model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            logger.info("모델 정보가 저장되었습니다.")
        except Exception as e:
            logger.warning(f"모델 정보 저장 중 오류: {str(e)}")
    
    def check_vector_db_exists(self) -> bool:
        """벡터 DB가 존재하는지 확인"""
        vector_db_path = self.get_vector_db_path()
        return os.path.exists(vector_db_path)
    
    def load_saved_model_info(self) -> dict:
        """저장된 모델 정보 로드"""
        try:
            if os.path.exists('vector_db_model_info.json'):
                with open('vector_db_model_info.json', 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"모델 정보 로드 실패: {str(e)}")
        return {} 

    def get_vector_db_info(self) -> dict:
        """벡터 DB 정보 조회"""
        try:
            vector_db_path = self.get_vector_db_path()
            if not os.path.exists(vector_db_path):
                return {
                    'exists': False,
                    'path': vector_db_path,
                    'document_count': 0,
                    'index_size': 0
                }
            
            # 임베딩 모델 로드 - 메타 텐서 오류 해결
            selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
            
            # 메타 텐서 오류 해결을 위한 설정
            import torch
            torch.set_num_threads(1)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 벡터 스토어 로드
            vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            
            # 문서 개수 확인
            document_count = len(vector_store.docstore._dict)
            
            # 인덱스 크기 확인 (MB)
            index_size = 0
            if os.path.exists(vector_db_path):
                for root, dirs, files in os.walk(vector_db_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        index_size += os.path.getsize(file_path)
                index_size = index_size / (1024 * 1024)  # MB로 변환
            
            return {
                'exists': True,
                'path': vector_db_path,
                'document_count': document_count,
                'index_size': index_size
            }
            
        except Exception as e:
            logger.error(f"벡터 DB 정보 조회 실패: {str(e)}")
            return {
                'exists': False,
                'path': self.get_vector_db_path(),
                'document_count': 0,
                'index_size': 0,
                'error': str(e)
            }
    
    def get_vector_db_samples(self, limit: int = 10) -> list:
        """벡터 DB에서 샘플 데이터 조회"""
        try:
            vector_db_path = self.get_vector_db_path()
            if not os.path.exists(vector_db_path):
                return []
            
            # 임베딩 모델 로드 - 메타 텐서 오류 해결
            selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
            
            # 메타 텐서 오류 해결을 위한 설정
            import torch
            torch.set_num_threads(1)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 벡터 스토어 로드
            vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            
            # 샘플 데이터 조회
            samples = []
            docstore = vector_store.docstore._dict
            
            for i, (doc_id, doc) in enumerate(docstore.items()):
                if i >= limit:
                    break
                
                # 문서 내용 일부 추출 (처음 200자)
                content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                samples.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': doc.metadata,
                    'length': len(doc.page_content)
                })
            
            return samples
            
        except Exception as e:
            logger.error(f"벡터 DB 샘플 조회 실패: {str(e)}")
            return []
    
    def search_vector_db(self, query: str, k: int = 5) -> list:
        """벡터 DB에서 검색"""
        try:
            vector_db_path = self.get_vector_db_path()
            if not os.path.exists(vector_db_path):
                return []
            
            # 임베딩 모델 로드 - 메타 텐서 오류 해결
            selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
            
            # 메타 텐서 오류 해결을 위한 설정
            import torch
            torch.set_num_threads(1)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 벡터 스토어 로드
            vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            
            # 검색 실행
            results = vector_store.similarity_search(query, k=k)
            
            search_results = []
            for i, doc in enumerate(results):
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                search_results.append({
                    'rank': i + 1,
                    'content': content,
                    'metadata': doc.metadata,
                    'length': len(doc.page_content)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"벡터 DB 검색 실패: {str(e)}")
            return [] 

    def get_embedding_model(self):
        """임베딩 모델 반환"""
        try:
            # 기본 임베딩 모델 설정
            model_name = "jhgan/ko-sroberta-multitask"
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"임베딩 모델 로드 완료: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {str(e)}")
            return None 