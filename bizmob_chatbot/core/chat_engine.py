"""
채팅 엔진 모듈
"""

import streamlit as st
import logging
import time
from typing import List, Tuple
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 로깅 설정
logger = logging.getLogger(__name__)


class ChatEngine:
    """채팅 엔진 클래스"""
    
    def __init__(self):
        if 'chat_engine_initialized' not in st.session_state:
            logger.info("ChatEngine 초기화 완료")
            st.session_state.chat_engine_initialized = True
    
    def get_embedding_model(self):
        """임베딩 모델 반환"""
        try:
            selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
            
            # 메타 텐서 오류 해결을 위한 설정
            import torch
            torch.set_num_threads(1)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'low_cpu_mem_usage': True,
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {str(e)}")
            return None
    
    def load_vector_store(self):
        """벡터 스토어 로드"""
        try:
            embeddings = self.get_embedding_model()
            if embeddings is None:
                return None
            
            # 벡터 DB 경로
            import os
            import re
            ai_model = st.session_state.get('selected_model', 'hyperclovax')
            safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
            vector_db_path = f"bizmob_faiss_index_{safe_model}"
            
            if not os.path.exists(vector_db_path):
                logger.warning("벡터 DB가 존재하지 않습니다")
                return None
            
            # FAISS 벡터 스토어 로드
            vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            return vector_store
            
        except Exception as e:
            logger.error(f"벡터 스토어 로드 실패: {str(e)}")
            return None
    
    def get_rag_chain(self) -> Runnable:
        """RAG 체인 생성"""
        try:
            # 선택된 모델 가져오기
            selected_model = st.session_state.get('selected_model', 'hyperclovax')
            
            # Ollama LLM 초기화
            llm = OllamaLLM(
                model=selected_model,
                temperature=0.1,
                top_p=0.9,
                max_tokens=2048
            )
            
            # 벡터 스토어 로드
            vector_store = self.load_vector_store()
            if vector_store is None:
                logger.error("벡터 스토어를 로드할 수 없습니다")
                return None
            
            # 프롬프트 템플릿
            template = """당신은 bizMOB Platform 전문가입니다. 
다음 컨텍스트를 사용하여 질문에 답변해주세요. 답변은 한글로 해주세요:

컨텍스트:
{context}

질문: {question}

답변:"""
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            # RAG 체인 생성
            chain = (
                {"context": vector_store.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"RAG 체인 생성 중 오류: {str(e)}")
            return None
    
    def process_question(self, user_question: str) -> Tuple[str, List[Document]]:
        """사용자 질문에 대한 RAG 처리"""
        try:
            logger.info(f"질문 처리 시작: {user_question}")
            
            # RAG 체인 선언
            chain = self.get_rag_chain()
            if chain is None:
                logger.error("RAG 체인 생성에 실패했습니다")
                return "RAG 체인을 생성할 수 없습니다. 벡터 DB가 초기화되었는지 확인해주세요.", []
            
            # 질문만 전달하여 RAG 체인 실행
            logger.info("RAG 체인 실행 중...")
            response = chain.invoke(user_question)
            
            # 관련 문서는 별도로 검색 (참조용)
            vector_store = self.load_vector_store()
            retrieve_docs = []
            if vector_store:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                retrieve_docs = retriever.invoke(user_question)
                logger.info(f"검색된 관련 문서: {len(retrieve_docs)}개")
            else:
                logger.warning("벡터 스토어를 로드할 수 없어 관련 문서를 검색할 수 없습니다")
            
            logger.info("질문 처리 완료")
            return response, retrieve_docs
            
        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {str(e)}")
            return f"질문 처리 중 오류가 발생했습니다: {str(e)}", []
    
    def generate_response(self, user_input: str) -> str:
        """사용자 입력에 대한 응답 생성"""
        try:
            logger.info("=== RAG 처리 과정 시작 ===")
            
            # 실제 RAG 처리
            response, context_docs = self.process_question(user_input)
            
            if response:
                logger.info("=== RAG 처리 과정 완료 ===")
                return response
            else:
                logger.error("RAG 처리 결과가 없습니다")
                return "죄송합니다. 응답을 생성할 수 없습니다. 벡터 DB가 초기화되었는지 확인해주세요."
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류: {str(e)}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}" 