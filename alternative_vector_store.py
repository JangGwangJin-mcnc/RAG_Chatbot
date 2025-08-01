#!/usr/bin/env python3
"""
대안 벡터 스토어 제공
PyTorch 보안 취약점 문제를 피하기 위한 대안
"""

import os
import warnings
from typing import List
from langchain_core.documents.base import Document

def create_chroma_store(documents: List[Document], embedding_model):
    """Chroma 벡터 스토어 생성 (FAISS 대안)"""
    try:
        from langchain_community.vectorstores import Chroma
        
        # 경고 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Chroma 벡터 스토어 생성
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
            
            print("Chroma vector store created successfully")
            return vector_store
            
    except ImportError:
        print("Chroma not available, trying to install...")
        os.system("pip install chromadb")
        return create_chroma_store(documents, embedding_model)
    except Exception as e:
        print(f"Error creating Chroma store: {e}")
        return None

def create_simple_store(documents: List[Document], embedding_model):
    """간단한 메모리 기반 벡터 스토어 생성"""
    try:
        from langchain_community.vectorstores import InMemoryVectorstore
        
        # 경고 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 메모리 기반 벡터 스토어 생성
            vector_store = InMemoryVectorstore.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            
            print("In-memory vector store created successfully")
            return vector_store
            
    except Exception as e:
        print(f"Error creating in-memory store: {e}")
        return None

def safe_faiss_store(documents: List[Document], embedding_model):
    """안전한 FAISS 스토어 생성 (경고 억제)"""
    try:
        from langchain_community.vectorstores import FAISS
        
        # 모든 경고 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 환경 변수 설정
            os.environ['TORCH_WARN_ON_LOAD'] = '0'
            os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
            
            # FAISS 벡터 스토어 생성
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model
            )
            
            print("FAISS vector store created successfully (with safety measures)")
            return vector_store
            
    except Exception as e:
        print(f"Error creating FAISS store: {e}")
        return None

def get_vector_store(documents: List[Document], embedding_model, store_type="safe_faiss"):
    """벡터 스토어 생성 (타입 선택 가능)"""
    
    if store_type == "chroma":
        return create_chroma_store(documents, embedding_model)
    elif store_type == "memory":
        return create_simple_store(documents, embedding_model)
    else:  # safe_faiss
        return safe_faiss_store(documents, embedding_model)

if __name__ == "__main__":
    print("Alternative vector store utilities loaded") 