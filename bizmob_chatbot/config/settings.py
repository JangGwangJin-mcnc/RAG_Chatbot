"""
Configuration settings for bizMOB Platform Chatbot
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Settings:
    """애플리케이션 설정"""
    
    # 모델 설정
    model_name: str = "llama3.2:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 벡터 DB 설정
    collection_name: str = "bizmob_documents"
    chroma_persist_directory: str = "./chroma_db"
    
    # 문서 설정
    default_document_folder: str = "PDF_bizMOB_Guide"
    supported_extensions: List[str] = None
    
    # UI 설정
    page_title: str = "bizMOB Platform Chatbot"
    page_icon: str = "🤖"
    
    # 인증 설정
    admin_password: str = "0000"
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.txt', '.docx', '.xlsx', '.pptx']
    
    @property
    def model_url(self) -> str:
        """Ollama 모델 URL"""
        return f"http://localhost:11434/api/generate"
    
    @property
    def embedding_url(self) -> str:
        """임베딩 모델 URL"""
        return f"http://localhost:11434/api/embeddings"
    
    def get_document_folder_path(self) -> str:
        """문서 폴더 경로 반환"""
        return os.path.join(os.getcwd(), self.default_document_folder)
    
    def get_chroma_path(self) -> str:
        """ChromaDB 경로 반환"""
        return os.path.abspath(self.chroma_persist_directory)


# 전역 설정 인스턴스
settings = Settings() 