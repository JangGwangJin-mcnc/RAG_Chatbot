#!/usr/bin/env python3
"""
벡터 데이터베이스 유틸리티 - ChromaDB 전용 버전
FAISS 의존성을 완전히 제거하고 ChromaDB만 사용
"""

import os
import json
import pandas as pd
import re
from typing import List, Dict, Any, Optional
import warnings

# 경고 억제
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ['TORCH_WARN_ON_LOAD'] = '0'
os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'

# ChromaDB 관련 import
try:
    from langchain_community.vectorstores import Chroma
    # HuggingFaceEmbeddings 대신 SafeSentenceTransformerEmbeddings 사용
    from langchain_community.embeddings import HuggingFaceEmbeddings
    CHROMADB_AVAILABLE = True
except ImportError:
    print("ChromaDB가 설치되지 않았습니다. pip install chromadb를 실행해주세요.")
    CHROMADB_AVAILABLE = False

# SafeSentenceTransformerEmbeddings 클래스 정의 (torch.load 취약점 방지)
class SafeSentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델을 안전하게 로드 (safetensors 직접 사용)"""
        try:
            # 환경 변수 설정으로 safetensors 강제 사용
            os.environ['SAFETENSORS_FAST_GPU'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            os.environ['TORCH_WEIGHTS_ONLY'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['TRANSFORMERS_USE_SAFETENSORS'] = '1'
            os.environ['TORCH_WARN_ON_LOAD'] = '0'
            os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
            os.environ['TRANSFORMERS_SAFE_SERIALIZATION'] = '1'
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
            
            # transformers와 safetensors를 직접 사용
            from transformers import AutoTokenizer, AutoModel
            import torch
            import safetensors
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True
            )
            
            # 모델 로드 (safetensors 강제 사용)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise Exception(f"모델 로딩 실패: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩"""
        if self.model is None:
            self._load_model()
        
        try:
            embeddings = []
            for text in texts:
                # 토크나이징
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # GPU로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 마지막 hidden state의 평균을 임베딩으로 사용
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(embedding.cpu().numpy().flatten().tolist())
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"임베딩 생성 실패: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩"""
        if self.model is None:
            self._load_model()
        
        try:
            # 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # GPU로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 마지막 hidden state의 평균을 임베딩으로 사용
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding.cpu().numpy().flatten().tolist()
                
        except Exception as e:
            raise Exception(f"쿼리 임베딩 실패: {e}")

def get_chroma_db_path():
    """ChromaDB 경로 반환"""
    return "./chroma_db"

def get_model_info_path(ai_model: str = 'llama3.2'):
    """모델 정보 파일 경로 반환"""
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
    return f"vector_db_model_info_{safe_model}.json"

def get_recommended_embedding_model(ai_model_name: str) -> str:
    """AI 모델에 따른 권장 임베딩 모델을 반환"""
    model_mapping = {
        'llama3.2': 'sentence-transformers/all-mpnet-base-v2',
        'llama3.2:3b': 'sentence-transformers/all-MiniLM-L6-v2',
        'gemma3': 'sentence-transformers/all-mpnet-base-v2',
        'gemma2': 'sentence-transformers/all-MiniLM-L6-v2',
        'mistral': 'sentence-transformers/all-mpnet-base-v2',
        'codellama': 'sentence-transformers/all-mpnet-base-v2'
    }
    
    for key, value in model_mapping.items():
        if key in ai_model_name.lower():
            return value
    return 'sentence-transformers/all-mpnet-base-v2'

def get_embedding_model(embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
    """임베딩 모델 반환 (SafeSentenceTransformerEmbeddings 사용)"""
    return SafeSentenceTransformerEmbeddings(model_name=embedding_model_name)

def initialize_chroma_db(ai_model: str = 'llama3.2', embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'):
    """ChromaDB 초기화"""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB가 설치되지 않았습니다.")
        return False
    
    try:
        # ChromaDB 디렉토리 생성
        chroma_path = get_chroma_db_path()
        os.makedirs(chroma_path, exist_ok=True)
        
        # 모델 정보 저장
        model_info = {
            'ai_model': ai_model,
            'embedding_model': embedding_model,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(get_model_info_path(ai_model), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print("✅ ChromaDB 벡터 데이터베이스 초기화 완료")
        return True
        
    except Exception as e:
        print(f"❌ 벡터 데이터베이스 초기화 실패: {e}")
        return False

def save_to_chroma_store(documents: list, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2') -> bool:
    """문서를 ChromaDB에 저장"""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB가 설치되지 않았습니다.")
        return False
    
    try:
        # SafeSentenceTransformerEmbeddings 사용 (torch.load 취약점 방지)
        embeddings = SafeSentenceTransformerEmbeddings(model_name=embedding_model)
        
        print(f"임베딩 모델 로딩 중: {embedding_model}")
        
        # ChromaDB에 저장
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=get_chroma_db_path()
        )
        vector_store.persist()
        
        print("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
        return True
        
    except Exception as e:
        print(f"❌ 벡터 데이터베이스 저장 실패: {e}")
        return False

def load_chroma_store(embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'):
    """ChromaDB에서 벡터 스토어 로드"""
    if not CHROMADB_AVAILABLE:
        print("ChromaDB가 설치되지 않았습니다.")
        return None
    
    try:
        # SafeSentenceTransformerEmbeddings 사용 (torch.load 취약점 방지)
        embeddings = SafeSentenceTransformerEmbeddings(model_name=embedding_model)
        vector_store = Chroma(
            persist_directory=get_chroma_db_path(),
            embedding_function=embeddings
        )
        return vector_store
    except Exception as e:
        print(f"❌ ChromaDB 로드 실패: {e}")
        return None

def get_chroma_info():
    """ChromaDB 정보 반환"""
    chroma_path = get_chroma_db_path()
    
    if not os.path.exists(chroma_path):
        return {
            'exists': False,
            'files': [],
            'size': 0
        }
    
    try:
        files = os.listdir(chroma_path)
        total_size = sum(os.path.getsize(os.path.join(chroma_path, f)) for f in files if os.path.isfile(os.path.join(chroma_path, f)))
        
        return {
            'exists': True,
            'files': files,
            'size': total_size,
            'path': chroma_path
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'files': [],
            'size': 0
        }

def get_model_info(ai_model: str = 'llama3.2'):
    """모델 정보 반환"""
    model_info_path = get_model_info_path(ai_model)
    
    if not os.path.exists(model_info_path):
        return None
    
    try:
        with open(model_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"모델 정보 파일 읽기 실패: {e}")
        return None

def clear_chroma_db():
    """ChromaDB 초기화"""
    chroma_path = get_chroma_db_path()
    
    if os.path.exists(chroma_path):
        try:
            import shutil
            shutil.rmtree(chroma_path)
            print("✅ ChromaDB 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ ChromaDB 초기화 실패: {e}")
            return False
    else:
        print("ChromaDB가 존재하지 않습니다.")
        return True

def search_chroma_documents(query: str, embedding_model: str = 'sentence-transformers/all-mpnet-base-v2', k: int = 5):
    """ChromaDB에서 문서 검색"""
    if not CHROMADB_AVAILABLE:
        return []
    
    try:
        vector_store = load_chroma_store(embedding_model)
        if vector_store is None:
            return []
        
        results = vector_store.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        print(f"문서 검색 실패: {e}")
        return []

if __name__ == "__main__":
    # 테스트 코드
    print("ChromaDB 벡터 데이터베이스 유틸리티")
    print(f"ChromaDB 사용 가능: {CHROMADB_AVAILABLE}")
    
    if CHROMADB_AVAILABLE:
        info = get_chroma_info()
        print(f"ChromaDB 정보: {info}")
        
        model_info = get_model_info()
        if model_info:
            print(f"모델 정보: {model_info}")
        else:
            print("모델 정보가 없습니다.") 