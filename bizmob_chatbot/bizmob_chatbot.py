#!/usr/bin/env python3
"""
bizMOB 챗봇 - ChromaDB 전용 버전
FAISS 의존성을 완전히 제거하고 ChromaDB만 사용
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

# 경고 억제
warnings.filterwarnings("ignore")

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"bizmob_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러 (UTF-8 인코딩)
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logging()

# 환경 변수 설정
os.environ['TORCH_WARN_ON_LOAD'] = '0'
os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'

# ChromaDB 관련 import
try:
    import chromadb
    from langchain_community.vectorstores import Chroma
    CHROMADB_AVAILABLE = True
except ImportError:
    st.error("ChromaDB가 설치되지 않았습니다. pip install chromadb를 실행해주세요.")
    CHROMADB_AVAILABLE = False

# NumPy 강제 설치 확인 및 재설치
try:
    import numpy
    logger.info(f"NumPy version: {numpy.__version__}")
    
    # NumPy가 제대로 작동하는지 테스트
    test_array = numpy.array([1, 2, 3])
    logger.info("NumPy test successful")
    
    # NumPy를 전역으로 설정하여 다른 모듈에서 사용할 수 있도록 함
    import sys
    sys.modules['numpy'] = numpy
    
    # 환경 변수 설정으로 NumPy 호환성 강화
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    # PyTorch와 NumPy 호환성 강제 설정
    try:
        import torch
        if hasattr(torch, 'set_default_tensor_type'):
            torch.set_default_tensor_type('torch.FloatTensor')
        logger.info("PyTorch NumPy compatibility set")
    except Exception as e:
        logger.warning(f"PyTorch NumPy compatibility setup failed: {e}")
    
    # Transformers 라이브러리에서 NumPy 설정
    try:
        import transformers
        if hasattr(transformers, 'np'):
            transformers.np = numpy
        logger.info("Transformers NumPy compatibility set")
    except Exception as e:
        logger.warning(f"Transformers NumPy compatibility setup failed: {e}")
    
    # SentenceTransformers에서 NumPy 설정
    try:
        import sentence_transformers
        if hasattr(sentence_transformers, 'np'):
            sentence_transformers.np = numpy
        logger.info("SentenceTransformers NumPy compatibility set")
    except Exception as e:
        logger.warning(f"SentenceTransformers NumPy compatibility setup failed: {e}")
    
except ImportError:
    logger.error("NumPy is not installed")
    st.error("NumPy가 설치되지 않았습니다. pip install numpy>=1.26.2를 실행해주세요.")
    st.stop()
except Exception as e:
    logger.error(f"NumPy error: {e}")
    st.error(f"NumPy 오류가 발생했습니다. pip install numpy>=1.26.2를 실행해주세요.")
    st.stop()

# 기타 필요한 import들
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import Ollama
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    st.stop()

# 파일 처리 관련 import
try:
    from file_utils import process_file, get_supported_extensions
except ImportError:
    st.error("file_utils.py 파일을 찾을 수 없습니다.")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="bizMOB 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .user-role {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .admin-only {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_user_role():
    """사용자 권한 확인"""
    # 실제 환경에서는 데이터베이스나 인증 시스템을 사용해야 합니다
    # 여기서는 간단한 예시로 관리자 권한을 확인합니다
    if 'user_role' not in st.session_state:
        # 기본값은 일반 사용자
        st.session_state.user_role = 'user'
    
    return st.session_state.user_role

def is_admin():
    """관리자 권한 확인"""
    return check_user_role() == 'admin'

def show_role_selector():
    """사용자 권한 선택기"""
    st.sidebar.subheader("👤 사용자 권한")
    
    role_options = {
        'user': '일반 사용자',
        'admin': '관리자'
    }
    
    current_role = st.session_state.get('user_role', 'user')
    selected_role = st.selectbox(
        "권한 선택",
        options=list(role_options.keys()),
        format_func=lambda x: role_options[x],
        index=0 if current_role == 'user' else 1
    )
    
    if selected_role != current_role:
        st.session_state.user_role = selected_role
        st.rerun()
    
    # 현재 권한 표시
    role_display = "관리자" if selected_role == 'admin' else "일반 사용자"
    st.sidebar.markdown(f'<div class="user-role">현재 권한: {role_display}</div>', unsafe_allow_html=True)

def get_chroma_db_path():
    """ChromaDB 경로 반환"""
    return "./chroma_db"

def get_model_info_path():
    """모델 정보 파일 경로 반환"""
    ai_model = st.session_state.get('selected_model', 'llama3.2')
    import re
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

def get_embedding_model():
    """임베딩 모델 반환"""
    selected_embedding = st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2')
    return HuggingFaceEmbeddings(model_name=selected_embedding)

def initialize_vector_db():
    """벡터 데이터베이스 초기화"""
    logger.info("Vector database initialization started")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return False
    
    try:
        # ChromaDB 디렉토리 생성
        chroma_path = get_chroma_db_path()
        logger.info(f"ChromaDB path: {chroma_path}")
        os.makedirs(chroma_path, exist_ok=True)
        logger.info("ChromaDB directory created successfully")
        
        # 모델 정보 저장
        model_info = {
            'ai_model': st.session_state.get('selected_model', 'llama3.2'),
            'embedding_model': st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        logger.info(f"Model info: {model_info}")
        
        model_info_path = get_model_info_path()
        logger.info(f"Model info file path: {model_info_path}")
        
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        st.session_state.vector_db_initialized = True
        st.success("✅ ChromaDB 벡터 데이터베이스 초기화 완료")
        logger.info("Vector database initialization successful")
        return True
        
    except Exception as e:
        error_msg = f"벡터 데이터베이스 초기화 실패: {e}"
        logger.error(f"Vector database initialization failed: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")
        return False

def save_to_chroma_store(documents: list) -> None:
    """문서를 ChromaDB에 저장"""
    logger.info(f"Vector database save started - document count: {len(documents)}")
    logger.info(f"Document preview: {[doc.page_content[:50] + '...' if len(doc.page_content) > 50 else doc.page_content for doc in documents[:3]]}")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return
    
    try:
        selected_embedding = st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        logger.info(f"Embedding model loading started: {selected_embedding}")
        
        # NumPy 재확인 및 강제 재설치 안내
        try:
            import numpy
            logger.info(f"NumPy recheck - version: {numpy.__version__}")
            
            # NumPy 기능 테스트
            test_array = numpy.array([1, 2, 3])
            test_result = numpy.sum(test_array)
            logger.info(f"NumPy function test successful - result: {test_result}")
            
        except ImportError:
            error_msg = "NumPy가 설치되지 않았습니다. 터미널에서 다음 명령어를 실행하세요: pip install numpy>=1.26.2"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return
        except Exception as e:
            error_msg = f"NumPy 오류가 발생했습니다: {e}. 터미널에서 다음 명령어를 실행하세요: pip uninstall numpy && pip install numpy>=1.26.2"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return
        
        # HuggingFaceEmbeddings 초기화 전에 NumPy 강제 설정
        try:
            import numpy as np
            import sys
            import os
            
            # 환경 변수 설정으로 NumPy 호환성 강화
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            
            # 모든 관련 모듈에서 NumPy 강제 설정
            sys.modules['numpy'] = np
            
            # SentenceTransformers에서 NumPy 강제 설정
            try:
                import sentence_transformers
                sentence_transformers.np = np
                logger.info("SentenceTransformers NumPy pre-override successful")
            except Exception as e:
                logger.warning(f"SentenceTransformers NumPy pre-override failed: {e}")
            
            # PyTorch에서 NumPy 호환성 강화
            try:
                import torch
                if hasattr(torch, 'set_default_tensor_type'):
                    torch.set_default_tensor_type('torch.FloatTensor')
                logger.info("PyTorch NumPy compatibility set")
            except Exception as e:
                logger.warning(f"PyTorch NumPy compatibility setup failed: {e}")
            
            # HuggingFaceEmbeddings 초기화 전에 모든 모듈에서 NumPy 설정
            try:
                import transformers
                if hasattr(transformers, 'np'):
                    transformers.np = np
                logger.info("Transformers NumPy override successful")
            except Exception as e:
                logger.warning(f"Transformers NumPy override failed: {e}")
            
            # HuggingFaceEmbeddings 초기화
            embeddings = HuggingFaceEmbeddings(model_name=selected_embedding)
            logger.info("Embedding model loading completed")
            
            st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        except Exception as e:
            logger.error(f"Embedding model loading failed: {e}")
            st.error(f"임베딩 모델 로딩 실패: {e}")
            return
        
        # ChromaDB에 저장
        logger.info("ChromaDB document save started")
        try:
            # NumPy 강제 재설정 및 호환성 해결
            import numpy as np
            import torch
            
            # PyTorch에서 NumPy 사용 가능하도록 설정
            if hasattr(torch, 'set_default_tensor_type'):
                torch.set_default_tensor_type('torch.FloatTensor')
            
            # NumPy 배열 테스트
            test_embeddings = np.array([[1.0, 2.0, 3.0]])
            logger.info(f"NumPy array test successful - shape: {test_embeddings.shape}")
            
            # PyTorch 텐서를 NumPy로 변환 테스트
            torch_tensor = torch.tensor([[1.0, 2.0, 3.0]])
            numpy_array = torch_tensor.numpy()
            logger.info(f"PyTorch -> NumPy conversion test successful - shape: {numpy_array.shape}")
            
            # SentenceTransformers에서 NumPy 사용 강제 설정
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
            
            # NumPy를 전역으로 설정하여 SentenceTransformers에서 사용할 수 있도록 함
            import sys
            sys.modules['numpy'] = np
            
            # SentenceTransformers 모듈에서 NumPy 재설정
            try:
                import sentence_transformers
                if hasattr(sentence_transformers, 'np'):
                    sentence_transformers.np = np
                logger.info("SentenceTransformers NumPy override successful")
            except Exception as e:
                logger.warning(f"SentenceTransformers NumPy override failed: {e}")
            
            # ChromaDB에서 NumPy 사용 강제 설정
            try:
                import chromadb
                if hasattr(chromadb, 'np'):
                    chromadb.np = np
                # ChromaDB 내부 모듈들에도 NumPy 설정
                try:
                    import chromadb.api
                    chromadb.api.np = np
                except:
                    pass
                try:
                    import chromadb.config
                    chromadb.config.np = np
                except:
                    pass
                logger.info("ChromaDB NumPy override successful")
            except Exception as e:
                logger.warning(f"ChromaDB NumPy override failed: {e}")
            
            # LangChain에서 NumPy 사용 강제 설정
            try:
                import langchain_community
                if hasattr(langchain_community, 'np'):
                    langchain_community.np = np
                # LangChain 내부 모듈들에도 NumPy 설정
                try:
                    import langchain_community.vectorstores
                    langchain_community.vectorstores.np = np
                except:
                    pass
                logger.info("LangChain NumPy override successful")
            except Exception as e:
                logger.warning(f"LangChain NumPy override failed: {e}")
            
            # 모든 모듈에서 NumPy 사용 가능하도록 강제 설정
            import builtins
            if not hasattr(builtins, '_numpy_original'):
                builtins._numpy_original = builtins.__dict__.get('numpy', None)
                builtins.numpy = np
                logger.info("Builtins NumPy override successful")
            
            # ChromaDB 초기화 전에 추가 환경 변수 설정
            import os
            os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
            os.environ['CHROMA_SERVER_HOST'] = 'localhost'
            os.environ['CHROMA_SERVER_HTTP_PORT'] = '8000'
            
            # ChromaDB 저장 전에 최종 NumPy 확인
            try:
                import numpy as final_np
                logger.info(f"Final NumPy check - version: {final_np.__version__}")
                
                # 새로운 ChromaDB 클라이언트 방식 사용
                chroma_path = get_chroma_db_path()
                
                # 새로운 ChromaDB 클라이언트 생성
                client = chromadb.Client()
                
                # 컬렉션 생성 또는 가져오기
                collection_name = "bizmob_documents"
                try:
                    collection = client.get_collection(name=collection_name)
                    logger.info("Existing collection found")
                except:
                    collection = client.create_collection(name=collection_name)
                    logger.info("New collection created")
                
                # 문서를 ChromaDB에 저장
                documents_texts = [doc.page_content for doc in documents]
                documents_metadatas = [doc.metadata for doc in documents]
                documents_ids = [f"doc_{i}" for i in range(len(documents))]
                
                # 임베딩 생성
                embeddings_list = embeddings.embed_documents(documents_texts)
                
                # ChromaDB에 추가
                collection.add(
                    documents=documents_texts,
                    embeddings=embeddings_list,
                    metadatas=documents_metadatas,
                    ids=documents_ids
                )
                
                logger.info("ChromaDB document save completed")
                st.success("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
                logger.info("Vector database save successful")
                
            except RuntimeError as e:
                if "Numpy is not available" in str(e):
                    # 마지막 시도: ChromaDB를 직접 초기화
                    try:
                        import chromadb
                        client = chromadb.Client()
                        collection = client.create_collection(name="bizmob_documents")
                        
                        # 문서를 직접 추가
                        for i, doc in enumerate(documents):
                            collection.add(
                                documents=[doc.page_content],
                                metadatas=[doc.metadata],
                                ids=[f"doc_{i}"]
                            )
                        
                        logger.info("ChromaDB direct save completed")
                        st.success("✅ 벡터 데이터베이스 저장 완료 (직접 저장)")
                        return
                    except Exception as direct_error:
                        error_msg = f"직접 저장도 실패: {direct_error}. 터미널에서 다음 명령어를 실행하세요: pip uninstall numpy torch sentence-transformers && pip install numpy>=1.26.2 torch>=2.0.0 sentence-transformers>=2.2.0"
                        logger.error(error_msg)
                        st.error(f"❌ {error_msg}")
                        return
                else:
                    raise e
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                error_msg = "NumPy 오류가 발생했습니다. 터미널에서 다음 명령어를 실행하세요: pip uninstall numpy torch sentence-transformers && pip install numpy>=1.26.2 torch>=2.0.0 sentence-transformers>=2.2.0"
                logger.error(error_msg)
                st.error(f"❌ {error_msg}")
                st.info("💡 팁: 가상환경을 사용 중이라면 가상환경을 비활성화하고 다시 활성화한 후 설치해보세요.")
            else:
                raise e
        
    except Exception as e:
        error_msg = f"벡터 데이터베이스 저장 실패: {e}"
        logger.error(f"Vector database save failed: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")

def load_chroma_store():
    """ChromaDB에서 벡터 스토어 로드"""
    logger.info("ChromaDB vector store loading started")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        chroma_path = get_chroma_db_path()
        logger.info(f"ChromaDB path: {chroma_path}")
        
        # ChromaDB 디렉토리 존재 확인
        if not os.path.exists(chroma_path):
            error_msg = f"ChromaDB 디렉토리가 존재하지 않습니다: {chroma_path}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
        
        logger.info("Embedding model loading started")
        embeddings = get_embedding_model()
        logger.info("Embedding model loading completed")
        
        # 새로운 ChromaDB 클라이언트 방식 사용
        client = chromadb.Client()
        
        # 컬렉션 가져오기
        collection_name = "bizmob_documents"
        try:
            collection = client.get_collection(name=collection_name)
            logger.info("Existing collection found")
        except:
            logger.warning("Collection not found, creating new one")
            collection = client.create_collection(name=collection_name)
        
        # LangChain Chroma 벡터 스토어 생성
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        logger.info("ChromaDB vector store creation completed")
        
        # 벡터 스토어 정보 로깅
        try:
            collection_count = collection.count()
            logger.info(f"ChromaDB collection document count: {collection_count}")
        except Exception as e:
            logger.warning(f"Collection info check failed: {e}")
        
        return vector_store
    except Exception as e:
        error_msg = f"ChromaDB 로드 실패: {e}"
        logger.error(f"ChromaDB load failed: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")
        return None

def get_rag_chain():
    """RAG 체인 생성"""
    logger.info("RAG chain creation started")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        logger.info(f"Selected AI model: {selected_model}")
        
        # Ollama LLM 초기화
        logger.info("Ollama LLM initialization started")
        llm = Ollama(model=selected_model)
        logger.info("Ollama LLM initialization completed")
        
        # ChromaDB 벡터 스토어 로드
        logger.info("ChromaDB vector store loading started")
        vector_store = load_chroma_store()
        if vector_store is None:
            error_msg = "벡터 스토어 로드 실패"
            logger.error(error_msg)
            return None
        logger.info("ChromaDB vector store loading completed")
        
        # 프롬프트 템플릿
        logger.info("Prompt template creation")
        prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요:

컨텍스트: {context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 생성
        logger.info("RAG chain creation started")
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("RAG chain creation completed")
        
        return chain
        
    except Exception as e:
        error_msg = f"RAG 체인 생성 실패: {e}"
        logger.error(f"RAG chain creation failed: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")
        return None

def process_question(question: str) -> str:
    """질문 처리"""
    logger.info(f"Question processing started: {question[:50]}...")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        return error_msg
    
    try:
        # RAG 체인 가져오기
        logger.info("RAG chain retrieval started")
        chain = get_rag_chain()
        if chain is None:
            error_msg = "벡터 데이터베이스를 로드할 수 없습니다."
            logger.error(error_msg)
            return error_msg
        logger.info("RAG chain retrieval completed")
        
        # 질문 처리
        logger.info("Question processing execution started")
        response = chain.invoke({"query": question})
        result = response.get("result", "답변을 생성할 수 없습니다.")
        logger.info(f"Question processing completed - answer length: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"질문 처리 중 오류 발생: {e}"
        logger.error(f"Question processing error occurred: {e}", exc_info=True)
        return error_msg

def main():
    """메인 함수"""
    # session_state 초기화
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'llama3.2'
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = 'sentence-transformers/all-mpnet-base-v2'
    if 'vector_db_initialized' not in st.session_state:
        st.session_state.vector_db_initialized = False
    if 'refresh_vector_db_info' not in st.session_state:
        st.session_state.refresh_vector_db_info = False
    if 'refresh_chroma_viewer' not in st.session_state:
        st.session_state.refresh_chroma_viewer = False
    if 'chroma_viewer_page' not in st.session_state:
        st.session_state.chroma_viewer_page = 1

    # 헤더
    st.markdown('<h1 class="main-header">bizMOB 챗봇</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">PDF_bizMOB_Guide 폴더의 bizMOB Platform 가이드 문서를 기반으로 질문에 답변합니다.</p>', unsafe_allow_html=True)
    
    # 동적으로 AI 모델명 안내
    ai_model_name = st.session_state.get('selected_model', 'llama3.2')
    if 'llama3.2' in ai_model_name.lower():
        model_display = 'Meta Llama 3.2 모델'
    else:
        model_display = f"Ollama AI 모델: {ai_model_name}"
    
    st.markdown(f'<p class="sub-header">현재 사용 중: {model_display}</p>', unsafe_allow_html=True)

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 사용자 권한 선택기
        show_role_selector()

        # AI 모델 선택
        st.subheader("🤖 AI 모델 선택")
        
        # 사용 가능한 모델 목록 가져오기
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                model_lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
                model_names = [line.split()[0] for line in model_lines if line.strip()]
            else:
                model_names = ['llama3.2', 'gemma3', 'mistral']
        except:
            model_names = ['llama3.2', 'gemma3', 'mistral']
        
        # model_names가 비어있으면 기본값 설정
        if not model_names:
            model_names = ['llama3.2', 'gemma3', 'mistral']
        
        # 저장된 모델 정보 불러오기
        model_info_path = get_model_info_path()
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    saved_info = json.load(f)
                    saved_ai_model = saved_info.get('ai_model', 'llama3.2')
                    saved_embedding_model = saved_info.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
                
                if saved_ai_model in model_names:
                    st.sidebar.success(f"✅ 저장된 모델 정보를 불러왔습니다: {saved_ai_model}")
                else:
                    # 저장된 모델이 없으면 llama3.2 또는 첫 번째 모델
                    default_index = 0
                    for i, name in enumerate(model_names):
                        if 'llama3.2' in name.lower():
                            default_index = i
                            break
                    saved_ai_model = model_names[default_index]
                    st.session_state.selected_model = model_names[default_index]
            except:
                # 저장된 정보가 없으면 llama3.2 또는 첫 번째 모델
                default_index = 0
                for i, name in enumerate(model_names):
                    if 'llama3.2' in name.lower():
                        default_index = i
                        break
                saved_ai_model = model_names[default_index]
                st.session_state.selected_model = model_names[default_index]
        else:
            # 저장된 정보가 없으면 llama3.2 또는 첫 번째 모델
            default_index = 0
            for i, name in enumerate(model_names):
                if 'llama3.2' in name.lower():
                    default_index = i
                    break
            saved_ai_model = model_names[default_index]
            st.session_state.selected_model = model_names[default_index]
        
        # 모델 선택 드롭다운
        selected_model = st.selectbox(
            "AI 모델 선택",
            model_names,
            index=model_names.index(saved_ai_model) if saved_ai_model in model_names else 0
        )
        
        if selected_model != st.session_state.get('selected_model'):
            st.session_state.selected_model = selected_model
            st.session_state.vector_db_initialized = False
        
        # 임베딩 모델 선택
        st.subheader("🔍 임베딩 모델 선택")
        
        embedding_models = [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ]
        
        # 현재 선택된 모델에 따른 권장 임베딩 모델
        current_embedding = get_recommended_embedding_model(selected_model)
        
        # 저장된 임베딩 모델 정보 불러오기
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    saved_info = json.load(f)
                    saved_embedding_model = saved_info.get('embedding_model', current_embedding)
            except:
                saved_embedding_model = current_embedding
        else:
            saved_embedding_model = current_embedding
        
        # 임베딩 모델 선택 드롭다운
        selected_embedding = st.selectbox(
            "임베딩 모델 선택",
            embedding_models,
            index=embedding_models.index(saved_embedding_model) if saved_embedding_model in embedding_models else 0
        )
        
        if selected_embedding != st.session_state.get('selected_embedding_model'):
            st.session_state.selected_embedding_model = selected_embedding
            st.session_state.vector_db_initialized = False
        
        # 벡터 DB 초기화 버튼 (관리자만)
        if is_admin():
            st.subheader("🗄️ 벡터 데이터베이스")
            
            if st.button("벡터 DB 초기화", type="primary"):
                if initialize_vector_db():
                    st.session_state.vector_db_initialized = True
            
            # 벡터 DB 상태 표시
            if st.session_state.get('vector_db_initialized', False):
                st.success("✅ 벡터 DB 초기화됨")
            else:
                st.warning("⚠️ 벡터 DB 초기화 필요")
        else:
            # 일반 사용자에게는 간단한 상태만 표시
            st.subheader("🗄️ 벡터 데이터베이스")
            if st.session_state.get('vector_db_initialized', False):
                st.success("✅ 벡터 DB 준비됨")
            else:
                st.warning("⚠️ 관리자가 벡터 DB를 초기화해야 합니다")

    # 사용자 권한에 따른 탭 구성
    user_role = check_user_role()
    
    if user_role == 'user':
        # 일반 사용자: 챗봇 기능만 표시
        st.markdown('<div class="admin-only">🔒 일반 사용자 모드: 챗봇 기능만 사용 가능합니다.</div>', unsafe_allow_html=True)
        
        # 챗봇 인터페이스
        st.subheader("💬 질문하기")
        
        # 현재 선택된 모델 정보 표시
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        if selected_model:
            # 저장된 모델 정보가 있는지 확인
            model_info_path = get_model_info_path()
            if os.path.exists(model_info_path):
                try:
                    with open(model_info_path, 'r', encoding='utf-8') as f:
                        model_info = json.load(f)
                        st.info(f"📊 현재 모델: {model_info.get('ai_model', 'Unknown')}")
                        st.info(f"🔍 임베딩 모델: {model_info.get('embedding_model', 'Unknown')}")
                        st.info(f"⏰ 생성 시간: {model_info.get('timestamp', 'Unknown')}")
                except:
                    st.warning("모델 정보를 불러올 수 없습니다.")
            else:
                st.warning("저장된 모델 정보가 없습니다.")
        
        # 질문 입력
        question = st.text_area("질문을 입력하세요:", height=100)
        
        if st.button("질문하기", type="primary"):
            if question.strip():
                with st.spinner("답변을 생성하는 중..."):
                    answer = process_question(question)
                    st.markdown("### 답변:")
                    st.write(answer)
            else:
                st.warning("질문을 입력해주세요.")
    
    else:
        # 관리자: 모든 기능 표시
        st.markdown('<div class="admin-only">🔧 관리자 모드: 모든 기능을 사용할 수 있습니다.</div>', unsafe_allow_html=True)
        
        # 메인 탭
        tab1, tab2, tab3, tab4 = st.tabs(["💬 챗봇", "📁 파일 업로드", "🗄️ 벡터 DB 관리", "ℹ️ 정보"])
        
        with tab1:
            # 현재 선택된 모델 정보 표시
            selected_model = st.session_state.get('selected_model', 'llama3.2')
            if selected_model:
                # 저장된 모델 정보가 있는지 확인
                model_info_path = get_model_info_path()
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                            st.info(f"📊 현재 모델: {model_info.get('ai_model', 'Unknown')}")
                            st.info(f"🔍 임베딩 모델: {model_info.get('embedding_model', 'Unknown')}")
                            st.info(f"⏰ 생성 시간: {model_info.get('timestamp', 'Unknown')}")
                    except:
                        st.warning("모델 정보를 불러올 수 없습니다.")
                else:
                    st.warning("저장된 모델 정보가 없습니다.")
            
            # 챗봇 인터페이스
            st.subheader("💬 질문하기")
            
            # 질문 입력
            question = st.text_area("질문을 입력하세요:", height=100)
            
            if st.button("질문하기", type="primary"):
                if question.strip():
                    with st.spinner("답변을 생성하는 중..."):
                        answer = process_question(question)
                        st.markdown("### 답변:")
                        st.write(answer)
                else:
                    st.warning("질문을 입력해주세요.")
        
        with tab2:
            st.subheader("📁 문서 업로드")
            
            # 지원되는 파일 형식 표시
            supported_extensions = get_supported_extensions()
            st.info(f"지원되는 파일 형식: {', '.join(supported_extensions)}")
            
            # 파일 업로드
            uploaded_files = st.file_uploader(
                "문서를 업로드하세요",
                type=supported_extensions,
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"업로드된 파일: {len(uploaded_files)}개")
                
                if st.button("문서 처리 및 벡터 DB 저장", type="primary"):
                    with st.spinner("문서를 처리하는 중..."):
                        all_documents = []
                        
                        for uploaded_file in uploaded_files:
                            try:
                                # 파일 처리
                                documents = process_file(uploaded_file)
                                all_documents.extend(documents)
                                st.success(f"✅ {uploaded_file.name} 처리 완료")
                            except Exception as e:
                                st.error(f"❌ {uploaded_file.name} 처리 실패: {e}")
                        
                        if all_documents:
                            # ChromaDB에 저장
                            save_to_chroma_store(all_documents)
                            st.session_state.vector_db_initialized = True
                        else:
                            st.warning("처리할 문서가 없습니다.")
        
        with tab3:
            st.subheader("🗄️ 벡터 DB 관리")
            
            # ChromaDB 상태 확인
            chroma_path = get_chroma_db_path()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 벡터 DB 정보")
                
                if os.path.exists(chroma_path):
                    try:
                        # ChromaDB 파일 정보
                        chroma_files = os.listdir(chroma_path)
                        total_size = sum(os.path.getsize(os.path.join(chroma_path, f)) for f in chroma_files if os.path.isfile(os.path.join(chroma_path, f)))
                        
                        st.success("✅ ChromaDB 존재")
                        st.info(f"📁 파일 수: {len(chroma_files)}개")
                        st.info(f"💾 크기: {total_size / 1024:.2f} KB")
                        
                        # 파일 목록 표시
                        with st.expander("📋 파일 목록 보기"):
                            for file in chroma_files:
                                file_path = os.path.join(chroma_path, file)
                                file_size = os.path.getsize(file_path) / 1024
                                st.write(f"• {file} ({file_size:.2f} KB)")
                        
                        # 벡터 DB 내용 검색
                        st.subheader("🔍 벡터 DB 내용 검색")
                        search_query = st.text_input("검색어를 입력하세요:")
                        
                        if search_query and st.button("검색", type="primary"):
                            try:
                                from vector_db_utils import search_chroma_documents
                                results = search_chroma_documents(search_query, st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2'))
                                
                                if results:
                                    st.success(f"검색 결과: {len(results)}개")
                                    for i, (doc, score) in enumerate(results):
                                        with st.expander(f"결과 {i+1} (유사도: {1/(1+score):.3f})"):
                                            st.write(f"**내용:** {doc.page_content[:200]}...")
                                            st.write(f"**메타데이터:** {doc.metadata}")
                                else:
                                    st.warning("검색 결과가 없습니다.")
                            except Exception as e:
                                st.error(f"검색 중 오류: {e}")
                        
                    except Exception as e:
                        st.error(f"ChromaDB 정보 확인 실패: {e}")
                else:
                    st.warning("⚠️ ChromaDB가 존재하지 않습니다.")
            
            with col2:
                st.subheader("💾 벡터 DB 다운로드")
                
                if os.path.exists(chroma_path):
                    try:
                        # ChromaDB 압축 다운로드
                        import zipfile
                        import tempfile
                        
                        if st.button("📦 ChromaDB 전체 다운로드", type="primary"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                                with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    for root, dirs, files in os.walk(chroma_path):
                                        for file in files:
                                            file_path = os.path.join(root, file)
                                            arcname = os.path.relpath(file_path, chroma_path)
                                            zipf.write(file_path, arcname)
                                
                                # 다운로드 버튼 생성
                                with open(tmp_file.name, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ 다운로드",
                                        data=f.read(),
                                        file_name=f"chroma_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip"
                                    )
                                
                                # 임시 파일 삭제
                                os.unlink(tmp_file.name)
                        
                        # 모델 정보 파일 다운로드
                        model_info_path = get_model_info_path()
                        if os.path.exists(model_info_path):
                            with open(model_info_path, 'r', encoding='utf-8') as f:
                                model_info_data = f.read()
                            
                            st.download_button(
                                label="📄 모델 정보 다운로드",
                                data=model_info_data,
                                file_name=f"model_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        # 벡터 DB 초기화
                        st.subheader("🗑️ 벡터 DB 초기화")
                        if st.button("⚠️ 벡터 DB 완전 삭제", type="secondary"):
                            try:
                                import shutil
                                shutil.rmtree(chroma_path)
                                st.success("✅ 벡터 DB가 삭제되었습니다.")
                                st.session_state.vector_db_initialized = False
                            except Exception as e:
                                st.error(f"삭제 실패: {e}")
                        
                    except Exception as e:
                        st.error(f"다운로드 기능 오류: {e}")
                else:
                    st.warning("다운로드할 ChromaDB가 없습니다.")
        
        with tab4:
            st.subheader("ℹ️ 시스템 정보")
            
            # 시스템 정보 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 환경 정보")
                st.info(f"Python 버전: {sys.version}")
                st.info(f"Streamlit 버전: {st.__version__}")
                
                # ChromaDB 상태 확인
                chroma_path = get_chroma_db_path()
                if os.path.exists(chroma_path):
                    st.success("✅ ChromaDB 디렉토리 존재")
                    
                    # ChromaDB 파일 목록
                    try:
                        chroma_files = os.listdir(chroma_path)
                        if chroma_files:
                            st.write("ChromaDB 파일:")
                            for file in chroma_files:
                                st.write(f"- {file}")
                        else:
                            st.warning("ChromaDB가 비어있습니다.")
                    except Exception as e:
                        st.error(f"ChromaDB 파일 목록 확인 실패: {e}")
                else:
                    st.warning("⚠️ ChromaDB 디렉토리가 없습니다.")
            
            with col2:
                st.subheader("📋 모델 정보")
                
                # 모델 정보 파일 확인
                model_info_path = get_model_info_path()
                if os.path.exists(model_info_path):
                    st.success("✅ 모델 정보 파일 존재")
                    try:
                        with open(model_info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                            st.json(model_info)
                    except Exception as e:
                        st.error(f"모델 정보 파일 읽기 실패: {e}")
                else:
                    st.warning("⚠️ 모델 정보 파일이 없습니다.")
                
                # 현재 세션 상태 정보
                st.subheader("🎛️ 현재 설정")
                st.info(f"선택된 AI 모델: {st.session_state.get('selected_model', 'llama3.2')}")
                st.info(f"선택된 임베딩 모델: {st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2')}")
                st.info(f"벡터 DB 초기화: {'✅ 완료' if st.session_state.get('vector_db_initialized', False) else '⚠️ 필요'}")

if __name__ == "__main__":
    main() 