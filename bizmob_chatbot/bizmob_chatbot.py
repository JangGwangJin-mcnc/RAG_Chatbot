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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 로거 초기화
logger = setup_logging()

# 환경 변수 설정
os.environ['TORCH_WARN_ON_LOAD'] = '0'
os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'

# ChromaDB 관련 import
try:
    from langchain_community.vectorstores import Chroma
    CHROMADB_AVAILABLE = True
except ImportError:
    st.error("ChromaDB가 설치되지 않았습니다. pip install chromadb를 실행해주세요.")
    CHROMADB_AVAILABLE = False

# NumPy 강제 설치 확인 및 재설치
try:
    import numpy
    logger.info(f"NumPy 버전: {numpy.__version__}")
    
    # NumPy가 제대로 작동하는지 테스트
    test_array = numpy.array([1, 2, 3])
    logger.info("NumPy 테스트 성공")
    
except ImportError:
    logger.error("NumPy가 설치되지 않았습니다.")
    st.error("NumPy가 설치되지 않았습니다. pip install numpy>=1.26.2를 실행해주세요.")
    st.stop()
except Exception as e:
    logger.error(f"NumPy 오류: {e}")
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
</style>
""", unsafe_allow_html=True)

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
    logger.info("벡터 데이터베이스 초기화 시작")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return False
    
    try:
        # ChromaDB 디렉토리 생성
        chroma_path = get_chroma_db_path()
        logger.info(f"ChromaDB 경로: {chroma_path}")
        os.makedirs(chroma_path, exist_ok=True)
        logger.info("ChromaDB 디렉토리 생성 완료")
        
        # 모델 정보 저장
        model_info = {
            'ai_model': st.session_state.get('selected_model', 'llama3.2'),
            'embedding_model': st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        logger.info(f"모델 정보: {model_info}")
        
        model_info_path = get_model_info_path()
        logger.info(f"모델 정보 파일 경로: {model_info_path}")
        
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        st.session_state.vector_db_initialized = True
        st.success("✅ ChromaDB 벡터 데이터베이스 초기화 완료")
        logger.info("벡터 데이터베이스 초기화 성공")
        return True
        
    except Exception as e:
        error_msg = f"벡터 데이터베이스 초기화 실패: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        return False

def save_to_chroma_store(documents: list) -> None:
    """문서를 ChromaDB에 저장"""
    logger.info(f"벡터 데이터베이스 저장 시작 - 문서 수: {len(documents)}")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return
    
    try:
        selected_embedding = st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        logger.info(f"임베딩 모델 로딩 시작: {selected_embedding}")
        
        # NumPy 재확인 및 강제 재설치 안내
        try:
            import numpy
            logger.info(f"NumPy 재확인: {numpy.__version__}")
            
            # NumPy 기능 테스트
            test_array = numpy.array([1, 2, 3])
            test_result = numpy.sum(test_array)
            logger.info(f"NumPy 기능 테스트 성공: {test_result}")
            
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
        
        embeddings = HuggingFaceEmbeddings(model_name=selected_embedding)
        logger.info("임베딩 모델 로딩 완료")
        
        st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        
        # ChromaDB에 저장
        logger.info("ChromaDB에 문서 저장 시작")
        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=get_chroma_db_path()
            )
            vector_store.persist()
            logger.info("ChromaDB 문서 저장 완료")
            
            st.success("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
            logger.info("벡터 데이터베이스 저장 성공")
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                error_msg = "NumPy 오류가 발생했습니다. 터미널에서 다음 명령어를 실행하세요: pip uninstall numpy && pip install numpy>=1.26.2"
                logger.error(error_msg)
                st.error(f"❌ {error_msg}")
                st.info("💡 팁: 가상환경을 사용 중이라면 가상환경을 비활성화하고 다시 활성화한 후 설치해보세요.")
            else:
                raise e
        
    except Exception as e:
        error_msg = f"벡터 데이터베이스 저장 실패: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")

def load_chroma_store():
    """ChromaDB에서 벡터 스토어 로드"""
    logger.info("ChromaDB 벡터 스토어 로드 시작")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        chroma_path = get_chroma_db_path()
        logger.info(f"ChromaDB 경로: {chroma_path}")
        
        # ChromaDB 디렉토리 존재 확인
        if not os.path.exists(chroma_path):
            error_msg = f"ChromaDB 디렉토리가 존재하지 않습니다: {chroma_path}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None
        
        logger.info("임베딩 모델 로딩 시작")
        embeddings = get_embedding_model()
        logger.info("임베딩 모델 로딩 완료")
        
        logger.info("ChromaDB 벡터 스토어 생성 시작")
        vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        logger.info("ChromaDB 벡터 스토어 생성 완료")
        
        # 벡터 스토어 정보 로깅
        try:
            collection_count = vector_store._collection.count()
            logger.info(f"ChromaDB 컬렉션 문서 수: {collection_count}")
        except Exception as e:
            logger.warning(f"컬렉션 정보 확인 실패: {e}")
        
        return vector_store
    except Exception as e:
        error_msg = f"ChromaDB 로드 실패: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        return None

def get_rag_chain():
    """RAG 체인 생성"""
    logger.info("RAG 체인 생성 시작")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        logger.info(f"선택된 AI 모델: {selected_model}")
        
        # Ollama LLM 초기화
        logger.info("Ollama LLM 초기화 시작")
        llm = Ollama(model=selected_model)
        logger.info("Ollama LLM 초기화 완료")
        
        # ChromaDB 벡터 스토어 로드
        logger.info("ChromaDB 벡터 스토어 로드 시작")
        vector_store = load_chroma_store()
        if vector_store is None:
            error_msg = "벡터 스토어 로드 실패"
            logger.error(error_msg)
            return None
        logger.info("ChromaDB 벡터 스토어 로드 완료")
        
        # 프롬프트 템플릿
        logger.info("프롬프트 템플릿 생성")
        prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요:

컨텍스트: {context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 생성
        logger.info("RAG 체인 생성 시작")
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("RAG 체인 생성 완료")
        
        return chain
        
    except Exception as e:
        error_msg = f"RAG 체인 생성 실패: {e}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ {error_msg}")
        return None

def process_question(question: str) -> str:
    """질문 처리"""
    logger.info(f"질문 처리 시작: {question[:50]}...")
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        return error_msg
    
    try:
        # RAG 체인 가져오기
        logger.info("RAG 체인 가져오기 시작")
        chain = get_rag_chain()
        if chain is None:
            error_msg = "벡터 데이터베이스를 로드할 수 없습니다."
            logger.error(error_msg)
            return error_msg
        logger.info("RAG 체인 가져오기 완료")
        
        # 질문 처리
        logger.info("질문 처리 실행 시작")
        response = chain.invoke({"query": question})
        result = response.get("result", "답변을 생성할 수 없습니다.")
        logger.info(f"질문 처리 완료 - 답변 길이: {len(result)}")
        return result
        
    except Exception as e:
        error_msg = f"질문 처리 중 오류 발생: {e}"
        logger.error(error_msg, exc_info=True)
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
        
        # 벡터 DB 초기화 버튼
        st.subheader("🗄️ 벡터 데이터베이스")
        
        if st.button("벡터 DB 초기화", type="primary"):
            if initialize_vector_db():
                st.session_state.vector_db_initialized = True
        
        # 벡터 DB 상태 표시
        if st.session_state.get('vector_db_initialized', False):
            st.success("✅ 벡터 DB 초기화됨")
        else:
            st.warning("⚠️ 벡터 DB 초기화 필요")

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