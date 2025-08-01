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
from datetime import datetime
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
    CHROMADB_AVAILABLE = True
except ImportError:
    st.error("ChromaDB가 설치되지 않았습니다. pip install chromadb를 실행해주세요.")
    CHROMADB_AVAILABLE = False

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
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB가 설치되지 않았습니다.")
        return False
    
    try:
        # ChromaDB 디렉토리 생성
        chroma_path = get_chroma_db_path()
        os.makedirs(chroma_path, exist_ok=True)
        
        # 모델 정보 저장
        model_info = {
            'ai_model': st.session_state.get('selected_model', 'llama3.2'),
            'embedding_model': st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(get_model_info_path(), 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        st.session_state.vector_db_initialized = True
        st.success("✅ ChromaDB 벡터 데이터베이스 초기화 완료")
        return True
        
    except Exception as e:
        st.error(f"❌ 벡터 데이터베이스 초기화 실패: {e}")
        return False

def save_to_chroma_store(documents: list) -> None:
    """문서를 ChromaDB에 저장"""
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB가 설치되지 않았습니다.")
        return
    
    try:
        selected_embedding = st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        embeddings = HuggingFaceEmbeddings(model_name=selected_embedding)
        
        st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        
        # ChromaDB에 저장
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=get_chroma_db_path()
        )
        vector_store.persist()
        
        st.success("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
        
    except Exception as e:
        st.error(f"❌ 벡터 데이터베이스 저장 실패: {e}")

def load_chroma_store():
    """ChromaDB에서 벡터 스토어 로드"""
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB가 설치되지 않았습니다.")
        return None
    
    try:
        embeddings = get_embedding_model()
        vector_store = Chroma(
            persist_directory=get_chroma_db_path(),
            embedding_function=embeddings
        )
        return vector_store
    except Exception as e:
        st.error(f"❌ ChromaDB 로드 실패: {e}")
        return None

def get_rag_chain():
    """RAG 체인 생성"""
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB가 설치되지 않았습니다.")
        return None
    
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        
        # Ollama LLM 초기화
        llm = Ollama(model=selected_model)
        
        # ChromaDB 벡터 스토어 로드
        vector_store = load_chroma_store()
        if vector_store is None:
            return None
        
        # 프롬프트 템플릿
        prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요:

컨텍스트: {context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RAG 체인 생성
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain
        
    except Exception as e:
        st.error(f"❌ RAG 체인 생성 실패: {e}")
        return None

def process_question(question: str) -> str:
    """질문 처리"""
    if not CHROMADB_AVAILABLE:
        return "ChromaDB가 설치되지 않았습니다."
    
    try:
        # RAG 체인 가져오기
        chain = get_rag_chain()
        if chain is None:
            return "벡터 데이터베이스를 로드할 수 없습니다."
        
        # 질문 처리
        response = chain.invoke({"query": question})
        return response.get("result", "답변을 생성할 수 없습니다.")
        
    except Exception as e:
        return f"질문 처리 중 오류 발생: {e}"

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
            st.rerun()
        
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
            st.rerun()
        
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
    tab1, tab2, tab3 = st.tabs(["💬 챗봇", "📁 파일 업로드", "ℹ️ 정보"])
    
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
        st.subheader("ℹ️ 시스템 정보")
        
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

if __name__ == "__main__":
    main() 