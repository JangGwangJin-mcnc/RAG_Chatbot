# 벡터DB 관련 유틸리티 함수 모듈
import os
import glob
import pandas as pd
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

EXTERNAL_SOURCE_EXTS = [
    ".py", ".js", ".scss", ".ts", ".vue", ".md", ".txt", ".rst", ".json", ".yaml", ".yml"
]

def get_vector_db_path():
    ai_model = st.session_state.get('selected_model', 'hyperclovax')
    import re
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
    return f"bizmob_faiss_index_{safe_model}"

def check_vector_db_exists():
    return os.path.exists(get_vector_db_path())

def chunk_documents(documents: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def load_all_documents_from_folder(folder_path: str = "PDF_bizMOB_Guide") -> list:
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"폴더가 존재하지 않습니다: {folder_path}")
        return documents
    supported_extensions = {
        '*.pdf': 'PDF', '*.xlsx': 'Excel', '*.xls': 'Excel', '*.pptx': 'PowerPoint', '*.ppt': 'PowerPoint', '*.docx': 'Word', '*.doc': 'Word'
    }
    all_files = []
    for pattern, file_type in supported_extensions.items():
        files = glob.glob(os.path.join(folder_path, pattern))
        all_files.extend([(f, file_type) for f in files])
    if not all_files:
        st.warning(f"{folder_path} 폴더에 지원하는 문서 파일이 없습니다.")
        st.info("지원 형식: PDF, Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), Word (.docx, .doc)")
        return documents
    for file_path, file_type in all_files:
        try:
            st.info(f"{file_type} 파일 로딩 중: {os.path.basename(file_path)}")
            # 실제 문서 로딩 함수는 원본에서 import 필요
            pass
        except Exception as e:
            st.error(f"❌ {os.path.basename(file_path)} ({file_type}) 로딩 실패: {str(e)}")
    ext_src_dir = "external_sources"
    if os.path.exists(ext_src_dir):
        for repo in os.listdir(ext_src_dir):
            repo_path = os.path.join(ext_src_dir, repo)
            for ext in EXTERNAL_SOURCE_EXTS:
                for file in glob.glob(f"{repo_path}/**/*{ext}", recursive=True):
                    try:
                        if os.path.getsize(file) > 2*1024*1024:
                            st.warning(f"{file} 파일은 2MB를 초과하여 벡터DB에 포함되지 않습니다.")
                            continue
                        with open(file, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={
                                'file_path': file,
                                'file_name': os.path.basename(file),
                                'file_type': 'Source',
                                'repo': repo
                            }
                        )
                        documents.append(doc)
                    except Exception as e:
                        st.warning(f"{file} 파일을 읽는 중 오류 발생: {e}")
    return documents

def save_to_vector_store(documents: list) -> None:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'))
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local(get_vector_db_path())
        st.success("✅ 벡터 데이터베이스 저장 완료 (선택된 임베딩 모델 사용)")
    except Exception as e:
        st.error(f"❌ 벡터 데이터베이스 저장 실패: {str(e)}")

def get_model_info_path():
    ai_model = st.session_state.get('selected_model', 'hyperclovax')
    import re
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
    return f"vector_db_model_info_{safe_model}.json"

def initialize_vector_db():
    with st.spinner("bizMOB Platform 가이드 문서들을 로딩하고 벡터 데이터베이스를 생성하는 중..."):
        documents = load_all_documents_from_folder()
        if not documents:
            st.error("로드할 문서가 없습니다.")
            return False
        chunked_documents = chunk_documents(documents)
        st.success(f"✅ {len(chunked_documents)}개의 청크로 분할 완료")
        st.info("벡터 데이터베이스에 저장하는 중...")
        save_to_vector_store(chunked_documents)
        try:
            model_info = {
                'ai_model': st.session_state.get('selected_model', 'hyperclovax'),
                'embedding_model': st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            import json
            with open(get_model_info_path(), 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            st.success("✅ 모델 정보가 저장되었습니다.")
        except Exception as e:
            st.warning(f"⚠️ 모델 정보 저장 중 오류: {str(e)}")
        return True

def load_saved_model_info():
    try:
        model_info_path = get_model_info_path()
        if os.path.exists(model_info_path):
            import json
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return model_info
        return None
    except Exception as e:
        st.warning(f"⚠️ 저장된 모델 정보 불러오기 실패: {str(e)}")
        return None 