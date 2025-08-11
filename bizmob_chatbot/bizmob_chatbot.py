#!/usr/bin/env python3
"""
bizMOB 챗봇 - ChromaDB 전용 버전
bizmob_chatbot_original.py의 모든 기능을 ChromaDB를 사용하여 구현
"""

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
import sys
import json
import pandas as pd
import re
import logging
import glob
import shutil
import subprocess
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from pptx import Presentation
from docx import Document as DocxDocument
from safetensors.torch import load_file
import sentence_transformers
import html

# 경고 억제
warnings.filterwarnings("ignore")

# 환경 변수 설정 - safetensors 강제 사용
os.environ['TORCH_WARN_ON_LOAD'] = '0'
os.environ['TORCH_LOAD_WARN_ONLY'] = '0'
os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['SAFETENSORS_FAST_GPU'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['TORCH_WEIGHTS_ONLY'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_USE_SAFETENSORS'] = '1'
# 추가 safetensors 강제 설정
os.environ['SAFETENSORS_FAST_GPU'] = '1'
os.environ['TRANSFORMERS_SAFE_SERIALIZATION'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './model_cache'
os.environ['HF_HOME'] = './huggingface'
os.environ['TORCH_HOME'] = './torch_cache'

# 외부 소스 확장자 리스트 상수 선언
EXTERNAL_SOURCE_EXTS = [
    ".py", ".js", ".scss", ".ts", ".vue", ".md", ".txt", ".rst", ".json", ".yaml", ".yml"
]

# ChromaDB 사용 가능 여부 확인
try:
    import chromadb
    from langchain_community.vectorstores import Chroma
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB가 설치되지 않았습니다. pip install chromadb를 실행해주세요.")

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    log_dir = "./logs"
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    try:
        # 로그 디렉토리 생성 시도
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"bizmob_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 파일 핸들러 (UTF-8 인코딩)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except (PermissionError, OSError) as e:
        # 로그 디렉토리 생성 실패 시 콘솔 로깅만 사용
        print(f"로그 디렉토리 생성 실패: {e}. 콘솔 로깅만 사용합니다.")
    
    # 콘솔 핸들러 (항상 추가)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# 로거 초기화
logger = setup_logging()

# UI 컴포넌트 모듈 import
from ui_components import (
    apply_css_styles, setup_page_config, initialize_chat_history,
    add_chat_message, display_chat_messages, check_user_role, is_admin,
    show_role_selector, show_sidebar_info, show_model_selector,
    show_embedding_model_info, show_file_upload_section,
    show_chat_interface, show_admin_interface, show_user_interface
)

# PyTorch 스레드 설정 (성능 최적화)
try:
    import torch
    torch.set_num_threads(1)
except Exception as e:
    logger.warning(f"PyTorch thread setup failed: {e}")

# 기타 필요한 import들
try:
    from langchain_core.documents.base import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import Runnable, RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import OllamaLLM
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore
    from langchain_core.embeddings import Embeddings
except ImportError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    st.stop()

# 환경변수 불러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()

# UI 스타일 및 페이지 설정 적용
apply_css_styles()
setup_page_config()



############################### 1단계 : 파일 업로드 및 관리 함수들 ##########################

def save_uploaded_file(uploaded_file: UploadedFile, folder_path: str = "PDF_bizMOB_Guide") -> str:
    """업로드된 파일을 지정된 폴더에 저장하고 파일 경로를 반환"""
    try:
        # 폴더가 없으면 생성
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 파일 경로 생성
        file_path = os.path.join(folder_path, uploaded_file.name)
        
        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"파일 저장 중 오류: {str(e)}")
        return None

def get_supported_file_types() -> dict:
    """지원하는 파일 형식과 설명을 반환"""
    return {
        'pdf': 'PDF 문서 (.pdf)',
        'xlsx': 'Excel 스프레드시트 (.xlsx)',
        'xls': 'Excel 스프레드시트 (.xls)',
        'pptx': 'PowerPoint 프레젠테이션 (.pptx)',
        'ppt': 'PowerPoint 프레젠테이션 (.ppt)',
        'docx': 'Word 문서 (.docx)',
        'doc': 'Word 문서 (.doc)'
    }

def validate_file_type(filename: str) -> bool:
    """파일 형식이 지원되는지 확인"""
    supported_extensions = ['.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.docx', '.doc']
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in supported_extensions

def upload_and_process_files() -> bool:
    """파일 업로드 및 처리 함수"""
    st.markdown("### 📁 파일 업로드")
    st.markdown("지원 형식: PDF, Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), Word (.docx, .doc)")
    
    # 파일 업로드 위젯
    uploaded_files = st.file_uploader(
        "문서 파일을 선택하세요",
        type=['pdf', 'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc'],
        accept_multiple_files=True,
        help="여러 파일을 동시에 선택할 수 있습니다."
    )
    
    if uploaded_files:
        st.markdown("#### 📋 업로드된 파일 목록")
        
        success_count = 0
        error_count = 0
        
        for uploaded_file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 {uploaded_file.name}")
            
            with col2:
                file_size = len(uploaded_file.getbuffer()) / 1024  # KB
                st.write(f"{file_size:.1f} KB")
            
            with col3:
                if validate_file_type(uploaded_file.name):
                    # 파일 저장
                    saved_path = save_uploaded_file(uploaded_file)
                    if saved_path:
                        st.success("✅")
                        success_count += 1
                    else:
                        st.error("❌")
                        error_count += 1
                else:
                    st.error("❌")
                    error_count += 1
        
        # 결과 요약
        if success_count > 0:
            st.success(f"✅ {success_count}개 파일이 성공적으로 업로드되었습니다.")
            
            # 벡터 데이터베이스 재초기화 옵션
            if st.button("🔄 업로드된 파일로 벡터 데이터베이스 재초기화", type="primary"):
                if initialize_vector_db_with_documents():
                    st.session_state.vector_db_initialized = True
                    st.success("벡터 데이터베이스가 새로운 파일들로 성공적으로 재초기화되었습니다!")
                    st.rerun()
                else:
                    st.error("벡터 데이터베이스 초기화에 실패했습니다.")
        
        if error_count > 0:
            st.error(f"❌ {error_count}개 파일 업로드에 실패했습니다.")
        
        return success_count > 0
    
    return False

def list_uploaded_files(folder_path: str = "PDF_bizMOB_Guide") -> dict:
    """업로드된 파일들을 형식별로 분류하여 반환"""
    if not os.path.exists(folder_path):
        return {}
    
    files_by_type = {
        'PDF': [],
        'Excel': [],
        'PowerPoint': [],
        'Word': []
    }
    
    # 지원하는 파일 확장자들
    supported_extensions = {
        '*.pdf': 'PDF',
        '*.xlsx': 'Excel',
        '*.xls': 'Excel',
        '*.pptx': 'PowerPoint',
        '*.ppt': 'PowerPoint',
        '*.docx': 'Word',
        '*.doc': 'Word'
    }
    
    for pattern, file_type in supported_extensions.items():
        files = glob.glob(os.path.join(folder_path, pattern))
        for file_path in files:
            file_size = os.path.getsize(file_path) / 1024  # KB
            files_by_type[file_type].append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': file_size
            })
    
    return files_by_type

def safe_key(filename: str) -> str:
    """파일명을 안전한 session state 키로 변환"""
    # 특수문자를 언더스코어로 변환
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    # 숫자로 시작하지 않도록 prefix 추가
    if safe_name and safe_name[0].isdigit():
        safe_name = 'file_' + safe_name
    return safe_name

def delete_file(file_path: str) -> bool:
    """파일 삭제 함수"""
    try:
        # 파일이 존재하는지 확인
        if not os.path.exists(file_path):
            st.error(f"파일이 존재하지 않습니다: {file_path}")
            return False
        
        # 파일 삭제
        os.remove(file_path)
        
        # 삭제 확인
        if os.path.exists(file_path):
            st.error(f"파일 삭제에 실패했습니다: {file_path}")
            return False
        
        return True
    except PermissionError:
        st.error(f"파일 삭제 권한이 없습니다: {file_path}")
        return False
    except Exception as e:
        st.error(f"파일 삭제 중 오류: {str(e)}")
        return False

def delete_file_with_confirmation(file_path: str, file_name: str) -> bool:
    """확인 후 파일 삭제 함수"""
    # 삭제 확인을 위한 session state 키
    confirm_key = f"confirm_delete_{file_name}"
    
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    
    if not st.session_state[confirm_key]:
        # 삭제 확인 버튼
        if st.button(f"🗑️ 삭제 확인", key=f"confirm_{file_name}"):
            st.session_state[confirm_key] = True
            return False
        return False
    else:
        # 실제 삭제 실행
        if delete_file(file_path):
            st.success(f"✅ {file_name} 파일이 삭제되었습니다.")
            # session state 정리
            del st.session_state[confirm_key]
            # 벡터 데이터베이스 재초기화 제안
            st.warning("⚠️ 삭제된 파일이 벡터 데이터베이스에 반영되려면 재초기화가 필요합니다.")
            if st.button("🔄 벡터DB 재초기화", key=f"reinit_after_delete_{file_name}"):
                if initialize_vector_db():
                    st.session_state.vector_db_initialized = True
                    st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
            return True
        else:
            st.error(f"❌ {file_name} 파일 삭제에 실패했습니다.")
            # session state 정리
            del st.session_state[confirm_key]
            return False

def manage_uploaded_files() -> None:
    """업로드된 파일 관리 인터페이스"""
    st.markdown("### 📂 업로드된 파일 관리")
    
    # 신규 파일 업로드 섹션 추가
    with st.expander("📁 신규 파일 업로드", expanded=False):
        st.markdown("#### 📤 파일 업로드")
        st.markdown("지원 형식: PDF, Excel (.xlsx, .xls), PowerPoint (.pptx, .ppt), Word (.docx, .doc)")
        
        # 파일 업로드 위젯
        uploaded_files = st.file_uploader(
            "문서 파일을 선택하세요",
            type=['pdf', 'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="여러 파일을 동시에 선택할 수 있습니다.",
            key="file_manager_uploader"
        )
        
        if uploaded_files:
            st.markdown("#### 📋 업로드할 파일 목록")
            
            success_count = 0
            error_count = 0
            
            for uploaded_file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📄 {uploaded_file.name}")
                
                with col2:
                    file_size = len(uploaded_file.getbuffer()) / 1024  # KB
                    st.write(f"{file_size:.1f} KB")
                
                with col3:
                    if validate_file_type(uploaded_file.name):
                        # 파일 저장
                        saved_path = save_uploaded_file(uploaded_file)
                        if saved_path:
                            st.success("✅")
                            success_count += 1
                        else:
                            st.error("❌")
                            error_count += 1
                    else:
                        st.error("❌")
                        error_count += 1
            
            # 결과 요약
            if success_count > 0:
                st.success(f"✅ {success_count}개 파일이 성공적으로 업로드되었습니다.")
                
                # 벡터 데이터베이스 재초기화 옵션
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🔄 벡터DB 재초기화", type="primary", key="file_manager_reinit"):
                        if initialize_vector_db_with_documents():
                            st.session_state.vector_db_initialized = True
                            st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
                        else:
                            st.error("벡터 데이터베이스 초기화에 실패했습니다.")
                
                with col2:
                    if st.button("🔄 페이지 새로고침", key="file_manager_refresh"):
                        pass  # 페이지 새로고침은 브라우저에서 직접 수행
            
            if error_count > 0:
                st.error(f"❌ {error_count}개 파일 업로드에 실패했습니다.")
    
    st.markdown("---")
    
    files_by_type = list_uploaded_files()
    
    if not any(files_by_type.values()):
        st.info("업로드된 파일이 없습니다.")
        st.markdown("---")
        st.markdown("### 📋 지원 파일 형식")
        st.markdown("- **PDF** (.pdf): 매뉴얼, 가이드 문서")
        st.markdown("- **Excel** (.xlsx, .xls): 데이터 시트, 분석 자료")
        st.markdown("- **PowerPoint** (.pptx, .ppt): 프레젠테이션, 교육 자료")
        st.markdown("- **Word** (.docx, .doc): 보고서, 문서, 매뉴얼")
        return
    
    # 파일 통계 정보
    total_files = sum(len(files) for files in files_by_type.values())
    total_size = sum(sum(file_info['size'] for file_info in files) for files in files_by_type.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 총 파일 수", f"{total_files}개")
    with col2:
        st.metric("💾 총 용량", f"{total_size:.1f} KB")
    with col3:
        if check_vector_db_exists():
            st.success("✅ 벡터DB 준비됨")
        else:
            st.warning("⚠️ 벡터DB 미준비")
    
    # 벡터DB 재초기화 버튼
    if st.button("🔄 벡터 데이터베이스 재초기화", type="primary", key="file_manager_main_reinit"):
        if initialize_vector_db_with_documents():
            st.session_state.vector_db_initialized = True
            st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
        else:
            st.error("벡터 데이터베이스 초기화에 실패했습니다.")
    
    st.markdown("---")
    
    # 파일 형식별로 탭 생성 (수정)
    file_types_with_files = [(file_type, files) for file_type, files in files_by_type.items() if files]
    tab_names = [f"{file_type} ({len(files)})" for file_type, files in file_types_with_files]
    if tab_names:
        tabs = st.tabs(tab_names)
        for i, (file_type, files) in enumerate(file_types_with_files):
            with tabs[i]:
                st.markdown(f"#### {file_type} 파일 목록")
                
                for file_info in files:
                    with st.expander(f"📄 {file_info['name']} ({file_info['size']:.1f} KB)"):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**파일명:** {file_info['name']}")
                            st.write(f"**크기:** {file_info['size']:.1f} KB")
                            st.write(f"**경로:** {file_info['path']}")
                        
                        with col2:
                            # 파일 다운로드 버튼
                            safe_download_key = safe_key(file_info['name'])
                            with open(file_info['path'], 'rb') as f:
                                st.download_button(
                                    label="📥 다운로드",
                                    data=f.read(),
                                    file_name=file_info['name'],
                                    key=f"download_{safe_download_key}"
                                )
                        
                        with col3:
                            # 파일 미리보기 버튼
                            safe_preview_key = safe_key(file_info['name'])
                            if st.button("👁️ 미리보기", key=f"preview_{safe_preview_key}"):
                                st.session_state.preview_file = file_info['path']
                                st.session_state.preview_file_type = file_type
                                st.session_state.preview_file_name = file_info['name']
                        
                        with col4:
                            # 파일 삭제 버튼 - 안전한 키 사용
                            safe_file_key = safe_key(file_info['name'])
                            delete_key = f"delete_{safe_file_key}"
                            
                            if delete_key not in st.session_state:
                                st.session_state[delete_key] = False
                            
                            if not st.session_state[delete_key]:
                                if st.button("🗑️ 삭제", key=f"btn_{safe_file_key}"):
                                    st.session_state[delete_key] = True
                            else:
                                st.warning(f"정말로 '{file_info['name']}' 파일을 삭제하시겠습니까?")
                                col_confirm1, col_confirm2 = st.columns(2)
                                with col_confirm1:
                                    if st.button("✅ 확인", key=f"confirm_{safe_file_key}"):
                                        if delete_file(file_info['path']):
                                            st.success(f"✅ {file_info['name']} 파일이 삭제되었습니다.")
                                            # 벡터DB 재초기화 제안
                                            st.warning("⚠️ 삭제된 파일이 벡터 데이터베이스에 반영되려면 재초기화가 필요합니다.")
                                            if st.button("🔄 벡터DB 재초기화", key=f"reinit_{safe_file_key}"):
                                                if initialize_vector_db_with_documents():
                                                    st.session_state.vector_db_initialized = True
                                                    st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
                                            # session state 정리
                                            del st.session_state[delete_key]
                                        else:
                                            st.error(f"❌ {file_info['name']} 파일 삭제에 실패했습니다.")
                                            del st.session_state[delete_key]
                                with col_confirm2:
                                    if st.button("❌ 취소", key=f"cancel_{safe_file_key}"):
                                        del st.session_state[delete_key]
                    
                    # 파일 미리보기 섹션
                    if 'preview_file' in st.session_state and st.session_state.preview_file_type == file_type:
                        st.markdown("---")
                        st.markdown(f"#### 👁️ 파일 미리보기: {st.session_state.preview_file_name}")
                        
                        try:
                            if file_type == 'PDF':
                                # PDF 미리보기
                                images = convert_pdf_to_images(st.session_state.preview_file)
                                if images:
                                    st.image(images[0], caption="첫 페이지 미리보기", width=400)
                                    st.info(f"총 {len(images)}페이지")
                            
                            elif file_type == 'Excel':
                                # Excel 미리보기
                                excel_file = pd.ExcelFile(st.session_state.preview_file)
                                sheet_names = excel_file.sheet_names
                                st.write(f"**시트 목록:** {', '.join(sheet_names)}")
                                
                                if sheet_names:
                                    selected_sheet = st.selectbox("시트 선택", sheet_names)
                                    df = pd.read_excel(st.session_state.preview_file, sheet_name=selected_sheet)
                                    st.dataframe(df.head(10), use_container_width=True)
                                    st.info(f"총 {len(df)}행, {len(df.columns)}열")
                            
                            elif file_type == 'PowerPoint':
                                # PowerPoint 미리보기
                                prs = Presentation(st.session_state.preview_file)
                                st.write(f"**총 슬라이드 수:** {len(prs.slides)}")
                                
                                if prs.slides:
                                    slide_content = ""
                                    for i, slide in enumerate(prs.slides[:3]):  # 처음 3개 슬라이드만
                                        slide_content += f"**슬라이드 {i+1}:**\n"
                                        for shape in slide.shapes:
                                            if hasattr(shape, "text") and shape.text.strip():
                                                slide_content += f"{shape.text}\n"
                                        slide_content += "\n"
                                    
                                    st.text_area("슬라이드 내용 미리보기", slide_content, height=200)
                            
                            elif file_type == 'Word':
                                # Word 미리보기
                                doc = DocxDocument(st.session_state.preview_file)
                                st.write(f"**제목:** {doc.core_properties.title or '제목 없음'}")
                                st.write(f"**작성자:** {doc.core_properties.author or '작성자 없음'}")
                                
                                # 처음 몇 단락만 미리보기
                                preview_text = ""
                                for para in doc.paragraphs[:10]:
                                    if para.text.strip():
                                        preview_text += para.text + "\n\n"
                                
                                st.text_area("문서 내용 미리보기", preview_text, height=200)
                        
                        except Exception as e:
                            st.error(f"파일 미리보기 중 오류: {str(e)}")
                        
                        # 미리보기 닫기 버튼
                        if st.button("❌ 미리보기 닫기"):
                            del st.session_state.preview_file
                            del st.session_state.preview_file_type
                            del st.session_state.preview_file_name 

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: PDF_bizMOB_Guide 폴더에서 모든 문서 파일을 찾아서 Document로 변환
def load_all_documents_from_folder(folder_path: str = "PDF_bizMOB_Guide") -> List[Document]:
    documents = []
    
    if not os.path.exists(folder_path):
        st.error(f"폴더가 존재하지 않습니다: {folder_path}")
        return documents
    
    # 지원하는 파일 확장자들
    supported_extensions = {
        '*.pdf': 'PDF',
        '*.xlsx': 'Excel',
        '*.xls': 'Excel',
        '*.pptx': 'PowerPoint',
        '*.ppt': 'PowerPoint',
        '*.docx': 'Word',
        '*.doc': 'Word'
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
            
            if file_type == 'PDF':
                loader = PyMuPDFLoader(file_path)
                doc = loader.load()
            elif file_type == 'Excel':
                doc = load_excel_file(file_path)
            elif file_type == 'PowerPoint':
                doc = load_powerpoint_file(file_path)
            elif file_type == 'Word':
                doc = load_word_file(file_path)
            
            for d in doc:
                d.metadata['file_path'] = file_path
                d.metadata['file_name'] = os.path.basename(file_path)
                d.metadata['file_type'] = file_type
            
            documents.extend(doc)
            st.success(f"✅ {os.path.basename(file_path)} ({file_type}) 로딩 완료")
        except Exception as e:
            st.error(f"❌ {os.path.basename(file_path)} ({file_type}) 로딩 실패: {str(e)}")
    
    # 외부 소스 코드(.py, .md 등)도 포함
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

def load_excel_file(file_path: str) -> List[Document]:
    """Excel 파일을 로드하여 Document 리스트로 변환"""
    documents = []
    try:
        # Excel 파일 읽기
        excel_file = pd.ExcelFile(file_path)
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 데이터프레임을 텍스트로 변환
            text_content = f"시트명: {sheet_name}\n\n"
            
            # 헤더 정보 추가
            if not df.empty:
                text_content += f"컬럼: {', '.join(df.columns.tolist())}\n\n"
                
                # 데이터 내용 추가 (처음 100행까지만)
                for idx, row in df.head(100).iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if row_text.strip():
                        text_content += f"행 {idx+1}: {row_text}\n"
            
            # Document 생성
            doc = Document(
                page_content=text_content,
                metadata={
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_type': 'Excel',
                    'sheet_name': sheet_name
                }
            )
            documents.append(doc)
            
    except Exception as e:
        st.error(f"Excel 파일 처리 중 오류: {str(e)}")
    
    return documents

def load_powerpoint_file(file_path: str) -> List[Document]:
    """PowerPoint 파일을 로드하여 Document 리스트로 변환"""
    documents = []
    try:
        # PowerPoint 파일 읽기
        prs = Presentation(file_path)
        
        for slide_num, slide in enumerate(prs.slides, 1):
            text_content = f"슬라이드 {slide_num}:\n\n"
            
            # 슬라이드의 모든 텍스트 추출
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content += f"{shape.text}\n\n"
            
            # 텍스트가 있는 경우에만 Document 생성
            if text_content.strip() and text_content != f"슬라이드 {slide_num}:\n\n":
                doc = Document(
                    page_content=text_content,
                    metadata={
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'file_type': 'PowerPoint',
                        'slide_number': slide_num
                    }
                )
                documents.append(doc)
                
    except Exception as e:
        st.error(f"PowerPoint 파일 처리 중 오류: {str(e)}")
    
    return documents

def load_word_file(file_path: str) -> List[Document]:
    """Word 파일을 로드하여 Document 리스트로 변환"""
    documents = []
    try:
        # Word 파일 읽기
        doc = DocxDocument(file_path)
        
        # 전체 문서를 하나의 Document로 처리
        text_content = ""
        
        # 제목 정보 추가
        if doc.core_properties.title:
            text_content += f"제목: {doc.core_properties.title}\n\n"
        
        # 단락별로 텍스트 추출
        for para in doc.paragraphs:
            if para.text.strip():
                text_content += para.text + "\n\n"
        
        # 표 내용 추출
        for table in doc.tables:
            text_content += "표 내용:\n"
            for row in table.rows:
                row_text = " | ".join([cell.text for cell in row.cells if cell.text.strip()])
                if row_text.strip():
                    text_content += row_text + "\n"
            text_content += "\n"
        
        # 텍스트가 있는 경우에만 Document 생성
        if text_content.strip():
            doc_chunk = Document(
                page_content=text_content,
                metadata={
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_type': 'Word',
                    'title': doc.core_properties.title or 'Unknown',
                    'author': doc.core_properties.author or 'Unknown'
                }
            )
            documents.append(doc_chunk)
            
    except Exception as e:
        st.error(f"Word 파일 처리 중 오류: {str(e)}")
    
    return documents

## 2: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 3: Document를 벡터DB로 저장 (ChromaDB 사용)
def save_to_vector_store(documents: List[Document]) -> None:
    try:
        # 선택된 임베딩 모델 사용
        embeddings = get_embedding_model()
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        
        st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        
        # ChromaDB에 저장
        save_to_chroma_store(documents)
        st.success("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
    except Exception as e:
        st.error(f"❌ 벡터 데이터베이스 저장 실패: {str(e)}")

## 4: 벡터DB 초기화 함수 (문서 로딩 포함)
def initialize_vector_db_with_documents():
    """PDF_bizMOB_Guide 폴더의 모든 문서를 로드하여 벡터DB를 초기화"""
    with st.spinner("bizMOB Platform 가이드 문서들을 로딩하고 벡터 데이터베이스를 생성하는 중..."):
        # 모든 문서들 로드 (PDF, Excel, PowerPoint)
        documents = load_all_documents_from_folder()
        
        if not documents:
            st.error("로드할 문서가 없습니다.")
            return False
        
        # 문서 청킹
        st.info("문서를 청크로 분할하는 중...")
        chunked_documents = chunk_documents(documents)
        st.success(f"✅ {len(chunked_documents)}개의 청크로 분할 완료")
        
        # 벡터DB 저장
        st.info("벡터 데이터베이스에 저장하는 중...")
        save_to_chroma_store(chunked_documents)
        
        # 성공적으로 초기화된 모델 정보를 파일에 저장
        try:
            model_info = {
                'ai_model': st.session_state.get('selected_model', 'exaone3.5'),
                'embedding_model': st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            import json
            with open('vector_db_model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            st.success("✅ 모델 정보가 저장되었습니다.")
        except Exception as e:
            st.warning(f"⚠️ 모델 정보 저장 중 오류: {str(e)}")
        
        return True

## 4-1: 벡터DB 초기화 함수 (ChromaDB만 초기화)
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
        
        # ChromaDB 클라이언트 생성 및 컬렉션 초기화
        import chromadb
        import time
        
        # 클라이언트 연결 시도
        max_retries = 3
        client = None
        
        for attempt in range(max_retries):
            try:
                client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chromadb.config.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Client connection attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
                else:
                    raise e
        
        if client is None:
            raise Exception("ChromaDB 클라이언트 연결 실패")
        
        collection_name = "bizmob_documents"
        
        # 기존 컬렉션이 있으면 삭제하고 새로 생성
        try:
            client.delete_collection(name=collection_name)
            logger.info("Existing collection deleted")
            time.sleep(0.5)  # 잠시 대기
        except:
            logger.info("No existing collection to delete")
        
        # 새 컬렉션 생성
        collection = client.create_collection(name=collection_name)
        logger.info("New collection created")
        
        # 모델 정보 저장
        model_info = {
            'ai_model': st.session_state.get('selected_model', 'exaone3.5'),
            'embedding_model': st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask'),
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
    
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return
    
    try:
        # 환경 변수 설정
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_DISABLE_WARNINGS'] = '1'
        
        # 임베딩 모델 로드
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        logger.info(f"Loading embedding model: {selected_embedding}")
        
        try:
            # SafeSentenceTransformerEmbeddings 사용 (torch.load 취약점 방지)
            embeddings = SafeSentenceTransformerEmbeddings(
                model_name=selected_embedding,
                device='cpu'
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Embedding model loading failed: {e}")
            st.error(f"임베딩 모델 로딩 실패: {e}")
            st.info("다른 임베딩 모델을 시도해보세요.")
            return
        
        # ChromaDB 클라이언트 생성 (지속적 저장을 위해 경로 지정)
        try:
            import chromadb
            import time
            chroma_path = get_chroma_db_path()
            os.makedirs(chroma_path, exist_ok=True)
            
            # session_state에서 기존 클라이언트 확인
            if 'chroma_client' in st.session_state:
                try:
                    # 기존 클라이언트 해제 시도
                    del st.session_state.chroma_client
                    import gc
                    gc.collect()
                    time.sleep(0.5)
                except:
                    pass
            
            # 지속적 저장을 위한 ChromaDB 클라이언트 생성 (재시도 로직)
            max_retries = 3
            client = None
            
            for attempt in range(max_retries):
                try:
                    client = chromadb.PersistentClient(
                        path=chroma_path,
                        settings=chromadb.config.Settings(
                            allow_reset=True,
                            anonymized_telemetry=False
                        )
                    )
                    # session_state에 클라이언트 저장
                    st.session_state.chroma_client = client
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Client connection attempt {attempt + 1} failed: {e}")
                        time.sleep(1)
                    else:
                        raise e
            
            if client is None:
                raise Exception("ChromaDB 클라이언트 연결 실패")
            
            collection_name = "bizmob_documents"
            
            # 기존 컬렉션이 있으면 삭제하고 새로 생성
            try:
                client.delete_collection(name=collection_name)
                logger.info("Existing collection deleted")
                time.sleep(0.5)  # 잠시 대기
            except:
                logger.info("No existing collection to delete")
            
            # 새 컬렉션 생성
            collection = client.create_collection(name=collection_name)
            logger.info("New collection created")
            
            # 문서 텍스트와 메타데이터 추출 (텍스트 정제)
            documents_texts = []
            documents_metadatas = []
            documents_ids = []
            
            for i, doc in enumerate(documents):
                # 텍스트 정제 (특수문자 및 인코딩 문제 해결)
                clean_text = doc.page_content.strip()
                if clean_text:
                    documents_texts.append(clean_text)
                    documents_metadatas.append(doc.metadata)
                    documents_ids.append(f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            if not documents_texts:
                st.warning("저장할 문서가 없습니다.")
                return
            
            # 임베딩 생성
            logger.info("Generating embeddings...")
            embeddings_list = embeddings.embed_documents(documents_texts)
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            
            # ChromaDB에 저장
            collection.add(
                documents=documents_texts,
                embeddings=embeddings_list,
                metadatas=documents_metadatas,
                ids=documents_ids
            )
            
            logger.info("ChromaDB document save completed")
            st.success(f"✅ {len(documents_texts)}개 문서가 벡터 데이터베이스에 저장되었습니다.")
            
            # 저장된 문서 수 확인
            try:
                count = collection.count()
                logger.info(f"Total documents in collection: {count}")
                st.info(f"📊 현재 벡터 DB에 저장된 문서 수: {count}개")
                
                # 저장된 문서 샘플 확인
                if count > 0:
                    sample_results = collection.get(limit=1)
                    if sample_results['documents']:
                        sample_text = sample_results['documents'][0][:100]
                        logger.info(f"Sample document: {sample_text}...")
                        st.info(f"📝 저장된 문서 샘플: {sample_text}...")
                
            except Exception as e:
                logger.warning(f"Could not get collection count: {e}")
            
        except Exception as e:
            logger.error(f"ChromaDB save failed: {e}")
            st.error(f"벡터 데이터베이스 저장 실패: {e}")
            return
        
    except Exception as e:
        error_msg = f"벡터 데이터베이스 저장 실패: {e}"
        logger.error(f"Vector database save failed: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")

def load_chroma_store():
    """ChromaDB에서 벡터 스토어 로드 (강화된 단일 인스턴스 관리)"""
    if not CHROMADB_AVAILABLE:
        error_msg = "ChromaDB가 설치되지 않았습니다."
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    try:
        # 캐시된 임베딩 모델 사용
        embeddings = get_embedding_model()
        if embeddings is None:
            logger.error("임베딩 모델 로드 실패")
            return None
        
        # ChromaDB 클라이언트 생성 (강화된 단일 인스턴스 관리)
        import chromadb
        from langchain_community.vectorstores import Chroma
        import time
        import gc
        
        chroma_path = get_chroma_db_path()
        
        # 디렉토리가 없으면 생성
        if not os.path.exists(chroma_path):
            os.makedirs(chroma_path, exist_ok=True)
        
        # 현재 선택된 모델과 임베딩으로 고유 키 생성
        selected_model = st.session_state.get('selected_model', 'exaone3.5')
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        global_vector_store_key = f"global_vector_store_{selected_model}_{selected_embedding}"
        
        # 기존 벡터 스토어가 있으면 재사용
        if global_vector_store_key in st.session_state:
            try:
                vector_store = st.session_state[global_vector_store_key]
                # 간단한 테스트로 연결 상태 확인
                test_collection = vector_store._collection
                test_collection.count()
                logger.info("기존 벡터 스토어 재사용 성공")
                return vector_store
            except Exception as e:
                logger.warning(f"기존 벡터 스토어 재사용 실패: {e}")
                # 기존 벡터 스토어 제거
                del st.session_state[global_vector_store_key]
                # 메모리 정리
                gc.collect()
                time.sleep(1)
        
        # ChromaDB 클라이언트 생성 (타임아웃 설정)
        try:
            client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.config.Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=chroma_path
                )
            )
            
            # 클라이언트 연결 테스트
            client.heartbeat()
            logger.info("ChromaDB 클라이언트 연결 성공")
            
        except Exception as e:
            logger.warning(f"ChromaDB 클라이언트 연결 실패, 재시도: {e}")
            # 기존 프로세스 정리 후 재시도
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if 'chroma' in proc.info['name'].lower() or any('chroma' in str(cmd).lower() for cmd in proc.info['cmdline'] or []):
                            proc.terminate()
                            logger.info(f"ChromaDB 프로세스 종료: {proc.info['pid']}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(2)
                
                client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chromadb.config.Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True,
                        persist_directory=chroma_path
                    )
                )
                client.heartbeat()
                logger.info("ChromaDB 클라이언트 재연결 성공")
                
            except Exception as e2:
                logger.error(f"ChromaDB 클라이언트 재연결 실패: {e2}")
                return None
        
        collection_name = "bizmob_documents"
        
        # 컬렉션 가져오기 (타임아웃 설정)
        try:
            collection = client.get_collection(name=collection_name)
            logger.info("기존 컬렉션 로드 성공")
        except Exception as e:
            logger.info("새 컬렉션 생성")
            collection = client.create_collection(name=collection_name)
        
        # LangChain Chroma 벡터 스토어 생성
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # 벡터 스토어를 전역에 저장
        st.session_state[global_vector_store_key] = vector_store
        
        logger.info("새 벡터 스토어 생성 완료")
        return vector_store
        
    except Exception as e:
        error_msg = f"ChromaDB 로드 실패: {e}"
        logger.error(f"ChromaDB 로드 실패: {e}", exc_info=True)
        st.error(f"❌ {error_msg}")
        return None

############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
def process_question(user_question, rag_chain=None):
    """사용자 질문을 처리하고 답변을 반환"""
    logger.info(f"=== 질문 처리 시작: {user_question[:50]}... ===")
    try:
        # RAG 체인이 전달되지 않았으면 가져오기
        if rag_chain is None:
            rag_chain = get_cached_rag_chain()
            if rag_chain is None:
                logger.error("RAG 체인 생성 실패")
            st.error("RAG 체인 생성에 실패했습니다.")
            return None, []
        
        # 질문 처리
        response = rag_chain.invoke(user_question)
        logger.info(f"질문 처리 완료, 응답 길이: {len(response) if response else 0}")
        
        # 관련 문서 검색 (캐시된 벡터 스토어 사용)
        retrieve_docs = []
        try:
            vector_store = get_cached_vector_store()
            if vector_store:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                retrieve_docs = retriever.invoke(user_question)
                logger.info(f"관련 문서 검색 완료, 문서 수: {len(retrieve_docs)}")
        except Exception as e:
            logger.warning(f"관련 문서 검색 실패: {e}")

        logger.info("=== 질문 처리 완료 ===")
        return response, retrieve_docs
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {str(e)}", exc_info=True)
        st.error(f"질문 처리 중 오류 발생: {str(e)}")
        return None, []

@st.cache_resource
def get_cached_vector_store():
    """캐시된 벡터 스토어 반환 (모델별 캐시)"""
    try:
        # 현재 선택된 모델과 임베딩 모델로 캐시 키 생성
        selected_model = st.session_state.get('selected_model', 'exaone3.5')
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        
        cache_key = f"vector_store_{selected_model}_{selected_embedding}"
        
        # 기존 캐시된 벡터 스토어가 있으면 재사용
        if cache_key in st.session_state:
            try:
                vector_store = st.session_state[cache_key]
                # 간단한 테스트로 연결 상태 확인
                test_collection = vector_store._collection
                test_collection.count()
                logger.info(f"기존 벡터 스토어 캐시 재사용: {cache_key}")
                return vector_store
            except Exception as e:
                logger.warning(f"기존 벡터 스토어 캐시 재사용 실패: {e}")
                # 기존 캐시 제거
                del st.session_state[cache_key]
        
        # 새 벡터 스토어 생성
        vector_store = load_chroma_store()
        if vector_store:
            st.session_state[cache_key] = vector_store
            logger.info(f"새 벡터 스토어 캐시 생성: {cache_key}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"벡터 스토어 캐시 생성 실패: {str(e)}", exc_info=True)
        return None

@st.cache_resource
def get_cached_rag_chain():
    """캐시된 RAG 체인 반환 (모델별 캐시)"""
    try:
        # 현재 선택된 모델로 캐시 키 생성
        selected_model = st.session_state.get('selected_model', 'exaone3.5')
        cache_key = f"rag_chain_{selected_model}"
        
        # 기존 캐시된 RAG 체인이 있으면 재사용
        if cache_key in st.session_state:
            logger.info(f"기존 RAG 체인 캐시 재사용: {cache_key}")
            return st.session_state[cache_key]
        
        # 새 RAG 체인 생성
        rag_chain = get_rag_chain()
        if rag_chain:
            st.session_state[cache_key] = rag_chain
            logger.info(f"새 RAG 체인 캐시 생성: {cache_key}")
        
        return rag_chain
        
    except Exception as e:
        logger.error(f"RAG 체인 캐시 생성 실패: {str(e)}", exc_info=True)
        return None

def get_rag_chain() -> Runnable:
    """RAG 체인 생성 (실제 구현)"""
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'exaone3.5')
        
        # Ollama LLM 초기화
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(
            model=selected_model,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048
        )
        
        # 캐시된 임베딩 모델 사용
        embeddings = get_embedding_model()
        if embeddings is None:
            logger.error("임베딩 모델 로드 실패")
            return None
        
        # 캐시된 벡터 스토어 사용
        vector_store = get_cached_vector_store()
        if vector_store is None:
            logger.error("벡터 스토어 로드 실패")
            return None
        
        # 프롬프트 템플릿
        from langchain_core.prompts import PromptTemplate
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
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        chain = (
            {"context": vector_store.as_retriever(search_kwargs={"k": 3}), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"RAG 체인 생성 중 오류: {str(e)}", exc_info=True)
        st.error(f"RAG 체인 생성 중 오류: {str(e)}")
        return None

############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################

@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지_bizmob"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  # 각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    try:
        image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
        st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)
    except Exception as e:
        st.error(f"이미지 표시 중 오류: {str(e)}")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def check_vector_db_exists():
    """벡터DB가 존재하는지 확인"""
    return os.path.exists(get_chroma_db_path())

def load_saved_model_info():
    """저장된 모델 정보를 불러옴"""
    try:
        if os.path.exists('vector_db_model_info.json'):
            import json
            with open('vector_db_model_info.json', 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            return model_info
        return None
    except Exception as e:
        st.warning(f"⚠️ 저장된 모델 정보 불러오기 실패: {str(e)}")
        return None

def check_ollama_models():
    """사용 가능한 Ollama 모델 확인"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        return False
    except:
        return False

def get_ollama_models():
    """Ollama에서 사용 가능한 모델 목록을 가져옴"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # 첫 번째 줄은 헤더이므로 제외
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        model_name = parts[0]
                        model_size = parts[1] if len(parts) > 1 else "Unknown"
                        models.append({
                            'name': model_name,
                            'size': model_size
                        })
            return models
        return []
    except Exception as e:
        st.error(f"Ollama 모델 목록 가져오기 실패: {str(e)}")
        return []

def get_available_embedding_models():
    """사용 가능한 임베딩 모델 목록을 반환"""
    return {
        'sentence-transformers/all-MiniLM-L6-v2': {
            'name': 'all-MiniLM-L6-v2',
            'description': '영어 전용, 빠르고 가벼운 모델',
            'language': 'English',
            'size': 'Small'
        },
        'jhgan/ko-sroberta-multitask': {
            'name': 'ko-sroberta-multitask',
            'description': '한국어 전용, 다중 작업 모델',
            'language': 'Korean',
            'size': 'Medium'
        },
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'description': '다국어 지원, 균형잡힌 성능',
            'language': 'Multilingual',
            'size': 'Medium'
        },
        'sentence-transformers/all-mpnet-base-v2': {
            'name': 'all-mpnet-base-v2',
            'description': '영어 전용, 고품질 임베딩',
            'language': 'English',
            'size': 'Large'
        },
        'intfloat/multilingual-e5-large': {
            'name': 'multilingual-e5-large',
            'description': '다국어 지원, 고품질 임베딩',
            'language': 'Multilingual',
            'size': 'Large'
        }
    }

class SafeSentenceTransformerEmbeddings(Embeddings):
    """safetensors를 직접 사용하는 안전한 임베딩 클래스"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델을 안전하게 로드 (safetensors 직접 사용)"""
        try:
            # 환경 변수 설정으로 safetensors 강제 사용
            import os
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
            st.error(f"모델 로딩 실패: {str(e)}")
            raise e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩"""
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
            st.error(f"임베딩 생성 실패: {str(e)}")
            raise e
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩"""
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
            st.error(f"쿼리 임베딩 실패: {str(e)}")
            raise e

@st.cache_resource
def get_embedding_model():
    """선택된 임베딩 모델을 반환 (safetensors 지원)"""
    selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
    logger.info(f"=== 임베딩 모델 로드 시작: {selected_embedding} ===")
    
    try:
        logger.info("1. SafeSentenceTransformerEmbeddings 시도")
        # 커스텀 임베딩 클래스 사용
        embeddings = SafeSentenceTransformerEmbeddings(
            model_name=selected_embedding,
            device='cpu'
        )
        logger.info("SafeSentenceTransformerEmbeddings 로드 성공")
        return embeddings
        
    except Exception as e:
        logger.warning(f"SafeSentenceTransformerEmbeddings 실패: {str(e)}")
        st.error(f"임베딩 모델 로딩 실패: {str(e)}")
        st.info("HuggingFaceEmbeddings로 재시도합니다...")
        
        try:
            logger.info("2. HuggingFaceEmbeddings 재시도")
            # HuggingFaceEmbeddings로 fallback (safetensors 사용)
            embeddings = HuggingFaceEmbeddings(
                model_name=selected_embedding,
                model_kwargs={
                    'device': 'cpu',
                    'torch_dtype': 'auto',
                    'low_cpu_mem_usage': True,
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info("HuggingFaceEmbeddings 로드 성공")
            st.success(f"✅ {selected_embedding} 모델을 HuggingFaceEmbeddings로 로드했습니다.")
            return embeddings
            
        except Exception as e2:
            logger.error(f"HuggingFaceEmbeddings 재시도도 실패: {str(e2)}")
            st.error(f"HuggingFaceEmbeddings 재시도도 실패: {str(e2)}")
            st.error("임베딩 모델을 로드할 수 없습니다. 다른 모델을 선택해주세요.")
            return None

def get_recommended_embedding_model(ai_model_name: str) -> str:
    """AI 모델에 따른 권장 임베딩 모델을 반환"""
    model_mapping = {
        'exaone3.5': 'jhgan/ko-sroberta-multitask',
        'llama3': 'sentence-transformers/all-mpnet-base-v2',
        'llama3.2': 'sentence-transformers/all-mpnet-base-v2',
        'llama3.2:3b': 'sentence-transformers/all-MiniLM-L6-v2',
        'llama3.2:8b': 'sentence-transformers/all-mpnet-base-v2',
        'llama3.2:70b': 'intfloat/multilingual-e5-large',
        'mistral': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'mistral:7b': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'mistral:instruct': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'codellama': 'sentence-transformers/all-mpnet-base-v2',
        'codellama:7b': 'sentence-transformers/all-MiniLM-L6-v2',
        'codellama:13b': 'sentence-transformers/all-mpnet-base-v2',
        'codellama:34b': 'intfloat/multilingual-e5-large',
        'gemma': 'sentence-transformers/all-mpnet-base-v2',
        'gemma:2b': 'sentence-transformers/all-MiniLM-L6-v2',
        'gemma:7b': 'sentence-transformers/all-mpnet-base-v2',
        'qwen': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'qwen:7b': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'qwen:14b': 'intfloat/multilingual-e5-large',
        'phi': 'sentence-transformers/all-mpnet-base-v2',
        'phi:2.7b': 'sentence-transformers/all-MiniLM-L6-v2',
        'phi:3.5': 'sentence-transformers/all-mpnet-base-v2',
    }
    # 정확 매칭
    if ai_model_name in model_mapping:
        return model_mapping[ai_model_name]
    # 부분 매칭
    for key, value in model_mapping.items():
        if key in ai_model_name.lower():
            return value
    return 'jhgan/ko-sroberta-multitask'

def get_chroma_db_path():
    """ChromaDB 경로 반환"""
    return "./chroma_db"

def get_model_info_path():
    """모델 정보 파일 경로 반환"""
    ai_model = st.session_state.get('selected_model', 'exaone3.5')
    import re
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
    return f"vector_db_model_info_{safe_model}.json" 

def main():
    """메인 애플리케이션 (관리자 전용)"""
    st.set_page_config(
        page_title="bizMOB 관리자",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 사용자 역할 초기화
    check_user_role()

    # 사이드바 정보 표시
    show_sidebar_info()
    
    # Ollama 상태 확인 및 모델 선택
    if check_ollama_models():
        st.sidebar.success("✅ Ollama 연결됨")
        
        # 사용 가능한 모델 목록 가져오기
        available_models = get_ollama_models()
        
        # 모델 선택기 표시
        show_model_selector(available_models, get_recommended_embedding_model, load_saved_model_info)
        
    else:
        st.sidebar.error("❌ Ollama 연결 실패")
        st.sidebar.info("Ollama가 설치되어 있고 실행 중인지 확인해주세요.")
    
    # 임베딩 모델 정보 표시
    show_embedding_model_info(get_available_embedding_models, load_saved_model_info, get_recommended_embedding_model)
    
    # 관리자만 파일 업로드 섹션 표시
    show_file_upload_section(save_uploaded_file, validate_file_type, initialize_vector_db_with_documents)
    
    # 채팅 페이지 링크
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 채팅 페이지")
    st.sidebar.markdown("[채팅 페이지로 이동](http://localhost:8501/Chat)")
    
    # 관리자 인터페이스만 표시
    show_admin_interface(display_chat_messages, check_vector_db_exists, initialize_vector_db_with_documents,
                       add_chat_message, process_question, manage_uploaded_files, load_saved_model_info)

if __name__ == "__main__":
    main() 