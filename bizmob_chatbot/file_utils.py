import os
import re
import glob
import pandas as pd
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
from pptx import Presentation
from docx import Document as DocxDocument
from langchain_core.documents.base import Document

def save_uploaded_file(uploaded_file: UploadedFile, folder_path: str = "PDF_bizMOB_Guide") -> str:
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"파일 저장 중 오류: {str(e)}")
        return None

def get_supported_file_types() -> dict:
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
    supported_extensions = ['.pdf', '.xlsx', '.xls', '.pptx', '.ppt', '.docx', '.doc']
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in supported_extensions

def list_uploaded_files(folder_path: str = "PDF_bizMOB_Guide") -> dict:
    if not os.path.exists(folder_path):
        return {}
    files_by_type = {
        'PDF': [],
        'Excel': [],
        'PowerPoint': [],
        'Word': []
    }
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
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename)
    if safe_name and safe_name[0].isdigit():
        safe_name = 'file_' + safe_name
    return safe_name

def delete_file(file_path: str) -> bool:
    try:
        if not os.path.exists(file_path):
            st.error(f"파일이 존재하지 않습니다: {file_path}")
            return False
        os.remove(file_path)
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
    confirm_key = f"confirm_delete_{file_name}"
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    if not st.session_state[confirm_key]:
        if st.button(f"🗑️ 삭제 확인", key=f"confirm_{file_name}"):
            st.session_state[confirm_key] = True
            return False
        return False
    else:
        if delete_file(file_path):
            st.success(f"✅ {file_name} 파일이 삭제되었습니다.")
            del st.session_state[confirm_key]
            st.warning("⚠️ 삭제된 파일이 벡터 데이터베이스에 반영되려면 재초기화가 필요합니다.")
            if st.button("🔄 벡터DB 재초기화", key=f"reinit_after_delete_{file_name}"):
                from .vector_db_utils import initialize_vector_db
                if initialize_vector_db():
                    st.session_state.vector_db_initialized = True
                    st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
            return True
        else:
            st.error(f"❌ {file_name} 파일 삭제에 실패했습니다.")
            del st.session_state[confirm_key]
            return False

def load_excel_file(file_path: str) -> List[Document]:
    documents = []
    try:
        excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text_content = f"시트명: {sheet_name}\n\n"
            if not df.empty:
                text_content += f"컬럼: {', '.join(df.columns.tolist())}\n\n"
                for idx, row in df.head(100).iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if row_text.strip():
                        text_content += f"행 {idx+1}: {row_text}\n"
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
    documents = []
    try:
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides, 1):
            text_content = f"슬라이드 {slide_num}:\n\n"
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_content += f"{shape.text}\n\n"
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
    documents = []
    try:
        doc = DocxDocument(file_path)
        text_content = ""
        if doc.core_properties.title:
            text_content += f"제목: {doc.core_properties.title}\n\n"
        for para in doc.paragraphs:
            if para.text.strip():
                text_content += para.text + "\n\n"
        for table in doc.tables:
            text_content += "표 내용:\n"
            for row in table.rows:
                row_text = " | ".join([cell.text for cell in row.cells if cell.text.strip()])
                if row_text.strip():
                    text_content += row_text + "\n"
            text_content += "\n"
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