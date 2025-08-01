## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.retrievers import BaseRetriever
from typing import List
import os
import fitz  # PyMuPDF
import re
import glob
import pandas as pd
from pptx import Presentation
from docx import Document as DocxDocument
import shutil
import subprocess
import sys, os

## 환경변수 불러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 외부 소스 확장자 리스트 상수 선언
EXTERNAL_SOURCE_EXTS = [
    ".py", ".js", ".scss", ".ts", ".vue", ".md", ".txt", ".rst", ".json", ".yaml", ".yml"
]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 하이브리드 검색을 위한 커스텀 리트리버 클래스
class HybridRetriever(BaseRetriever):
    """시멘틱 검색 70% + 키워드 검색 30%를 결합한 하이브리드 리트리버"""
    
    def __init__(self, vector_store, semantic_weight=0.7, keyword_weight=0.3, k=3):
        # BaseRetriever 초기화 시 필드 충돌을 방지하기 위해 kwargs 사용
        super().__init__()
        # private 변수로 저장하여 BaseRetriever와의 충돌 방지
        self._vector_store = vector_store
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._k = k
    
    def _semantic_search(self, query: str, k: int = 3):
        """시멘틱 검색 (벡터 유사도)"""
        try:
            # 벡터 스토어의 기본 검색 사용
            semantic_results = self._vector_store.similarity_search_with_score(query, k=k)
            
            # 결과 형식 통일
            results = []
            for doc, score in semantic_results:
                # FAISS 점수는 거리이므로 유사도로 변환 (1 / (1 + distance))
                similarity_score = 1 / (1 + score)
                results.append({
                    'document': doc,
                    'score': similarity_score
                })
            
            return results
        except Exception as e:
            st.warning(f"시멘틱 검색 실패: {str(e)}")
            return []
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """관련 문서 `검색 (하이브리드 방식)"""
        try:
            # 시멘틱 검색 수행 (더 많은 결과 가져오기)
            semantic_results = self._semantic_search(query, k=self._k * 2)
            
            # 키워드 매칭 점수 계산
            keyword_boosted_results = []
            for result in semantic_results:
                doc = result['document']
                semantic_score = result['score']
                
                # 키워드 매칭 점수 계산
                keyword_score = self._calculate_keyword_score(query, doc.page_content)
                
                # 가중 평균 계산 (시멘틱 70% + 키워드 30%)
                combined_score = (
                    self._semantic_weight * semantic_score +
                    self._keyword_weight * keyword_score
                )
                
                keyword_boosted_results.append({
                    'document': doc,
                    'combined_score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score
                })
            
            # 결합된 점수로 정렬
            sorted_results = sorted(
                keyword_boosted_results,
                key=lambda x: x['combined_score'],
                reverse=True
            )
            
            # 상위 k개 문서 반환
            top_documents = []
            for result in sorted_results[:self._k]:
                top_documents.append(result['document'])
            
            return top_documents
        except Exception as e:
            st.warning(f"하이브리드 검색 실패: {str(e)}")
            # 실패 시 기본 벡터 검색으로 폴백
            try:
                return self._vector_store.similarity_search(query, k=self._k)
            except:
                return []
    
    def _calculate_keyword_score(self, query: str, document_content: str) -> float:
        """키워드 매칭 점수 계산"""
        try:
            # 쿼리에서 키워드 추출 (한글, 영문, 숫자)
            query_keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
            
            if not query_keywords:
                return 0.0
            
            # 문서 내용에서 키워드 매칭 확인
            doc_content_lower = document_content.lower()
            matched_keywords = 0
            
            for keyword in query_keywords:
                if len(keyword) > 1 and keyword in doc_content_lower:  # 1글자 키워드는 제외
                    matched_keywords += 1
            
            # 키워드 매칭 비율 계산
            keyword_score = matched_keywords / len(query_keywords)
            
            return keyword_score
        except Exception as e:
            return 0.0

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
                if initialize_vector_db():
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
                        if initialize_vector_db():
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
    if st.button("🔄 벡터 데이터베이스 재초기화", type="primary", key="file_manager_main_reinit_1"):
        if initialize_vector_db():
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
                                                if initialize_vector_db():
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

## 3: Document를 벡터DB로 저장 (로컬 임베딩 모델 사용)
def save_to_vector_store(documents: List[Document]) -> None:
    try:
        # 선택된 임베딩 모델 사용
        embeddings = get_embedding_model()
        selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
        
        st.info(f"임베딩 모델 로딩 중: {selected_embedding}")
        
        # PyTorch 보안 취약점 해결을 위해 ChromaDB 사용
        try:
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            vector_store.persist()
            st.success("✅ 벡터 데이터베이스 저장 완료 (ChromaDB 사용)")
        except ImportError:
            # ChromaDB가 없는 경우 FAISS 사용 (경고 억제)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vector_store = FAISS.from_documents(documents, embedding=embeddings)
                vector_store.save_local(get_vector_db_path())
            st.success("✅ 벡터 데이터베이스 저장 완료 (FAISS 사용)")
            
    except Exception as e:
        st.error(f"❌ 벡터 데이터베이스 저장 실패: {str(e)}")

## 4: 벡터DB 초기화 함수
def initialize_vector_db():
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
        save_to_vector_store(chunked_documents)
        
        # 성공적으로 초기화된 모델 정보를 파일에 저장
        try:
            model_info = {
                'ai_model': st.session_state.get('selected_model', 'llama3.2'),
                'embedding_model': st.session_state.get('selected_embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            import json
            with open('vector_db_model_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            st.success("✅ 모델 정보가 저장되었습니다.")
        except Exception as e:
            st.warning(f"⚠️ 모델 정보 저장 중 오류: {str(e)}")
        
        return True

############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):
    try:
        # RAG 체인 선언
        chain = get_rag_chain()
        if chain is None:
            st.error("RAG 체인 생성에 실패했습니다.")
            return None, []
        
        # 질문만 전달하여 RAG 체인 실행
        response = chain.invoke(user_question)
        
        # 관련 문서는 하이브리드 검색으로 검색 (참조용)
        embeddings = get_embedding_model()
        
        # ChromaDB 또는 FAISS 사용
        try:
            from langchain_community.vectorstores import Chroma
            new_db = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
        except ImportError:
            # ChromaDB가 없는 경우 FAISS 사용 (경고 억제)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_db = FAISS.load_local(get_vector_db_path(), embeddings, allow_dangerous_deserialization=True)
        
        hybrid_retriever = HybridRetriever(
            vector_store=new_db,
            semantic_weight=0.5,
            keyword_weight=0.5,
            k=3
        )
        retrieve_docs: List[Document] = hybrid_retriever.invoke(user_question)
        return response, retrieve_docs
    except Exception as e:
        st.error(f"질문 처리 중 오류 발생: {str(e)}")
        return None, []

def get_rag_chain() -> Runnable:
    """RAG 체인 생성"""
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        
        # Ollama LLM 초기화
        llm = OllamaLLM(
            model=selected_model,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048
        )
        
        # 선택된 임베딩 모델 사용
        embeddings = get_embedding_model()
        
        # 벡터 스토어 로드 (ChromaDB 우선 사용)
        try:
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
        except ImportError:
            # ChromaDB가 없는 경우 FAISS 사용 (경고 억제)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vector_store = FAISS.load_local(get_vector_db_path(), embeddings, allow_dangerous_deserialization=True)
        
        # 프롬프트 템플릿
        template = """당신은 bizMOB Platform 전문가입니다. 
다음 컨텍스트를 사용하여 질문에 답변해주세요:

컨텍스트:
{context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # 하이브리드 리트리버 생성 (시멘틱 50% + 키워드 50%)
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            semantic_weight=0.5,
            keyword_weight=0.5,
            k=3
        )
        
        # RAG 체인 생성
        chain = (
            {"context": hybrid_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
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
    vector_db_path = get_vector_db_path()
    # index.faiss 파일이 존재하는지 확인
    faiss_file_path = os.path.join(vector_db_path, 'index.faiss')
    return os.path.exists(faiss_file_path)

def load_saved_model_info():
    """저장된 모델 정보를 불러옴"""
    try:
        # 현재 디렉토리와 상위 디렉토리에서 모델 정보 파일 찾기
        possible_paths = [
            'vector_db_model_info.json',
            '../vector_db_model_info.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                import json
                with open(path, 'r', encoding='utf-8') as f:
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

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-multitask',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_recommended_embedding_model(ai_model_name: str) -> str:
    """AI 모델에 따른 권장 임베딩 모델을 반환"""
    model_mapping = {
        'llama3.2': 'sentence-transformers/all-mpnet-base-v2',
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
    return 'sentence-transformers/all-mpnet-base-v2'

def get_vector_db_path():
    """현재 선택된 AI 모델에 맞는 벡터DB 경로 반환"""
    ai_model = st.session_state.get('selected_model', 'llama3.2')
    # 파일명에 사용할 수 없는 문자는 언더스코어로 대체
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', ai_model)
    vector_db_path = f"bizmob_faiss_index_{safe_model}"
    
    # 기존에 존재하는 FAISS 인덱스가 있는지 확인
    existing_indices = []
    for item in os.listdir('.'):
        if item.startswith('bizmob_faiss_index_') and os.path.isdir(item):
            existing_indices.append(item)
    
    # 기존 인덱스가 있으면 우선 사용
    if existing_indices:
        # 현재 모델에 맞는 인덱스가 있으면 사용
        for index in existing_indices:
            if safe_model in index or ai_model in index:
                return index
        # 없으면 첫 번째 기존 인덱스 사용
        return existing_indices[0]
    
    return vector_db_path

def main():
    st.set_page_config("bizMOB Platform 챗봇", layout="wide", page_icon="📱")

    # session_state 초기화
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'llama3.2'
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    if 'vector_db_initialized' not in st.session_state:
        st.session_state.vector_db_initialized = False
    if 'refresh_vector_db_info' not in st.session_state:
        st.session_state.refresh_vector_db_info = False
    if 'refresh_faiss_viewer' not in st.session_state:
        st.session_state.refresh_faiss_viewer = False
    if 'faiss_viewer_page' not in st.session_state:
        st.session_state.faiss_viewer_page = 1

    # 사이드바에 제목과 설명
    st.sidebar.title("📱 bizMOB Platform 챗봇")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**기능**:")
    st.sidebar.markdown("- bizMOB Platform 가이드 문서 기반 질의응답")
    st.sidebar.markdown("- 플랫폼 사용법 및 기능 안내")
    st.sidebar.markdown("- 실시간 문서 참조")
    st.sidebar.markdown("- **Ollama 설치 모델 사용**")
    st.sidebar.markdown("- **파일 업로드 및 관리**")
    
    # Ollama 상태 확인 및 모델 선택
    if check_ollama_models():
        st.sidebar.success("✅ Ollama 연결됨")
        
        # 사용 가능한 모델 목록 가져오기
        available_models = get_ollama_models()
        
        if available_models:
            st.sidebar.markdown("### 🤖 AI 모델 선택")
            
            # 모델 선택을 위한 드롭다운
            model_options = [f"{model['name']} ({model['size']})" for model in available_models]
            model_names = [model['name'] for model in available_models]
            
            # 기본 모델 설정
            if 'selected_model' not in st.session_state:
                # 저장된 모델 정보가 있으면 불러오기
                saved_model_info = load_saved_model_info()
                if saved_model_info and saved_model_info.get('ai_model'):
                    saved_ai_model = saved_model_info['ai_model']
                    # 저장된 모델이 현재 사용 가능한 모델 목록에 있는지 확인
                    if saved_ai_model in model_names:
                        st.session_state.selected_model = saved_ai_model
                        # 저장된 임베딩 모델도 설정
                        if saved_model_info.get('embedding_model'):
                            st.session_state.selected_embedding_model = saved_model_info['embedding_model']
                        st.sidebar.success(f"✅ 저장된 모델 정보를 불러왔습니다: {saved_ai_model}")
                    else:
                        # 저장된 모델이 없으면 llama3.2 또는 첫 번째 모델
                        default_index = 0
                        for i, name in enumerate(model_names):
                            if 'llama3.2' in name.lower():
                                default_index = i
                                break
                        st.session_state.selected_model = model_names[default_index]
                else:
                    # 저장된 정보가 없으면 llama3.2 또는 첫 번째 모델
                    default_index = 0
                    for i, name in enumerate(model_names):
                        if 'llama3.2' in name.lower():
                            default_index = i
                            break
                    st.session_state.selected_model = model_names[default_index]
            
            # 현재 선택된 모델의 인덱스 찾기
            current_index = 0
            for i, name in enumerate(model_names):
                if name == st.session_state.selected_model:
                    current_index = i
                    break
            
            # 모델 선택 드롭다운
            selected_index = st.sidebar.selectbox(
                "사용할 AI 모델을 선택하세요:",
                options=range(len(model_options)),
                index=current_index,
                format_func=lambda x: model_options[x]
            )
            # 선택된 모델 업데이트
            if selected_index != current_index:
                st.session_state.selected_model = model_names[selected_index]
                recommended_embedding = get_recommended_embedding_model(model_names[selected_index])
                st.session_state.selected_embedding_model = recommended_embedding
                st.sidebar.success(f"✅ 모델이 변경되었습니다: {model_names[selected_index]}")
                st.sidebar.info(f"🔤 권장 임베딩 모델로 자동 변경: {recommended_embedding}")
                st.session_state['refresh_vector_db_info'] = True
                st.session_state['refresh_faiss_viewer'] = True
                st.session_state['faiss_viewer_page'] = 1  # 모델 변경 시 페이지네이션 초기화
            
            # 현재 선택된 모델 정보 표시
            selected_model_info = available_models[selected_index]
            st.sidebar.info(f"**현재 모델**: {selected_model_info['name']}")
            st.sidebar.info(f"**모델 크기**: {selected_model_info['size']}")
            
        else:
            st.sidebar.warning("⚠️ 사용 가능한 모델이 없습니다.")
            st.sidebar.info("Ollama에 모델을 설치해주세요.")
    else:
        st.sidebar.error("❌ Ollama 연결 실패")
        st.sidebar.info("Ollama가 설치되어 있고 실행 중인지 확인해주세요.")
    
    # 임베딩 모델 자동 입력 및 정보 표시 (드롭다운 제거)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔤 임베딩 모델 정보")
    
    # 저장된 임베딩 모델이 있으면 사용, 없으면 AI 모델에 맞는 권장 모델 사용
    if 'selected_embedding_model' in st.session_state:
        current_embedding = st.session_state.selected_embedding_model
    else:
        # selected_model이 초기화되지 않은 경우 기본값 사용
        selected_model = st.session_state.get('selected_model', 'llama3.2')
        # selected_model이 None이거나 빈 문자열인 경우 기본값 사용
        if not selected_model:
            selected_model = 'llama3.2'
            st.session_state.selected_model = selected_model
        current_embedding = get_recommended_embedding_model(selected_model)
        st.session_state.selected_embedding_model = current_embedding
    
    available_embedding_models = get_available_embedding_models()
    selected_embedding_info = available_embedding_models.get(current_embedding, {})
    
    # 저장된 모델 정보가 있는지 확인
    saved_model_info = load_saved_model_info()
    if saved_model_info and saved_model_info.get('embedding_model'):
        st.sidebar.success(f"✅ **저장된 임베딩 모델**: {selected_embedding_info.get('name', current_embedding)}")
    else:
        st.sidebar.info(f"🔤 **권장 임베딩 모델**: {selected_embedding_info.get('name', current_embedding)}")
    
    st.sidebar.info(f"**언어**: {selected_embedding_info.get('language', 'Unknown')}")
    st.sidebar.info(f"**크기**: {selected_embedding_info.get('size', 'Unknown')}")
    st.sidebar.caption(f"**설명**: {selected_embedding_info.get('description', '')}")
    
    # 사이드바에 파일 업로드 섹션 추가
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 파일 업로드")
    
    # 파일 업로드 위젯
    uploaded_files = st.sidebar.file_uploader(
        "문서 파일을 선택하세요",
        type=['pdf', 'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc'],
        accept_multiple_files=True,
        help="여러 파일을 동시에 선택할 수 있습니다."
    )
    
    if uploaded_files:
        st.sidebar.markdown("#### 📋 업로드된 파일")
        
        success_count = 0
        error_count = 0
        
        for uploaded_file in uploaded_files:
            if validate_file_type(uploaded_file.name):
                # 파일 저장
                saved_path = save_uploaded_file(uploaded_file)
                if saved_path:
                    st.sidebar.success(f"✅ {uploaded_file.name}")
                    success_count += 1
                else:
                    st.sidebar.error(f"❌ {uploaded_file.name}")
                    error_count += 1
            else:
                st.sidebar.error(f"❌ {uploaded_file.name} (지원하지 않는 형식)")
                error_count += 1
        
        # 결과 요약
        if success_count > 0:
            st.sidebar.success(f"✅ {success_count}개 파일 업로드 완료")
            
            # 벡터 데이터베이스 재초기화 옵션
            if st.sidebar.button("🔄 벡터DB 재초기화", type="primary"):
                if initialize_vector_db():
                    st.session_state.vector_db_initialized = True
                    st.sidebar.success("벡터 데이터베이스 재초기화 완료!")
                else:
                    st.sidebar.error("벡터 데이터베이스 초기화 실패")
        
        if error_count > 0:
            st.sidebar.error(f"❌ {error_count}개 파일 업로드 실패")
    
    # 메인 컨텐츠
    left_column, right_column = st.columns([1, 1])
    
    with left_column:
        st.header("📱 bizMOB Platform 챗봇")
        st.markdown("PDF_bizMOB_Guide 폴더의 bizMOB Platform 가이드 문서를 기반으로 질문에 답변합니다.")
        # 동적으로 AI 모델명 안내
        ai_model_name = st.session_state.get('selected_model', 'llama3.2')
        if 'llama3.2' in ai_model_name.lower():
            model_display = 'Meta Llama 3.2 모델'
        else:
            model_display = f"Ollama AI 모델: {ai_model_name}"
        st.info(f"💡 **{model_display}를 사용하여 PDF, Excel, PowerPoint, Word 문서의 내용을 분석하고 질문에 답변합니다.**")
        
        # 탭 생성 (벡터DB 생성 탭을 가장 오른쪽으로 이동)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 챗봇", "📂 파일 관리", "🔗 소스 관리", "🧊 FAISS 뷰어", "🗂️ 벡터DB 생성"])
        
        with tab1:
            # 현재 선택된 모델 정보 표시
            selected_model = st.session_state.get('selected_model', 'llama3.2')
            if selected_model:
                # 저장된 모델 정보가 있는지 확인
                saved_model_info = load_saved_model_info()
                if saved_model_info and saved_model_info.get('ai_model') == selected_model:
                    st.success(f"🤖 **현재 사용 중인 AI 모델 (저장됨)**: {selected_model}")
                else:
                    st.info(f"🤖 **현재 사용 중인 AI 모델**: {selected_model}")
            
            # 현재 선택된 임베딩 모델 정보 표시
            if 'selected_embedding_model' in st.session_state:
                available_embedding_models = get_available_embedding_models()
                selected_embedding_info = available_embedding_models.get(st.session_state.selected_embedding_model, {})
                embedding_name = selected_embedding_info.get('name', st.session_state.selected_embedding_model)
                embedding_language = selected_embedding_info.get('language', 'Unknown')
                
                # 저장된 임베딩 모델 정보가 있는지 확인
                saved_model_info = load_saved_model_info()
                if saved_model_info and saved_model_info.get('embedding_model') == st.session_state.selected_embedding_model:
                    st.success(f"🔤 **현재 사용 중인 임베딩 모델 (저장됨)**: {embedding_name} ({embedding_language})")
                else:
                    st.info(f"🔤 **현재 사용 중인 임베딩 모델**: {embedding_name} ({embedding_language})")
            
            # 벡터DB 상태 표시 및 초기화 버튼
            if check_vector_db_exists():
                st.success("✅ 벡터 데이터베이스가 준비되었습니다 (AI 모델별)")
            else:
                st.warning("⚠️ 벡터 데이터베이스가 초기화되지 않았습니다. 아래 버튼을 클릭해주세요.")
                if st.button("🔄 벡터 데이터베이스 초기화", type="primary"):
                    if initialize_vector_db():
                        st.session_state.vector_db_initialized = True
                        st.success("벡터 데이터베이스가 성공적으로 초기화되었습니다!")
            # 벡터DB 재초기화 버튼
            if st.button("🔄 벡터 데이터베이스 재초기화", type="primary", key="file_manager_main_reinit_2"):
                if initialize_vector_db():
                    st.session_state.vector_db_initialized = True
                    st.success("벡터 데이터베이스가 성공적으로 재초기화되었습니다!")
                else:
                    st.error("벡터 데이터베이스 초기화에 실패했습니다.")
            
            st.markdown("---")
            
            # 질문 입력 처리 함수
            def handle_question_submit():
                if st.session_state.get('user_question_input', '').strip():
                    st.session_state['submit_question'] = True

            # 질문 입력 + 요청 아이콘 버튼 (한 줄에 배치) → 입력창만 남김
            user_question = st.text_area(
                "bizMOB Platform에 대해 질문해 주세요",
                placeholder="bizMOB Platform의 주요 기능은 무엇인가요?",
                key="user_question_input",
                on_change=handle_question_submit,
                height=80
            )
            
            # 질문 처리 (텍스트 변경 또는 Enter 키 입력 시)
            if (user_question and check_vector_db_exists()) or st.session_state.get('submit_question', False):
                # Enter 키로 제출된 경우 처리 후 상태 초기화
                if st.session_state.get('submit_question', False):
                    st.session_state['submit_question'] = False
                
                with st.spinner("질문을 처리하는 중..."):
                    response, context = process_question(user_question)
                    
                    if response:
                        st.markdown("### 🤖 AI 답변")
                        st.write(response)
                        
                        # 관련 문서 표시
                        if context:
                            st.markdown("### 📄 참조 문서")
                            for i, document in enumerate(context):
                                with st.expander(f"📋 관련 문서 {i+1}"):
                                    st.write(document.page_content)
                                    file_name = document.metadata.get('file_name', 'Unknown')
                                    file_type = document.metadata.get('file_type', 'Unknown')
                                    
                                    # 파일 타입에 따른 참조 정보 표시
                                    if file_type == 'PDF':
                                        page_number = document.metadata.get('page', 0) + 1
                                        st.caption(f"출처: {file_name} (PDF 페이지 {page_number})")
                                        
                                        # PDF 페이지 보기 버튼
                                        button_key = f"view_page_{file_name}_{page_number}_{i}"
                                        if st.button(f"📖 PDF 페이지 보기", key=button_key):
                                            st.session_state.page_number = str(page_number)
                                            st.session_state.pdf_file = file_name
                                            st.session_state.file_type = 'PDF'
                                            
                                    elif file_type == 'Excel':
                                        sheet_name = document.metadata.get('sheet_name', 'Unknown')
                                        st.caption(f"출처: {file_name} (Excel 시트: {sheet_name})")
                                        
                                        # Excel 시트 보기 버튼
                                        button_key = f"view_excel_{file_name}_{sheet_name}_{i}"
                                        if st.button(f"📊 Excel 시트 보기", key=button_key):
                                            st.session_state.excel_file = file_name
                                            st.session_state.sheet_name = sheet_name
                                            st.session_state.file_type = 'Excel'
                                            
                                    elif file_type == 'PowerPoint':
                                        slide_number = document.metadata.get('slide_number', 0)
                                        st.caption(f"출처: {file_name} (PowerPoint 슬라이드 {slide_number})")
                                        
                                        # PowerPoint 슬라이드 보기 버튼
                                        button_key = f"view_ppt_{file_name}_{slide_number}_{i}"
                                        if st.button(f"📽️ PPT 슬라이드 보기", key=button_key):
                                            st.session_state.ppt_file = file_name
                                            st.session_state.slide_number = str(slide_number)
                                            st.session_state.file_type = 'PowerPoint'
                                            
                                    elif file_type == 'Word':
                                        title = document.metadata.get('title', 'Unknown')
                                        author = document.metadata.get('author', 'Unknown')
                                        st.caption(f"출처: {file_name} (Word 문서: {title}, 작성자: {author})")
                                        
                                        # Word 문서 보기 버튼
                                        button_key = f"view_word_{file_name}_{i}"
                                        if st.button(f"📄 Word 문서 보기", key=button_key):
                                            st.session_state.word_file = file_name
                                            st.session_state.file_type = 'Word'
                                            
                                    else:
                                        st.caption(f"출처: {file_name}")
                    else:
                        st.error("답변을 생성할 수 없습니다. 다시 시도해주세요.")
            elif user_question and not check_vector_db_exists():
                st.error("벡터 데이터베이스가 초기화되지 않았습니다. 먼저 초기화 버튼을 클릭해주세요.")
        
        with tab2:
            # 파일 관리 인터페이스
            manage_uploaded_files()
        
        with tab3:
            st.header("🔗 외부 소스(GitHub) 관리")
            st.markdown("GitHub 저장소 경로를 입력하면 소스를 다운로드하여 벡터DB 생성에 포함할 수 있습니다.")
            github_url = st.text_input("GitHub 저장소 URL 입력", placeholder="https://github.com/username/repo")
            if st.button("⬇️ 소스 다운로드", key="download_github_btn"):
                if github_url.strip().startswith("https://github.com/"):
                    repo_name = github_url.rstrip('/').split('/')[-1]
                    dest_dir = os.path.join("external_sources", repo_name)
                    os.makedirs("external_sources", exist_ok=True)
                    if os.path.exists(dest_dir):
                        st.info(f"이미 다운로드된 저장소입니다: {dest_dir}")
                    else:
                        with st.spinner("저장소를 다운로드 중입니다..."):
                            try:
                                subprocess.run(["git", "clone", github_url, dest_dir], check=True)
                                st.success(f"✅ 저장소 다운로드 완료: {dest_dir}")
                            except Exception as e:
                                st.error(f"❌ 저장소 다운로드 실패: {e}")
                else:
                    st.error("올바른 GitHub URL을 입력하세요.")
            # 다운로드된 소스 목록 표시
            if os.path.exists("external_sources"):
                st.markdown("### 📂 다운로드된 소스 목록")
                for repo in os.listdir("external_sources"):
                    repo_path = os.path.join("external_sources", repo)
                    st.write(f"- {repo_path}")

        with tab4:
            st.header("🧊 FAISS 벡터DB 뷰어")
            # 모델 변경 시 리플래시
            if st.session_state.get('refresh_faiss_viewer', False):
                st.session_state['refresh_faiss_viewer'] = False
                st.rerun()
            if not check_vector_db_exists():
                st.warning("벡터DB가 아직 생성되지 않았습니다. 먼저 문서를 업로드하고 벡터DB를 생성하세요.")
            else:
                try:
                    embeddings = get_embedding_model()
                    db = FAISS.load_local(get_vector_db_path(), embeddings, allow_dangerous_deserialization=True)
                    docs = list(db.docstore._dict.values())
                    # 페이지네이션 상태
                    page_size = 100
                    total_docs = len(docs)
                    total_pages = max(1, (total_docs + page_size - 1) // page_size)
                    page = st.session_state.get('faiss_viewer_page', 1)
                    if page < 1:
                        page = 1
                    if page > total_pages:
                        page = total_pages
                    start = (page - 1) * page_size
                    end = start + page_size
                    rows = []
                    for doc in docs[start:end]:
                        meta = doc.metadata.copy()
                        meta['content_len'] = len(doc.page_content)
                        meta['content_sample'] = doc.page_content[:100].replace('\n', ' ') + ("..." if len(doc.page_content) > 100 else "")
                        rows.append(meta)
                    if total_docs == 0:
                        st.info("벡터DB에 저장된 문서가 없습니다. 문서를 업로드하고 벡터DB를 생성하세요.")
                    elif rows:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(rows))
                        st.caption(f"{total_docs}개 중 {start+1}~{min(end, total_docs)}개 문서(청크) 미리보기")
                    else:
                        st.info("벡터DB에 저장된 문서가 없습니다.")
                    # 페이지 네비게이션바
                    col_prev, col_page, col_next = st.columns([1,2,1])
                    with col_prev:
                        if st.button("⬅️ 이전", key="faiss_prev"):
                            if page > 1:
                                st.session_state['faiss_viewer_page'] = page - 1
                                st.rerun()
                    with col_page:
                        st.markdown(f"<div style='text-align:center;'>페이지 {page} / {total_pages}</div>", unsafe_allow_html=True)
                    with col_next:
                        if st.button("다음 ➡️", key="faiss_next"):
                            if page < total_pages:
                                st.session_state['faiss_viewer_page'] = page + 1
                                st.rerun()
                except Exception as e:
                    st.error(f"FAISS 벡터DB를 불러오는 중 오류: {e}")

        with tab5:
            st.header("🗂️ 벡터DB 생성/초기화")
            # 모델 변경 시 리플래시
            if st.session_state.get('refresh_vector_db_info', False):
                st.session_state['refresh_vector_db_info'] = False
                st.rerun()
            st.markdown("문서 업로드 후, 아래 버튼을 눌러 벡터 데이터베이스를 생성하거나 초기화할 수 있습니다.")
            st.info("벡터DB는 PDF, Excel, PowerPoint, Word 문서의 내용을 임베딩하여 검색을 빠르게 해줍니다.")
            # 벡터DB 상태
            if check_vector_db_exists():
                st.success("✅ 벡터 데이터베이스가 이미 생성되어 있습니다.")
            else:
                st.warning("⚠️ 벡터 데이터베이스가 아직 생성되지 않았습니다.")
            # 벡터DB 생성/초기화 버튼
            if st.button("🗂️ 벡터DB 생성/초기화", type="primary", key="vector_db_create_btn"):
                with st.spinner("문서를 분석하고 벡터 데이터베이스를 생성하는 중입니다..."):
                    result = initialize_vector_db()
                if result:
                    st.success("✅ 벡터 데이터베이스가 성공적으로 생성/초기화되었습니다!")
                else:
                    st.error("❌ 벡터 데이터베이스 생성/초기화에 실패했습니다. 문서가 업로드되어 있는지 확인하세요.")
            # 벡터DB 정보 표시 (모델별)
            model_info = load_saved_model_info()
            st.markdown("---")
            st.markdown(f"### 현재 선택된 AI 모델 정보")
            if model_info:
                st.markdown(f"**AI 모델:** {model_info.get('ai_model', '-')}")
                st.markdown(f"**임베딩 모델:** {model_info.get('embedding_model', '-')}")
                st.markdown(f"**생성 시각:** {model_info.get('timestamp', '-')}")
            else:
                st.info("이 모델로 생성된 벡터DB 정보가 없습니다. 먼저 벡터DB를 생성하세요.")

if __name__ == "__main__":
    main() 