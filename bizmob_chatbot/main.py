"""
bizMOB Platform Chatbot - Main Application
모듈화된 구조로 리팩토링된 메인 애플리케이션
"""

import streamlit as st
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config.settings import settings
from utils.logging_config import setup_logging
from core.auth import AuthManager
from ui.chat_interface import ChatInterface
from ui.admin_panel import AdminPanel
from ui.file_upload import FileUpload


def main():
    """메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide"
    )
    
    # 로깅 설정
    logger = setup_logging()
    
    # 컴포넌트 초기화
    auth_manager = AuthManager()
    chat_interface = ChatInterface()
    admin_panel = AdminPanel()
    file_upload = FileUpload()
    
    # 사이드바 설정
    setup_sidebar(auth_manager, admin_panel, file_upload)
    
    # 메인 화면 - 채팅 중심
    display_main_chat_interface(auth_manager, chat_interface)


def setup_sidebar(auth_manager, admin_panel, file_upload):
    """사이드바 설정"""
    with st.sidebar:
        st.title("🤖 bizMOB Chatbot")
        st.markdown("---")
        
        # 사용자 모드 표시
        display_user_mode(auth_manager)
        st.markdown("---")
        
        # 관리자 로그인 (일반 사용자일 때만)
        if not auth_manager.is_admin():
            admin_panel.display_login_section()
            st.markdown("---")
        
        # 관리자 메뉴 (관리자일 때만)
        if auth_manager.is_admin():
            display_admin_sidebar_menu(auth_manager, admin_panel, file_upload)


def display_user_mode(auth_manager):
    """사용자 모드 표시"""
    if auth_manager.is_admin():
        st.success("👨‍💼 관리자 모드")
        st.info(f"역할: {auth_manager.get_user_role()}")
    else:
        st.info("👤 일반 사용자 모드")
        st.info("관리자 기능을 사용하려면 로그인하세요.")


def display_admin_sidebar_menu(auth_manager, admin_panel, file_upload):
    """관리자 사이드바 메뉴"""
    st.markdown("### ⚙️ 관리자 도구")
    
    # 로그아웃 버튼
    if st.button("🚪 로그아웃", key="logout_sidebar"):
        auth_manager.logout()
        st.rerun()
    
    st.markdown("---")
    
    # 관리자 메뉴 탭
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📊 상태", "📁 파일", "🛠️ 도구"])
    
    with admin_tab1:
        display_system_status()
    
    with admin_tab2:
        display_file_management(file_upload)
    
    with admin_tab3:
        display_admin_tools(admin_panel)


def display_system_status():
    """시스템 상태 표시"""
    st.markdown("#### 📊 시스템 상태")
    
    # 벡터 DB 상태 - 실제 존재 여부 확인
    from core.vector_db_manager import VectorDBManager
    vector_db_manager = VectorDBManager()
    vector_db_exists = vector_db_manager.check_vector_db_exists()
    
    if vector_db_exists:
        st.success("✅ 벡터 DB가 존재합니다")
        st.session_state.vector_db_initialized = True
    else:
        st.warning("⚠️ 벡터 데이터베이스가 초기화되지 않았습니다.")
        st.session_state.vector_db_initialized = False
    
    # 채팅 히스토리 상태
    if 'chat_history' in st.session_state:
        message_count = len(st.session_state.chat_history)
        st.info(f"채팅 메시지: {message_count}개")
    
    # 업로드된 파일 상태
    folder_path = "PDF_bizMOB_Guide"
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        st.info(f"업로드된 파일: {len(files)}개")
    else:
        st.info("업로드된 파일: 0개")


def display_file_management(file_upload):
    """파일 관리"""
    st.markdown("#### 📁 파일 관리")
    
    # 파일 업로드
    uploaded_files = file_upload.display_file_upload_section()
    
    # 업로드된 파일 정보
    file_upload.display_uploaded_files_info()
    
    # 벡터 DB 업데이트 버튼
    st.markdown("---")
    st.markdown("#### 🔄 벡터 DB 관리")
    
    if st.button("🔄 업로드된 파일로 벡터 DB 업데이트", type="primary", key="update_vector_db_from_files"):
        file_upload.update_vector_db()
        st.rerun()


def display_admin_tools(admin_panel):
    """관리자 도구"""
    st.markdown("#### 🛠️ 관리 도구")
    
    # 채팅 히스토리 초기화
    if st.button("🗑️ 채팅 히스토리 초기화", key="clear_chat"):
        if 'chat_history' in st.session_state:
            st.session_state.chat_history.clear()
        st.success("채팅 히스토리가 초기화되었습니다.")
        st.rerun()
    
    # 업로드된 파일 정보 초기화
    if st.button("🗑️ 업로드 정보 초기화", key="clear_upload"):
        if 'uploaded_folders' in st.session_state:
            st.session_state.uploaded_folders.clear()
        st.success("업로드 정보가 초기화되었습니다.")
        st.rerun()
    
    # 벡터 DB 상태 리셋
    if st.button("🔄 벡터 DB 상태 리셋", key="reset_vector_db"):
        if 'vector_db_initialized' in st.session_state:
            del st.session_state.vector_db_initialized
        st.success("벡터 DB 상태가 리셋되었습니다.")
        st.rerun()
    
    # 벡터 DB 재생성
    if st.button("🔄 벡터 DB 재생성", type="primary", key="rebuild_vector_db"):
        admin_panel.rebuild_vector_db()
        st.rerun()


def display_main_chat_interface(auth_manager, chat_interface):
    """메인 채팅 인터페이스 표시"""
    # 관리자 모드일 때는 탭으로 구성
    if auth_manager.is_admin():
        tab1, tab2 = st.tabs(["💬 채팅", "🔍 벡터 DB 데이터"])
        
        with tab1:
            # 채팅 인터페이스
            chat_interface.display_chat_input_area()
        
        with tab2:
            # 벡터 DB 데이터 조회
            display_vector_db_data_tab(chat_interface.vector_db_manager)
    else:
        # 일반 사용자는 채팅만
        chat_interface.display_chat_input_area()


def display_vector_db_data_tab(vector_db_manager):
    """벡터 DB 데이터 조회 탭"""
    st.markdown("### 🔍 벡터 DB 데이터 조회")
    
    # 벡터 DB 정보 조회
    vector_db_info = vector_db_manager.get_vector_db_info()
    
    if vector_db_info['exists']:
        st.success("✅ 벡터 DB가 존재합니다")
        
        # 기본 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 문서 개수", vector_db_info['document_count'])
        with col2:
            st.metric("💾 인덱스 크기", f"{vector_db_info['index_size']:.2f} MB")
        with col3:
            st.metric("📁 저장 경로", vector_db_info['path'].split('/')[-1])
        
        # 저장된 모델 정보 표시
        model_info = vector_db_manager.load_saved_model_info()
        if model_info:
            st.info(f"**AI 모델**: {model_info.get('ai_model', 'N/A')}")
            st.info(f"**임베딩 모델**: {model_info.get('embedding_model', 'N/A')}")
            st.info(f"**생성 시간**: {model_info.get('timestamp', 'N/A')}")
        
        # 샘플 데이터 조회
        st.markdown("#### 📋 샘플 데이터")
        samples = vector_db_manager.get_vector_db_samples(limit=5)
        
        if samples:
            for i, sample in enumerate(samples):
                with st.expander(f"📄 샘플 {i+1} (길이: {sample['length']}자)"):
                    st.write("**내용:**")
                    st.text(sample['content'])
                    st.write("**메타데이터:**")
                    st.json(sample['metadata'])
        else:
            st.warning("샘플 데이터를 조회할 수 없습니다")
        
        # 검색 기능
        st.markdown("#### 🔍 벡터 DB 검색")
        search_query = st.text_input("검색어를 입력하세요:", placeholder="예: bizmob, 플랫폼, 기능")
        
        if search_query:
            if st.button("🔍 검색"):
                search_results = vector_db_manager.search_vector_db(search_query, k=5)
                
                if search_results:
                    st.success(f"'{search_query}'에 대한 검색 결과 ({len(search_results)}개)")
                    
                    for result in search_results:
                        with st.expander(f"🏆 순위 {result['rank']} (길이: {result['length']}자)"):
                            st.write("**내용:**")
                            st.text(result['content'])
                            st.write("**메타데이터:**")
                            st.json(result['metadata'])
                else:
                    st.warning("검색 결과가 없습니다")
    else:
        st.warning("⚠️ 벡터 DB가 존재하지 않습니다")
        if 'error' in vector_db_info:
            st.error(f"오류: {vector_db_info['error']}")


if __name__ == "__main__":
    main() 