#!/usr/bin/env python3
"""
bizMOB 챗봇 UI/UX 컴포넌트 모듈
UI 관련 함수들과 스타일링을 분리하여 관리
"""

import streamlit as st
import html
import re
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional
from streamlit.runtime.uploaded_file_manager import UploadedFile

# CSS 스타일 정의
CSS_STYLES = """
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
    .chat-message-user {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    .chat-message-assistant {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    .chat-bubble-user {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .chat-bubble-assistant {
        background-color: #F0F0F0;
        color: black;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .chat-timestamp {
        font-size: 0.7em;
        opacity: 0.7;
        margin-top: 5px;
    }
</style>
"""

def apply_css_styles():
    """CSS 스타일 적용"""
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

def setup_page_config():
    """페이지 설정"""
    st.set_page_config(
        page_title="bizMOB Platform 챗봇",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_chat_history():
    """채팅 기록 초기화"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_chat_message(role: str, content: str, timestamp=None):
    """채팅 메시지 추가"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({
        'role': role,
        'content': content,
        'timestamp': timestamp
    })

def display_chat_messages():
    """채팅 메시지들 표시"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 채팅 컨테이너 생성
    chat_container = st.container()
    
    with chat_container:
        # 채팅 메시지들 표시
        for message in st.session_state.chat_history:
            # 메시지 내용을 안전하게 처리
            content = message['content']
            
            # HTML 태그 제거
            content = re.sub(r'<[^>]+>', '', content)
            
            # 특수 문자 이스케이프
            content = html.escape(content)
            
            # 줄바꿈을 <br> 태그로 변환
            content = content.replace('\n', '<br>')
            
            if message['role'] == 'user':
                # 사용자 메시지 (오른쪽 정렬, 파란색 배경)
                st.markdown(f"""
                <div class="chat-message-user">
                    <div class="chat-bubble-user">
                        {content}
                        <div class="chat-timestamp">{message['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI 메시지 (왼쪽 정렬, 회색 배경)
                st.markdown(f"""
                <div class="chat-message-assistant">
                    <div class="chat-bubble-assistant">
                        {content}
                        <div class="chat-timestamp">{message['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def check_user_role():
    """사용자 역할 확인"""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'general'

def is_admin():
    """관리자 여부 확인"""
    return st.session_state.get('user_role') == 'admin'

def show_role_selector():
    """역할 선택기 표시"""
    st.sidebar.markdown("### 👤 사용자 역할")
    
    role = st.sidebar.selectbox(
        "역할을 선택하세요",
        ["일반 사용자", "관리자"],
        index=0 if st.session_state.get('user_role') == 'general' else 1
    )
    
    if role == "관리자":
        password = st.sidebar.text_input("관리자 비밀번호", type="password")
        if st.sidebar.button("로그인"):
            if password == "0000":  # 관리자 비밀번호
                st.session_state.user_role = 'admin'
                st.sidebar.success("관리자로 로그인되었습니다!")
                st.rerun()
            else:
                st.sidebar.error("비밀번호가 올바르지 않습니다.")
    else:
        st.session_state.user_role = 'general'
        if st.sidebar.button("일반 사용자로 설정"):
            st.rerun()

def show_sidebar_info():
    """사이드바 정보 표시"""
    st.sidebar.title("📱 bizMOB Platform 챗봇")
    st.sidebar.markdown("---")
    
    # 역할 선택기 표시
    show_role_selector()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**기능**:")
    st.sidebar.markdown("- bizMOB Platform 가이드 문서 기반 질의응답")
    st.sidebar.markdown("- 플랫폼 사용법 및 기능 안내")
    st.sidebar.markdown("- 실시간 문서 참조")
    st.sidebar.markdown("- **Ollama 설치 모델 사용**")
    
    # 관리자만 파일 업로드 기능 표시
    if is_admin():
        st.sidebar.markdown("- **파일 업로드 및 관리**")
        st.sidebar.markdown("- **소스 관리**")
        st.sidebar.markdown("- **ChromaDB 뷰어**")
        st.sidebar.markdown("- **벡터DB 생성**")

def show_model_selector(available_models, get_recommended_embedding_model, load_saved_model_info):
    """모델 선택기 표시"""
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
                    # 저장된 모델이 없으면 exaone3.5 또는 첫 번째 모델
                    default_index = 0
                    for i, name in enumerate(model_names):
                        if 'exaone3.5' in name.lower():
                            default_index = i
                            break
                    st.session_state.selected_model = model_names[default_index]
            else:
                # 저장된 정보가 없으면 exaone3.5 또는 첫 번째 모델
                default_index = 0
                for i, name in enumerate(model_names):
                    if 'exaone3.5' in name.lower():
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
            old_model = st.session_state.selected_model
            st.session_state.selected_model = model_names[selected_index]
            recommended_embedding = get_recommended_embedding_model(model_names[selected_index])
            st.session_state.selected_embedding_model = recommended_embedding
            
            # 모델이 변경되면 벡터 DB 재생성 필요
            if old_model != model_names[selected_index]:
                st.sidebar.warning("⚠️ 모델이 변경되어 벡터 DB를 재생성해야 합니다.")
                
                # 자동 재생성 옵션 제공
                if st.sidebar.button("🔄 벡터 DB 자동 재생성", key="auto_rebuild_vector_db"):
                    try:
                        # 기존 벡터 DB 초기화
                        import chromadb
                        chroma_path = "./chroma_db"
                        if os.path.exists(chroma_path):
                            client = chromadb.PersistentClient(path=chroma_path)
                            try:
                                client.delete_collection(name="bizmob_documents")
                                st.sidebar.success("✅ 기존 벡터 DB 삭제 완료")
                            except:
                                pass
                        
                        # 새 벡터 DB 생성
                        from bizmob_chatbot import initialize_vector_db_with_documents
                        if initialize_vector_db_with_documents():
                            st.sidebar.success("✅ 벡터 DB 재생성 완료")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ 벡터 DB 재생성 실패")
                    except Exception as e:
                        st.sidebar.error(f"❌ 벡터 DB 재생성 중 오류: {str(e)}")
                else:
                    st.sidebar.info("📁 '벡터DB 생성/초기화' 버튼을 클릭해주세요.")
                
                # 캐시 초기화
                st.session_state['refresh_vector_db_info'] = True
                st.session_state['refresh_faiss_viewer'] = True
                st.session_state['faiss_viewer_page'] = 1
                # 벡터 스토어 캐시 제거
                for key in list(st.session_state.keys()):
                    if key.startswith('global_vector_store_') or key.startswith('vector_store_') or key.startswith('rag_chain_'):
                        del st.session_state[key]
            
            st.sidebar.success(f"✅ 모델이 변경되었습니다: {model_names[selected_index]}")
            st.sidebar.info(f"🔤 권장 임베딩 모델로 자동 변경: {recommended_embedding}")
        
        # 현재 선택된 모델 정보 표시
        selected_model_info = available_models[selected_index]
        st.sidebar.info(f"**현재 모델**: {selected_model_info['name']}")
        st.sidebar.info(f"**모델 크기**: {selected_model_info['size']}")
        
    else:
        st.sidebar.warning("⚠️ 사용 가능한 모델이 없습니다.")
        st.sidebar.info("Ollama에 모델을 설치해주세요.")

def show_embedding_model_info(get_available_embedding_models, load_saved_model_info, get_recommended_embedding_model):
    """임베딩 모델 정보 표시"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔤 임베딩 모델 정보")
    
    # 저장된 임베딩 모델이 있으면 사용, 없으면 AI 모델에 맞는 권장 모델 사용
    if 'selected_embedding_model' in st.session_state:
        current_embedding = st.session_state.selected_embedding_model
    else:
        current_embedding = get_recommended_embedding_model(st.session_state.selected_model)
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

def show_file_upload_section(save_uploaded_file, validate_file_type, initialize_vector_db_with_documents):
    """파일 업로드 섹션 표시 (관리자용)"""
    if is_admin():
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 파일 업로드")
        
        # 파일 업로드 위젯
        uploaded_files = st.sidebar.file_uploader(
            "문서 파일을 선택하세요",
            type=['pdf', 'xlsx', 'xls', 'pptx', 'ppt', 'docx', 'doc'],
            accept_multiple_files=True,
            help="여러 파일을 동시에 선택할 수 있습니다.",
            key="main_file_uploader"
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
                if st.sidebar.button("🔄 벡터DB 재초기화", type="primary", key="main_reinit"):
                    if initialize_vector_db_with_documents():
                        st.session_state.vector_db_initialized = True
                        st.sidebar.success("벡터 데이터베이스 재초기화 완료!")
                    else:
                        st.sidebar.error("벡터 데이터베이스 초기화에 실패")
            
            if error_count > 0:
                st.sidebar.error(f"❌ {error_count}개 파일 업로드에 실패")

def show_chat_interface(display_chat_messages, check_vector_db_exists, initialize_vector_db_with_documents, 
                       add_chat_message, process_question):
    """채팅 인터페이스 표시"""
    st.markdown("### 💬 채팅")
    
    # 채팅 메시지들 표시
    display_chat_messages()
    
    # 벡터DB 상태 표시 및 초기화 버튼
    if check_vector_db_exists():
        st.success("✅ 벡터 데이터베이스가 준비되었습니다 (AI 모델별)")
    else:
        st.warning("⚠️ 벡터 데이터베이스가 초기화되지 않았습니다. 아래 버튼을 클릭해주세요.")
        if st.button("🔄 벡터 데이터베이스 초기화", type="primary", key="admin_tab1_vector_db_init"):
            if initialize_vector_db_with_documents():
                st.session_state.vector_db_initialized = True
                st.success("벡터 데이터베이스가 성공적으로 초기화되었습니다!")
    
    st.markdown("---")
    
    # 질문 입력 처리 함수
    def handle_question_submit():
        if st.session_state.get('user_question_input', '').strip():
            st.session_state['submit_question'] = True

    # 질문 입력
    # 동적 키를 사용하여 입력창 초기화
    input_key = f"user_question_input_{st.session_state.get('input_counter', 0)}"
    user_question = st.text_area(
        "bizMOB Platform에 대해 질문해 주세요",
        placeholder="bizMOB Platform의 주요 기능은 무엇인가요?",
        key=input_key,
        on_change=handle_question_submit,
        height=80
    )
    
    # 질문 처리
    if user_question and user_question.strip():
        # 로그 추가
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"=== 사용자 질문 수신: {user_question[:50]}... ===")
        
        # 사용자 메시지를 채팅 기록에 추가
        add_chat_message('user', user_question)
        
        # 벡터DB 상태 확인
        if check_vector_db_exists():
            logger.info("벡터DB 존재 확인됨, 질문 처리 시작")
            with st.spinner("질문을 처리하는 중..."):
                try:
                    logger.info("process_question 함수 호출")
                    response, context = process_question(user_question)
                    
                    if response:
                        logger.info(f"답변 생성 완료, 길이: {len(response)}")
                        # AI 답변을 채팅 기록에 추가
                        add_chat_message('assistant', response)
                        
                        # 입력창 초기화를 위한 카운터 증가
                        st.session_state['input_counter'] = st.session_state.get('input_counter', 0) + 1
                        
                        logger.info("채팅 메시지 추가 완료, 페이지 새로고침")
                        # 화면 새로고침
                        st.rerun()
                    else:
                        logger.warning("답변 생성 실패")
                        st.error("답변을 생성할 수 없습니다. 다시 시도해주세요.")
                except Exception as e:
                    logger.error(f"질문 처리 중 오류: {str(e)}", exc_info=True)
                    st.error(f"질문 처리 중 오류가 발생했습니다: {str(e)}")
        else:
            logger.warning("벡터DB가 초기화되지 않음")
            st.error("벡터 데이터베이스가 초기화되지 않았습니다. 먼저 초기화 버튼을 클릭해주세요.")

def show_admin_interface(display_chat_messages, check_vector_db_exists, initialize_vector_db_with_documents,
                        add_chat_message, process_question, manage_uploaded_files, load_saved_model_info):
    """관리자 인터페이스 표시"""
    # 관리자: 전체 기능 접근
    left_column, right_column = st.columns([1, 1])
    
    with left_column:
        st.header("📁 bizMOB Platform 파일 관리")
        st.markdown("PDF_bizMOB_Guide 폴더의 bizMOB Platform 가이드 문서를 관리하고 벡터 데이터베이스를 구축합니다.")
        # 동적으로 AI 모델명 안내
        ai_model_name = st.session_state.get('selected_model', 'exaone3.5')
        if 'exaone3.5' in ai_model_name.lower():
            model_display = 'ExaOne 3.5 모델'
        else:
            model_display = f"Ollama AI 모델: {ai_model_name}"
        st.info(f"💡 **{model_display}를 사용하여 PDF, Excel, PowerPoint, Word 문서를 처리하고 벡터 데이터베이스를 생성합니다.**")
        
        # 탭 생성 (3개 탭으로 축소, 벡터DB에 뷰어와 생성 기능 통합)
        tab1, tab2, tab3 = st.tabs(["📂 파일 관리", "🔗 소스 관리", "🗂️ 벡터DB"])
        
        with tab1:
            # 파일 관리 인터페이스
            manage_uploaded_files()
        
        with tab2:
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
        
        with tab3:
            st.header("🗂️ 벡터DB")
            
            # 벡터DB 상태 및 생성 섹션
            st.markdown("### 🔄 벡터DB 생성/초기화")
            # 모델 변경 시 리플래시
            if st.session_state.get('refresh_vector_db_info', False):
                st.session_state['refresh_vector_db_info'] = False
                st.rerun()
            
            # 벡터DB 상태 확인
            if check_vector_db_exists():
                st.success("✅ 벡터 데이터베이스가 이미 생성되어 있습니다.")
            else:
                st.warning("⚠️ 벡터 데이터베이스가 아직 생성되지 않았습니다.")
            
            # 벡터DB 생성/초기화 버튼
            if st.button("🔄 벡터 데이터베이스 초기화", type="primary", key="admin_tab3_vector_db_init"):
                if initialize_vector_db_with_documents():
                    st.session_state.vector_db_initialized = True
                    st.success("벡터 데이터베이스가 성공적으로 초기화되었습니다!")
                    st.rerun()
                else:
                    st.error("벡터 데이터베이스 초기화에 실패했습니다.")
            
            # 저장된 모델 정보 표시
            saved_model_info = load_saved_model_info()
            if saved_model_info:
                st.markdown("### 📋 저장된 모델 정보")
                st.info(f"**AI 모델**: {saved_model_info.get('ai_model', 'Unknown')}")
                st.info(f"**임베딩 모델**: {saved_model_info.get('embedding_model', 'Unknown')}")
                st.info(f"**생성 시간**: {saved_model_info.get('timestamp', 'Unknown')}")
            else:
                st.info("이 모델로 생성된 벡터DB 정보가 없습니다. 먼저 벡터DB를 생성하세요.")
            
            st.markdown("---")
            
            # ChromaDB 뷰어 섹션
            st.markdown("### 👁️ ChromaDB 뷰어")
            # 모델 변경 시 리플래시
            if st.session_state.get('refresh_chroma_viewer', False):
                st.session_state['refresh_chroma_viewer'] = False
                st.rerun()
            
            # ChromaDB 뷰어
            try:
                import chromadb
                import threading
                chroma_db_path = "./chroma_db"
                
                if os.path.exists(chroma_db_path):
                    # 전역 벡터 스토어 공유 (강화된 관리)
                    global_vector_store_key = "global_vector_store"
                    
                    # 기존 벡터 스토어가 있으면 재사용
                    if global_vector_store_key in st.session_state:
                        try:
                            vector_store = st.session_state[global_vector_store_key]
                            # 간단한 테스트로 연결 상태 확인
                            test_collection = vector_store._collection
                            test_collection.count()
                            chroma_client = vector_store._client
                        except Exception as e:
                            # 기존 벡터 스토어 제거
                            del st.session_state[global_vector_store_key]
                            # 기존 ChromaDB 프로세스 정리
                            try:
                                import psutil
                                import time
                                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                    try:
                                        if 'chroma' in proc.info['name'].lower() or any('chroma' in str(cmd).lower() for cmd in proc.info['cmdline'] or []):
                                            proc.terminate()
                                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                                        pass
                                time.sleep(2)
                            except Exception:
                                pass
                            chroma_client = chromadb.PersistentClient(
                                path=chroma_db_path,
                                settings=chromadb.config.Settings(
                                    allow_reset=True,
                                    anonymized_telemetry=False,
                                    is_persistent=True,
                                    persist_directory=chroma_db_path
                                )
                            )
                    else:
                        # 새 클라이언트 생성
                        chroma_client = chromadb.PersistentClient(
                            path=chroma_db_path,
                            settings=chromadb.config.Settings(
                                allow_reset=True,
                                anonymized_telemetry=False,
                                is_persistent=True,
                                persist_directory=chroma_db_path
                            )
                        )
                    
                    collections = chroma_client.list_collections()
                    
                    if collections:
                        st.success(f"✅ ChromaDB 연결됨: {len(collections)}개 컬렉션")
                        
                        # 컬렉션 선택
                        collection_names = [col.name for col in collections]
                        selected_collection = st.selectbox(
                            "컬렉션을 선택하세요:",
                            collection_names,
                            key="chroma_collection_selector_tab3"
                        )
                        
                        if selected_collection:
                            collection = chroma_client.get_collection(selected_collection)
                            count = collection.count()
                            st.info(f"📊 선택된 컬렉션: {selected_collection} ({count}개 문서)")
                            
                            # 페이지네이션
                            items_per_page = 10
                            total_pages = (count + items_per_page - 1) // items_per_page
                            
                            if total_pages > 1:
                                current_page = st.selectbox(
                                    f"페이지 선택 (총 {total_pages}페이지):",
                                    range(1, total_pages + 1),
                                    key="chroma_page_selector_tab3"
                                )
                            else:
                                current_page = 1
                            
                            # 데이터 조회
                            offset = (current_page - 1) * items_per_page
                            results = collection.get(
                                limit=items_per_page,
                                offset=offset,
                                include=['documents', 'metadatas', 'embeddings']
                            )
                            
                            if results['documents']:
                                st.markdown("### 📄 문서 목록")
                                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                                    with st.expander(f"문서 {offset + i + 1}: {metadata.get('file_name', 'Unknown')}"):
                                        st.markdown(f"**메타데이터**: {metadata}")
                                        st.markdown(f"**내용**: {doc[:500]}{'...' if len(doc) > 500 else ''}")
                            else:
                                st.warning("해당 페이지에 문서가 없습니다.")
                    else:
                        st.warning("⚠️ ChromaDB에 컬렉션이 없습니다.")
                else:
                    st.warning("⚠️ ChromaDB가 초기화되지 않았습니다.")
            except Exception as e:
                st.error(f"ChromaDB 벡터DB를 불러오는 중 오류: {e}")

def show_user_interface(display_chat_messages, check_vector_db_exists, add_chat_message, process_question):
    """일반 사용자 인터페이스 표시"""
    st.header("📱 bizMOB Platform 챗봇")
    st.markdown("bizMOB Platform에 대해 질문해 주세요!")
    
    # 카카오톡 스타일 채팅 인터페이스
    st.markdown("### �� 채팅")
    
    # 채팅 메시지들 표시
    display_chat_messages()
    
    # 벡터DB 상태 확인
    if not check_vector_db_exists():
        st.warning("⚠️ 벡터 데이터베이스가 초기화되지 않았습니다. 관리자에게 문의하세요.")
        return
    
    st.markdown("---")
    
    # 질문 입력 처리 함수
    def handle_question_submit():
        if st.session_state.get('user_question_input', '').strip():
            st.session_state['submit_question'] = True

    # 질문 입력
    # 동적 키를 사용하여 입력창 초기화
    input_key = f"user_question_input_{st.session_state.get('input_counter', 0)}"
    user_question = st.text_area(
        "bizMOB Platform에 대해 질문해 주세요",
        placeholder="bizMOB Platform의 주요 기능은 무엇인가요?",
        key=input_key,
        on_change=handle_question_submit,
        height=80
    )
    
    # 질문 처리
    if user_question and user_question.strip():
        # 로그 추가
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"=== 일반 사용자 질문 수신: {user_question[:50]}... ===")
        
        # 사용자 메시지를 채팅 기록에 추가
        add_chat_message('user', user_question)
        
        # 벡터DB 상태 확인
        if check_vector_db_exists():
            logger.info("벡터DB 존재 확인됨, 질문 처리 시작")
            with st.spinner("질문을 처리하는 중..."):
                try:
                    logger.info("process_question 함수 호출")
                    response, context = process_question(user_question)
                    
                    if response:
                        logger.info(f"답변 생성 완료, 길이: {len(response)}")
                        # AI 답변을 채팅 기록에 추가
                        add_chat_message('assistant', response)
                        
                        # 입력창 초기화를 위한 카운터 증가
                        st.session_state['input_counter'] = st.session_state.get('input_counter', 0) + 1
                        
                        logger.info("채팅 메시지 추가 완료, 페이지 새로고침")
                        # 화면 새로고침
                        st.rerun()
                    else:
                        logger.warning("답변 생성 실패")
                        st.error("답변을 생성할 수 없습니다. 다시 시도해주세요.")
                except Exception as e:
                    logger.error(f"질문 처리 중 오류: {str(e)}", exc_info=True)
                    st.error(f"질문 처리 중 오류가 발생했습니다: {str(e)}")
        else:
            logger.warning("벡터DB가 초기화되지 않음")
            st.error("벡터 데이터베이스가 초기화되지 않았습니다. 관리자에게 문의하세요.") 