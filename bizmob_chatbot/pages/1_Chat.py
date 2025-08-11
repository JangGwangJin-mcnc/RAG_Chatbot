#!/usr/bin/env python3
"""
bizMOB 간단 채팅 페이지 - 모든 사용자가 바로 사용 가능
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime
import subprocess

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
        log_file = os.path.join(log_dir, f"chat_page_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 파일 핸들러 (UTF-8 인코딩)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except (PermissionError, OSError) as e:
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

# 페이지 설정
st.set_page_config(
    page_title="bizMOB 채팅",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .chat-message-user {
        background-color: #007AFF;
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 70%;
        margin: 10px 0;
        margin-left: auto;
        word-wrap: break-word;
    }
    .chat-message-assistant {
        background-color: #F0F0F0;
        color: black;
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 70%;
        margin: 10px 0;
        margin-right: auto;
        word-wrap: break-word;
    }
    .status-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session():
    """세션 상태 초기화"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'exaone3.5'
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0

def add_message(role, content):
    """채팅 메시지 추가"""
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_messages.append({
        'role': role,
        'content': content,
        'timestamp': timestamp
    })

def display_chat():
    """채팅 메시지 표시"""
    for message in st.session_state.chat_messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message-user">
                {message['content']}
                <div style="font-size: 0.7em; opacity: 0.7; margin-top: 5px;">
                    {message['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message-assistant">
                {message['content']}
                <div style="font-size: 0.7em; opacity: 0.7; margin-top: 5px;">
                    {message['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)

def simple_chat_response(user_question):
    """간단한 채팅 응답 생성 - Ollama 직접 호출"""
    logger.info(f"=== 채팅 응답 생성 시작: {user_question[:50]}... ===")
    
    try:
        # bizMOB 전문가 프롬프트 구성
        prompt = f"""당신은 bizMOB Platform 전문가입니다. 
bizMOB Platform은 모바일 앱 개발을 위한 플랫폼으로, 다음과 같은 특징이 있습니다:

주요 기능:
- 모바일 앱 개발 및 배포
- 백엔드 서비스 연동
- 사용자 관리 및 인증
- 데이터베이스 연동
- API 개발 및 관리

사용자의 질문에 대해 친절하고 정확하게 답변해주세요.
답변은 한글로 해주세요.

질문: {user_question}

답변:"""
        
        logger.info(f"프롬프트 구성 완료, 모델: {st.session_state.selected_model}")
        
        # Ollama 호출
        logger.info("Ollama 호출 시작...")
        result = subprocess.run([
            'ollama', 'run', st.session_state.selected_model, prompt
        ], capture_output=True, text=True, timeout=60)
        
        logger.info(f"Ollama 호출 완료, returncode: {result.returncode}")
        
        if result.returncode == 0:
            response = result.stdout.strip()
            if response:
                logger.info(f"응답 생성 성공, 길이: {len(response)}")
                return response
            else:
                logger.warning("응답이 비어있습니다")
                return "죄송합니다. 응답을 생성할 수 없습니다."
        else:
            error_msg = f"모델 호출 중 오류가 발생했습니다: {result.stderr}"
            logger.error(f"Ollama 오류: {result.stderr}")
            return error_msg
            
    except subprocess.TimeoutExpired:
        logger.error("응답 시간 초과")
        return "응답 시간이 초과되었습니다. 다시 시도해주세요."
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}", exc_info=True)
        return f"오류가 발생했습니다: {str(e)}"

def check_ollama():
    """Ollama 상태 확인"""
    logger.info("Ollama 상태 확인 시작")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Ollama 연결 성공")
            return True
        else:
            logger.warning(f"Ollama 연결 실패, returncode: {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"Ollama 상태 확인 중 오류: {str(e)}")
        return False

def get_available_models():
    """사용 가능한 모델 목록 가져오기"""
    logger.info("사용 가능한 모델 목록 조회 시작")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # 첫 번째 줄은 헤더이므로 제외
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        model_name = parts[0]
                        models.append(model_name)
            logger.info(f"사용 가능한 모델 {len(models)}개 발견: {models}")
            return models
        else:
            logger.warning(f"모델 목록 조회 실패, returncode: {result.returncode}")
            return []
    except Exception as e:
        logger.error(f"모델 목록 조회 중 오류: {str(e)}")
        return []

def main():
    """메인 함수"""
    logger.info("=== bizMOB 채팅 페이지 시작 ===")
    
    st.markdown('<h1 class="main-header">🤖 bizMOB Platform 챗봇</h1>', unsafe_allow_html=True)
    
    # 세션 초기화
    logger.info("세션 초기화 시작")
    initialize_session()
    logger.info("세션 초기화 완료")
    
    # 사이드바 - 간단한 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # Ollama 상태 확인
        if check_ollama():
            st.success("✅ Ollama 연결됨")
            
            # 모델 선택
            available_models = get_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "AI 모델 선택",
                    available_models,
                    index=0 if 'exaone3.5' in available_models else 0
                )
                st.session_state.selected_model = selected_model
                st.info(f"선택된 모델: {selected_model}")
            else:
                st.warning("⚠️ 사용 가능한 모델이 없습니다.")
        else:
            st.error("❌ Ollama 연결 실패")
            st.info("Ollama가 설치되어 있고 실행 중인지 확인해주세요.")
            return
        
        st.markdown("---")
        st.markdown("### 💡 사용법")
        st.markdown("""
        1. 질문을 입력하세요
        2. '질문하기' 버튼을 클릭하세요
        3. AI가 bizMOB Platform에 대해 답변합니다
        """)
    
    # 메인 채팅 인터페이스
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 채팅 메시지 표시
        display_chat()
        
        # 질문 입력
        st.markdown("---")
        user_question = st.text_area(
            "bizMOB Platform에 대해 질문해 주세요",
            placeholder="bizMOB Platform의 주요 기능은 무엇인가요?",
            key=f"input_{st.session_state.input_counter}",
            height=100
        )
        
        # 질문 제출 버튼
        if st.button("질문하기", type="primary", use_container_width=True):
            logger.info("질문하기 버튼 클릭됨")
            
            if user_question and user_question.strip():
                logger.info(f"사용자 질문 수신: {user_question.strip()}")
                
                # 사용자 메시지 추가
                add_message('user', user_question.strip())
                logger.info("사용자 메시지 추가 완료")
                
                # AI 응답 생성
                with st.spinner("답변을 생성하는 중..."):
                    logger.info("AI 응답 생성 시작")
                    response = simple_chat_response(user_question.strip())
                    logger.info(f"AI 응답 생성 완료: {response[:100]}...")
                    
                    add_message('assistant', response)
                    logger.info("AI 메시지 추가 완료")
                
                # 입력창 초기화
                st.session_state.input_counter += 1
                logger.info("입력창 초기화 완료, 페이지 새로고침")
                st.rerun()
            else:
                logger.warning("빈 질문 입력 시도")
                st.error("질문을 입력해주세요.")
    
    with col2:
        st.markdown("### 📊 채팅 정보")
        st.info(f"총 메시지: {len(st.session_state.chat_messages)}개")
        
        if st.session_state.chat_messages:
            user_count = len([m for m in st.session_state.chat_messages if m['role'] == 'user'])
            assistant_count = len([m for m in st.session_state.chat_messages if m['role'] == 'assistant'])
            
            st.metric("사용자 질문", user_count)
            st.metric("AI 답변", assistant_count)
        
        # 채팅 기록 초기화
        if st.button("🗑️ 채팅 기록 초기화", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 사용 팁")
        st.markdown("""
        - bizMOB Platform에 대한 질문을 자유롭게 해보세요
        - 구체적인 질문일수록 더 정확한 답변을 받을 수 있습니다
        - 채팅 기록은 브라우저 세션 동안 유지됩니다
        - 별도의 설정 없이 바로 사용 가능합니다
        """)

if __name__ == "__main__":
    main() 