"""
채팅 인터페이스 UI 컴포넌트
"""

import streamlit as st
from datetime import datetime
import sys
import os
import time
import threading
import logging

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from utils.text_processing import format_chat_message
from core.chat_engine import ChatEngine

# 로깅 설정
logger = logging.getLogger(__name__)


class ChatInterface:
    """채팅 인터페이스 관리"""
    
    def __init__(self):
        self.setup_chat_history()
        # ChatEngine을 한 번만 초기화
        if 'chat_engine' not in st.session_state:
            st.session_state.chat_engine = ChatEngine()
            logger.info("ChatInterface 초기화 완료")
        self.chat_engine = st.session_state.chat_engine
        
        # VectorDBManager 추가
        from core.vector_db_manager import VectorDBManager
        self.vector_db_manager = VectorDBManager()
    
    def setup_chat_history(self):
        """채팅 히스토리 초기화"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'is_loading' not in st.session_state:
            st.session_state.is_loading = False
        if 'loading_dots' not in st.session_state:
            st.session_state.loading_dots = 0
        if 'chat_interface_initialized' not in st.session_state:
            st.session_state.chat_interface_initialized = True
            logger.info("채팅 히스토리 초기화 완료")
    
    def display_chat_messages(self):
        """채팅 메시지 표시"""
        # 채팅 컨테이너 스타일
        st.markdown("""
        <style>
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #007AFF;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 채팅 메시지 영역
        st.markdown("### 채팅 기록")
        
        # 채팅 히스토리가 없을 때 안내 메시지
        if not st.session_state.chat_history:
            st.info("채팅을 시작해보세요!")
            return
        
        # 메시지들 표시
        for i, message in enumerate(st.session_state.chat_history):
            content = message['content']
            role = message['role']
            timestamp = message['timestamp']
            
            # HTML 포맷팅
            formatted_message = format_chat_message(content, role, timestamp)
            st.markdown(formatted_message, unsafe_allow_html=True)
        
        # 로딩 상태 표시
        if st.session_state.is_loading:
            self.display_loading_message()
    
    def display_loading_message(self):
        """로딩 메시지 표시 - 실제 빙글빙글 애니메이션"""
        # 로딩 점 애니메이션
        dots = "." * (st.session_state.loading_dots % 4)
        
        loading_html = f"""
        <div style="display: flex; justify-content: flex-start; margin: 4px 0;">
            <div style="background-color: #F0F0F0; color: black; padding: 6px 10px; border-radius: 15px; display: inline-block; max-width: 60%; word-wrap: break-word; position: relative;">
                <div style="margin-bottom: 2px; display: flex; align-items: center;">
                    <div class="loading-spinner"></div>
                    <span>챗봇이 정보를 찾고 있습니다{dots}</span>
                </div>
                <div style="font-size: 0.6em; opacity: 0.7;">응답 생성 중...</div>
                <!-- 왼쪽 뾰족한 꼬리 -->
                <div style="position: absolute; left: -6px; top: 8px; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 6px solid #F0F0F0;"></div>
            </div>
        </div>
        """
        st.markdown(loading_html, unsafe_allow_html=True)
    
    def add_user_message(self, content: str):
        """사용자 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M")
        message = {
            'role': 'user',
            'content': content,
            'timestamp': timestamp
        }
        st.session_state.chat_history.append(message)
        logger.info(f"사용자 메시지 추가: '{content}' ({timestamp})")
    
    def add_ai_message(self, content: str):
        """AI 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M")
        message = {
            'role': 'assistant',
            'content': content,
            'timestamp': timestamp
        }
        st.session_state.chat_history.append(message)
        logger.info(f"AI 메시지 추가: '{content[:50]}...' ({timestamp})")
    
    def clear_chat_history(self):
        """채팅 히스토리 초기화"""
        st.session_state.chat_history = []
        logger.info("채팅 히스토리 초기화 완료")
    
    def get_chat_input(self) -> str:
        """채팅 입력 받기"""
        user_input = st.chat_input("메시지를 입력하세요...")
        return user_input
    
    def process_user_input(self, user_input: str):
        """사용자 입력 처리 및 응답 생성"""
        if not user_input:
            return
        
        logger.info(f"사용자 입력 처리 시작: '{user_input}'")
        
        # 사용자 메시지 추가
        self.add_user_message(user_input)
        
        # 로딩 상태 시작
        st.session_state.is_loading = True
        st.session_state.loading_dots = 0
        logger.info("로딩 상태 시작")
        
        # 실제 응답 생성 (동기적으로 처리)
        try:
            logger.info("ChatEngine에 응답 생성 요청")
            
            # RAG 처리 과정 로깅
            logger.info("=== RAG 처리 과정 시작 ===")
            ai_response = self.chat_engine.generate_response(user_input)
            logger.info("=== RAG 처리 과정 완료 ===")
            
            # AI 응답 추가
            logger.info(f"응답 생성 완료: '{ai_response[:50]}...'")
            self.add_ai_message(ai_response)
            logger.info(f"AI 메시지 추가 완료: '{ai_response[:50]}...'")
            
            # 로딩 상태 종료
            st.session_state.is_loading = False
            logger.info("로딩 상태 종료 완료")
            
            # 강제 리렌더링
            st.rerun()
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류: {str(e)}")
            st.session_state.is_loading = False
            st.error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")
            st.rerun()
    
    def display_chat_input_area(self):
        """채팅 입력 영역 표시"""
        # 채팅 메시지 표시
        self.display_chat_messages()
        
        # 로딩 상태 확인 및 업데이트
        if st.session_state.is_loading:
            st.session_state.loading_dots += 1
            if st.session_state.loading_dots > 10:  # 5초 후 타임아웃
                st.session_state.is_loading = False
                logger.warning("로딩 타임아웃 - 응답이 너무 오래 걸립니다")
        
        # 입력 받기
        user_input = self.get_chat_input()
        
        if user_input:
            logger.info(f"사용자 입력 받음: '{user_input}'")
            self.process_user_input(user_input)
        
        return user_input
    
    def display_chat_stats(self):
        """채팅 통계 표시"""
        if st.session_state.chat_history:
            st.markdown("### 채팅 통계")
            total_messages = len(st.session_state.chat_history)
            user_messages = len([m for m in st.session_state.chat_history if m['role'] == 'user'])
            ai_messages = len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
            
            logger.info(f"채팅 통계 - 총: {total_messages}, 사용자: {user_messages}, AI: {ai_messages}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 메시지", total_messages)
            with col2:
                st.metric("사용자 메시지", user_messages)
            with col3:
                st.metric("AI 메시지", ai_messages) 