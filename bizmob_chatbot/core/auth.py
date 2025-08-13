"""
인증 및 권한 관리 모듈
"""

import streamlit as st
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from config.settings import settings


class AuthManager:
    """인증 및 권한 관리"""
    
    def __init__(self):
        self.admin_password = settings.admin_password
    
    def is_admin(self) -> bool:
        """관리자 권한 확인"""
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
        
        return st.session_state.is_admin
    
    def login(self, password: str) -> bool:
        """관리자 로그인"""
        if password == self.admin_password:
            st.session_state.is_admin = True
            return True
        return False
    
    def logout(self):
        """로그아웃"""
        st.session_state.is_admin = False
    
    def require_admin(self):
        """관리자 권한 필요 시 호출"""
        if not self.is_admin():
            st.error("관리자 권한이 필요합니다.")
            st.stop()
    
    def get_user_role(self) -> str:
        """사용자 역할 반환"""
        return "admin" if self.is_admin() else "user"
    
    def can_access_admin_features(self) -> bool:
        """관리자 기능 접근 권한 확인"""
        return self.is_admin()
    
    def can_upload_files(self) -> bool:
        """파일 업로드 권한 확인"""
        return self.is_admin()
    
    def can_manage_vector_db(self) -> bool:
        """벡터 DB 관리 권한 확인"""
        return self.is_admin() 