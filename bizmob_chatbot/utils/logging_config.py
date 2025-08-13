"""
로깅 설정 모듈
"""

import logging
from datetime import datetime


def setup_logging():
    """로깅 설정 - 콘솔 로깅만 사용"""
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러만 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = None):
    """로거 인스턴스 반환"""
    return logging.getLogger(name or __name__) 