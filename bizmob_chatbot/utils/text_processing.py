"""
텍스트 처리 유틸리티
"""

import re
import html


def clean_text(text: str) -> str:
    """텍스트 정제"""
    if not text:
        return ""
    
    # HTML 태그 제거
    text = remove_html_tags(text)
    
    # 특수 문자 이스케이프
    text = html.escape(text)
    
    # 줄바꿈을 <br> 태그로 변환
    text = text.replace('\n', '<br>')
    
    return text


def remove_html_tags(text: str) -> str:
    """HTML 태그 제거"""
    if not text:
        return ""
    
    # HTML 태그 제거
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text


def format_chat_message(content: str, role: str, timestamp: str) -> str:
    """채팅 메시지 HTML 포맷팅 - 카카오톡 스타일"""
    content = clean_text(content)
    
    if role == 'user':
        # 사용자 메시지 (오른쪽, 파란색)
        return f"""
        <div style="display: flex; justify-content: flex-end; margin: 4px 0;">
            <div style="background-color: #007AFF; color: white; padding: 6px 10px; border-radius: 15px; display: inline-block; max-width: 60%; word-wrap: break-word; position: relative;">
                <div style="margin-bottom: 2px;">{content}</div>
                <div style="font-size: 0.6em; opacity: 0.8; text-align: right;">{timestamp}</div>
            </div>
        </div>
        """
    else:
        # AI 메시지 (왼쪽, 회색, 뾰족한 꼬리)
        return f"""
        <div style="display: flex; justify-content: flex-start; margin: 4px 0;">
            <div style="background-color: #F0F0F0; color: black; padding: 6px 10px; border-radius: 15px; display: inline-block; max-width: 60%; word-wrap: break-word; position: relative;">
                <div style="margin-bottom: 2px;">{content}</div>
                <div style="font-size: 0.6em; opacity: 0.7;">{timestamp}</div>
                <!-- 왼쪽 뾰족한 꼬리 -->
                <div style="position: absolute; left: -6px; top: 8px; width: 0; height: 0; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 6px solid #F0F0F0;"></div>
            </div>
        </div>
        """ 