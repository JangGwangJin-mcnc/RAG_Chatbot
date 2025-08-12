#!/usr/bin/env python3
"""
캐시 초기화 스크립트
"""

import streamlit as st
import os
import sys

# 상위 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def clear_all_caches():
    """모든 캐시 초기화"""
    print("=== 캐시 초기화 시작 ===")
    
    # Streamlit 캐시 초기화
    try:
        # 세션 상태에서 캐시 관련 키들 제거
        cache_keys = []
        for key in list(st.session_state.keys()):
            if any(prefix in key for prefix in [
                'global_vector_store_',
                'vector_store_',
                'rag_chain_',
                'embedding_model_',
                'cached_'
            ]):
                cache_keys.append(key)
                del st.session_state[key]
        
        print(f"제거된 캐시 키들: {cache_keys}")
        
        # Streamlit 캐시 데코레이터 초기화
        st.cache_resource.clear()
        st.cache_data.clear()
        
        print("✅ Streamlit 캐시 초기화 완료")
        
    except Exception as e:
        print(f"❌ 캐시 초기화 중 오류: {e}")
    
    # ChromaDB 관련 파일들 정리
    try:
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            import shutil
            shutil.rmtree(chroma_path)
            print("✅ ChromaDB 디렉토리 삭제 완료")
        
        # 모델 정보 파일들 삭제
        model_info_files = [f for f in os.listdir('.') if f.startswith('vector_db_model_info') and f.endswith('.json')]
        for file in model_info_files:
            os.remove(file)
            print(f"✅ 모델 정보 파일 삭제: {file}")
            
    except Exception as e:
        print(f"❌ 파일 정리 중 오류: {e}")
    
    print("=== 캐시 초기화 완료 ===")

if __name__ == "__main__":
    clear_all_caches() 