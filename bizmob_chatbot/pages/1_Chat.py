#!/usr/bin/env python3
"""
bizMOB 간단 채팅 페이지 - RAG 기반 답변 생성
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime
import subprocess

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# bizmob_chatbot 모듈 import
try:
    from bizmob_chatbot import (
        get_cached_rag_chain,
        get_cached_vector_store,
        get_hybrid_retriever,
        process_question,
        get_hybrid_search_config,
        update_hybrid_search_config
    )
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"RAG 모듈 import 실패: {e}")
    RAG_AVAILABLE = False

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"chat_page_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    
    return logger

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
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
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
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        border-bottom-right-radius: 0.3rem;
    }
    .chat-message-assistant {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        border-bottom-left-radius: 0.3rem;
    }
    .stButton > button {
        width: 100%;
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
    if 'hybrid_search_config' not in st.session_state:
        st.session_state.hybrid_search_config = {'bm25_weight': 0.3, 'vector_weight': 0.7}
    if 'last_retrieved_docs' not in st.session_state:
        st.session_state.last_retrieved_docs = []
    
    # 벡터 스토어 관련 세션 상태 자동 초기화
    auto_clear_vector_store_cache()

def auto_clear_vector_store_cache():
    """벡터 스토어 관련 세션 상태 자동 정리"""
    try:
        logger.info("=== 자동 벡터 스토어 캐시 정리 시작 ===")
        
        # 벡터 스토어 관련 캐시 키 찾기
        vector_cache_keys = []
        for key in list(st.session_state.keys()):
            if (key.startswith('vector_store_') or 
                key.startswith('global_vector_store_') or
                key.startswith('rag_chain_')):
                vector_cache_keys.append(key)
        
        # 캐시된 벡터 스토어 자동 제거
        for key in vector_cache_keys:
            del st.session_state[key]
            logger.info(f"자동 캐시 제거: {key}")
        
        # Streamlit 캐시 자동 무효화
        try:
            st.cache_resource.clear()
            logger.info("Streamlit 캐시 자동 무효화 완료")
        except Exception as e:
            logger.warning(f"Streamlit 캐시 자동 무효화 실패: {e}")
        
        # 정리된 캐시 수 로깅
        if vector_cache_keys:
            logger.info(f"자동 캐시 정리 완료: {len(vector_cache_keys)}개 키 제거")
        else:
            logger.info("정리할 벡터 스토어 캐시가 없습니다.")
        
        logger.info("=== 자동 벡터 스토어 캐시 정리 완료 ===")
        
    except Exception as e:
        logger.error(f"자동 캐시 정리 실패: {e}")
        # 오류가 발생해도 앱 실행은 계속

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
            
            # AI 응답인 경우 관련 문서 내용 표시
            if message['role'] == 'assistant' and 'last_retrieved_docs' in st.session_state:
                # 마지막 AI 응답의 문서인지 확인
                if len(st.session_state.chat_messages) > 0:
                    last_ai_index = None
                    for i, msg in enumerate(st.session_state.chat_messages):
                        if msg['role'] == 'assistant':
                            last_ai_index = i
                    
                    if last_ai_index == st.session_state.chat_messages.index(message):
                        st.markdown("---")
                        st.markdown("### 📚 **관련 문서 상세 내용**")
                        st.info("아래 문서를 클릭하면 자세한 내용을 볼 수 있습니다.")
                        
                        for i, doc in enumerate(st.session_state.last_retrieved_docs):
                            display_document_content(doc, i+1)

def rag_chat_response(user_question):
    """RAG 기반 채팅 응답 생성"""
    logger.info(f"=== RAG 채팅 응답 생성 시작: {user_question[:50]}... ===")
    
    if not RAG_AVAILABLE:
        logger.error("RAG 모듈을 사용할 수 없습니다")
        return "죄송합니다. RAG 기능을 사용할 수 없습니다."
    
    try:
        # 현재 하이브리드 검색 설정 가져오기
        current_config = st.session_state.hybrid_search_config
        bm25_weight = current_config['bm25_weight']
        vector_weight = current_config['vector_weight']
        logger.info(f"현재 하이브리드 검색 설정: BM25={bm25_weight:.2f}, Vector={vector_weight:.2f}")
        
        # RAG 설정을 현재 세션 설정으로 업데이트 (매번 검색 시마다)
        try:
            config = {
                'bm25_weight': bm25_weight,
                'vector_weight': vector_weight,
                'initial_k': 8,
                'final_k': 3,
                'enable_reranking': True,
                'metadata_boost': True,
                'recency_boost': True
            }
            update_hybrid_search_config(config)
            logger.info(f"RAG 하이브리드 검색 설정 업데이트 완료: BM25={bm25_weight:.2f}, Vector={vector_weight:.2f}")
        except Exception as e:
            logger.warning(f"RAG 설정 업데이트 실패, 기본값 사용: {e}")
        
        # RAG 체인으로 질문 처리 (안전한 벡터 스토어 사용)
        logger.info("RAG 체인으로 질문 처리 시작...")
        
        # 안전한 벡터 스토어 확인
        vector_store = safe_get_vector_store()
        if not vector_store:
            logger.warning("벡터 스토어를 가져올 수 없습니다. 기본 RAG 체인 사용")
        
        result = process_question(user_question)
        
        if result and len(result) == 2:
            response, retrieve_docs = result
            logger.info(f"RAG 응답 생성 성공, 길이: {len(response) if response else 0}")
            
            # 관련 문서 정보 추가 (클릭 가능한 형태로)
            if retrieve_docs:
                doc_info = "\n\n📚 **참고 문서:**\n"
                for i, doc in enumerate(retrieve_docs[:3]):
                    source = doc.metadata.get('source', 'Unknown')
                    title = doc.metadata.get('title', 'No Title')
                    relevance = doc.metadata.get('relevance_score', 'N/A')
                    doc_info += f"{i+1}. {title} (출처: {source}, 관련성: {relevance})\n"
                
                if response:
                    response += doc_info
                else:
                    response = doc_info
                logger.info(f"참고 문서 {len(retrieve_docs)}개 추가 완료")
                
                # 문서 내용을 세션에 저장하여 UI에서 표시할 수 있도록 함
                st.session_state.last_retrieved_docs = retrieve_docs[:3]
            
            if not response:
                response = "죄송합니다. bizMOB Platform에 대한 정보를 찾을 수 없습니다."
            
            return response
        else:
            logger.warning("RAG 응답이 비어있거나 잘못된 형식입니다")
            return "죄송합니다. bizMOB Platform에 대한 정보를 찾을 수 없습니다."
            
    except Exception as e:
        logger.error(f"RAG 응답 생성 실패: {e}", exc_info=True)
        return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

def fallback_chat_response(user_question):
    """폴백: Ollama 직접 호출"""
    logger.info(f"=== 폴백 채팅 응답 생성: {user_question[:50]}... ===")
    
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
        
        logger.info(f"폴백 프롬프트 구성 완료, 모델: {st.session_state.selected_model}")
        
        # Ollama 호출
        logger.info("Ollama 호출 시작...")
        result = subprocess.run([
            'ollama', 'run', st.session_state.selected_model, prompt
        ], capture_output=True, text=True, timeout=120, encoding='utf-8')
        
        logger.info(f"Ollama 호출 완료, returncode: {result.returncode}")
        
        if result.returncode == 0:
            if result.stdout is not None:
                response = result.stdout.strip()
                if response:
                    logger.info(f"폴백 응답 생성 성공, 길이: {len(response)}")
                    return response
                else:
                    logger.warning("폴백 응답이 비어있습니다")
                    return "죄송합니다. 응답을 생성할 수 없습니다."
            else:
                logger.warning("stdout이 None입니다")
                return "죄송합니다. 응답을 생성할 수 없습니다."
        else:
            error_msg = f"모델 호출 중 오류가 발생했습니다: {result.stderr}"
            logger.error(f"Ollama 오류: {result.stderr}")
            return error_msg
            
    except subprocess.TimeoutExpired:
        logger.error("응답 시간 초과 (2분)")
        return "응답 시간이 초과되었습니다 (2분). 다시 시도해주세요."
    except Exception as e:
        logger.error(f"폴백 응답 생성 중 예상치 못한 오류: {str(e)}", exc_info=True)
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

def update_rag_config():
    """RAG 하이브리드 검색 설정 업데이트"""
    if RAG_AVAILABLE:
        try:
            current_config = st.session_state.hybrid_search_config
            config = {
                'bm25_weight': current_config['bm25_weight'],
                'vector_weight': current_config['vector_weight'],
                'initial_k': 8,
                'final_k': 3,
                'enable_reranking': True,
                'metadata_boost': True,
                'recency_boost': True
            }
            update_hybrid_search_config(config)
            logger.info(f"RAG 설정 자동 업데이트 완료: BM25={current_config['bm25_weight']:.2f}, Vector={current_config['vector_weight']:.2f}")
        except Exception as e:
            logger.warning(f"RAG 설정 자동 업데이트 실패: {e}")

def get_available_models():
    """사용 가능한 모델 목록 가져오기"""
    logger.info("사용 가능한 모델 목록 조회 시작")
    
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            values = []
            
            for line in lines[1:]:  # 첫 번째 줄은 헤더이므로 제외
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        model_name = parts[0]
                        values.append(model_name)
            logger.info(f"사용 가능한 모델 {len(values)}개 발견: {values}")
            return values
        else:
            logger.warning(f"모델 목록 조회 실패, returncode: {result.returncode}")
            return []
    except Exception as e:
        logger.error(f"모델 목록 조회 중 오류: {str(e)}")
        return []

def safe_get_vector_store():
    """안전한 벡터 스토어 가져오기 (자동 복구 포함)"""
    try:
        logger.info("=== 안전한 벡터 스토어 가져오기 시작 ===")
        
        # 먼저 자동 캐시 정리 수행
        auto_clear_vector_store_cache()
        
        # 벡터 스토어 가져오기
        from bizmob_chatbot import get_cached_vector_store
        vector_store = get_cached_vector_store()
        
        if vector_store:
            logger.info("벡터 스토어 가져오기 성공")
            return vector_store
        else:
            logger.warning("벡터 스토어 가져오기 실패")
            return None
            
    except Exception as e:
        logger.error(f"안전한 벡터 스토어 가져오기 실패: {e}")
        return None

def display_document_content(doc, doc_index):
    """개별 문서 내용 표시"""
    with st.expander(f"📄 {doc_index}. {doc.metadata.get('title', 'No Title')} (출처: {doc.metadata.get('source', 'Unknown')})", expanded=False):
        # 문서 메타데이터 표시
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**출처:** {doc.metadata.get('source', 'Unknown')}")
        with col2:
            st.info(f"**관련성:** {doc.metadata.get('relevance_score', 'N/A')}")
        
        # 추가 메타데이터가 있으면 표시
        if 'page' in doc.metadata:
            st.info(f"**페이지:** {doc.metadata['page']}")
        if 'chunk_id' in doc.metadata:
            st.info(f"**청크 ID:** {doc.metadata['chunk_id']}")
        
        # 문서 내용 표시
        st.markdown("### 📝 문서 내용")
        st.markdown(f"```\n{doc.page_content}\n```")
        
        # 원본 파일 다운로드 버튼 (가능한 경우)
        source = doc.metadata.get('source', '')
        if source and os.path.exists(source):
            try:
                with open(source, 'rb') as f:
                    file_bytes = f.read()
                    file_name = os.path.basename(source)
                    st.download_button(
                        label=f"📥 {file_name} 다운로드",
                        data=file_bytes,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.warning(f"파일 다운로드 준비 실패: {e}")
        else:
            st.warning("원본 파일을 찾을 수 없습니다.")

def main():
    """메인 함수"""
    logger.info("=== bizMOB RAG 채팅 페이지 시작 ===")
    
    # 페이지 진입 시 자동 세션 초기화
    logger.info("페이지 진입 시 자동 세션 초기화 시작")
    auto_clear_vector_store_cache()
    logger.info("페이지 진입 시 자동 세션 초기화 완료")
    
    st.markdown('<h1 class="main-header">🤖 bizMOB Platform RAG 챗봇</h1>', unsafe_allow_html=True)
    
    # 세션 초기화
    logger.info("세션 초기화 시작")
    initialize_session()
    logger.info("세션 초기화 완료")
    
    # 사이드바 - 설정 및 정보
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # RAG 상태 표시
        if RAG_AVAILABLE:
            st.success("✅ RAG 기능 사용 가능")
        else:
            st.warning("⚠️ RAG 기능 사용 불가")
        
        # 하이브리드 검색 설정
        st.markdown("#### 🔍 하이브리드 검색 설정")
        
        # BM25 가중치 슬라이더
        bm25_weight = st.slider(
            "BM25 가중치",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.hybrid_search_config['bm25_weight'],
            step=0.1,
            key="bm25_weight_slider"
        )
        
        # Vector 가중치 자동 계산 (BM25와 합이 1.0이 되도록)
        vector_weight = round(1.0 - bm25_weight, 1)
        
        # Vector 가중치 표시 (자동 동기화)
        st.info(f"🔗 **벡터 가중치: {vector_weight:.1f}** (자동 동기화)")
        
        # 가중치 합계 표시 (항상 1.0이어야 함)
        total_weight = bm25_weight + vector_weight
        st.success(f"✅ 가중치 합계: {total_weight:.1f} (자동 동기화)")
        
        # 세션 상태 업데이트
        st.session_state.hybrid_search_config = {
            'bm25_weight': bm25_weight,
            'vector_weight': vector_weight
        }
        
        # RAG 설정 자동 업데이트 (가중치가 변경될 때마다)
        if RAG_AVAILABLE:
            try:
                config = {
                    'bm25_weight': bm25_weight,
                    'vector_weight': vector_weight,
                    'initial_k': 8,
                    'final_k': 3,
                    'enable_reranking': True,
                    'metadata_boost': True,
                    'recency_boost': True
                }
                update_hybrid_search_config(config)
                st.success(f"✅ RAG 설정이 자동으로 업데이트되었습니다! (BM25: {bm25_weight:.1f}, Vector: {vector_weight:.1f})")
                logger.info(f"RAG 설정 자동 업데이트 완료: BM25={bm25_weight:.2f}, Vector={vector_weight:.2f}")
            except Exception as e:
                st.warning(f"⚠️ RAG 설정 자동 업데이트 실패: {e}")
                logger.error(f"RAG 설정 자동 업데이트 실패: {e}")
        else:
            st.success(f"✅ 하이브리드 검색 설정이 저장되었습니다! (BM25: {bm25_weight:.1f}, Vector: {vector_weight:.1f})")
        
        # 현재 설정 상태 표시
        st.info(f"**현재 설정:** BM25: {bm25_weight:.1f}, Vector: {vector_weight:.1f}")
        
        # 설정 변경 안내
        st.caption("💡 BM25 슬라이더를 조정하면 Vector 가중치가 자동으로 동기화되고 RAG 설정이 업데이트됩니다.")
        
        # 디버깅 정보 표시 (개발용)
        if st.checkbox("🔍 디버깅 정보 표시", key="debug_info"):
            st.json(st.session_state.hybrid_search_config)
            
            # 가중치 테스트 버튼
            if st.button("🧪 가중치 테스트", key="test_weights"):
                try:
                    from bizmob_chatbot import test_hybrid_search, get_cached_vector_store
                    vector_store = get_cached_vector_store()
                    if vector_store:
                        test_result = test_hybrid_search(
                            "bizMOB 4.0", 
                            vector_store,
                            bm25_weight=bm25_weight,
                            vector_weight=vector_weight
                        )
                        st.success(f"테스트 완료! 검색 시간: {test_result.get('search_time', 'N/A')}초")
                        st.json(test_result)
                    else:
                        st.warning("벡터 스토어를 사용할 수 없습니다.")
                except Exception as e:
                    st.error(f"테스트 실패: {e}")
        
        # 설정 저장 버튼 (수동 저장용)
        if st.button("💾 수동 설정 저장", use_container_width=True):
            st.session_state.hybrid_search_config = {
                'bm25_weight': bm25_weight,
                'vector_weight': vector_weight
            }
            
            # RAG 설정 업데이트
            if RAG_AVAILABLE:
                try:
                    config = {
                        'bm25_weight': bm25_weight,
                        'vector_weight': vector_weight,
                        'initial_k': 8,
                        'final_k': 3,
                        'enable_reranking': True,
                        'metadata_boost': True,
                        'recency_boost': True
                    }
                    update_hybrid_search_config(config)
                    st.success("하이브리드 검색 설정이 저장되었습니다!")
                except Exception as e:
                    st.warning(f"RAG 설정 업데이트 실패: {e}")
            else:
                st.success("하이브리드 검색 설정이 저장되었습니다!")
        
        st.markdown("---")
        
        # AI 모델 설정
        st.markdown("#### 🤖 AI 모델 설정")
        
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
        
        st.markdown("---")
        
        # 채팅 정보
        st.markdown("#### 📊 채팅 정보")
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
    st.markdown("### 💡 사용법")
    st.markdown("""
    1. 질문을 입력하세요
    2. '질문하기' 버튼을 클릭하세요
    3. AI가 bizMOB Platform 문서를 기반으로 답변합니다
    4. RAG 기능으로 관련 문서를 참고하여 정확한 답변을 제공합니다
    """)

    # 메인 채팅 인터페이스
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
            
            # AI 응답 생성 (RAG 우선, 폴백으로 Ollama 직접 호출)
            with st.spinner("답변을 생성하는 중..."):
                logger.info("AI 응답 생성 시작")
                
                if RAG_AVAILABLE:
                    logger.info("RAG 기반 응답 생성 시도...")
                    response = rag_chat_response(user_question.strip())
                else:
                    logger.info("RAG 사용 불가, 폴백 응답 생성...")
                    response = fallback_chat_response(user_question.strip())
                
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

    # 하단 안내
    st.markdown("---")
    st.markdown("### 💡 사용 팁")
    st.markdown("""
    - bizMOB Platform에 대한 질문을 자유롭게 해보세요
    - RAG 기능으로 관련 문서를 기반으로 정확한 답변을 제공합니다
    - 구체적인 질문일수록 더 정확한 답변을 받을 수 있습니다
    - 채팅 기록은 브라우저 세션 동안 유지됩니다
    """)

if __name__ == "__main__":
    main() 