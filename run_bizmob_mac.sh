#!/bin/bash

# bizMOB 챗봇 맥북 실행 스크립트
# 사용법: ./run_bizmob_mac.sh

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 스크립트 시작
echo "=========================================="
echo "🚀 bizMOB 챗봇 실행 스크립트"
echo "=========================================="

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_info "스크립트 위치: $SCRIPT_DIR"

# Python 가상환경 확인
if [ -d "$SCRIPT_DIR/bizmob_env" ]; then
    log_info "가상환경 발견: $SCRIPT_DIR/bizmob_env"
    VENV_PATH="$SCRIPT_DIR/bizmob_env"
elif [ -d "$HOME/bizmob_project/bizmob_env" ]; then
    log_info "가상환경 발견: $HOME/bizmob_project/bizmob_env"
    VENV_PATH="$HOME/bizmob_project/bizmob_env"
else
    log_warning "가상환경을 찾을 수 없습니다."
    log_info "가상환경을 생성하시겠습니까? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "가상환경 생성 중..."
        python3 -m venv "$SCRIPT_DIR/bizmob_env"
        VENV_PATH="$SCRIPT_DIR/bizmob_env"
        log_success "가상환경 생성 완료"
        
        # 의존성 설치
        log_info "의존성 설치 중..."
        source "$VENV_PATH/bin/activate"
        pip install --upgrade pip
        pip install -r "$SCRIPT_DIR/bizmob_chatbot/requirements_multiformat.txt"
        log_success "의존성 설치 완료"
    else
        log_error "가상환경이 필요합니다. 설치 가이드를 참조하세요."
        exit 1
    fi
fi

# Ollama 서비스 확인
log_info "Ollama 서비스 상태 확인 중..."
if ! brew services list | grep -q "ollama.*started"; then
    log_warning "Ollama 서비스가 실행되지 않았습니다."
    log_info "Ollama 서비스를 시작하시겠습니까? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Ollama 서비스 시작 중..."
        brew services start ollama
        sleep 3
        log_success "Ollama 서비스 시작 완료"
    else
        log_error "Ollama 서비스가 필요합니다."
        exit 1
    fi
else
    log_success "Ollama 서비스가 실행 중입니다."
fi

# 가상환경 활성화
log_info "가상환경 활성화 중..."
source "$VENV_PATH/bin/activate"
log_success "가상환경 활성화 완료"

# bizmob_chatbot 디렉토리로 이동
if [ -d "$SCRIPT_DIR/bizmob_chatbot" ]; then
    cd "$SCRIPT_DIR/bizmob_chatbot"
    log_info "작업 디렉토리: $(pwd)"
else
    log_error "bizmob_chatbot 디렉토리를 찾을 수 없습니다."
    exit 1
fi

# Streamlit 실행
log_info "bizMOB 챗봇 시작 중..."
log_info "웹 브라우저에서 http://localhost:8080 으로 접속하세요."
log_info "종료하려면 Ctrl+C를 누르세요."
echo ""

# Streamlit 실행
streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0 