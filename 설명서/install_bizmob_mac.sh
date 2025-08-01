#!/bin/bash
# bizMOB 챗봇 맥북 자동 설치 스크립트

set -e  # 오류 발생 시 스크립트 중단

echo "=== bizMOB 챗봇 맥북 설치 시작 ==="
echo "설치 시간: $(date)"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
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

# 1. 시스템 확인
log_info "시스템 정보 확인 중..."
echo "macOS 버전: $(sw_vers -productVersion)"
echo "아키텍처: $(uname -m)"
echo "사용자: $(whoami)"
echo ""

# 2. Homebrew 설치 확인 및 설치
log_info "Homebrew 설치 확인 중..."
if ! command -v brew &> /dev/null; then
    log_info "Homebrew 설치 중..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Apple Silicon Mac의 경우 PATH 설정
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        source ~/.zshrc
        log_info "Apple Silicon Mac용 PATH 설정 완료"
    fi
    log_success "Homebrew 설치 완료"
else
    log_success "Homebrew가 이미 설치되어 있습니다"
fi

# Homebrew 업데이트
log_info "Homebrew 업데이트 중..."
brew update
log_success "Homebrew 업데이트 완료"

# 3. Python 설치
log_info "Python 3.9 설치 중..."
brew install python@3.9
log_success "Python 3.9 설치 완료"

# Python 경로 확인
PYTHON_PATH=$(which python3)
log_info "Python 경로: $PYTHON_PATH"
log_info "Python 버전: $(python3 --version)"

# 4. Git 설치
log_info "Git 설치 확인 중..."
if ! command -v git &> /dev/null; then
    log_info "Git 설치 중..."
    brew install git
    log_success "Git 설치 완료"
else
    log_success "Git이 이미 설치되어 있습니다"
fi

# 5. Ollama 설치
log_info "Ollama 설치 중..."
brew install ollama
log_success "Ollama 설치 완료"

# Ollama 서비스 시작
log_info "Ollama 서비스 시작 중..."
brew services start ollama
sleep 5  # 서비스 시작 대기

# Ollama 상태 확인
if brew services list | grep ollama | grep started > /dev/null; then
    log_success "Ollama 서비스가 정상적으로 시작되었습니다"
else
    log_error "Ollama 서비스 시작 실패"
    exit 1
fi

# 6. 프로젝트 설정
log_info "프로젝트 디렉토리 설정 중..."
cd ~
mkdir -p bizmob_project
cd bizmob_project
log_success "프로젝트 디렉토리 생성 완료"

# 7. 가상환경 생성
log_info "Python 가상환경 생성 중..."
python3 -m venv bizmob_env
source bizmob_env/bin/activate
log_success "가상환경 생성 및 활성화 완료"

# pip 업그레이드
log_info "pip 업그레이드 중..."
pip install --upgrade pip
log_success "pip 업그레이드 완료"

# 8. 프로젝트 파일 복사 (현재 디렉토리에서)
log_info "프로젝트 파일 복사 중..."
if [ -d "/c%3A/Users/alfus/OneDrive/Desktop/my_ai_project-main" ]; then
    cp -r "/c%3A/Users/alfus/OneDrive/Desktop/my_ai_project-main"/* .
    log_success "프로젝트 파일 복사 완료"
else
    log_warning "프로젝트 파일을 찾을 수 없습니다. 수동으로 복사해주세요."
    echo "프로젝트 파일을 ~/bizmob_project/ 디렉토리에 복사한 후 계속하세요."
    read -p "파일 복사가 완료되었으면 Enter를 눌러주세요..."
fi

# 9. 의존성 패키지 설치
log_info "의존성 패키지 설치 중..."
if [ -f "bizmob_chatbot/requirements_multiformat.txt" ]; then
    pip install -r bizmob_chatbot/requirements_multiformat.txt
    log_success "기본 의존성 패키지 설치 완료"
else
    log_error "requirements_multiformat.txt 파일을 찾을 수 없습니다"
    exit 1
fi

# 추가 패키지 설치
log_info "추가 패키지 설치 중..."
pip install python-dotenv psutil requests
log_success "추가 패키지 설치 완료"

# 10. 환경 설정
log_info "환경 설정 중..."
cat > .env << EOF
# 토크나이저 병렬화 비활성화
TOKENIZERS_PARALLELISM=false

# Ollama 서버 주소
OLLAMA_HOST=http://localhost:11434

# Streamlit 설정
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 로그 레벨
LOG_LEVEL=INFO
EOF
log_success ".env 파일 생성 완료"

# 필요한 디렉토리 생성
log_info "필요한 디렉토리 생성 중..."
mkdir -p PDF_bizMOB_Guide
mkdir -p external_sources
mkdir -p bizmob_faiss_index_hyperclovax
mkdir -p PDF_이미지_bizmob
log_success "디렉토리 생성 완료"

# 권한 설정
log_info "권한 설정 중..."
chmod +x bizmob_chatbot/bizmob_chatbot.py
chmod 755 PDF_bizMOB_Guide external_sources
log_success "권한 설정 완료"

# 11. AI 모델 다운로드
log_info "AI 모델 다운로드 중..."
ollama pull hyperclovax
log_success "hyperclovax 모델 다운로드 완료"

# 12. 설치 확인
log_info "설치 확인 중..."

# Python 패키지 확인
if python -c "import streamlit, langchain, faiss, openai" 2>/dev/null; then
    log_success "주요 Python 패키지 설치 확인 완료"
else
    log_error "일부 Python 패키지 설치에 문제가 있습니다"
fi

# Ollama 연결 확인
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log_success "Ollama 연결 확인 완료"
else
    log_error "Ollama 연결에 문제가 있습니다"
fi

# 13. 설치 완료 메시지
echo ""
echo "=========================================="
log_success "bizMOB 챗봇 맥북 설치가 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 명령어로 애플리케이션을 실행하세요:"
echo ""
echo "1. 프로젝트 디렉토리로 이동:"
echo "   cd ~/bizmob_project"
echo ""
echo "2. 가상환경 활성화:"
echo "   source bizmob_env/bin/activate"
echo ""
echo "3. 애플리케이션 실행:"
echo "   cd bizmob_chatbot"
echo "   streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0"
echo ""
echo "4. 웹 브라우저에서 접속:"
echo "   http://localhost:8080"
echo ""
echo "설치 시간: $(date)"
echo ""

# 실행 스크립트 생성
log_info "실행 스크립트 생성 중..."
cat > ~/bizmob_project/run_bizmob.sh << 'EOF'
#!/bin/bash
cd ~/bizmob_project
source bizmob_env/bin/activate
cd bizmob_chatbot
streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0
EOF

chmod +x ~/bizmob_project/run_bizmob.sh
log_success "실행 스크립트 생성 완료: ~/bizmob_project/run_bizmob.sh"

echo "간편 실행을 위해 다음 명령어를 사용할 수도 있습니다:"
echo "   ~/bizmob_project/run_bizmob.sh"
echo ""

log_success "설치 과정이 모두 완료되었습니다!" 