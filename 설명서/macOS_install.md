# bizMOB 챗봇 맥북 설치 가이드

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [개발 도구 설치](#개발-도구-설치)
3. [Ollama 설치](#ollama-설치)
4. [프로젝트 설정](#프로젝트-설정)
5. [의존성 패키지 설치](#의존성-패키지-설치)
6. [환경 설정](#환경-설정)
7. [애플리케이션 실행](#애플리케이션-실행)
8. [문제 해결](#문제-해결)
9. [접속 및 사용](#접속-및-사용)

## 시스템 요구사항

### 최소 사양
- **OS**: macOS 10.15 (Catalina) 이상
- **CPU**: Intel Core i5 또는 Apple Silicon M1 이상
- **RAM**: 8GB 이상 (16GB 권장)
- **저장공간**: 20GB 이상
- **Python**: 3.9 이상

### 권장 사양
- **CPU**: Apple Silicon M1 Pro/Max 또는 Intel Core i7 이상
- **RAM**: 16GB 이상
- **저장공간**: 50GB 이상 (모델 다운로드용)
- **macOS**: 12.0 (Monterey) 이상

## 개발 도구 설치

### 1. Homebrew 설치
```bash
# Homebrew 설치 (이미 설치되어 있다면 건너뛰기)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# PATH 설정 (Apple Silicon Mac의 경우)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Homebrew 업데이트
brew update
```

### 2. Python 설치
```bash
# Python 3.9 설치
brew install python@3.9

# Python 경로 확인
python3 --version
which python3
```

### 3. Git 설치
```bash
# Git 설치 (이미 설치되어 있을 수 있음)
brew install git

# Git 설정
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 4. 추가 도구 설치
```bash
# 유용한 도구들 설치
brew install wget
brew install tree
brew install htop
```

## Ollama 설치

### 1. Ollama 다운로드 및 설치
```bash
# Ollama 설치
brew install ollama

# 또는 공식 설치 스크립트 사용
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Ollama 서비스 시작
```bash
# Ollama 시작
ollama serve

# 백그라운드에서 실행하려면
brew services start ollama
```

### 3. 기본 모델 다운로드
```bash
# hyperclovax 모델 다운로드 (권장)
ollama pull hyperclovax

# 또는 다른 모델들
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull gemma:2b
```

### 4. Ollama 연결 테스트
```bash
# 모델 목록 확인
ollama list

# 간단한 테스트
ollama run hyperclovax "안녕하세요"
```

## 프로젝트 설정

### 1. 프로젝트 디렉토리 생성
```bash
# 홈 디렉토리로 이동
cd ~

# 프로젝트 디렉토리 생성
mkdir bizmob_project
cd bizmob_project
```

### 2. 가상환경 생성
```bash
# Python 가상환경 생성
python3 -m venv bizmob_env

# 가상환경 활성화
source bizmob_env/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

### 3. 프로젝트 파일 복사
```bash
# 프로젝트 파일들을 현재 디렉토리로 복사
# (Git에서 클론하거나 파일을 직접 복사)

# 예시: Git에서 클론하는 경우
# git clone https://github.com/your-repo/bizmob-chatbot.git
# cd bizmob-chatbot

# 또는 파일을 직접 복사하는 경우
# cp -r /path/to/your/project/* .
```

## 의존성 패키지 설치

### 1. 기본 의존성 설치
```bash
# requirements.txt 설치
pip install -r bizmob_chatbot/requirements_multiformat.txt
```

### 2. 추가 패키지 설치
```bash
# 환경 변수 관리
pip install python-dotenv

# 추가 의존성 (필요시)
pip install psutil
pip install requests
```

### 3. 설치 확인
```bash
# 설치된 패키지 목록 확인
pip list

# Python 경로 확인
which python
```

## 환경 설정

### 1. 환경 변수 파일 생성
```bash
# .env 파일 생성
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
```

### 2. 디렉토리 구조 생성
```bash
# 필요한 디렉토리들 생성
mkdir -p PDF_bizMOB_Guide
mkdir -p external_sources
mkdir -p bizmob_faiss_index_hyperclovax
mkdir -p PDF_이미지_bizmob
```

### 3. 권한 설정
```bash
# 실행 권한 부여
chmod +x bizmob_chatbot/bizmob_chatbot.py

# 디렉토리 권한 설정
chmod 755 PDF_bizMOB_Guide
chmod 755 external_sources
```

## 애플리케이션 실행

### 1. 기본 실행
```bash
# bizmob_chatbot 디렉토리로 이동
cd bizmob_chatbot

# Streamlit 앱 실행
streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0
```

### 2. 백그라운드 실행
```bash
# nohup을 사용한 백그라운드 실행
nohup streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# 프로세스 확인
ps aux | grep streamlit
```

### 3. 방화벽 설정 (필요시)
```bash
# macOS 방화벽에서 8080 포트 허용
# 시스템 환경설정 > 보안 및 개인 정보 보호 > 방화벽 > 방화벽 옵션
# 또는 터미널에서:
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
```

## 문제 해결

### 1. 메모리 부족 문제
```bash
# 메모리 사용량 확인
top -l 1 | grep PhysMem

# 활성 프로세스 확인
ps aux | grep -E "(streamlit|ollama|python)" | head -10
```

### 2. 포트 충돌 문제
```bash
# 포트 사용 확인
lsof -i :8080

# 다른 포트 사용
streamlit run bizmob_chatbot.py --server.port 8081
```

### 3. 권한 문제
```bash
# 파일 권한 확인
ls -la bizmob_chatbot/

# 권한 수정
chmod 755 bizmob_chatbot/
chmod +x bizmob_chatbot/bizmob_chatbot.py
```

### 4. Python 패키지 문제
```bash
# 가상환경 재활성화
deactivate
source bizmob_env/bin/activate

# 패키지 재설치
pip install --force-reinstall -r bizmob_chatbot/requirements_multiformat.txt
```

### 5. Ollama 연결 문제
```bash
# Ollama 서비스 상태 확인
brew services list | grep ollama

# Ollama 재시작
brew services restart ollama

# 연결 테스트
curl http://localhost:11434/api/tags
```

### 6. Apple Silicon 관련 문제
```bash
# Rosetta 2 설치 (Intel 앱 실행용)
softwareupdate --install-rosetta

# Python 패키지 재설치 (Apple Silicon 최적화)
pip install --force-reinstall --no-cache-dir -r bizmob_chatbot/requirements_multiformat.txt
```

### 7. Homebrew 관련 문제
```bash
# Homebrew 진단
brew doctor

# Homebrew 업데이트
brew update && brew upgrade

# 캐시 정리
brew cleanup
```

## 접속 및 사용

### 1. 로컬 접속
- **URL**: `http://localhost:8080`
- **브라우저**: Safari, Chrome, Firefox 등

### 2. 원격 접속
- **URL**: `http://맥북IP:8080`
- **IP 확인**: `ifconfig | grep "inet " | grep -v 127.0.0.1`

### 3. 초기 설정
1. 웹 브라우저에서 접속
2. 사이드바에서 AI 모델 선택
3. 파일 업로드 탭에서 문서 업로드
4. 벡터DB 생성 탭에서 벡터 데이터베이스 초기화
5. 챗봇 탭에서 질문 시작

### 4. 로그 확인
```bash
# Streamlit 로그 확인
tail -f streamlit.log

# 시스템 로그 확인
log show --predicate 'process == "streamlit"' --last 1h
```

## 유지보수

### 1. 정기 업데이트
```bash
# Homebrew 업데이트
brew update && brew upgrade

# Python 패키지 업데이트
pip install --upgrade pip
pip install --upgrade -r bizmob_chatbot/requirements_multiformat.txt
```

### 2. 백업
```bash
# 프로젝트 백업
tar -czf bizmob_backup_$(date +%Y%m%d).tar.gz bizmob_project/

# 벡터DB 백업
cp -r bizmob_faiss_index_* ~/Desktop/backup/
```

### 3. 모니터링
```bash
# 시스템 리소스 모니터링
top

# 디스크 사용량 확인
df -h

# 메모리 사용량 확인
vm_stat
```

## 자동 시작 설정 (선택사항)

### 1. LaunchAgent 설정
```bash
# LaunchAgent 디렉토리 생성
mkdir -p ~/Library/LaunchAgents

# plist 파일 생성
cat > ~/Library/LaunchAgents/com.bizmob.chatbot.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.bizmob.chatbot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/env</string>
        <string>bash</string>
        <string>-c</string>
        <string>cd ~/bizmob_project/bizmob_chatbot && source ~/bizmob_project/bizmob_env/bin/activate && streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/bizmob_chatbot.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/bizmob_chatbot_error.log</string>
</dict>
</plist>
EOF

# LaunchAgent 로드
launchctl load ~/Library/LaunchAgents/com.bizmob.chatbot.plist
```

### 2. 서비스 관리
```bash
# 서비스 시작
launchctl start com.bizmob.chatbot

# 서비스 중지
launchctl stop com.bizmob.chatbot

# 서비스 상태 확인
launchctl list | grep bizmob
```

## 지원 및 문의

문제가 발생하거나 추가 지원이 필요한 경우:
1. 로그 파일 확인 (`/tmp/bizmob_chatbot.log`)
2. 시스템 리소스 상태 점검
3. 네트워크 연결 상태 확인
4. 개발팀에 문의

---

**설치 완료 후 반드시 테스트를 진행하여 모든 기능이 정상 작동하는지 확인하세요.**

## 빠른 설치 스크립트

전체 설치 과정을 자동화하는 스크립트를 제공합니다:

```bash
#!/bin/bash
# bizMOB 챗봇 맥북 자동 설치 스크립트

echo "=== bizMOB 챗봇 맥북 설치 시작 ==="

# 1. Homebrew 설치 확인
if ! command -v brew &> /dev/null; then
    echo "Homebrew 설치 중..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
    source ~/.zshrc
fi

# 2. Python 설치
echo "Python 설치 중..."
brew install python@3.9

# 3. Ollama 설치
echo "Ollama 설치 중..."
brew install ollama
brew services start ollama

# 4. 프로젝트 설정
echo "프로젝트 설정 중..."
cd ~
mkdir -p bizmob_project
cd bizmob_project

# 5. 가상환경 생성
echo "가상환경 생성 중..."
python3 -m venv bizmob_env
source bizmob_env/bin/activate

# 6. 의존성 설치
echo "의존성 패키지 설치 중..."
pip install --upgrade pip
pip install -r bizmob_chatbot/requirements_multiformat.txt
pip install python-dotenv

# 7. 환경 설정
echo "환경 설정 중..."
cat > .env << EOF
TOKENIZERS_PARALLELISM=false
OLLAMA_HOST=http://localhost:11434
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=INFO
EOF

mkdir -p PDF_bizMOB_Guide external_sources bizmob_faiss_index_hyperclovax PDF_이미지_bizmob

# 8. 모델 다운로드
echo "AI 모델 다운로드 중..."
ollama pull hyperclovax

echo "=== 설치 완료 ==="
echo "다음 명령어로 애플리케이션을 실행하세요:"
echo "cd ~/bizmob_project/bizmob_chatbot"
echo "source ~/bizmob_project/bizmob_env/bin/activate"
echo "streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0"
```

이 스크립트를 `install_bizmob.sh`로 저장하고 실행 권한을 부여한 후 실행하세요:

```bash
chmod +x install_bizmob.sh
./install_bizmob.sh
``` 