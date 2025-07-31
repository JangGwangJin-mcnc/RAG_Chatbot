# bizMOB 챗봇 리눅스 설치 가이드

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [시스템 패키지 설치](#시스템-패키지-설치)
3. [Ollama 설치](#ollama-설치)
4. [프로젝트 설정](#프로젝트-설정)
5. [의존성 패키지 설치](#의존성-패키지-설치)
6. [환경 설정](#환경-설정)
7. [애플리케이션 실행](#애플리케이션-실행)
8. [서비스 등록 (선택사항)](#서비스-등록-선택사항)
9. [문제 해결](#문제-해결)
10. [접속 및 사용](#접속-및-사용)

## 시스템 요구사항

### 최소 사양
- **OS**: Ubuntu 20.04 LTS 이상, CentOS 8 이상, Debian 11 이상
- **CPU**: 4코어 이상 (AI 모델 실행용)
- **RAM**: 8GB 이상 (16GB 권장)
- **저장공간**: 20GB 이상
- **Python**: 3.9 이상

### 권장 사양
- **CPU**: 8코어 이상
- **RAM**: 16GB 이상
- **저장공간**: 50GB 이상 (모델 다운로드용)
- **GPU**: NVIDIA GPU (선택사항, 가속용)

## 시스템 패키지 설치

### 1. 시스템 업데이트
```bash
# 패키지 목록 업데이트
sudo apt update

# 시스템 업그레이드
sudo apt upgrade -y

# 재부팅 (필요시)
sudo reboot
```

### 2. 필수 패키지 설치
```bash
# Python 및 개발 도구
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y build-essential python3-dev

# Git 설치
sudo apt install -y git

# curl 설치 (Ollama 설치용)
sudo apt install -y curl

# wget 설치
sudo apt install -y wget
```

### 3. 추가 도구 설치
```bash
# 압축 도구
sudo apt install -y unzip

# 파일 시스템 도구
sudo apt install -y tree

# 네트워크 도구
sudo apt install -y net-tools
```

## Ollama 설치

### 1. Ollama 다운로드 및 설치
```bash
# Ollama 설치 스크립트 실행
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Ollama 서비스 시작
```bash
# Ollama 서비스 시작
sudo systemctl start ollama

# 부팅 시 자동 시작 설정
sudo systemctl enable ollama

# 서비스 상태 확인
sudo systemctl status ollama
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

### 3. 방화벽 설정
```bash
# UFW 방화벽 활성화
sudo ufw enable

# 8080 포트 허용
sudo ufw allow 8080

# 방화벽 상태 확인
sudo ufw status
```

## 서비스 등록 (선택사항)

### 1. systemd 서비스 파일 생성
```bash
# 서비스 파일 생성
sudo tee /etc/systemd/system/bizmob-chatbot.service << EOF
[Unit]
Description=bizMOB Chatbot Service
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$HOME/bizmob_project/bizmob_chatbot
Environment=PATH=$HOME/bizmob_project/bizmob_env/bin
Environment=PYTHONPATH=$HOME/bizmob_project
ExecStart=$HOME/bizmob_project/bizmob_env/bin/streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### 2. 서비스 활성화
```bash
# systemd 데몬 리로드
sudo systemctl daemon-reload

# 서비스 활성화
sudo systemctl enable bizmob-chatbot

# 서비스 시작
sudo systemctl start bizmob-chatbot

# 서비스 상태 확인
sudo systemctl status bizmob-chatbot
```

### 3. 서비스 관리 명령어
```bash
# 서비스 시작
sudo systemctl start bizmob-chatbot

# 서비스 중지
sudo systemctl stop bizmob-chatbot

# 서비스 재시작
sudo systemctl restart bizmob-chatbot

# 로그 확인
sudo journalctl -u bizmob-chatbot -f
```

## 문제 해결

### 1. 메모리 부족 문제
```bash
# 메모리 사용량 확인
free -h

# 스왑 메모리 추가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 스왑 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. 포트 충돌 문제
```bash
# 포트 사용 확인
sudo netstat -tlnp | grep 8080

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
sudo systemctl status ollama

# Ollama 재시작
sudo systemctl restart ollama

# 연결 테스트
curl http://localhost:11434/api/tags
```

## 접속 및 사용

### 1. 로컬 접속
- **URL**: `http://localhost:8080`
- **브라우저**: Chrome, Firefox, Safari 등

### 2. 원격 접속
- **URL**: `http://서버IP:8080`
- **예시**: `http://192.168.1.100:8080`

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
sudo journalctl -u bizmob-chatbot -f
```

## 유지보수

### 1. 정기 업데이트
```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 패키지 업데이트
pip install --upgrade pip
pip install --upgrade -r bizmob_chatbot/requirements_multiformat.txt
```

### 2. 백업
```bash
# 프로젝트 백업
tar -czf bizmob_backup_$(date +%Y%m%d).tar.gz bizmob_project/

# 벡터DB 백업
cp -r bizmob_faiss_index_* /backup/
```

### 3. 모니터링
```bash
# 시스템 리소스 모니터링
htop

# 디스크 사용량 확인
df -h

# 메모리 사용량 확인
free -h
```

## 지원 및 문의

문제가 발생하거나 추가 지원이 필요한 경우:
1. 로그 파일 확인
2. 시스템 리소스 상태 점검
3. 네트워크 연결 상태 확인
4. 개발팀에 문의

---

**설치 완료 후 반드시 테스트를 진행하여 모든 기능이 정상 작동하는지 확인하세요.** 