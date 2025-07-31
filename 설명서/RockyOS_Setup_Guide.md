# RockyOS에서 bizMOB 챗봇 프로젝트 초기 설정 가이드

## 개요
이 가이드는 RockyOS에서 bizMOB 챗봇 프로젝트를 처음부터 설정하여 서버가 정상적으로 실행되도록 하는 완전한 설정 과정을 다룹니다.

## 사전 요구사항
- RockyOS 8.x 또는 9.x
- 인터넷 연결
- sudo 권한이 있는 사용자 계정

---

## 1단계: 시스템 업데이트 및 기본 도구 설치

### 시스템 업데이트
```bash
# 시스템 패키지 업데이트
sudo dnf update -y

# 시스템 재부팅 (권장)
sudo reboot
```

### 기본 개발 도구 설치
```bash
# Python 및 개발 도구 설치
sudo dnf groupinstall "Development Tools" -y
sudo dnf install python3 python3-pip python3-devel -y

# Git 설치 (프로젝트 클론용)
sudo dnf install git -y

# 추가 의존성 패키지 설치
sudo dnf install gcc gcc-c++ make cmake -y
sudo dnf install libffi-devel openssl-devel -y
sudo dnf install redhat-rpm-config -y
```

---

## 2단계: Python 가상환경 설정

### 프로젝트 디렉토리 생성
```bash
# 프로젝트 디렉토리 생성 및 이동
mkdir -p ~/projects
cd ~/projects

# 프로젝트 클론 (GitHub에서 가져온 경우)
# git clone <repository-url> my_ai_project
# cd my_ai_project

# 또는 기존 프로젝트 복사
# cp -r /path/to/your/project ~/projects/my_ai_project
# cd my_ai_project
```

### Python 가상환경 생성
```bash
# Python 가상환경 생성
python3 -m venv bizmob_env

# 가상환경 활성화
source bizmob_env/bin/activate

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

---

## 3단계: Python 패키지 설치

### 필수 패키지 설치
```bash
# 가상환경이 활성화된 상태에서
cd bizmob_chatbot

# 필요한 패키지 설치
pip install -r requirements_multiformat.txt

# 추가로 필요한 패키지들
pip install python-dotenv
pip install streamlit
```

### 설치 확인
```bash
# 설치된 패키지 확인
pip list

# Python 버전 확인
python --version
```

---

## 4단계: Ollama 설치 (로컬 AI 모델용)

### Ollama 설치
```bash
# Ollama 설치 스크립트 다운로드 및 실행
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서비스 시작
sudo systemctl start ollama
sudo systemctl enable ollama

# Ollama 상태 확인
sudo systemctl status ollama
```

### 기본 모델 다운로드 (선택사항)
```bash
# 기본 모델 다운로드
ollama pull llama2
ollama pull hyperclovax

# 사용 가능한 모델 확인
ollama list
```

---

## 5단계: 환경 설정 파일 생성

### .env 파일 생성
```bash
# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
TOKENIZERS_PARALLELISM=false
EOF

# 파일 권한 설정
chmod 600 .env
```

### 환경 변수 설정
```bash
# .bashrc에 환경 변수 추가
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
source ~/.bashrc
```

---

## 6단계: 필요한 디렉토리 생성

### 프로젝트 구조 생성
```bash
# 프로젝트 루트에서
mkdir -p PDF_bizMOB_Guide
mkdir -p external_sources
mkdir -p bizmob_faiss_index_hyperclovax

# 디렉토리 권한 설정
chmod 755 PDF_bizMOB_Guide
chmod 755 external_sources
chmod 755 bizmob_faiss_index_hyperclovax
```

---

## 7단계: 권한 설정

### 실행 권한 부여
```bash
# 실행 권한 부여
chmod +x bizmob_chatbot.py

# 디렉토리 권한 설정
chmod 755 PDF_bizMOB_Guide
chmod 755 external_sources
```

### 사용자 그룹 설정 (필요시)
```bash
# 사용자를 wheel 그룹에 추가
sudo usermod -aG wheel $USER

# 변경사항 적용을 위해 재로그인
# 또는 다음 명령으로 그룹 변경사항 적용
newgrp wheel
```

---

## 8단계: 서버 실행 테스트

### 기본 실행
```bash
# 가상환경이 활성화된 상태에서
cd bizmob_chatbot

# Streamlit 서버 실행
streamlit run bizmob_chatbot.py --server.port 8501 --server.address 0.0.0.0
```

### 백그라운드 실행 (선택사항)
```bash
# nohup을 사용한 백그라운드 실행
nohup streamlit run bizmob_chatbot.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# 프로세스 확인
ps aux | grep streamlit
```

---

## 9단계: 방화벽 설정

### RockyOS 방화벽 설정
```bash
# 방화벽에서 포트 8501 허용
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload

# 방화벽 상태 확인
sudo firewall-cmd --list-ports
```

### SELinux 설정 (필요시)
```bash
# SELinux 상태 확인
sestatus

# SELinux가 활성화된 경우 설정
sudo setsebool -P httpd_can_network_connect 1
sudo setsebool -P httpd_can_network_relay 1
```

---

## 10단계: 서비스 자동 시작 설정 (선택사항)

### systemd 서비스 파일 생성
```bash
# systemd 서비스 파일 생성
sudo tee /etc/systemd/system/bizmob-chatbot.service > /dev/null << EOF
[Unit]
Description=bizMOB Chatbot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/projects/my_ai_project-main/bizmob_chatbot
Environment=PATH=$HOME/projects/my_ai_project-main/bizmob_env/bin
ExecStart=$HOME/projects/my_ai_project-main/bizmob_env/bin/streamlit run bizmob_chatbot.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### 서비스 활성화
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

---

## 11단계: 접속 확인

### 로컬 접속 테스트
```bash
# curl을 사용한 접속 테스트
curl -I http://localhost:8501
```

### 웹 브라우저 접속
- **로컬 접속**: `http://localhost:8501`
- **원격 접속**: `http://서버IP:8501`

---

## 문제 해결

### 1. 포트 충돌 문제
```bash
# 다른 포트 사용
streamlit run bizmob_chatbot.py --server.port 8502

# 또는 사용 중인 포트 확인
sudo netstat -tlnp | grep :8501
```

### 2. 메모리 부족 문제
```bash
# 메모리 사용량 확인
free -h

# 스왑 메모리 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 영구 스왑 설정
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. 권한 문제
```bash
# 파일 권한 확인
ls -la bizmob_chatbot.py

# 권한 수정
chmod +x bizmob_chatbot.py
chmod 755 PDF_bizMOB_Guide/
```

### 4. Python 패키지 문제
```bash
# 가상환경 재활성화
deactivate
source bizmob_env/bin/activate

# 패키지 재설치
pip install --force-reinstall -r requirements_multiformat.txt
```

### 5. Ollama 연결 문제
```bash
# Ollama 서비스 상태 확인
sudo systemctl status ollama

# Ollama 재시작
sudo systemctl restart ollama

# Ollama 로그 확인
sudo journalctl -u ollama -f
```

---

## 모니터링 및 유지보수

### 로그 확인
```bash
# Streamlit 로그 확인
tail -f streamlit.log

# systemd 서비스 로그 확인
sudo journalctl -u bizmob-chatbot -f
```

### 성능 모니터링
```bash
# 시스템 리소스 사용량 확인
htop

# 디스크 사용량 확인
df -h

# 메모리 사용량 확인
free -h
```

### 정기 업데이트
```bash
# 시스템 업데이트
sudo dnf update -y

# Python 패키지 업데이트
pip list --outdated
pip install --upgrade package_name
```

---

## 보안 고려사항

### 방화벽 설정
```bash
# 필요한 포트만 허용
sudo firewall-cmd --permanent --remove-service=ssh
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### 사용자 권한
```bash
# 최소 권한 원칙 적용
sudo chown -R $USER:$USER ~/projects/my_ai_project-main/
chmod 600 .env
```

---

## 완료 확인 체크리스트

- [ ] Python 3.9+ 설치 완료
- [ ] 가상환경 생성 및 활성화
- [ ] 모든 패키지 설치 완료
- [ ] Ollama 설치 및 실행
- [ ] .env 파일 생성
- [ ] 필요한 디렉토리 생성
- [ ] 권한 설정 완료
- [ ] 방화벽 설정 완료
- [ ] 서버 실행 테스트 성공
- [ ] 웹 브라우저 접속 확인
- [ ] 서비스 자동 시작 설정 (선택사항)

---

## 추가 정보

### 유용한 명령어
```bash
# 프로세스 확인
ps aux | grep streamlit

# 포트 사용량 확인
sudo netstat -tlnp

# 서비스 상태 확인
sudo systemctl status bizmob-chatbot

# 로그 실시간 확인
tail -f streamlit.log
```

### 지원 및 문의
- 프로젝트 이슈: GitHub Issues
- RockyOS 지원: Red Hat Customer Portal
- Streamlit 문서: https://docs.streamlit.io/

---

**참고**: 이 가이드는 RockyOS 8.x/9.x를 기준으로 작성되었습니다. 다른 버전에서는 일부 명령어가 다를 수 있습니다. 