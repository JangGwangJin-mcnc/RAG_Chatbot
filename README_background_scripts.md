# bizMOB 챗봇 백그라운드 실행 스크립트

맥북에서 bizMOB 챗봇을 백그라운드에서 실행하기 위한 스크립트들입니다.

## 스크립트 파일들

### 1. `run_bizmob_background.sh`
- bizMOB 챗봇을 백그라운드에서 실행
- 자동으로 PyTorch 보안 취약점 해결 적용
- ChromaDB 자동 설치
- PID 파일과 로그 파일 생성

### 2. `stop_bizmob.sh`
- 백그라운드에서 실행 중인 bizMOB 챗봇 중지
- PID 파일을 통한 안전한 프로세스 종료

### 3. `view_bizmob_logs.sh`
- 실행 중인 챗봇의 로그 확인
- 실시간 로그 추적 가능

## 사용법

### 챗봇 시작
```bash
# 스크립트 실행 권한 부여 (최초 1회)
chmod +x run_bizmob_background.sh
chmod +x stop_bizmob.sh
chmod +x view_bizmob_logs.sh

# 백그라운드에서 챗봇 시작
./run_bizmob_background.sh
```

### 챗봇 중지
```bash
./stop_bizmob.sh
```

### 로그 확인
```bash
# 최근 로그 확인
./view_bizmob_logs.sh

# 실시간 로그 추적
tail -f /Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot.log
```

### 프로세스 상태 확인
```bash
# PID 파일 확인
cat /Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot.pid

# 실행 중인 프로세스 확인
ps aux | grep "streamlit run bizmob_chatbot.py"
```

## 접속 정보

- **URL**: http://localhost:8080
- **로그 파일**: `/Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot.log`
- **PID 파일**: `/Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot.pid`

## 주의사항

1. 스크립트는 `/Users/mcnc/chatbot/RAG_Chatbot` 디렉토리에서 실행되어야 합니다.
2. 가상환경 `venv310`이 해당 디렉토리에 있어야 합니다.
3. 백그라운드 실행 시 터미널을 닫아도 챗봇이 계속 실행됩니다.
4. 챗봇을 중지하려면 반드시 `stop_bizmob.sh` 스크립트를 사용하세요.

## 문제 해결

### 챗봇이 시작되지 않는 경우
```bash
# 로그 확인
./view_bizmob_logs.sh

# 가상환경 확인
ls -la /Users/mcnc/chatbot/RAG_Chatbot/venv310

# 수동으로 실행해보기
cd /Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot
source ../venv310/bin/activate
streamlit run bizmob_chatbot.py --server.port 8080
```

### 프로세스가 강제 종료된 경우
```bash
# 남은 프로세스 정리
pkill -f "streamlit run bizmob_chatbot.py"
rm -f /Users/mcnc/chatbot/RAG_Chatbot/bizmob_chatbot.pid
``` 