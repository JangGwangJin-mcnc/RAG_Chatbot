# bizMOB RAG Chatbot - 포트 80 실행 가이드

## 📋 개요
이 스크립트들은 bizMOB RAG Chatbot을 포트 80에서 백그라운드로 실행하고 관리하기 위한 도구입니다.

**실행 파일**: `bizmob_chatbot.py` (메인 관리자 페이지)

## 🚀 실행 스크립트

### 1. `run_chatbot_80.sh` - Chatbot 시작
```bash
./run_chatbot_80.sh
```

**기능:**
- 포트 80에서 Streamlit 서버 시작
- **메인 파일**: `bizmob_chatbot.py` 실행 (관리자 페이지)
- 백그라운드에서 실행
- 자동 로그 파일 생성
- 이전 프로세스 자동 종료

**실행 후 출력:**
```
Chatbot이 백그라운드에서 시작되었습니다.
프로세스 ID: 12345
PID 파일: ../chatbot_80.pid
로그 파일: logs/chatbot_80_20250108_143022.log

웹 브라우저에서 http://localhost:80 또는 http://서버IP:80 으로 접속하세요.
관리자 페이지가 표시됩니다.
```

### 2. `stop_chatbot_80.sh` - Chatbot 종료
```bash
./stop_chatbot_80.sh
```

**기능:**
- 실행 중인 chatbot 프로세스 종료
- 포트 80 사용 중인 모든 프로세스 정리
- PID 파일 자동 삭제

### 3. `status_chatbot_80.sh` - 상태 확인
```bash
./status_chatbot_80.sh
```

**기능:**
- 프로세스 실행 상태 확인
- 포트 80 사용 상태 확인
- 로그 파일 상태 확인
- 가상환경 상태 확인

## 🌐 접속 방법

### 로컬 접속
```
http://localhost:80
http://127.0.0.1:80
```

### 외부 접속
```
http://서버IP:80
http://도메인:80
```

**접속 시 표시되는 페이지**: bizMOB 관리자 페이지 (파일 업로드, 벡터DB 관리, 채팅 등)

## 📁 파일 구조
```
RAG_Chatbot/
├── run_chatbot_80.sh          # 실행 스크립트
├── stop_chatbot_80.sh         # 종료 스크립트
├── status_chatbot_80.sh       # 상태 확인 스크립트
├── chatbot_80.pid             # 프로세스 ID (자동 생성)
├── bizmob_chatbot/
│   ├── venv310/               # 가상환경
│   ├── bizmob_chatbot.py      # 메인 관리자 페이지
│   ├── pages/1_Chat.py        # 채팅 전용 페이지
│   └── logs/                  # 로그 디렉토리
└── README_chatbot_80.md       # 이 파일
```

## 📊 모니터링

### 실시간 로그 확인
```bash
# 로그 파일 실시간 모니터링
tail -f bizmob_chatbot/logs/chatbot_80_YYYYMMDD_HHMMSS.log

# 또는 상태 확인 스크립트 사용
./status_chatbot_80.sh
```

### 프로세스 확인
```bash
# 프로세스 상태 확인
ps aux | grep streamlit

# 포트 사용 확인
sudo lsof -i :80
```

## ⚠️ 주의사항

1. **포트 80 사용**: HTTP 기본 포트이므로 관리자 권한이 필요할 수 있습니다.
2. **방화벽 설정**: 외부 접속을 위해 방화벽에서 포트 80을 열어야 합니다.
3. **가상환경**: 스크립트 실행 전 `bizmob_chatbot/venv310` 가상환경이 준비되어 있어야 합니다.
4. **의존성**: `requirements.txt`의 모든 패키지가 설치되어 있어야 합니다.
5. **실행 파일**: `bizmob_chatbot.py`가 메인 파일로 실행됩니다 (관리자 페이지).

## 🔧 문제 해결

### 포트 80 권한 오류
```bash
# 포트 80 사용 권한 확인
sudo lsof -i :80

# 필요시 포트 변경 (예: 8080)
# run_chatbot_80.sh 파일에서 --server.port 80을 8080으로 수정
```

### 프로세스가 시작되지 않는 경우
```bash
# 가상환경 상태 확인
./status_chatbot_80.sh

# 수동으로 가상환경 활성화 후 테스트
cd bizmob_chatbot
source venv310/bin/activate
streamlit run bizmob_chatbot.py --server.port 80
```

### 로그 파일이 생성되지 않는 경우
```bash
# 로그 디렉토리 권한 확인
ls -la bizmob_chatbot/logs/

# 필요시 디렉토리 생성
mkdir -p bizmob_chatbot/logs
```

## 📝 로그 파일

로그 파일은 `bizmob_chatbot/logs/` 디렉토리에 다음 형식으로 생성됩니다:
```
chatbot_80_20250108_143022.log
chatbot_80_20250108_150000.log
chatbot_80_20250108_160000.log
```

## 🎯 사용 시나리오

1. **개발/테스트**: 로컬에서 `./run_chatbot_80.sh` 실행
2. **프로덕션**: 서버에서 백그라운드로 실행 후 외부 접속 허용
3. **모니터링**: `./status_chatbot_80.sh`로 정기적 상태 확인
4. **유지보수**: `./stop_chatbot_80.sh`로 서비스 중단 후 재시작

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. `./status_chatbot_80.sh` 실행 결과
2. 로그 파일의 오류 메시지
3. 가상환경과 의존성 설치 상태
4. 포트 80 사용 권한
5. `bizmob_chatbot.py` 파일이 존재하는지 확인 