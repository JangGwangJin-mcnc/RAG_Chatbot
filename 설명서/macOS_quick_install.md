# bizMOB 챗봇 맥북 빠른 설치 가이드

## 🚀 5분 설치 가이드

### 1단계: 자동 설치 스크립트 실행

```bash
# 터미널 열기 (Spotlight에서 '터미널' 검색)

# 설치 스크립트 다운로드 및 실행
curl -fsSL https://raw.githubusercontent.com/your-repo/bizmob-chatbot/main/설명서/install_bizmob_mac.sh | bash
```

### 2단계: 수동 설치 (권장)

```bash
# 1. Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Python 및 Ollama 설치
brew install python@3.9 ollama

# 3. Ollama 시작
brew services start ollama

# 4. 프로젝트 설정
cd ~
mkdir bizmob_project && cd bizmob_project
python3 -m venv bizmob_env
source bizmob_env/bin/activate

# 5. 프로젝트 파일 복사 (수동으로 복사)
# Finder에서 프로젝트 파일을 ~/bizmob_project/로 복사

# 6. 의존성 설치
pip install -r bizmob_chatbot/requirements_multiformat.txt

# 7. AI 모델 다운로드
ollama pull hyperclovax

# 8. 실행
cd bizmob_chatbot
streamlit run bizmob_chatbot.py --server.port 8080
```

### 3단계: 웹 브라우저에서 접속

- **URL**: `http://localhost:8080`
- **브라우저**: Safari, Chrome, Firefox

## 📋 시스템 요구사항

- **macOS**: 10.15 (Catalina) 이상
- **RAM**: 8GB 이상 (16GB 권장)
- **저장공간**: 20GB 이상
- **인터넷**: 안정적인 연결 필요

## 🔧 문제 해결

### Ollama 연결 오류
```bash
# Ollama 재시작
brew services restart ollama

# 상태 확인
brew services list | grep ollama
```

### 포트 충돌
```bash
# 다른 포트 사용
streamlit run bizmob_chatbot.py --server.port 8081
```

### 메모리 부족
```bash
# 메모리 사용량 확인
top -l 1 | grep PhysMem

# 불필요한 프로세스 종료
killall -9 streamlit ollama
```

## 📞 지원

문제가 발생하면:
1. 터미널에서 `~/bizmob_project/run_bizmob.sh` 실행
2. 로그 확인: `tail -f ~/bizmob_project/streamlit.log`
3. 개발팀에 문의

---

**설치 완료 후 웹 브라우저에서 `http://localhost:8080`으로 접속하여 테스트하세요!** 