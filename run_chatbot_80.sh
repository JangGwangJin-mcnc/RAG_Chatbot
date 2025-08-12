#!/bin/bash

# bizMOB RAG Chatbot 백그라운드 실행 스크립트 (포트 80)
# 작성일: 2025-01-08

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")/bizmob_chatbot"

# 가상환경 활성화
source venv310/bin/activate

# 이전 프로세스 종료 (포트 80 사용 중인 경우)
echo "포트 80 사용 중인 프로세스 확인 및 종료..."
sudo lsof -ti:80 | xargs -r sudo kill -9 2>/dev/null

# 로그 디렉토리 생성
mkdir -p logs

# 현재 시간으로 로그 파일명 생성
LOG_FILE="logs/chatbot_80_$(date +%Y%m%d_%H%M%S).log"

echo "=== bizMOB RAG Chatbot 시작 ===" | tee -a "$LOG_FILE"
echo "시작 시간: $(date)" | tee -a "$LOG_FILE"
echo "포트: 80" | tee -a "$LOG_FILE"
echo "로그 파일: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

# Streamlit을 백그라운드에서 실행 (포트 80) - bizmob_chatbot.py 실행
nohup streamlit run bizmob_chatbot.py \
    --server.port 80 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --logger.level info \
    > "$LOG_FILE" 2>&1 &

# 프로세스 ID 저장
CHATBOT_PID=$!
echo $CHATBOT_PID > ../chatbot_80.pid

echo "Chatbot이 백그라운드에서 시작되었습니다."
echo "프로세스 ID: $CHATBOT_PID"
echo "PID 파일: ../chatbot_80.pid"
echo "로그 파일: $LOG_FILE"
echo ""
echo "서버 상태 확인: tail -f $LOG_FILE"
echo "프로세스 종료: kill $CHATBOT_PID"
echo "포트 확인: sudo lsof -i :80"
echo ""
echo "웹 브라우저에서 http://localhost:80 또는 http://서버IP:80 으로 접속하세요."
echo "관리자 메인 페이지가 표시됩니다. (/Chat 페이지에서 채팅 사용 가능)" 