#!/bin/bash

# bizMOB 챗봇 백그라운드 실행 스크립트 (맥북용)
echo "Starting bizMOB chatbot in background..."

# 스크립트 디렉토리 설정
SCRIPT_DIR="/Users/mcnc/chatbot/RAG_Chatbot"

# 가상환경 경로 설정
VENV_PATH="$SCRIPT_DIR/venv310"

# 로그 파일 경로
LOG_FILE="$SCRIPT_DIR/bizmob_chatbot.log"
PID_FILE="$SCRIPT_DIR/bizmob_chatbot.pid"

# 프로세스가 이미 실행 중인지 확인
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "bizMOB chatbot is already running with PID: $PID"
        echo "To stop it, run: kill $PID"
        exit 1
    else
        echo "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# 가상환경 활성화 및 실행
cd "$SCRIPT_DIR/bizmob_chatbot"

# PyTorch 보안 취약점 해결 적용
echo "Applying PyTorch security fixes..."
python ../fix_torch_vulnerability.py

# ChromaDB 확인 및 설치
echo "Checking ChromaDB availability..."
python -c "import chromadb" 2>/dev/null || {
    echo "Installing ChromaDB..."
    pip install chromadb
}

# 환경 변수 설정
export TORCH_WARN_ON_LOAD=0
export TORCH_LOAD_WARN_ONLY=0
export PYTORCH_DISABLE_WARNINGS=1

# 백그라운드에서 Streamlit 실행
echo "Starting Streamlit in background..."
nohup streamlit run bizmob_chatbot.py --server.port 8080 --server.address 0.0.0.0 > "$LOG_FILE" 2>&1 &

# PID 저장
echo $! > "$PID_FILE"

echo "bizMOB chatbot started in background"
echo "PID: $(cat $PID_FILE)"
echo "Log file: $LOG_FILE"
echo "Access URL: http://localhost:8080"
echo ""
echo "To stop the chatbot:"
echo "kill $(cat $PID_FILE)"
echo ""
echo "To view logs:"
echo "tail -f $LOG_FILE" 