#!/bin/bash

# bizMOB 챗봇 로그 확인 스크립트 (맥북용)
echo "Viewing bizMOB chatbot logs..."

# 스크립트 디렉토리 설정
SCRIPT_DIR="/Users/mcnc/chatbot/RAG_Chatbot"
LOG_FILE="$SCRIPT_DIR/bizmob_chatbot.log"
PID_FILE="$SCRIPT_DIR/bizmob_chatbot.pid"

# 로그 파일 확인
if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    echo "The chatbot may not have been started yet"
    exit 1
fi

# 프로세스 상태 확인
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "bizMOB chatbot is running (PID: $PID)"
    else
        echo "bizMOB chatbot is not running (stale PID file)"
    fi
else
    echo "PID file not found - checking for running processes..."
    PIDS=$(ps aux | grep "streamlit run bizmob_chatbot.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        echo "Found running processes: $PIDS"
    else
        echo "No bizMOB chatbot processes found"
    fi
fi

echo ""
echo "=== Recent Logs ==="
echo "Log file: $LOG_FILE"
echo ""

# 로그 파일 크기 확인
LOG_SIZE=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
if [ "$LOG_SIZE" -gt 0 ]; then
    echo "Last 50 lines of logs:"
    echo "----------------------------------------"
    tail -50 "$LOG_FILE"
    echo "----------------------------------------"
    echo ""
    echo "To follow logs in real-time, run:"
    echo "tail -f $LOG_FILE"
else
    echo "Log file is empty or not accessible"
fi 