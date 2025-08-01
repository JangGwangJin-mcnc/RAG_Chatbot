#!/bin/bash

# bizMOB 챗봇 중지 스크립트 (맥북용)
echo "Stopping bizMOB chatbot..."

# 스크립트 디렉토리 설정
SCRIPT_DIR="/Users/mcnc/chatbot/RAG_Chatbot"
PID_FILE="$SCRIPT_DIR/bizmob_chatbot.pid"

# PID 파일 확인
if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found. Checking for running processes..."
    
    # 실행 중인 streamlit 프로세스 찾기
    PIDS=$(ps aux | grep "streamlit run bizmob_chatbot.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PIDS" ]; then
        echo "No bizMOB chatbot process found"
        exit 0
    else
        echo "Found running processes: $PIDS"
        for PID in $PIDS; do
            echo "Killing process $PID"
            kill $PID
        done
        echo "All bizMOB chatbot processes stopped"
        exit 0
    fi
fi

# PID 읽기
PID=$(cat "$PID_FILE")

# 프로세스가 실행 중인지 확인
if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping bizMOB chatbot (PID: $PID)"
    kill $PID
    
    # 프로세스가 완전히 종료될 때까지 대기
    sleep 2
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing process..."
        kill -9 $PID
    fi
    
    echo "bizMOB chatbot stopped"
else
    echo "Process with PID $PID is not running"
fi

# PID 파일 삭제
rm -f "$PID_FILE"
echo "PID file removed" 