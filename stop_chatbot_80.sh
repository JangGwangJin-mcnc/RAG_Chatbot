#!/bin/bash

# bizMOB RAG Chatbot 종료 스크립트 (포트 80)
# 작성일: 2025-01-08

echo "=== bizMOB RAG Chatbot 종료 ==="

# PID 파일에서 프로세스 ID 읽기
if [ -f "chatbot_80.pid" ]; then
    PID=$(cat chatbot_80.pid)
    echo "PID 파일에서 프로세스 ID: $PID"
    
    # 프로세스가 실행 중인지 확인
    if ps -p $PID > /dev/null 2>&1; then
        echo "프로세스 $PID 종료 중..."
        kill $PID
        
        # 5초 대기 후 강제 종료
        sleep 5
        if ps -p $PID > /dev/null 2>&1; then
            echo "프로세스 강제 종료 중..."
            kill -9 $PID
        fi
        
        echo "프로세스 $PID 종료 완료"
    else
        echo "프로세스 $PID는 이미 종료되었습니다."
    fi
    
    # PID 파일 삭제
    rm -f chatbot_80.pid
else
    echo "PID 파일을 찾을 수 없습니다."
fi

# 포트 80 사용 중인 프로세스 확인 및 종료
echo "포트 80 사용 중인 프로세스 확인..."
PORT_PIDS=$(sudo lsof -ti:80 2>/dev/null)

if [ -n "$PORT_PIDS" ]; then
    echo "포트 80 사용 중인 프로세스들: $PORT_PIDS"
    echo "프로세스들 종료 중..."
    echo $PORT_PIDS | xargs -r sudo kill -9
    echo "포트 80 프로세스들 종료 완료"
else
    echo "포트 80을 사용 중인 프로세스가 없습니다."
fi

echo "=== Chatbot 종료 완료 ===" 