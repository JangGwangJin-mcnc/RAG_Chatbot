#!/bin/bash

# bizMOB RAG Chatbot 상태 확인 스크립트 (포트 80)
# 작성일: 2025-01-08

echo "=== bizMOB RAG Chatbot 상태 확인 ==="
echo "확인 시간: $(date)"
echo ""

# PID 파일 확인
if [ -f "chatbot_80.pid" ]; then
    PID=$(cat chatbot_80.pid)
    echo "PID 파일: chatbot_80.pid"
    echo "프로세스 ID: $PID"
    
    # 프로세스 상태 확인
    if ps -p $PID > /dev/null 2>&1; then
        echo "프로세스 상태: 실행 중 ✅"
        echo "프로세스 정보:"
        ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem
    else
        echo "프로세스 상태: 종료됨 ❌"
        echo "PID 파일은 있지만 프로세스가 실행되지 않음"
    fi
else
    echo "PID 파일: 없음"
fi

echo ""

# 포트 80 사용 상태 확인
echo "포트 80 사용 상태:"
if sudo lsof -i:80 > /dev/null 2>&1; then
    echo "포트 80: 사용 중 ✅"
    echo "사용 중인 프로세스:"
    sudo lsof -i:80
else
    echo "포트 80: 사용되지 않음 ❌"
fi

echo ""

# 로그 파일 확인
echo "로그 파일 상태:"
if [ -d "bizmob_chatbot/logs" ]; then
    echo "로그 디렉토리: bizmob_chatbot/logs ✅"
    echo "최근 로그 파일들:"
    ls -la bizmob_chatbot/logs/ | tail -5
else
    echo "로그 디렉토리: 없음 ❌"
fi

echo ""

# 가상환경 상태 확인
echo "가상환경 상태:"
if [ -d "bizmob_chatbot/venv310" ]; then
    echo "가상환경: bizmob_chatbot/venv310 ✅"
    echo "Python 버전:"
    bizmob_chatbot/venv310/bin/python --version
else
    echo "가상환경: 없음 ❌"
fi

echo ""
echo "=== 상태 확인 완료 ===" 