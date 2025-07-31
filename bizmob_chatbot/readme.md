
# 파이썬으로 RAG 웹서비스 만들기 : bizMOB 플랫폼 챗봇

### 파이썬 버전 : 3.9.6 
- Mac Sequoia 15.5
  
### bizMOB 관련 문서 : PDF_bizMOB_Guide
## 활용 기술 : SystemArchitectureDiagram.png

## 예시 화면

## 사전 준비사항

### 파이썬 가상 환경 설정  
Python 가상환경 설정 가이드 : python-set-venv.md

### OpenAI API Key 발급
OpenAI API Key 발급 방법 : get-openai-api-key.md


## 시작 방법  

1. 다음 명령어로 필요한 패키지 설치:
    ```bash
    pip3 install -r requirements_multiformat.txt
    ```
2. `.env` 파일을 생성하고 OpenAI API Key 입력:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    ```
3. 다음 명령어로 웹 서비스 실행:
    ```bash
    streamlit run bizmob_chatbot.py
    ```

## 코드 설명  
- **실행 코드**: `bizmob_chatbot.py` 
- **완성 코드**: `completed.py` 

## 예시 질문
1. bizMOB Client의 요소는?

## 기타
- FAISS 뷰어 : https://faissviewer-hu2g6bbyxgcdjjumbsfysz.streamlit.app
