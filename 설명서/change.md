# bizMOB Chatbot 프로젝트 변경점 상세 기록

## 개요
이 문서는 bizMOB Chatbot 프로젝트의 주요 변경점들을 상세하게 기록한 문서입니다. 
각 변경사항의 문제점, 해결방법, 코드 변경 내용을 포함합니다.

## 주요 변경사항 목록

### 1. FAISS 벡터DB 경로 문제 해결

#### 문제점
- 애플리케이션이 존재하지 않는 `bizmob_faiss_index_llama3_2_latest` 디렉토리를 찾으려고 시도
- 실제로는 `bizmob_faiss_index_gemma3_latest` 디렉토리가 존재함
- `vector_db_model_info.json` 파일의 경로 불일치

#### 해결방법
**파일**: `bizmob_chatbot/bizmob_chatbot.py`

1. **`get_vector_db_path()` 함수 수정**
```python
def get_vector_db_path():
    """벡터DB 경로를 동적으로 결정"""
    # 현재 디렉토리에서 FAISS 인덱스 찾기
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    # 가능한 FAISS 인덱스 디렉토리들
    possible_paths = [
        os.path.join(current_dir, "bizmob_faiss_index_gemma3_latest"),
        os.path.join(parent_dir, "bizmob_faiss_index_gemma3_latest"),
        os.path.join(current_dir, "bizmob_faiss_index_llama3_2_latest"),
        os.path.join(parent_dir, "bizmob_faiss_index_llama3_2_latest"),
    ]
    
    # 존재하는 첫 번째 경로 반환
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 기본값 반환
    return "bizmob_faiss_index_gemma3_latest"
```

2. **`check_vector_db_exists()` 함수 수정**
```python
def check_vector_db_exists():
    """벡터DB가 존재하는지 확인"""
    vector_db_path = get_vector_db_path()
    # index.faiss 파일이 존재하는지 확인
    faiss_file_path = os.path.join(vector_db_path, 'index.faiss')
    return os.path.exists(faiss_file_path)
```

3. **`load_saved_model_info()` 함수 수정**
```python
def load_saved_model_info():
    """저장된 모델 정보 로드"""
    # 현재 디렉토리와 상위 디렉토리에서 검색
    search_paths = [
        os.getcwd(),
        os.path.dirname(os.getcwd())
    ]
    
    for path in search_paths:
        model_info_path = os.path.join(path, 'vector_db_model_info.json')
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"모델 정보 로드 실패: {str(e)}")
    
    return None
```

### 2. Streamlit 키 중복 오류 해결

#### 문제점
```
streamlit.errors.StreamlitDuplicateElementKey: There are multiple elements with the same key='file_manager_main_reinit'
```

#### 해결방법
**파일**: `bizmob_chatbot/bizmob_chatbot.py`

```python
# 변경 전
key="file_manager_main_reinit"

# 변경 후
key="file_manager_main_reinit_1"  # 라인 446
key="file_manager_main_reinit_2"  # 라인 1379
```

### 3. PyTorch 메타 텐서 오류 해결

#### 문제점
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
```

#### 해결방법

1. **requirements_multiformat.txt 수정**
```diff
--- a/bizmob_chatbot/requirements_multiformat.txt
+++ b/bizmob_chatbot/requirements_multiformat.txt
@@ -10,4 +10,6 @@
 openpyxl
 python-pptx
 python-docx
-unstructured
+unstructured
+torch>=2.0.0
+transformers>=4.30.0
```

2. **get_embedding_model() 함수 수정**
```python
def get_embedding_model():
    """선택된 임베딩 모델을 반환"""
    selected_embedding = st.session_state.get('selected_embedding_model', 'jhgan/ko-sroberta-multitask')
    
    # 모델별 설정 (PyTorch 메타 텐서 오류 해결)
    model_configs = {
        'jhgan/ko-sroberta-multitask': {
            'model_kwargs': {
                'device': 'cpu'
            },
            'encode_kwargs': {'normalize_embeddings': True}
        },
        'sentence-transformers/all-MiniLM-L6-v2': {
            'model_kwargs': {
                'device': 'cpu'
            },
            'encode_kwargs': {'normalize_embeddings': True}
        },
        # ... 다른 모델들도 동일하게 수정
    }
    
    config = model_configs.get(selected_embedding, {})
    
    try:
        # PyTorch 메타 텐서 오류 방지를 위한 추가 설정
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        # 기본 설정에 device 정보 추가
        if 'model_kwargs' not in config:
            config['model_kwargs'] = {}
        config['model_kwargs'].update({
            'device': device
        })
        
        return HuggingFaceEmbeddings(
            model_name=selected_embedding,
            **config
        )
    except Exception as e:
        st.error(f"임베딩 모델 로드 실패: {str(e)}")
        # 폴백: 기본 설정으로 재시도
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
```

### 4. HybridRetriever 클래스 구현

#### 목적
시멘틱 검색 70% + 키워드 검색 30%를 결합한 하이브리드 검색 시스템 구현

#### 구현 내용
**파일**: `bizmob_chatbot/bizmob_chatbot.py`

```python
class HybridRetriever(BaseRetriever):
    """시멘틱 검색 70% + 키워드 검색 30%를 결합한 하이브리드 리트리버"""
    
    def __init__(self, vector_store, semantic_weight=0.7, keyword_weight=0.3, k=3):
        # BaseRetriever 초기화 시 필드 충돌을 방지하기 위해 kwargs 사용
        super().__init__()
        # private 변수로 저장하여 BaseRetriever와의 충돌 방지
        self._vector_store = vector_store
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._k = k
    
    def _semantic_search(self, query: str, k: int = 3):
        """시멘틱 검색 (벡터 유사도)"""
        try:
            # 벡터 스토어의 기본 검색 사용
            semantic_results = self._vector_store.similarity_search_with_score(query, k=k)
            
            # 결과 형식 통일
            results = []
            for doc, score in semantic_results:
                # FAISS 점수는 거리이므로 유사도로 변환 (1 / (1 + distance))
                similarity_score = 1 / (1 + score)
                results.append({
                    'document': doc,
                    'score': similarity_score
                })
            
            return results
        except Exception as e:
            st.warning(f"시멘틱 검색 실패: {str(e)}")
            return []
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """관련 문서 검색 (하이브리드 방식)"""
        try:
            # 시멘틱 검색 수행 (더 많은 결과 가져오기)
            semantic_results = self._semantic_search(query, k=self._k * 2)
            
            # 키워드 매칭 점수 계산
            keyword_boosted_results = []
            for result in semantic_results:
                doc = result['document']
                semantic_score = result['score']
                
                # 키워드 매칭 점수 계산
                keyword_score = self._calculate_keyword_score(query, doc.page_content)
                
                # 가중 평균 계산 (시멘틱 70% + 키워드 30%)
                combined_score = (
                    self._semantic_weight * semantic_score +
                    self._keyword_weight * keyword_score
                )
                
                keyword_boosted_results.append({
                    'document': doc,
                    'combined_score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score
                })
            
            # 결합된 점수로 정렬
            sorted_results = sorted(
                keyword_boosted_results,
                key=lambda x: x['combined_score'],
                reverse=True
            )
            
            # 상위 k개 문서 반환
            top_documents = []
            for result in sorted_results[:self._k]:
                top_documents.append(result['document'])
            
            return top_documents
        except Exception as e:
            st.warning(f"하이브리드 검색 실패: {str(e)}")
            # 실패 시 기본 벡터 검색으로 폴백
            try:
                return self._vector_store.similarity_search(query, k=self._k)
            except:
                return []
    
    def _calculate_keyword_score(self, query: str, document_content: str) -> float:
        """키워드 매칭 점수 계산"""
        try:
            # 쿼리에서 키워드 추출 (한글, 영문, 숫자)
            query_keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
            
            if not query_keywords:
                return 0.0
            
            # 문서 내용에서 키워드 매칭 확인
            doc_content_lower = document_content.lower()
            matched_keywords = 0
            
            for keyword in query_keywords:
                if len(keyword) > 1 and keyword in doc_content_lower:  # 1글자 키워드는 제외
                    matched_keywords += 1
            
            # 키워드 매칭 비율 계산
            keyword_score = matched_keywords / len(query_keywords)
            
            return keyword_score
        except Exception as e:
            return 0.0
```

### 5. BaseRetriever 필드 충돌 문제 해결

#### 문제점
```
"HybridRetriever" object has no field "vector_store"
```

#### 해결방법
- 모든 변수를 private 변수로 변경 (`_vector_store`, `_semantic_weight` 등)
- `BaseRetriever`와의 필드 충돌 방지

```python
# 변경 전
self.vector_store = vector_store
self.semantic_weight = semantic_weight
self.keyword_weight = keyword_weight
self.k = k

# 변경 후
self._vector_store = vector_store
self._semantic_weight = semantic_weight
self._keyword_weight = keyword_weight
self._k = k
```

### 6. RAG 체인 통합

#### 변경 내용
**파일**: `bizmob_chatbot/bizmob_chatbot.py` - `get_rag_chain()` 함수

```python
def get_rag_chain() -> Runnable:
    """RAG 체인 생성"""
    try:
        # 선택된 모델 가져오기
        selected_model = st.session_state.get('selected_model', 'hyperclovax')
        
        # Ollama LLM 초기화
        llm = OllamaLLM(
            model=selected_model,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048
        )
        
        # 선택된 임베딩 모델 사용
        embeddings = get_embedding_model()
        
        # 벡터 스토어 로드 (allow_dangerous_deserialization=True 추가)
        vector_store = FAISS.load_local(get_vector_db_path(), embeddings, allow_dangerous_deserialization=True)
        
        # 프롬프트 템플릿
        template = """당신은 bizMOB Platform 전문가입니다. 
다음 컨텍스트를 사용하여 질문에 답변해주세요:

컨텍스트:
{context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        # 하이브리드 리트리버 생성 (시멘틱 70% + 키워드 30%)
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            semantic_weight=0.7,
            keyword_weight=0.3,
            k=3
        )
        
        # RAG 체인 생성
        chain = (
            {"context": hybrid_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        st.error(f"RAG 체인 생성 중 오류: {str(e)}")
        return None
```

### 7. process_question 함수 수정

#### 변경 내용
**파일**: `bizmob_chatbot/bizmob_chatbot.py` - `process_question()` 함수

```python
@st.cache_data
def process_question(user_question):
    try:
        # RAG 체인 선언
        chain = get_rag_chain()
        if chain is None:
            st.error("RAG 체인 생성에 실패했습니다.")
            return None, []
        
        # 질문만 전달하여 RAG 체인 실행
        response = chain.invoke(user_question)
        
        # 관련 문서는 하이브리드 검색으로 검색 (참조용)
        embeddings = get_embedding_model()
        new_db = FAISS.load_local(get_vector_db_path(), embeddings, allow_dangerous_deserialization=True)
        hybrid_retriever = HybridRetriever(
            vector_store=new_db,
            semantic_weight=0.7,
            keyword_weight=0.3,
            k=3
        )
        retrieve_docs: List[Document] = hybrid_retriever.invoke(user_question)

        return response, retrieve_docs
    except Exception as e:
        st.error(f"질문 처리 중 오류 발생: {str(e)}")
        return None, []
```

## 하이브리드 검색 알고리즘 상세

### 1. 시멘틱 검색 (70%)
- FAISS 벡터 유사도 검색 사용
- 쿼리를 임베딩하여 벡터 공간에서 유사한 문서 검색
- FAISS 거리 점수를 유사도 점수로 변환: `1 / (1 + distance)`

### 2. 키워드 검색 (30%)
- 정규표현식을 사용한 키워드 매칭
- 한글, 영문, 숫자 키워드 추출
- 1글자 키워드는 제외하여 노이즈 방지
- 키워드 매칭 비율 계산: `matched_keywords / total_keywords`

### 3. 가중 평균 계산
```python
combined_score = (
    self._semantic_weight * semantic_score +  # 70%
    self._keyword_weight * keyword_score      # 30%
)
```

### 4. 결과 정렬 및 반환
- 결합된 점수로 내림차순 정렬
- 상위 k개 문서 반환

## 성능 최적화

### 1. 캐싱 적용
- `@st.cache_data` 데코레이터로 중복 계산 방지
- 임베딩 모델 로드 캐싱
- 벡터DB 로드 캐싱

### 2. 오류 처리 및 폴백
- 각 단계별 예외 처리
- 시멘틱 검색 실패 시 기본 벡터 검색으로 폴백
- 임베딩 모델 로드 실패 시 기본 모델로 폴백

### 3. 동적 경로 탐지
- FAISS 인덱스 경로 동적 탐지
- 모델 정보 파일 자동 검색
- 유연한 파일 구조 지원

## 테스트 및 검증

### 1. 실행 방법
```powershell
cd bizmob_chatbot
python -m streamlit run bizmob_chatbot.py
```

### 2. 확인 사항
- 애플리케이션 정상 실행
- FAISS 벡터DB 정상 로드
- 하이브리드 검색 정상 작동
- 오류 메시지 없음

## 향후 개선 방향

### 1. 성능 최적화
- 벡터 검색 속도 개선
- 메모리 사용량 최적화
- 캐싱 전략 개선

### 2. 기능 확장
- 더 다양한 문서 형식 지원
- 다국어 지원 확대
- 검색 가중치 동적 조정

### 3. 사용자 경험
- 검색 결과 시각화 개선
- 실시간 검색 제안
- 검색 히스토리 관리

---

**작성일**: 2024년 12월
**버전**: 1.0
**작성자**: AI Assistant 

---

## [2024-07-24] 롤백 및 안정화 작업

### 변경점 및 수정점

- 부모자식(계층형) 검색 관련 import 및 코드 완전 삭제
- sys.path.append 등 import 경로 조작 코드 삭제
- 임베딩 모델을 항상 'jhgan/ko-sroberta-multitask'로 고정 (meta tensor 오류 방지)
- HybridRetriever만 남기고 계층형 관련 함수/클래스/코드/주석 모두 삭제
- 전체적으로 import, 함수, 클래스 등 불필요하거나 문제 있는 부분 정리
- 최소한의 안정적인 하이브리드 검색 챗봇 코드로 복원

--- 

### 정확도 향상 원인

- 계층형(부모자식) 검색 코드 완전 제거: parent/child 매핑 오류, 불완전한 메타데이터, 계층형 DB 미생성 등으로 인한 노이즈가 사라지고, 실제 질문과 가장 유사한 문단만 반환됨
- 임베딩 모델을 'jhgan/ko-sroberta-multitask'로 고정: CPU에서 항상 잘 동작하는 한국어 특화 모델로, 검색 품질이 일관되고 오류가 사라짐
- 불필요한 코드/경로/캐시 문제 제거: 벡터DB와 임베딩 모델이 항상 정상적으로 로드되어 검색 품질이 안정화됨
- 청킹(chunking) 방식은 동일(800자+100자 오버랩)
- 결과적으로, 불필요한 계층형/복잡한 코드 제거와 안정적인 임베딩 모델 고정으로 인해 가장 유사한 문단만 정확하게 반환되어 검색 정확도가 체감상 상승함

--- 