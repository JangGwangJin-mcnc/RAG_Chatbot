# Hybrid Search (하이브리드 서치) 동작 원리 및 실제 검색 흐름

이 문서는 bizMOB 프로젝트에서 사용되는 하이브리드 서치(HybridRetriever) 기반의 검색 방식과, 사용자가 질문을 입력했을 때 실제로 어떤 과정으로 검색이 이루어지는지 상세히 설명합니다.

---

## 1. 하이브리드 서치란?

- **시멘틱(의미 기반) 검색**과 **키워드(문자열) 검색**을 결합한 검색 방식입니다.
- 시멘틱 검색은 문장의 의미적 유사도를, 키워드 검색은 쿼리와 문서의 단어 일치 정도를 반영합니다.
- 두 점수를 가중 평균하여, 더 정확하고 실용적인 검색 결과를 제공합니다.

---

## 2. 주요 클래스 및 함수 구조

### HybridRetriever 클래스
- `BaseRetriever`를 상속받아 커스텀 구현
- 주요 파라미터:
  - `vector_store`: FAISS 등 벡터DB 인스턴스
  - `semantic_weight`: 시멘틱 점수 가중치 (기본 0.7)
  - `keyword_weight`: 키워드 점수 가중치 (기본 0.3)
  - `k`: 반환할 문서 개수

#### 주요 메서드
- `_semantic_search(query, k)`: 벡터DB에서 의미 기반 유사도 검색
- `_calculate_keyword_score(query, document_content)`: 쿼리와 문서의 키워드 매칭 점수 계산
- `_get_relevant_documents(query)`: 두 점수를 결합해 상위 k개 문서 반환 (실제 검색 핵심)

---

## 3. 실제 검색 동작 흐름

### 1) 사용자가 질문 입력
- Streamlit UI에서 사용자가 질문을 입력합니다.

### 2) RAG 체인에서 HybridRetriever 호출
- 질문이 들어오면, RAG(Retrieval-Augmented Generation) 체인에서 HybridRetriever의 `_get_relevant_documents`가 호출됩니다.

### 3) 시멘틱 검색 수행
- `_semantic_search`에서 벡터DB(FAISS)의 `similarity_search_with_score`로 쿼리와 의미적으로 유사한 문서들을 k*2개 정도 가져옵니다.
- 각 결과는 (문서, 거리) 형태로 반환되며, 거리를 유사도로 변환(1/(1+거리))합니다.

### 4) 키워드 점수 계산
- 각 문서에 대해 `_calculate_keyword_score`를 호출하여, 쿼리와 문서의 키워드(한글, 영문, 숫자) 일치 정도를 0~1 사이 점수로 계산합니다.

### 5) 점수 결합 및 정렬
- 시멘틱 점수와 키워드 점수를 가중 평균하여(`semantic_weight * 시멘틱 + keyword_weight * 키워드`),
- 결합 점수로 내림차순 정렬 후 상위 k개 문서를 최종 결과로 반환합니다.

### 6) LLM에 컨텍스트로 전달
- 최종적으로 선택된 문서들이 LLM(예: Llama3, Gemma3 등)에 컨텍스트로 전달되어, 답변 생성에 활용됩니다.

---

## 4. 점수 계산 예시

- 예를 들어, 쿼리가 "전자결재 승인 방법"이고, 문서 A의 시멘틱 점수가 0.8, 키워드 점수가 0.5라면:
  - 결합 점수 = 0.7 * 0.8 + 0.3 * 0.5 = 0.56 + 0.15 = 0.71
- 여러 문서 중 결합 점수가 높은 순서대로 상위 k개가 선택됩니다.

---

## 5. 실제 코드 예시 (요약)

```python
class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        semantic_results = self._semantic_search(query, k=self._k * 2)
        keyword_boosted_results = []
        for result in semantic_results:
            doc = result['document']
            semantic_score = result['score']
            keyword_score = self._calculate_keyword_score(query, doc.page_content)
            combined_score = (
                self._semantic_weight * semantic_score +
                self._keyword_weight * keyword_score
            )
            keyword_boosted_results.append({
                'document': doc,
                'combined_score': combined_score
            })
        sorted_results = sorted(keyword_boosted_results, key=lambda x: x['combined_score'], reverse=True)
        return [r['document'] for r in sorted_results[:self._k]]
```

---

## 6. 요약
- 하이브리드 서치는 의미와 키워드의 장점을 모두 살려, 실제 업무 문서 검색에서 높은 정확도와 실용성을 제공합니다.
- 사용자는 자연어로 질문만 입력하면, 시스템이 자동으로 두 방식의 검색을 결합해 최적의 답변 근거를 찾아줍니다.

---

문의사항이나 추가 설명이 필요하면 언제든 말씀해 주세요! 

---

## 7. 키워드 점수 계산 로직 상세

### 실제 코드 (HybridRetriever 클래스 내부)

```python
    def _calculate_keyword_score(self, query: str, document_content: str) -> float:
        """키워드 매칭 점수 계산"""
        try:
            # 쿼리에서 한글, 영문, 숫자 키워드 추출
            query_keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query.lower())
            
            if not query_keywords:
                return 0.0
            
            # 문서 내용 소문자 변환
            doc_content_lower = document_content.lower()
            matched_keywords = 0
            
            # 각 키워드가 문서에 포함되어 있으면 카운트 (1글자 키워드는 제외)
            for keyword in query_keywords:
                if len(keyword) > 1 and keyword in doc_content_lower:
                    matched_keywords += 1
            
            # 키워드 매칭 비율 계산 (0~1 사이)
            keyword_score = matched_keywords / len(query_keywords)
            
            return keyword_score
        except Exception as e:
            return 0.0
```

### 동작 원리
1. **쿼리에서 키워드 추출**: 한글, 영문, 숫자로 이루어진 단어(2글자 이상)만 추출
2. **문서 내용 소문자 변환**: 대소문자 구분 없이 비교
3. **키워드 매칭 개수 카운트**: 각 키워드가 문서에 포함되어 있으면 카운트
4. **최종 점수 계산**: 매칭된 키워드 개수 / 전체 키워드 개수 (0~1 사이 실수)
5. **예외 처리**: 오류 발생 시 0.0 반환

### 예시
- 쿼리: "전자결재 승인 방법"
- 문서: "전자결재 시스템에서 승인 절차는 다음과 같습니다..."
- 추출 키워드: ["전자결재", "승인", "방법"]
- 문서에 포함된 키워드: 2개 (전자결재, 승인)
- **키워드 점수 = 2 / 3 ≈ 0.67**

---

## 8. 개선 방향 및 예시 코드

### 개선 아이디어
- 불용어(Stopwords) 제거
- 부분 일치/유사어 허용 (형태소 분석, 어간 추출, 오타 허용 등)
- 키워드별 가중치 부여
- TF-IDF 등 통계적 방법 활용
- 문서 내 등장 빈도 반영
- 문장 단위 매칭 등

### 개선 예시 코드 (TF-IDF 활용)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def improved_keyword_score(query, document_content):
    # 불용어 제거, 형태소 분석 등 추가 가능
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([query, document_content])
    # 코사인 유사도 등으로 점수 산출
    score = (tfidf * tfidf.T).A[0,1]
    return score
```

---

이처럼 키워드 점수 계산은 단순 매칭부터 통계적/자연어처리 기반까지 다양하게 확장할 수 있습니다. 실제 서비스 품질을 높이고 싶다면 위의 개선 방향을 참고해 적용해보세요! 