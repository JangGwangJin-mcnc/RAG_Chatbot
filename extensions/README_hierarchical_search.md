# 🔍 부모-자식 전략 계층적 검색

## 📋 개요

기존의 단일 벡터DB 검색을 부모-자식 계층 구조로 개선하여 더 정확하고 관련성 높은 검색 결과를 제공합니다.

## 🏗️ 아키텍처

```
부모 벡터DB (개요/개념)
├── 자식 벡터DB 1 (설정/가이드)
├── 자식 벡터DB 2 (매뉴얼/튜토리얼)
└── 자식 벡터DB 3 (문제해결/예제)
```

## 🔄 검색 프로세스

1. **키워드 검색** → 부모 벡터DB에서 관련 주제 찾기
2. **부모 결과 분석** → 연결된 자식 벡터DB들 식별
3. **자식 DB 검색** → 관련 상세 내용 검색
4. **결과 통합** → 부모 + 자식 결과를 가중치로 조합
5. **순위 조정** → 관련성 점수 기반 최종 순위 결정

## 📁 파일 구조

```
extensions/hierarchical_search/
├── __init__.py                    # 모듈 초기화
├── metadata_manager.py           # 메타데이터 관리
├── vector_db_hierarchy.py        # 계층적 벡터DB 관리
├── search_combiner.py            # 검색 결과 통합
├── parent_child_strategy.py      # 메인 전략 클래스
└── README_hierarchical_search.md # 이 파일

extensions/
├── hierarchical_search_example.py # 사용 예시
└── README_hierarchical_search.md  # 이 파일
```

## 🎯 주요 기능

### **1. 자동 문서 분류**
- 키워드 기반 부모/자식 문서 분류
- 파일명 패턴 분석
- 내용 기반 주제 추출

### **2. 계층적 벡터DB 관리**
- 부모-자식 관계 자동 매핑
- 주제별 벡터DB 분리 저장
- 계층 정보 자동 저장/로드

### **3. 스마트 검색**
- 부모 DB에서 개요/개념 검색
- 관련 자식 DB에서 상세 정보 검색
- 가중치 기반 결과 통합

### **4. 결과 순위 조정**
- 관련성 점수 계산
- 부모/자식 가중치 적용
- 중복 제거 및 필터링

## 🚀 사용 방법

### **1. 기본 사용**
```python
from extensions.hierarchical_search import ParentChildSearchStrategy
from langchain_huggingface import HuggingFaceEmbeddings

# 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask"
)

# 부모-자식 전략 초기화
strategy = ParentChildSearchStrategy(embedding_model)

# 계층적 벡터DB 생성
strategy.setup_hierarchy(documents)

# 검색 수행
results, summary = strategy.search("bizMOB Client 설정 방법")
```

### **2. 기존 벡터DB 로드**
```python
# 저장된 계층적 벡터DB 로드
if strategy.load_hierarchy():
    results, summary = strategy.search(query)
```

### **3. 통계 정보 확인**
```python
# 벡터DB 통계
stats = strategy.get_statistics()
print(f"부모 문서: {stats['parent_documents']}개")
print(f"자식 DB: {stats['child_databases']}개")

# 계층 구조 정보
hierarchy_info = strategy.get_hierarchy_info()
```

## ⚙️ 설정 옵션

### **가중치 설정**
```python
# SearchCombiner에서 조정 가능
self.parent_weight = 0.4    # 부모 문서 가중치
self.child_weight = 0.6     # 자식 문서 가중치
self.relevance_threshold = 0.7  # 관련성 임계값
```

### **검색 개수 설정**
```python
# ParentChildSearchStrategy에서 조정 가능
self.parent_search_k = 2    # 부모 DB 검색 개수
self.child_search_k = 3     # 자식 DB 검색 개수
```

### **부모-자식 매핑**
```python
# MetadataManager에서 조정 가능
self.parent_child_mapping = {
    'bizMOB_Client': ['client_setup', 'client_config', 'client_troubleshooting'],
    'bizMOB_Server': ['server_setup', 'server_config', 'server_maintenance'],
    # ... 추가 매핑
}
```

## 📊 검색 결과 예시

### **입력 쿼리**: "bizMOB Client 설정 방법"

### **검색 과정**:
1. **부모 DB 검색**: "bizMOB Client" 관련 개요 문서 찾기
2. **자식 DB 검색**: 
   - `client_setup` DB에서 설치 가이드
   - `client_config` DB에서 설정 매뉴얼
   - `client_troubleshooting` DB에서 문제해결
3. **결과 통합**: 개요 + 상세 설정 정보

### **결과 형태**:
```
📋 검색 결과 요약:
- 총 문서: 8개
- 부모 문서: 2개 (개요/개념)
- 자식 문서: 6개 (설정/가이드/문제해결)
- 관련 주제: bizMOB_Client
- 평균 관련성: 0.85
```

## 🔧 확장 가능성

### **1. 다중 계층 지원**
- 부모 → 자식 → 손자 계층 구조
- 더 세분화된 분류 체계

### **2. 동적 가중치**
- 쿼리 유형에 따른 가중치 자동 조정
- 사용자 피드백 기반 학습

### **3. 실시간 업데이트**
- 새 문서 추가 시 자동 계층 재구성
- 증분 벡터DB 업데이트

## 🎯 장점

1. **정확성 향상**: 부모-자식 관계로 더 관련성 높은 결과
2. **컨텍스트 제공**: 개요 + 상세 정보의 조합
3. **확장성**: 새로운 주제/카테고리 쉽게 추가
4. **유연성**: 가중치 및 매핑 자유롭게 조정

## 🚀 실행 예시

```bash
# 예시 코드 실행
streamlit run extensions/hierarchical_search_example.py
```

## 📝 주의사항

1. **초기 설정**: 문서 분류 규칙을 도메인에 맞게 조정 필요
2. **메모리 사용**: 여러 벡터DB로 인한 메모리 사용량 증가
3. **검색 시간**: 계층적 검색으로 인한 약간의 성능 저하
4. **데이터 품질**: 부모-자식 분류의 정확도가 결과 품질에 영향 