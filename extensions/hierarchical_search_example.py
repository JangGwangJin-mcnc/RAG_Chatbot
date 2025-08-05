# 부모-자식 전략 사용 예시
import streamlit as st
import os
from extensions.hierarchical_search import ParentChildSearchStrategy

# SafeSentenceTransformerEmbeddings 클래스 정의 (torch.load 취약점 방지)
class SafeSentenceTransformerEmbeddings:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self._load_model()
    
    def _load_model(self):
        """모델을 안전하게 로드 (safetensors 사용)"""
        try:
            # 환경 변수 설정으로 safetensors 강제 사용
            os.environ['SAFETENSORS_FAST_GPU'] = '1'
            os.environ['TRANSFORMERS_USE_SAFETENSORS'] = '1'
            os.environ['TORCH_WEIGHTS_ONLY'] = '1'
            os.environ['TRANSFORMERS_SAFE_SERIALIZATION'] = '1'
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
        except Exception as e:
            raise Exception(f"모델 로딩 실패: {e}")
    
    def embed_documents(self, texts):
        """문서 임베딩"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str):
        """쿼리 임베딩"""
        return self.model.encode([text])[0].tolist()

def hierarchical_search_example():
    """부모-자식 전략 사용 예시"""
    
    st.header("🔍 부모-자식 전략 검색 예시")
    
    # 1. 임베딩 모델 초기화
    st.subheader("1. 임베딩 모델 초기화")
    try:
        # SafeSentenceTransformerEmbeddings 사용 (torch.load 취약점 방지)
        embedding_model = SafeSentenceTransformerEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            device='cpu'
        )
        st.success("✅ 임베딩 모델 로드 완료")
    except Exception as e:
        st.error(f"❌ 임베딩 모델 로드 실패: {str(e)}")
        return
    
    # 2. 부모-자식 전략 초기화
    st.subheader("2. 부모-자식 전략 초기화")
    strategy = ParentChildSearchStrategy(embedding_model)
    
    # 3. 기존 벡터DB 로드 시도
    st.subheader("3. 기존 계층적 벡터DB 확인")
    if strategy.exists():
        st.success("✅ 기존 계층적 벡터DB 발견")
        
        # 통계 정보 표시
        stats = strategy.get_statistics()
        st.info("📊 벡터DB 통계:")
        st.write(f"- 부모 문서: {stats['parent_documents']}개")
        st.write(f"- 자식 DB: {stats['child_databases']}개")
        st.write(f"- 자식 문서 총합: {stats['total_child_documents']}개")
        st.write(f"- 주제: {', '.join(stats['topics'])}")
        
        # 계층 정보 표시
        hierarchy_info = strategy.get_hierarchy_info()
        st.info("🏗️ 계층 구조:")
        for parent, children in hierarchy_info['parent_child_mapping'].items():
            st.write(f"- **{parent}**: {', '.join(children)}")
    
    else:
        st.warning("⚠️ 기존 계층적 벡터DB가 없습니다")
        st.info("💡 새로운 계층적 벡터DB를 생성하려면 문서를 업로드하세요")
    
    # 4. 검색 테스트
    st.subheader("4. 계층적 검색 테스트")
    
    if strategy.exists():
        # 검색 쿼리 입력
        query = st.text_input(
            "검색할 키워드를 입력하세요:",
            placeholder="예: bizMOB Client 설정 방법"
        )
        
        if st.button("🔍 계층적 검색 실행") and query:
            with st.spinner("계층적 검색을 수행하는 중..."):
                # 검색 실행
                results, summary = strategy.search(query)
                
                # 결과 표시
                st.subheader("📋 검색 결과")
                
                # 요약 정보
                st.info("📊 검색 결과 요약:")
                st.write(f"- 총 문서: {summary['total_count']}개")
                st.write(f"- 부모 문서: {summary['parent_count']}개")
                st.write(f"- 자식 문서: {summary['child_count']}개")
                st.write(f"- 관련 주제: {', '.join(summary['topics'])}")
                st.write(f"- 평균 관련성: {summary['avg_relevance']:.3f}")
                
                # 상세 결과
                if results:
                    st.subheader("📄 상세 결과")
                    for i, doc in enumerate(results[:5]):  # 상위 5개만 표시
                        with st.expander(f"결과 {i+1}: {doc.metadata.get('file_name', 'Unknown')}"):
                            st.write(f"**타입**: {doc.metadata.get('source_type', 'unknown')}")
                            st.write(f"**주제**: {doc.metadata.get('topic', 'unknown')}")
                            st.write(f"**관련성 점수**: {doc.metadata.get('relevance_score', 0):.3f}")
                            st.write(f"**가중 점수**: {doc.metadata.get('weighted_score', 0):.3f}")
                            st.write("**내용**:")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                else:
                    st.warning("검색 결과가 없습니다")
    else:
        st.info("계층적 벡터DB가 없어서 검색을 수행할 수 없습니다")

# 사용 예시
if __name__ == "__main__":
    hierarchical_search_example() 