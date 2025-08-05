# 계층적 벡터DB 관리
from typing import List, Dict, Any, Optional
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
import os
import json
import re

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

class VectorDBHierarchy:
    """계층적 벡터DB 관리 클래스"""
    
    def __init__(self, embedding_model: SafeSentenceTransformerEmbeddings, base_path: str = "hierarchical_vector_db"):
        self.embedding_model = embedding_model
        self.base_path = base_path
        self.parent_db = None
        self.child_dbs = {}
        self.hierarchy_info = {}
        
        # 디렉토리 생성
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(f"{base_path}/parent", exist_ok=True)
        os.makedirs(f"{base_path}/children", exist_ok=True)
    
    def setup_hierarchy(self, documents: List[Document], metadata_manager) -> None:
        """문서들을 부모-자식으로 분류하고 벡터DB 생성"""
        parent_docs = []
        child_docs_by_topic = {}
        
        # 문서 분류
        for doc in documents:
            # 메타데이터 업데이트
            updated_metadata = metadata_manager.classify_document(doc)
            doc.metadata = updated_metadata
            
            # 부모/자식 분류
            if updated_metadata.get('doc_type') == 'parent':
                parent_docs.append(doc)
            else:
                topic = updated_metadata.get('topic', 'general')
                if topic not in child_docs_by_topic:
                    child_docs_by_topic[topic] = []
                child_docs_by_topic[topic].append(doc)
        
        # 부모 벡터DB 생성
        if parent_docs:
            self.create_parent_vector_db(parent_docs)
        
        # 자식 벡터DB들 생성
        for topic, docs in child_docs_by_topic.items():
            self.create_child_vector_db(topic, docs)
        
        # 계층 정보 저장
        self.save_hierarchy_info(metadata_manager)
    
    def create_parent_vector_db(self, parent_docs: List[Document]) -> None:
        """부모 벡터DB 생성"""
        try:
            self.parent_db = FAISS.from_documents(parent_docs, self.embedding_model)
            parent_db_path = f"{self.base_path}/parent/parent_index"
            self.parent_db.save_local(parent_db_path)
            print(f"✅ 부모 벡터DB 생성 완료: {len(parent_docs)}개 문서")
        except Exception as e:
            print(f"❌ 부모 벡터DB 생성 실패: {str(e)}")
    
    def create_child_vector_db(self, topic: str, child_docs: List[Document]) -> None:
        """자식 벡터DB 생성"""
        try:
            child_db = FAISS.from_documents(child_docs, self.embedding_model)
            child_db_path = f"{self.base_path}/children/{topic}_index"
            child_db.save_local(child_db_path)
            self.child_dbs[topic] = child_db
            print(f"✅ 자식 벡터DB 생성 완료 ({topic}): {len(child_docs)}개 문서")
        except Exception as e:
            print(f"❌ 자식 벡터DB 생성 실패 ({topic}): {str(e)}")
    
    def load_hierarchy(self) -> bool:
        """저장된 계층적 벡터DB 로드"""
        try:
            # 부모 DB 로드
            parent_db_path = f"{self.base_path}/parent/parent_index"
            if os.path.exists(parent_db_path):
                self.parent_db = FAISS.load_local(parent_db_path, self.embedding_model)
                print("✅ 부모 벡터DB 로드 완료")
            
            # 자식 DB들 로드
            children_path = f"{self.base_path}/children"
            if os.path.exists(children_path):
                for folder in os.listdir(children_path):
                    if folder.endswith('_index'):
                        topic = folder.replace('_index', '')
                        child_db_path = f"{children_path}/{folder}"
                        self.child_dbs[topic] = FAISS.load_local(child_db_path, self.embedding_model)
                        print(f"✅ 자식 벡터DB 로드 완료 ({topic})")
            
            # 계층 정보 로드
            self.load_hierarchy_info()
            return True
        except Exception as e:
            print(f"❌ 계층적 벡터DB 로드 실패: {str(e)}")
            return False
    
    def save_hierarchy_info(self, metadata_manager) -> None:
        """계층 정보를 파일로 저장"""
        hierarchy_data = {
            'parent_child_mapping': metadata_manager.parent_child_mapping,
            'child_dbs': list(self.child_dbs.keys()),
            'base_path': self.base_path
        }
        
        hierarchy_path = f"{self.base_path}/hierarchy_info.json"
        with open(hierarchy_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_data, f, ensure_ascii=False, indent=2)
    
    def load_hierarchy_info(self) -> None:
        """저장된 계층 정보 로드"""
        hierarchy_path = f"{self.base_path}/hierarchy_info.json"
        if os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'r', encoding='utf-8') as f:
                self.hierarchy_info = json.load(f)
    
    def get_parent_db(self) -> Optional[FAISS]:
        """부모 벡터DB 반환"""
        return self.parent_db
    
    def get_child_db(self, topic: str) -> Optional[FAISS]:
        """특정 주제의 자식 벡터DB 반환"""
        return self.child_dbs.get(topic)
    
    def get_all_child_dbs(self) -> Dict[str, FAISS]:
        """모든 자식 벡터DB 반환"""
        return self.child_dbs.copy()
    
    def get_related_child_topics(self, parent_topic: str) -> List[str]:
        """부모 주제에 관련된 자식 주제들 반환"""
        if 'parent_child_mapping' in self.hierarchy_info:
            return self.hierarchy_info['parent_child_mapping'].get(parent_topic, [])
        return []
    
    def exists(self) -> bool:
        """계층적 벡터DB가 존재하는지 확인"""
        parent_path = f"{self.base_path}/parent/parent_index"
        hierarchy_path = f"{self.base_path}/hierarchy_info.json"
        return os.path.exists(parent_path) and os.path.exists(hierarchy_path) 