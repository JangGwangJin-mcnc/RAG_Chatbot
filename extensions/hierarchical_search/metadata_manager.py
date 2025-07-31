# 메타데이터 관리 및 부모-자식 관계 설정
from typing import List, Dict, Any, Optional
from langchain_core.documents.base import Document
import re
import json
import os

class MetadataManager:
    """문서 메타데이터 관리 및 부모-자식 관계 설정"""
    
    def __init__(self):
        # 부모-자식 관계 매핑
        self.parent_child_mapping = {
            'bizMOB_Client': ['client_setup', 'client_config', 'client_troubleshooting'],
            'bizMOB_Server': ['server_setup', 'server_config', 'server_maintenance'],
            'bizMOB_Platform': ['platform_overview', 'platform_features', 'platform_architecture'],
            'SSL_VPN': ['vpn_setup', 'vpn_config', 'vpn_troubleshooting'],
            'API': ['api_documentation', 'api_examples', 'api_integration'],
            'Database': ['db_setup', 'db_config', 'db_maintenance'],
            'Security': ['security_guide', 'security_config', 'security_best_practices']
        }
        
        # 키워드 기반 분류 규칙
        self.classification_keywords = {
            'parent': [
                '개요', '소개', '개념', '아키텍처', '구조', '설명', '개발', '플랫폼',
                'overview', 'introduction', 'concept', 'architecture', 'structure', 
                'explanation', 'development', 'platform'
            ],
            'child': [
                '설정', '설치', '구성', '매뉴얼', '가이드', '튜토리얼', '예제', '문제해결',
                'setup', 'install', 'configuration', 'manual', 'guide', 'tutorial', 
                'example', 'troubleshooting', 'error', 'issue'
            ]
        }
    
    def classify_document(self, doc: Document) -> Dict[str, Any]:
        """문서를 부모/자식으로 분류하고 메타데이터 추가"""
        content = doc.page_content.lower()
        file_name = doc.metadata.get('file_name', '').lower()
        
        # 기본 메타데이터 복사
        metadata = doc.metadata.copy()
        
        # 문서 타입 분류 (부모/자식)
        doc_type = self._determine_document_type(content, file_name)
        metadata['doc_type'] = doc_type
        
        # 주제 분류
        topic = self._extract_topic(content, file_name)
        metadata['topic'] = topic
        
        # 부모 문서인 경우
        if doc_type == 'parent':
            metadata['parent_id'] = f"parent_{topic}_{hash(file_name)}"
            metadata['children'] = self.parent_child_mapping.get(topic, [])
        
        # 자식 문서인 경우
        elif doc_type == 'child':
            parent_topic = self._find_parent_topic(topic)
            metadata['parent_topic'] = parent_topic
            metadata['parent_id'] = f"parent_{parent_topic}_{hash(parent_topic)}"
            metadata['child_category'] = topic
        
        return metadata
    
    def _determine_document_type(self, content: str, file_name: str) -> str:
        """문서가 부모인지 자식인지 판단"""
        parent_score = 0
        child_score = 0
        
        # 키워드 기반 점수 계산
        for keyword in self.classification_keywords['parent']:
            if keyword in content or keyword in file_name:
                parent_score += 1
        
        for keyword in self.classification_keywords['child']:
            if keyword in content or keyword in file_name:
                child_score += 1
        
        # 파일명 패턴 기반 분류
        if any(pattern in file_name for pattern in ['overview', 'intro', 'guide']):
            parent_score += 2
        if any(pattern in file_name for pattern in ['setup', 'config', 'manual']):
            child_score += 2
        
        # 점수 비교
        if parent_score > child_score:
            return 'parent'
        else:
            return 'child'
    
    def _extract_topic(self, content: str, file_name: str) -> str:
        """문서에서 주제 추출"""
        # 파일명에서 주제 추출
        for topic in self.parent_child_mapping.keys():
            if topic.lower() in file_name:
                return topic
        
        # 내용에서 주제 추출
        for topic in self.parent_child_mapping.keys():
            if topic.lower() in content:
                return topic
        
        # 기본값
        return 'general'
    
    def _find_parent_topic(self, child_topic: str) -> str:
        """자식 주제에 해당하는 부모 주제 찾기"""
        for parent, children in self.parent_child_mapping.items():
            if child_topic in children:
                return parent
        return 'general'
    
    def get_related_child_topics(self, parent_topic: str) -> List[str]:
        """부모 주제에 관련된 자식 주제들 반환"""
        return self.parent_child_mapping.get(parent_topic, [])
    
    def save_mapping(self, file_path: str = "hierarchical_mapping.json"):
        """부모-자식 매핑 정보를 파일로 저장"""
        mapping_data = {
            'parent_child_mapping': self.parent_child_mapping,
            'classification_keywords': self.classification_keywords
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    def load_mapping(self, file_path: str = "hierarchical_mapping.json"):
        """저장된 매핑 정보를 파일에서 로드"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                self.parent_child_mapping = mapping_data.get('parent_child_mapping', {})
                self.classification_keywords = mapping_data.get('classification_keywords', {}) 