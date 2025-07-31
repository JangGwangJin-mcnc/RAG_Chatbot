# SharePoint 연동 유틸리티 (기본 예시)
# Office365-REST-Python-Client 필요: pip install Office365-REST-Python-Client

from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.user_credential import UserCredential
import os

# SharePoint 접속 정보 (사용자 맞춤 입력 필요)
SHAREPOINT_SITE_URL = "https://yourcompany.sharepoint.com/sites/your-site-name"
SHAREPOINT_DOC_LIB = "Shared Documents"  # 문서 라이브러리 이름
SHAREPOINT_USERNAME = "your-email@yourcompany.com"
SHAREPOINT_PASSWORD = "your-password"


def get_sharepoint_context():
    ctx = ClientContext(SHAREPOINT_SITE_URL).with_credentials(
        UserCredential(SHAREPOINT_USERNAME, SHAREPOINT_PASSWORD)
    )
    return ctx


def list_files_in_folder(folder_path="/"):
    """
    SharePoint 문서 라이브러리 내 폴더의 파일 목록을 반환
    """
    ctx = get_sharepoint_context()
    folder = ctx.web.get_folder_by_server_relative_url(f"{SHAREPOINT_DOC_LIB}{folder_path}")
    files = folder.files
    ctx.load(files)
    ctx.execute_query()
    return [f.properties for f in files]


def download_file_from_sharepoint(server_relative_path, local_path):
    """
    SharePoint에서 파일을 다운로드하여 local_path에 저장
    server_relative_path 예시: '/Shared Documents/테스트.pdf'
    """
    ctx = get_sharepoint_context()
    with open(local_path, "wb") as output_file:
        response = ctx.web.get_file_by_server_relative_url(server_relative_path).download(output_file).execute_query()
    return local_path

# 사용 예시 (직접 실행 시)
if __name__ == "__main__":
    # 파일 목록 조회
    files = list_files_in_folder("/")
    for f in files:
        print(f["Name"], f["ServerRelativeUrl"])
    # 파일 다운로드 예시
    # download_file_from_sharepoint('/Shared Documents/테스트.pdf', '로컬저장경로/테스트.pdf') 