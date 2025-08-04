import os
import shutil
import time
import gc
import subprocess

def test_chroma_deletion():
    chroma_path = "./chroma_db"
    
    print(f"테스트 시작: {chroma_path}")
    print(f"폴더 존재: {os.path.exists(chroma_path)}")
    
    if os.path.exists(chroma_path):
        print("폴더 내용:")
        for root, dirs, files in os.walk(chroma_path):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  - {file_path}")
        
        # 1단계: 파일 속성 변경
        print("\n1단계: 파일 속성 변경")
        for root, dirs, files in os.walk(chroma_path):
            for file in files:
                if file.endswith('.sqlite3') or file.endswith('.db'):
                    file_path = os.path.join(root, file)
                    try:
                        subprocess.run(['attrib', '-R', '-H', '-S', file_path], 
                                      shell=True, capture_output=True, timeout=3)
                        print(f"  속성 변경 완료: {file_path}")
                    except Exception as e:
                        print(f"  속성 변경 실패: {file_path} - {e}")
        
        # 2단계: shutil.rmtree 시도
        print("\n2단계: shutil.rmtree 시도")
        try:
            shutil.rmtree(chroma_path, ignore_errors=True)
            if not os.path.exists(chroma_path):
                print("  ✅ shutil.rmtree 성공")
                return True
            else:
                print("  ❌ shutil.rmtree 실패")
        except Exception as e:
            print(f"  ❌ shutil.rmtree 오류: {e}")
        
        # 3단계: 개별 파일 삭제
        print("\n3단계: 개별 파일 삭제")
        for root, dirs, files in os.walk(chroma_path, topdown=False):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    os.chmod(file_path, 0o777)
                    os.remove(file_path)
                    print(f"  파일 삭제 완료: {file_path}")
                except Exception as e:
                    print(f"  파일 삭제 실패: {file_path} - {e}")
        
        # 4단계: 폴더 삭제
        print("\n4단계: 폴더 삭제")
        for root, dirs, files in os.walk(chroma_path, topdown=False):
            for dir in dirs:
                try:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
                    print(f"  폴더 삭제 완료: {dir_path}")
                except Exception as e:
                    print(f"  폴더 삭제 실패: {dir_path} - {e}")
        
        # 5단계: 최종 확인
        if not os.path.exists(chroma_path):
            print("  ✅ 최종 삭제 성공")
            return True
        else:
            print("  ❌ 최종 삭제 실패")
            
            # 6단계: 강제 삭제
            print("\n5단계: 강제 삭제")
            try:
                result = subprocess.run(['rmdir', '/S', '/Q', chroma_path], 
                                      shell=True, capture_output=True, timeout=15)
                print(f"  rmdir 결과: {result.returncode}")
                
                if result.returncode != 0 and os.path.exists(chroma_path):
                    ps_command = f'Remove-Item -Path "{chroma_path}" -Recurse -Force -ErrorAction SilentlyContinue'
                    result2 = subprocess.run(['powershell', '-Command', ps_command], 
                                           capture_output=True, timeout=15)
                    print(f"  PowerShell 결과: {result2.returncode}")
                
                if not os.path.exists(chroma_path):
                    print("  ✅ 강제 삭제 성공")
                    return True
                else:
                    print("  ❌ 강제 삭제 실패")
                    return False
            except Exception as e:
                print(f"  ❌ 강제 삭제 오류: {e}")
                return False
    else:
        print("삭제할 폴더가 없습니다.")
        return True

if __name__ == "__main__":
    success = test_chroma_deletion()
    print(f"\n최종 결과: {'성공' if success else '실패'}") 