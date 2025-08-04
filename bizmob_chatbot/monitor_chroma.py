import psutil
import time
import os

def monitor_chroma_processes():
    """ChromaDB 파일을 사용하는 프로세스를 실시간으로 모니터링"""
    chroma_path = './chroma_db'
    
    print("=== ChromaDB 프로세스 모니터링 시작 ===")
    print(f"모니터링 경로: {os.path.abspath(chroma_path)}")
    print("Ctrl+C로 종료")
    print("-" * 50)
    
    try:
        while True:
            # ChromaDB 관련 프로세스 찾기
            chroma_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline']).lower()
                        if 'chroma' in cmdline or 'sqlite' in cmdline:
                            chroma_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # ChromaDB 파일을 직접 열고 있는 프로세스 찾기
            file_processes = []
            if os.path.exists(chroma_path):
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        for file in proc.open_files():
                            if chroma_path in file.path:
                                file_processes.append(proc)
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            
            # 결과 출력
            if chroma_processes or file_processes:
                print(f"\n[{time.strftime('%H:%M:%S')}] ChromaDB 관련 프로세스 발견:")
                
                if chroma_processes:
                    print("  ChromaDB 명령어 관련 프로세스:")
                    for proc in chroma_processes:
                        print(f"    PID: {proc.pid}, Name: {proc.name()}")
                        try:
                            print(f"    Cmdline: {' '.join(proc.cmdline())}")
                        except:
                            pass
                
                if file_processes:
                    print("  ChromaDB 파일을 열고 있는 프로세스:")
                    for proc in file_processes:
                        print(f"    PID: {proc.pid}, Name: {proc.name()}")
                        try:
                            for file in proc.open_files():
                                if chroma_path in file.path:
                                    print(f"    파일: {file.path}")
                        except:
                            pass
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] ChromaDB 관련 프로세스 없음", end="")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")

if __name__ == "__main__":
    monitor_chroma_processes() 