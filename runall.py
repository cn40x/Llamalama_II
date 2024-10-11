import os
from threading import Thread

def run_streamlit():
    os.system(r"streamlit run ./stream4.py --server.port 80")
def main():
    t=Thread(target=run_streamlit)
    t.start()
    
if __name__=="__main__":
    main()