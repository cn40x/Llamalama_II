import streamlit as st
import time
import glob
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from tools import transcribe_audio, clean_think_tag, OllamaEmbeddings, get_llm_response, clean_text, text_to_speech,pull_model,voice_convert
from streamlit_mic_recorder import mic_recorder
import nltk
import random
import torch
import whisper


def handle_file_upload(uploaded_files,DOCS_FOLDER:str="./docs"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded {uploaded_file.name} successfully!")
        
        st.session_state.vector_db = process_pdfs_and_create_vector_db()
        
def load_or_create_vector_db(VECTOR_DB_PATH:str='./local_vector_db'):
    if os.path.exists(VECTOR_DB_PATH):
        with st.spinner("Loading existing vector database..."):
            # โหลดฐานข้อมูลจากพาธที่กำหนด
            st.session_state.vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),collection_name="local-rag")
            st.success("Vector database loaded successfully!")
    else:
        st.session_state.vector_db = process_pdfs_and_create_vector_db()

def process_pdfs_and_create_vector_db(VECTOR_DB_PATH:str='./local_vector_db'):
    with st.spinner("Processing PDFs and creating vector database..."):
        all_chunks = []
        for pdf_file in glob.glob(os.path.join("./docs/*.pdf")):
            loader = UnstructuredPDFLoader(file_path=pdf_file)
            data = loader.load()
            st.write(f"🔍 Loaded from {pdf_file}:\n", data[0].page_content[:500])  # แสดงบางส่วนของเนื้อหา
            st.write(f"Vector DB contains {len(st.session_state.vector_db._collection.get()['documents'])} documents") #ลองใส่ st.write() หลังจาก Chroma ถูกสร้าง
            # ปรับการใช้ split_documents() ให้มีขนาดใหญ่ขึ้น
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            all_chunks.extend(chunks)

        embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
        # สร้างฐานข้อมูลเวกเตอร์จากเอกสาร
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_function,
            collection_name="local-rag",
            persist_directory=VECTOR_DB_PATH,
        )
        st.success("Vector database created successfully!")

        return vector_db

def main():
    st.set_page_config(page_title="Llamalama II ChatLLM App", page_icon="🤖🦙")
    st.title("Llamalama II 🤖🦙")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'transcription' not in st.session_state:
        st.session_state.transcription = None

    if 'vector_db' not in st.session_state:
        vector_db_path = './local_vector_db'
        load_or_create_vector_db()

    st.sidebar.header("📂 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    # แสดงชื่อไฟล์ที่อัปโหลด
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Uploaded File: {uploaded_file.name}")

    if st.sidebar.button("Process Files"):
        handle_file_upload(uploaded_files)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your question 🤗 (Englienglish only)")

    prompt = None
    with st.container():
        audio = mic_recorder(start_prompt="to record ⏺️", stop_prompt="to stop⏹️", key='recorder', just_once=True)
        
    if audio:
        st.session_state.transcription = None
        st.audio(audio['bytes'], format="audio/wav")
        transcription = transcribe_audio(audio['bytes'])
        st.session_state.transcription = transcription
        st.write(transcription)
        audio = None 

    if st.session_state.transcription:
        prompt = st.session_state.transcription
        st.session_state.transcription = None

    elif user_input:
        prompt = user_input
        
    else:
        prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            start_time = time.time()
            with st.spinner("Thinking🦙🤔🧐"):
                llm_response = get_llm_response(prompt,agent_type="teacher", vector_db=st.session_state.vector_db,rag=True)
                llm_response = clean_think_tag(text=llm_response)
            for chunk in llm_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        clean_response = clean_text(full_response)
        audio_bytes =text_to_speech("lama!" + clean_response + "lama")
        st.audio(audio_bytes, format="audio/wav")
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Response time: {elapsed_time:.2f} seconds")


        if random.randint(0,10)>7:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                start_time = time.time()
                with st.spinner("Thinking🦙🤔🧐"):
                    student_response = get_llm_response(prompt,agent_type="student",rag=False,t_res=clean_response)
                    student_response = clean_think_tag(text=student_response)
                message_placeholder.markdown(student_response)
                
                audio_bytes =text_to_speech(student_response)
                out_path=voice_convert(audio_bytes)
                if out_path:
                    with open(out_path, "rb") as f:
                        converted_audio_bytes = f.read()
                    st.audio(converted_audio_bytes, format="audio/wav")

                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Response time: {elapsed_time:.2f} seconds")

            
            
if __name__ == "__main__":
    main()