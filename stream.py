import streamlit as st
import time
import glob
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from tools import transcribe_audio, OllamaEmbeddings, get_llm_response, clean_text, text_to_speech
from streamlit_mic_recorder import mic_recorder

def main():
    st.set_page_config(page_title="Llamalama II ChatLLM App", page_icon="🤖🦙")
    st.title("Llamalama II 🤖🦙")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'transcription' not in st.session_state:
        st.session_state.transcription = None

    if 'vector_db' not in st.session_state:
        vector_db_path = './local_vector_db'  # Path where the vector database will be saved

        # if os.path.exists(vector_db_path):
        #     with st.spinner("Loading existing vector database..."):
        #         st.session_state.vector_db = Chroma.load(vector_db_path)
        #         st.success("Vector database loaded successfully!")
        # else:
        with st.spinner("Loading PDF and creating vector database..."):
            all_chunks = []
            for i in glob.glob("./docs/*.pdf"):
                loader = UnstructuredPDFLoader(file_path=i)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                all_chunks.extend(chunks)

            embedding_function = OllamaEmbeddings(model='mxbai-embed-large')
            st.session_state.vector_db = Chroma.from_documents(
                documents=all_chunks,
                embedding=embedding_function,
                collection_name="local-rag",
                persist_directory="./local_vector_db",
            )
            st.session_state.vector_db.persist()
        st.success("Vector database created and saved successfully!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your question /พิมเพื่อถาม🤗")

    prompt = None
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
            with st.spinner("Thinking/กำลังคิดอยู่...🦙🤔🧐"):
                llm_response = get_llm_response(prompt, vector_db=st.session_state.vector_db)

            for chunk in llm_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Response time: {elapsed_time:.2f} seconds")

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        clean_response = clean_text(full_response)
        audio_bytes = text_to_speech("lama!" + clean_response + "lama")
        st.audio(audio_bytes, format="audio/wav")


if __name__ == "__main__":
    main()
