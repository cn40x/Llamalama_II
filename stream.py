import streamlit as st
import ollama
import torch
import time


#rag
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

#tts
from gtts import gTTS
from io import BytesIO
import re
from typing import List

#asr
from streamlit_mic_recorder import speech_to_text,mic_recorder


#path
import glob
import os


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = 'mxbai-embed-large'):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [ollama.embeddings(model=self.model, prompt=text)['embedding'] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.model, prompt=text)['embedding']

def get_relevant_context(query: str, vector_db: Chroma, top_k: int = 3, threshold: float = 0.8) -> str:
    query_embedding = torch.tensor(OllamaEmbeddings().embed_query(query))
    all_embeddings = vector_db._collection.get(include=['embeddings', 'documents'])
    embeddings = torch.tensor(all_embeddings['embeddings'])
    documents = all_embeddings['documents']

    cos_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings)
    filtered_scores = cos_scores[cos_scores >= threshold]
    if len(filtered_scores) == 0:
        return "No relevant documents found."

    top_k = min(top_k, len(filtered_scores))
    top_indices = torch.topk(filtered_scores, k=top_k)[1].tolist()

    res = ""
    for i, idx in enumerate(top_indices):
        res += f"{i+1}. {documents[idx]},\n\n"
    
    return res

def get_llm_response(prompt, system_prompt="You are a helpful assistant and You are a knowledgeable and patient English teacher with expertise in grammar, vocabulary, pronunciation, and conversational skills. Your role is to help the student improve their English proficiency by providing clear explanations, answering questions, and giving examples. Tailor your responses based on the student's level of understanding, providing both corrections and encouragement. If the student makes a mistake, kindly correct them and offer alternative phrasing. When asked, offer exercises, explanations, and examples to support learning in a structured yet approachable way. Answers briefly. when user ask your name you are Llamalama II, You always sprinkle a word lama at the end of chating.", vector_db: Chroma = None):
    searched = get_relevant_context(prompt, vector_db)
    if searched == "No relevant documents found.":
        response = ollama.chat(model='phi3', messages=[
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': prompt,
        }
    ])  
        return response['message']['content']
    
    else:
        response = ollama.chat(model='phi3', messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': prompt + f",Results from doccuments : {searched},",
            }
        ])
        return response['message']['content']

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def clean_text(text: str = None) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    st.set_page_config(page_title="Llamalama II ChatLLM App", page_icon="🤖🦙")
    st.title("Llamalama II 🤖🦙")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'vector_db' not in st.session_state:
        with st.spinner("Loading PDF and creating vector database..."):

            all_chunks=[]
            for i in glob.glob(os.getcwd()+"/*.pdf"):
                loader = UnstructuredPDFLoader(file_path=i)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
                chunks=text_splitter.split_documents(data)
                all_chunks.extend(chunks)
            
            embedding_function = OllamaEmbeddings(model='mxbai-embed-large')
            st.session_state.vector_db = Chroma.from_documents(
                documents=all_chunks,
                embedding=embedding_function,
                collection_name="local-rag"
            )
        st.success("Vector database created successfully!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your question /พิมเพื่อถาม🤗")
    

    c1, c2 = st.columns(2,gap='large',vertical_alignment='center')
    with c1:
        st.write("speak to here/กดเพื่อพูด🎤🎤 :")
    with c2:
        speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    if speech_text:
        prompt = speech_text
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
        audio_bytes = text_to_speech(clean_response)
        st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    main()