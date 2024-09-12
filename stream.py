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

#request
import requests
from dotenv import load_dotenv

#api and fishaudio things

from typing import Annotated, AsyncGenerator, Literal
import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, conint

load_dotenv()
token = os.getenv("env_token")
url = "https://api.fish.audio/v1/tts"
text_ref="lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!"


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    references: list[ServeReferenceAudio] = []
    reference_id: str | None = "94748c82e544406f9b7a3e4b348e66b6"
    normalize: bool = True
    latency: Literal["normal", "balanced"] = "normal"



class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = 'mxbai-embed-large'):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [ollama.embeddings(model=self.model, prompt=text)['embedding'] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.model, prompt=text)['embedding']

def get_relevant_context(query: str, vector_db: Chroma, top_k: int = 3, threshold: float = 0.7) -> str:
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

def get_llm_response(prompt, system_prompt="You are a female helpful assistant and English teacher. Try to answer briefly within 500 characters of letter but if asked for generate article or story answer full content. You always sprinkle a word lama! at the beginning and the end of chating.", vector_db: Chroma = None):
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
                'content': prompt + f",Results from searching doccuments : {searched},",
            }
        ])
        return response['message']['content']

def text_to_speech(text):
    if len(text) < 500:
        try:
            request = ServeTTSRequest(
                text=text,
                references=[
                    ServeReferenceAudio(
                        audio=open("./bailu_llama.mp3", "rb").read(),
                        text="lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!",
                    )
                ],
            )

            with httpx.Client() as client:
                response = client.post(
                    "https://api.fish.audio/v1/tts",
                    content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
                    headers={
                        "authorization": f"Bearer {token}",
                        "content-type": "application/msgpack",
                    },
                    timeout=None,
                )
                
                with open("output.mp3", "wb") as f:
                    f.write(response.content)
                
                return read_mp3_to_bytes("output.mp3")

        except Exception as e:
            print(f"Error using fish.audio API: {e}")
            return fallback_tts(text)
    else:
        return fallback_tts(text)

def fallback_tts(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def read_mp3_to_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()
        

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
