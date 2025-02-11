
import ollama
import torch

import re

#rag
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from io import BytesIO

from typing import List

#tts
from gtts import gTTS


#asr
import whisper



whisper_model = whisper.load_model("small")
def transcribe_audio(audio_bytes):
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    transcription = whisper_model.transcribe("temp_audio.wav")
    return transcription['text']

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

def get_llm_response(prompt, system_prompt="You are a female helpful assistant and English teacher. Try to answer briefly within 500 characters of letter but if asked for generate article or story answer full content.", vector_db: Chroma = None):
    searched = get_relevant_context(prompt, vector_db)
    if searched == "No relevant documents found.":
        response = ollama.chat(model='phi3.5', messages=[
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': prompt,
        }
    ],options={'temperature': 0.6})  
        return response['message']['content']
    
    else:
        response = ollama.chat(model='phi3.5', messages=[
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
    # if len(text) < 500:
    #     try:
    #         request = ServeTTSRequest(
    #             text=text,
    #             references=[
    #                 ServeReferenceAudio(
    #                     audio=open("./bailu_llama.mp3", "rb").read(),
    #                     text="lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!",
    #                 )
    #             ],
    #         )

    #         with httpx.Client() as client:
    #             response = client.post(
    #                 "https://api.fish.audio/v1/tts",
    #                 content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
    #                 headers={
    #                     "authorization": f"Bearer {token}",
    #                     "content-type": "application/msgpack",
    #                 },
    #                 timeout=None,
    #             )
                
    #             with open("output.mp3", "wb") as f:
    #                 f.write(response.content)
                
    #             return read_mp3_to_bytes("output.mp3")

    #     except Exception as e:
    #         print(f"Error using fish.audio API: {e}")
    #         return fallback_tts(text)
    # else:
    #     return fallback_tts(text)
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