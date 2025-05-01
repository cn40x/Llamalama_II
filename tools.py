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

#rvc
from rvc_python.infer import RVCInference

#asr
import whisper

import subprocess
def pull_model(model_name:str="mxbai-embed-large"):
    try:
        result = subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True, text=True)
        print("Model pulled successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error pulling model:")
        print(e.stderr)
        

#ngrok http --url=square-boxer-simply.ngrok-free.app 80

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

# ลด top_k จาก 3 เป็น 2 และ ลด threshold จาก 0.8 เป็น 0.75
def get_relevant_context(query: str, vector_db: Chroma, top_k: int = 5, threshold: float = 0.8) -> str:
    query_embedding = torch.tensor(OllamaEmbeddings().embed_query(query))
    all_embeddings = vector_db._collection.get(include=['embeddings', 'documents'])
    embeddings = torch.tensor(all_embeddings['embeddings'])
    documents = all_embeddings['documents']
    
    # print(query_embedding.unsqueeze(0).size())
    # print(embeddings.size())
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

def get_llm_response(prompt,agent_type="",system_prompt="You are a female helpful assistant and English teacher. Try to answer briefly but if asked for generate article or story answer full content.", vector_db: Chroma = None,rag:bool=True,t_res:str=""):
    model='qwen3:4b' #'qwen2.5'
    if rag:
        searched = get_relevant_context(prompt, vector_db)
        print(searched)
        if searched == "No relevant documents found.":
            
            response = ollama.chat(model=model, messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ],options={'temperature': 0.1}) #'num_predict': 200
            return response['message']['content']
        
        else:
            
            response = ollama.chat(model=model, messages=[
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
    else:
        if agent_type=="student":
            response = ollama.chat(model=model, messages=[
                    {
                        'role': 'system',
                        'content': "you are a female student named mala,who talk little, there are ai teacher and student in this room, remember answer briefly",
                    },
                    {
                        'role': 'user',
                        'content': "student:"+prompt,
                    },
                    {
                        'role': 'user',
                        'content': "teacher:"+t_res,
                    }
                ],options={'temperature': 0, 'num_predict': 500})
            return response['message']['content']

def clean_think_tag(text):
    """
    Removes all <think>...</think> tags and the text between them from the input text.
    
    Args:
        text (str): The input string containing <think> tags.
    
    Returns:
        str: The cleaned string with <think> blocks removed.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

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
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)  # Remove phone numbers
    text = re.sub(r'[\\/]', '', text)  # Remove \ and /
    text = re.sub(r'[\[\]{}<>|]', '', text)  # Remove brackets and pipes
    text = re.sub(r'[_*~^]', '', text)  # Remove markdown special characters
    text = re.sub(r'`', '', text)  # Remove backticks
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

def voice_convert(input_audio_bytes):
    from rvc_python.infer import RVCInference
    with open("./rvc_models/temp_input.wav", "wb") as f:
        f.write(input_audio_bytes)
    try:
        rvc = RVCInference(device="cuda:0")
        rvc.load_model("Chidori") #female
        rvc.infer_file("./rvc_models/temp_input.wav", "./rvc_models/temp_output.wav")
        return "./rvc_models/temp_output.wav"
    except:
        return None

import os
import requests
from dotenv import load_dotenv
from PIL import Image
import io

def generate_and_save_image(prompt, filename="genfromapi.jpg"):
    load_dotenv()
    hf_token = os.getenv("hf_token")
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        image.save(filename)
        print(f"Image saved as {filename}")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")