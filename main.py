# chat
import ollama

# rag
import torch
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# tts
from gtts import gTTS
from playsound import playsound
import os
import tempfile

import re


from typing import List


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = 'mxbai-embed-large'):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [ollama.embeddings(model=self.model, prompt=text)['embedding'] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.model, prompt=text)['embedding']
    
def get_relevant_context(query: str, vector_db: Chroma, top_k: int = 3) -> List[str]:
    query_embedding = torch.tensor(OllamaEmbeddings().embed_query(query))
    
    all_embeddings = vector_db._collection.get(include=['embeddings', 'documents'])
    embeddings = torch.tensor(all_embeddings['embeddings'])
    documents = all_embeddings['documents']
    

    cos_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings)
    
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    res=""
    for i,idx in enumerate(top_indices):
        res=res+str(i+1)+"."+documents[idx]+",\n\n"
    
    return res



def get_llm_response(prompt, system_prompt="You are a helpful assistant who helps me study English and answers briefly. when user ask your name you are Llamalama II, You must say lamalama at the end of chating.", vector_db: Chroma = None):
    
    searched = get_relevant_context(prompt, vector_db)
    if searched=="No relevant documents found.":
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
                'content': prompt + f",Here the User's query (unnecessary): {searched},",
            }
        ])
        return response['message']['content']


def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
        playsound(fp.name)
    os.unlink(fp.name)


def clean_text(text:str=None)->str:
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #remove https naa
    
    
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text) #remove phone naa
    
    
    text = re.sub(r'\s+', ' ', text).strip() # remove extra whitespace
    
    return text


def main():
    loader = UnstructuredPDFLoader(file_path="./Grammar_Cheatsheet.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks = text_splitter.split_documents(data)
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large')

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name="local-rag"
    )

    while True:
        user_input = input("Enter your prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            vector_db.delete_collection()
            break
        
        llm_response = get_llm_response(user_input,vector_db=vector_db)
        print("LLM response:", llm_response)
        text_to_speech(clean_text(llm_response))

    
if __name__ == "__main__":
    main()
