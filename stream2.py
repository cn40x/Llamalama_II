import streamlit as st
import ollama
import torch
import time

#tts
from voice import VoiceService
from io import BytesIO
import re
from typing import List

#asr
from streamlit_mic_recorder import speech_to_text

#path
import glob
import os

import random

if 'messages' not in st.session_state:
    st.session_state.messages = []

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def get_llm_response(prompt,charactor,system_prompt="You are a female helpful assistant and English teacher. Try to answer briefly within 500 characters of letter but if asked for generate article or story answer full content. You always sprinkle a word lama! at the beginning and the end of chating."):
    if charactor=="Aeliana":
        system_prompt="You are Aeliana, a brave and curious 14-year-old adventurer in a fantasy world who want to be a tanker,a paladin to protect others with shield also can use some light magic. You’re too young sometimes you too scare to protect others and can not take much of damage. You’re full of energy and optimism,you think creatively and approach every conversation with excitement, remember answer user for shortly"
    
    if charactor=="Elara":
        system_prompt="You are Elara, an witch class 18-year-old book-loving, nerdy girl with glasses. You’re deeply knowledgeable, spending most of your time studying ancient tomes. you can go to your libary dimension anywhere anytime , you must go to the libary dimention when you want to hide from ghost monster. Quiet but insightful, you approach challenges with curiosity. sometime you don't answer just write .... for the whole chat and remember answer user for shortly"
    
    if charactor=="Lyra":
        system_prompt="You are Lyra, a 25-year-old who acts as the caring big sister in a fantasy world. at start of you sentense you must say Ara Ara.You love taking care of others, whether by cooking delicious meals or healing wounds with your skills. remember answer user for shortly"
    
    if charactor=="battle":
        system_prompt="you are a system agent in battle turnbase rpg game, help me valuate damage number from user's skills and help me valuate healing number from user's skills, the number should be integers and explain why give value like that."

    response = ollama.chat(model='llama3.1', messages=[
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


def text_to_speech(text,vs,charactor="B"):
    vs.fishspeech(text,charactor=charactor)

def read_mp3_wave_to_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()
        
def clean_text(text: str = None) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text= re.sub(r"\([^)]*\)", "", text).strip()
    text = re.sub(r'/([^/]+)/', r'\1', text)
    return text


def valuate_agent(text:str)->str:
    return get_llm_response(prompt=clean_text(text),charactor="battle")

def battle_agent(message_placeholder):
    turn=1
    hp_me=200
    hp_Aeliana=190
    hp_Elara=160
    hp_Lyra=180
    hp_monster=200
    
    while(hp_me>0 or hp_Aeliana>0 or hp_Lyra>0 or hp_Elara>0):
        #message_placeholder.markdown(valuate_agent())
        if hp_monster<0:
            break
    pass

def system_agent_rb():
    
    pass

def main():
    st.set_page_config(page_title="Llamalama II ChatLLM App", page_icon="🤖🦙")
    st.title("Llamalama II 🤖🦙")
    
    vs = VoiceService()

    try:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    except AttributeError:
        st.session_state.messages = []


    user_input = st.chat_input("Type your question /พิมเพื่อถาม🤗")

    c1, c2 = st.columns(2)
    
    with c1:
        st.write("Speak here/กดเพื่อพูด🎤🎤 :")
        
    with c2:
        speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    prompt = speech_text if speech_text else user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            charactor_list = ["Aeliana", "Lyra", "Elara"]
            c = random.choice(charactor_list)
            start_time = time.time()
            
            with st.spinner("Thinking/กำลังคิดอยู่...🦙🤔🧐"):
                llm_response = get_llm_response(prompt, charactor=c)

            for chunk in llm_response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Response time: {elapsed_time:.2f} seconds")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        clean_response = clean_text(full_response)
        start_time=time.time()
        text_to_speech(clean_response, vs, charactor=c)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        audio_bytes = read_mp3_wave_to_bytes("outputs/output0.wav")
        
        st.audio(audio_bytes, format="audio/wav")

        st.write(f"Response time: {elapsed_time:.2f} seconds")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

