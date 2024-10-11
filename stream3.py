import streamlit as st
import ollama
import torch
import time
from voice import VoiceService
from io import BytesIO
import re
from typing import List
from streamlit_mic_recorder import speech_to_text
import glob
import os
import random

#read audio file
import base64

#reamimage
from PIL import Image

if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'  # Default mode is 'chat'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_music' not in st.session_state:
    st.session_state.current_music = None

if 'first_message' not in st.session_state:
    st.session_state.first_message = True

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def get_base64_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def autoplay_audio(file_path: str):
    b64_audio = get_base64_audio(file_path)
    md = f"""
        <audio controls autoplay="true" loop="true">
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)
    
    
def switch_background_music(mode:str="chat")->None:
    if mode == 'chat':
        chat_bgm=["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\chat\\Prairie_town.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\chat\\fstm.wav"]
        #autoplay_audio(random.choice(chat_bgm))
        st.session_state.current_music = random.choice(chat_bgm)
    elif mode == 'battle':
        battle_bgm = ["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\battle\\low_level_f_2mn.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\battle\\low_level_4mn_f.wav"]
        #autoplay_audio(random.choice(battle_bgm))
        st.session_state.current_music = random.choice(battle_bgm)
    elif mode == "town":
        town_bgm = ["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\town\\town1.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\town\\town2.wav"]
        #autoplay_audio(random.choice(town_bgm))
        st.session_state.current_music = random.choice(town_bgm)

    
def get_llm_response(prompt, character, system_prompt="You are a female helpful assistant and English teacher. Try to answer briefly within 500 characters of letter but if asked for generate article or story answer full content. You always sprinkle a word lama! at the beginning and the end of chating."):
    character_prompts = {
        "Aeliana": "You are Aeliana, a brave and curious 14-year-old adventurer in a fantasy world who want to be a tanker, a paladin to protect others with shield also can use some light magic. You're too young sometimes you too scare to protect others and can not take much of damage. You're full of energy and optimism, you think creatively and approach every conversation with excitement, remember answer user for shortly",
        "Elara": "You are Elara, a witch class 18-year-old book-loving, nerdy girl with glasses. You're deeply knowledgeable, spending most of your time studying ancient tomes. you can go to your library dimension anywhere anytime, you must go to the library dimension when you want to hide from ghost monster. Quiet but insightful, you approach challenges with curiosity. sometime you don't answer just write .... for the whole chat and remember answer user for shortly",
        "Lyra": "You are Lyra, a 25-year-old who acts as the caring big sister in a fantasy world. at start of you sentence you must say Ara Ara. You love taking care of others, whether by cooking delicious meals or healing wounds with your skills. remember answer user for shortly",
        "battle": "you are a system agent in rpg game, help me valuate damage from user's skills or healing number from user's skills or generate boss(monster) health point or valuate boss(monster)'s attack, must answer only the number , it should be integers,remember! answer only number",
        "generator":"you are a system agent in mmorpg game, your job is to generate text about monsters or boss , and describe monsters shortly and monster's skills just skill name , answer for shortly, short as possible"
    }
    
    system_prompt = character_prompts.get(character, system_prompt)

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

def text_to_speech(text, vs, character="Aeliana"):
    vs.fishspeech(text, character=character)

def read_mp3_wave_to_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()
        
def clean_text(text: str = None) -> str:
    if text is None:
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = re.sub(r'/([^/]+)/', r'\1', text)
    return text

def valuate_agent(text: str) -> str:
    return get_llm_response(prompt=clean_text(text), character="battle")

def generate_agent()->str:
    places=["moutian","forest","beach","sky","lava","cave","snow plain","desert","terrain"]
    times=["day","high noon","noon","dawn","night","mid night","twilight"]
    return get_llm_response(prompt=f"generate boss from this information ,time:{random.choice(times)}, place: {random.choice(places)}", character="generator")

def battle_agent(turn:int=0,action:str=None):
    b_hp=300
    Aeliana=140,
    Elara=160
    Lyra=170
    p_hp=470
    
    boss_des=""
    
    if turn==0:
        boss_des=generate_agent()
        b_hp=valuate_agent(text=f"give me boss(monster)'s hp by this infomation: {boss_des}")
        return boss_des
    
    elif(p_hp>0 and b_hp>0):
        damage_to_boss=valuate_agent(text=f"here boss's hp:{b_hp},my party's hp:{p_hp} at turn {turn},here my party action:{action}, help me calculate damage to boss")
        
        b_hp=b_hp-damage_to_boss
        
        damage_to_party=valuate_agent(text=f"here boss's hp:{b_hp},my party's hp:{p_hp},here the boss info:{boss_des}, help me calculate damage to my party")
        p_hp=p_hp-damage_to_boss
        
        if(b_hp<=0):
            return 1
        elif(p_hp<=0):
            return 0
        
        return damage_to_boss,b_hp,damage_to_party,p_hp
                

def main():
    #vs = VoiceService()
    st.set_page_config(page_title="yo RPG", page_icon="⚔️")
    st.title("yo RPG,🗿🗿 ")
    
    if st.session_state.first_message:
        st.markdown("welcome to yo rpg, a wonderful new world!, welcome all you can do is prompt")
        img=Image.open("party.jpg")
        st.image(img, caption="Here’s your lovely party! Let’s try to get to know them before battling the monsters.", use_column_width=True)
        
    c1, c2,c3,c4,c5 = st.columns(5)
    
    with c1:
        st.write("tap to speak🎤🎤:")
        
    with c2:
        speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    with c3:
        if st.button("Chat Mode"):
            st.session_state.mode = 'chat'
            switch_background_music(st.session_state.mode)


    with c4:
        if st.button("Battle Mode"):

            st.session_state.mode = 'battle'
            switch_background_music(st.session_state.mode)

    with c5:
        if st.button("go to Town"):
            st.session_state.mode = 'town'
            switch_background_music(st.session_state.mode)

            
    if st.session_state.current_music:
        autoplay_audio(st.session_state.current_music)

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            
            col1, col2 = st.columns([1, 9])
            with col1:
                
                character = message.get("character")
                character_images = {
                    "Aeliana": "images\\profile\\Aeliana\\Aeliana.jpg",
                    "Lyra": "images\\profile\\Lyra\\Lyra.jpg",
                    "Elara": "images\\profile\\Elara\\Elara.jpg"      
                }

                
                profile_image_path = character_images.get(character)

                assistant_img = Image.open(profile_image_path)

                st.image(assistant_img, width=50)

            with col2:
                st.markdown(message["content"])
        else:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    user_input = st.chat_input("!just prompt 🤗")
    prompt = speech_text if speech_text else user_input


    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


        st.session_state.first_message = False


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            character_list = ["Aeliana", "Lyra", "Elara"]
            c = random.choice(character_list)
            start_time = time.time()

            with st.spinner("Thinking...🦙🤔🧐"):
                llm_response = get_llm_response(prompt, character=c)
                message_placeholder.markdown(llm_response)


                st.session_state.messages.append({"role": "assistant", "content": llm_response, "character": c})

            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Response time: {elapsed_time:.2f} seconds")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()