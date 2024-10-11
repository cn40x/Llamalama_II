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

#
import base64

if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'
    
if 'messages' not in st.session_state:
    st.session_state.messages = []
    

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
    
    
def switch_background_music(mode:str="chat"):
    if mode == 'chat':
        chat_bgm=["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\chat\\Prairie_town.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\chat\\fstm.wav"]
        autoplay_audio(random.choice(chat_bgm))
    elif mode == 'battle':
        battle_bgm = ["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\battle\\low_level_f_2mn.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\battle\\low_level_4mn_f.wav"]
        autoplay_audio(random.choice(battle_bgm))
    elif mode == "town":
        town_bgm = ["C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\town\\town1.wav","C:\\Users\\User\\code\\Llamalama_II\\sounds\\bg\\town\\town2.wav"]
        autoplay_audio(random.choice(town_bgm))

    
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
    st.set_page_config(page_title="Llamalama II ChatLLM App", page_icon="⚔️")

    # if st.button(f"Switch to {'battle' if st.session_state.mode == 'chat' else 'Normal'} Mode"):
    #     st.session_state.mode = 'battle' if st.session_state.mode == 'chat' else 'normal'
    #     switch_background_music(mode=st.session_state.mode)  # Change music based on mode

    # Load background music
    #switch_background_music(mode=st.session_state.mode)
    #mode=st.selectbox("mode chat/battle/town", ("chat", "battle","town"))
    # st.session_state.mode=mode
    
    # print(st.session_state.mode)
    
    # if st.session_state.mode == "chat":
    #     st.title("Llamalama II 🦙 - Chat Mode")
    ##     switch_background_music(st.session_state.mode)
    # elif st.session_state.mode == "battle":
    #     st.title("Llamalama II ⚔️🦙 - Battle Mode")
    #     switch_background_music(st.session_state.mode)
    # elif st.session_state.mode == "town":
    #     st.title("Llamalama II 🦙 - Town Mode")
    #     switch_background_music(st.session_state.mode)
        

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("!just prompt 🤗")
    full_response=""
    c1, c2,c3,c4,c5 = st.columns(5)
        
    with c1:
        st.write("tap to speak🎤🎤:")
        
    with c2:
        speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    
    with c3:
        if st.button("Switch to Chat Mode"):
            st.session_state.mode = 'chat'
            switch_background_music(st.session_state.mode)

    with c4:
        if st.button("Switch to Battle Mode"):
            st.session_state.mode = 'battle'
            switch_background_music(st.session_state.mode)

    with c5:
        if st.button("Switch to Town Mode"):
            st.session_state.mode = 'town'
            switch_background_music(st.session_state.mode)
            
    if st.session_state.mode == "chat":
        st.title("Llamalama II 🦙 - Chat Mode")
        switch_background_music(st.session_state.mode)
        prompt = speech_text if speech_text else user_input

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                character_list = ["Aeliana", "Lyra", "Elara"]
                c = random.choice(character_list)
                start_time = time.time()
                
                with st.spinner("Thinking...🦙🤔🧐"):
                    llm_response = get_llm_response(prompt, character=c)

                for chunk in llm_response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                
                message_placeholder.markdown(full_response)

                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Response time: {elapsed_time:.2f} seconds")
            
    elif st.session_state.mode == "battle":
        st.title("Llamalama II ⚔️🦙 - Battle Mode")
        #switch_background_music(st.session_state.mode)
        
        turn=0
        print(turn)
        prompt = speech_text if speech_text else user_input

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown("your party turn")
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                character_list = ["Aeliana", "Lyra", "Elara"]
                c = random.choice(character_list)
                start_time = time.time()
                
                with st.spinner(f"{c} is Thinking...🦙🤔🧐"):
                    llm_response = get_llm_response(prompt, character=c)
                    turn+=1
                for chunk in llm_response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
                
                
            with st.chat_message("monster"):
                message_placeholder = st.empty()
                full_response = ""
                
                
                with st.spinner(f"monster is Thinking...🦙🤔🧐"):
                    llm_response = battle_agent(turn=turn,action=llm_response)
                    turn+=1
                if llm_response!=0 and llm_response !=1:
                    for chunk in llm_response.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                        
                elif llm_response==1:
                    message_placeholder.markdown("you won")
                    
                elif llm_response==0:
                    message_placeholder.markdown("you loose")
                    
                else:
                    damage_to_boss,b_hp,damage_to_party,p_hp=llm_response
                    message_placeholder.markdown(f"damage_to_boss:{damage_to_boss}, boss hp:{b_hp}, damage_to_party:{damage_to_party}, party hp:{p_hp}")

                message_placeholder.markdown(full_response)
            
    elif st.session_state.mode == "town":
        st.title("Llamalama II 🦙 - Town Mode")
        #switch_background_music(st.session_state.mode)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()