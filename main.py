import ollama
import pyttsx3

def get_llm_response(prompt):
    response = ollama.chat(model='phi3', messages=[
        {
            'role': 'user',
            'content': prompt,
        }
    ])
    return response['message']['content']

from gtts import gTTS
from playsound import playsound
import os
import tempfile

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
        playsound(fp.name)
    os.unlink(fp.name)

def main():
    while True:
        user_input = input("Enter your prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        llm_response = get_llm_response(user_input)
        print("LLM response:", llm_response)
        
        text_to_speech(llm_response)

if __name__ == "__main__":
    main()
