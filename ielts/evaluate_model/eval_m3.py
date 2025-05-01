from datasets import load_dataset
import ollama
import os
import glob
import re


#ds = load_dataset("openthaigpt/thai-onet-m6-exam", "english",split="train+test")
#ds.save_to_disk("C:\\Users\\User\\code\\Llamalama_II\\datasets\\m6_eng")

def get_llm_response(prompt, system_prompt="You are a helpful assistant and You are a knowledgeable and patient English teacher with expertise in grammar, vocabulary, pronunciation, and conversational skills."):
    
    response = ollama.chat(model='phi3', messages=[
    {
        'role': 'system',
        'content': """You are a helpful assistant and You are a knowledgeable and patient English teacher with expertise in grammar, vocabulary, pronunciation, and conversational skills\
        here are the example:<question>"Dialog1 : At Rose's House\n              Rose :  Are you superstitious?\n              Suda : _1_\n              Rose : Er..it's a belief based on fear or false ideas. For example,\n                          many people believe that Friday the 13th is an unlucky\n                          day. _2_\n             Suda : Well, I _3_ I'm a bit superstitious then. I had a car\n                         accident last Friday and it was Friday the13th.\n             Rose : Did you get hurt?\n             Suda : _4_ How about some coffee?\n             Rose : _5_"<question>\n\n
        <choices>1. I don't agree.\n2. I know exactly.\n3. What's that?\n4. How do you spell it?\n5. What's new?<choices>\n\n
        <answer>3. What's that?<answer>""",
    },
    {
        'role': 'user',
        'content': prompt,
    }
])  
    return response['message']['content']

if __name__=="__main__":
    get_llm_response()