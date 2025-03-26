# from rvc_python.infer import RVCInference

# rvc = RVCInference(device="cuda:0")
# rvc.load_model("Chidori")
# rvc.infer_file("./rvc_models/temp_input.wav", "./rvc_models/temp_output.wav")


# import requests
# import base64

# url = "http://localhost:5050/convert"
# with open("input.wav", "rb") as audio_file:
#     audio_data = base64.b64encode(audio_file.read()).decode()

# response = requests.post(url, json={"audio_data": audio_data})

# with open("output.wav", "wb") as output_file:
#     output_file.write(response.content)