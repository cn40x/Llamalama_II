import os
import requests

from dotenv import load_dotenv

load_dotenv()
text = "lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!"

url = "https://api.fish.audio/v1/tts"
token = os.getenv("env_token")
# audio_file_path = "./bailu_llama.mp3"  # Replace with actual path


# payload = {
#     "references": [
#         {
#             "text": text,
#             "audio": audio_file_path,  # Changed from {} to None
#         }
#     ],
#     "text": text,
#     "reference_id": "94748c82e544406f9b7a3e4b348e66b6",
#     "chunk_length": 499
# }

# headers = {
#     "Authorization": f"Bearer {token}",  # Use the token from environment variable
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)

# print(response.status_code)
# print(response.text)


from typing import Annotated, AsyncGenerator, Literal

import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, conint


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "mp3"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = "94748c82e544406f9b7a3e4b348e66b6"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"


request = ServeTTSRequest(
    text="lama hello i am your english teacher now lama!",
    references=[
        ServeReferenceAudio(
            audio=open("./bailu_llama.mp3", "rb").read(),
            text="lama - let us celebrate our diversity as it adds color to life itself and creates a world that is more beautiful than ever before – lama! So go ahead; don't be afraid of being yourself or loving whomever you wish. After all, love knows no boundaries in this vast universe—lama - let's continue celebrating our diversity together with open hearts and minds- Lama!!! Let us take pride where we are today while working towards an even more inclusive tomorrow – lama!",
        )
    ],
)

with (
    httpx.Client() as client,
    open("output.mp3", "wb") as f,
):
    with client.stream(
        "POST",
        "https://api.fish.audio/v1/tts",
        content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/msgpack",
        },
        timeout=None,
    ) as response:
        for chunk in response.iter_bytes():
            f.write(chunk)
