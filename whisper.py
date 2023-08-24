import openai
import os
import requests
import config
import base64

#import speech_recognition as sr
openai.api_key ='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'
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 20  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 



with open("output.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
print(transcript)
prompt = transcript
height = 512
width = 512
steps = 50


api_host = 'https://api.stability.ai'
api_key = 'sk-x4I6zjSYgSVfYmeRujWu5lBPQV0UCRktG63qeZpNzKsOK4oA'

engine_id = 'stable-diffusion-xl-beta-v2-2-2'


def getModelList():
    url = f"{api_host}/v1/engines/list"
    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})

    if response.status_code == 200:
        payload = response.json()
        print(payload)

getModelList()


def generateStableDiffusionImage(prompt, height, width, steps):
    url = f"{api_host}/v1/generation/{engine_id}/text-to-image"
    headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    payload = {}
    payload['text_prompts'] = [{"text": f"{prompt}"}]
    payload['cfg_scale'] = 7
    payload['clip_guidance_preset'] = 'FAST_BLUE'
    payload['height'] = height
    payload['width'] = width
    payload['samples'] = 1
    payload['steps'] = steps

    response = requests.post(url,headers=headers,json=payload)

    #Processing the response
    if response.status_code == 200:
        data = response.json()
        for i, image in enumerate(data["artifacts"]):
            with open(f"{prompt[7:30]}.png", "ab") as f:
                f.write(base64.b64decode(image["base64"]))
                f.close()

generateStableDiffusionImage(prompt, height, width, steps)
