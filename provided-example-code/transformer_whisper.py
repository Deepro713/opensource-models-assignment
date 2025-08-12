from transformers import pipeline
import sys

def transcribe(audio_path):
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = asr(audio_path)
    print(result["text"])

if __name__ == "__main__":
    transcribe(sys.argv[1])