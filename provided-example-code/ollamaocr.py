import ollama
import base64
import argparse
from PIL import Image
import io

MODEL = 'minicpm-v:8b'

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def perform_ocr(image_path):
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path)

        # Prompt input
        prompt=input("Enter a prompt for OCR (or press Enter to use default): ")

        # Send request to MiniCPM-V model
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            images=[image_base64]
        )

        # Print the extracted text
        print("Extracted Text:", response['response'])

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform OCR on an image using MiniCPM-V 2.6 model")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()
    perform_ocr(args.image_path)