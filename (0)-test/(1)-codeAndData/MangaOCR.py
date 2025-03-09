from manga_ocr import MangaOcr
import fugashi
import os

# Function to clean up OCR text using fugashi tokenizer
def clean_japanese_text(text):
    tagger = fugashi.Tagger()
    return "".join([word.surface for word in tagger(text)])  # Keep original spacing

# Initialize MangaOCR
ocr = MangaOcr()

# Construct correct image path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "test_jpg", "DL-Raw.Net_111 (2).jpg")

# Run MangaOCR on the image
text_result = ocr(image_path)

# Clean up and print text results
cleaned_text = clean_japanese_text(text_result)  # Use fugashi to improve text structure
print(f"Detected text: {cleaned_text}")
