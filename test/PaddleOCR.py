import cv2
import re
import csv
from paddleocr import PaddleOCR
import fugashi
import os


# function to clean up OCR text using fugashi tokenizer
def clean_japanese_text(text):
    tagger = fugashi.Tagger()
    return "".join([word.surface for word in tagger(text)])


# function to filter out non-Japanese characters (kanji, hiragana, katakana)
def clean_non_japanese(text):
    return re.sub(r'[^ぁ-んァ-ン一-龯々〆〤]+', '', text)


# initialize PaddleOCR with optimized parameters
ocr = PaddleOCR(
    lang="japan",
    rec_algorithm="SVTR_LCNet",
    use_angle_cls=True,
    rec_batch_num=50,
    rec_char_type="jp",
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=2,
    use_mp=True,
    rec_max_len=30,
    drop_score=0.65,
    use_dilation=True
)

# construct correct folder and CSV paths
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, "test_jpg")
csv_file = os.path.join(script_dir, "test_data.csv")

# open CSV file and prepare for writing
with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # write header only if file is empty
    if file.tell() == 0:
        writer.writerow(["Extracted Text", "Image Name", "Confidence"])

    # process all JPG images in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Skipping {filename}: Unable to read image.")
                continue

            # convert to grayscale and apply denoising
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 30, 7, 21)

            # run OCR on the processed image
            results = ocr.ocr(img_denoised, cls=True)
            data_written = False  # Track if any text was written

            # extract and clean text results
            if results and results[0]:
                for result in results[0]:
                    if isinstance(result, list) and len(result) > 1:
                        _, (text, confidence) = result
                        cleaned_text = clean_japanese_text(text)
                        cleaned_text = clean_non_japanese(cleaned_text)

                        if len(cleaned_text) >= 2:
                            # write results to CSV
                            writer.writerow([cleaned_text, filename, f"{confidence:.2f}"])
                            data_written = True

            # print success message 
            if data_written:
                print()
                print(f"{filename} successfully read into CSV")
                print()
