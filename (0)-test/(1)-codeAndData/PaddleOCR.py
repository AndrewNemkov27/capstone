import cv2
import re
import csv
import os
import fugashi
from collections import Counter
from paddleocr import PaddleOCR
from googletrans import Translator

# Note: this code is not included in project blog post.

# Initialize the fugashi tokenizer
tagger = fugashi.Tagger()

# POS tag mapping (Japanese to English)
pos_mapping = {
    "名詞": "noun",
    "動詞": "verb",
    "形容詞": "adjective",
    "副詞": "adverb",
    "助詞": "particle",
    "助動詞": "auxiliary verb",
    "連体詞": "adnominal",
    "接続詞": "conjunction",
    "感動詞": "interjection",
    "接頭詞": "prefix",
    "接尾詞": "suffix",
    "その他": "other"
}


# function to clean non-Japanese characters
def clean_non_japanese(text):
    return re.sub(r'[^ぁ-んァ-ン一-龯々〆〤]+', '', text)


# initialize Google Translator
translator = Translator()
translation_cache = {}


# function to translate Japanese to English
def get_english_translation(word):
    if word in translation_cache:
        return translation_cache[word]

    try:
        translated = translator.translate(word, src="ja", dest="en")
        translation_cache[word] = translated.text
        return translated.text
    except Exception as e:
        print(f"Error translating {word}: {e}")
        return "N/A"


# function to tokenize Japanese text into words with POS tagging
def tokenize_japanese(text):
    words_with_pos = []
    for word in tagger(text):
        pos = word.feature.pos1
        surface = word.surface

        # ignore grammatical particles (助詞)
        if pos == "助詞":
            continue

        # ignore single-character hiragana or katakana words
        if len(surface) == 1 and re.match(r'^[ぁ-んァ-ン]$', surface):
            continue

        pos_english = pos_mapping.get(pos, "unknown")
        words_with_pos.append((surface, pos_english))

    return words_with_pos


# define character sets for categorization
HIRAGANA = set("ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖ")
KATAKANA = set("ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ")
KANJI_RANGE = (ord("一"), ord("龯"))  # unicode range for Kanji characters


# function to compute character ratios
def compute_character_ratios(word):
    total_chars = len(word)
    if total_chars == 0:
        return 0.0, 0.0, 0.0

    hiragana_count = sum(1 for char in word if char in HIRAGANA)
    katakana_count = sum(1 for char in word if char in KATAKANA)
    kanji_count = sum(1 for char in word if KANJI_RANGE[0] <= ord(char) <= KANJI_RANGE[1])

    return round(hiragana_count / total_chars, 4), round(katakana_count / total_chars, 4), round(kanji_count / total_chars, 4)


# initialize PaddleOCR
ocr = PaddleOCR(
    lang="japan",
    rec_algorithm="SVTR_LCNet",
    use_angle_cls=True,
    rec_batch_num=50,  # batch size processing - increases speed of processing
    rec_char_type="jp",
    det_db_box_thresh=0.3,  # box detection confidence - increase only captures clear to see text boxes
    det_db_unclip_ratio=2,  # expand detected text boxes - increases detection boxes for more words
    use_mp=True,
    rec_max_len=30,  # maximum length of extracted phrase - increases size of extracted phrase
    drop_score=0.65,  # limit to confidence score - increases to give only higher confidence phrases
    use_dilation=True
)

code_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.dirname(code_dir)

image_folder = os.path.join(image_dir, "(0)-test_jpg")
csv_file = os.path.join(code_dir, "test_data.csv")

# extract parent folder name for img_series
img_series = os.path.basename(image_folder)

# word frequency counter
word_counts = Counter()
total_words = 0

# Step 1: count word occurrences
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

        # run OCR
        results = ocr.ocr(img_denoised, cls=True)

        if results and results[0]:
            for result in results[0]:
                if isinstance(result, list) and len(result) > 1:
                    _, (text, _) = result
                    cleaned_text = clean_non_japanese(text)

                    if len(cleaned_text) >= 2:
                        words_with_pos = tokenize_japanese(cleaned_text)

                        # count each word
                        for word, _ in words_with_pos:
                            word_counts[word] += 1
                            total_words += 1

# initialize counter for success message
processed_count = 0

# Step 2: write to CSV with frequency and ratios
with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # write header only if file is empty
    if file.tell() == 0:
        writer.writerow(["word_JAP", "word_US", "word_POS", "phrase_JAP", "img_title", "img_series", "length", "confidence", "word_freq", "hiragana_ratio", "katakana_ratio", "kanji_ratio"])

    # process images again to write the data
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)

            if img is None:
                continue

            # convert to grayscale and denoise
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 30, 7, 21)

            # OCR
            results = ocr.ocr(img_denoised, cls=True)
            data_written = False  # track if data was written

            if results and results[0]:
                for result in results[0]:
                    if isinstance(result, list) and len(result) > 1:
                        _, (text, confidence) = result
                        cleaned_text = clean_non_japanese(text)

                        if len(cleaned_text) >= 2:
                            words_with_pos = tokenize_japanese(cleaned_text)
                            translations = {word: get_english_translation(word) for word, _ in words_with_pos}

                            for word, pos in words_with_pos:
                                translation = translations.get(word, "N/A")
                                word_length = len(word)
                                word_frequency = round(word_counts[word] / total_words, 4)
                                hiragana_ratio, katakana_ratio, kanji_ratio = compute_character_ratios(word)

                                writer.writerow([word, translation, pos, cleaned_text, filename, img_series, word_length, f"{confidence:.2f}", word_frequency, hiragana_ratio, katakana_ratio, kanji_ratio])
                                data_written = True

            if data_written:
                processed_count += 1
                print()
                print(f"#{processed_count}: {filename} was successfully processed.")

