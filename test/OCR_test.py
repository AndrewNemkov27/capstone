from paddleocr import PaddleOCR
import cv2
import fugashi


# Function to resize image while keeping aspect ratio
def resize_image(image_path, max_width=1920):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    if width > max_width:
        ratio = max_width / width
        new_dimensions = (max_width, int(height * ratio))
        resized = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
        return resized
    return image


# Function to clean up text using fugashi tokenizer
def clean_japanese_text(text):
    tagger = fugashi.Tagger()
    return "".join([word.surface for word in tagger(text)])  # Keep original spacing


# Initialize PaddleOCR with improved parameters
ocr = PaddleOCR(
    lang="japan",
    rec_algorithm="CRNN",  # Use CRNN for better sequential text recognition
    use_angle_cls=True,  # Detect rotated text
    rec_batch_num=25,  # Increase batch size for more text detection
    rec_char_type="jp",
    det_db_box_thresh=0.2,  # Lower threshold to detect smaller text
    det_db_unclip_ratio=4,  # Expand detected text regions
    rec_max_len=50,  # Allow longer detected text
    use_dilation=True  # Improve text connectivity
)

# Load and preprocess the image
image_path = r"C:\Users\andne\Downloads\test\DL-Raw.Net_166 (2).jpg"
resized_image = resize_image(image_path)

# Save preprocessed image for OCR input
temp_path = "temp_resized.jpg"
cv2.imwrite(temp_path, resized_image)

# Run OCR on resized image
results = ocr.ocr(temp_path, cls=True)

# Extract, clean up, and print text results
for result in results[0]:
    _, (text, confidence) = result
    cleaned_text = clean_japanese_text(text)  # Use fugashi without spaces
    print(f"Detected text: {cleaned_text} (Confidence: {confidence:.2f})")
