import cv2
import torch
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading TrOCR model...")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(device)

print("TrOCR loaded.")


def remove_ruled_lines(gray):
    th = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

    detected = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    no_lines = cv2.subtract(th, detected)

    return 255 - no_lines


def extract_lines(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cleaned = remove_ruled_lines(gray)

    th = cv2.threshold(
        cleaned,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    horizontal_sum = np.sum(th, axis=1)

    lines = []
    start = None

    for i, value in enumerate(horizontal_sum):

        if value > 1000 and start is None:
            start = i

        elif value <= 1000 and start is not None:

            if i - start > 15:
                line_img = cleaned[start:i, :]
                lines.append(line_img)

            start = None

    return lines


def ocr_line(line_img):

    pil_img = Image.fromarray(line_img).convert("RGB")

    pixel_values = processor(
        pil_img,
        return_tensors="pt"
    ).pixel_values.to(device)

    ids = model.generate(
        pixel_values,
        max_length=64,
        num_beams=4
    )

    text = processor.batch_decode(
        ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


def extract_text(image_path):

    lines = extract_lines(image_path)

    results = []

    for line in lines:
        txt = ocr_line(line)
        if len(txt) > 2:
            results.append(txt)

    return "\n".join(results)