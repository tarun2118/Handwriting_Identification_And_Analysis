import cv2
import os
from pathlib import Path

# Input: path to PNG page
# Output: multiple line images saved in writer-wise directory

def remove_ruled_lines(img_gray):
    # threshold
    th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # detect ruled lines (horizontal)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected = cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # subtract to remove
    cleaned = cv2.subtract(th, detected)

    cleaned = 255 - cleaned  # invert back to normal
    return cleaned

def segment_page_into_lines(page_path, out_folder):
    img = cv2.imread(str(page_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cleaned = remove_ruled_lines(gray)

    # binarize for segmentation
    th = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # dilate to join text in one line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,3))
    dil = cv2.dilate(th, kernel, iterations=1)

    # find contours = lines
    cnts = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])  # sort by y

    os.makedirs(out_folder, exist_ok=True)

    line_no = 1
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < 15:
            continue
        line_img = gray[y:y+h, :]
        cv2.imwrite(f"{out_folder}/line_{line_no}.png", line_img)
        line_no += 1

def process_all_writers():
    base = Path("Dataset")
    writers = ["WRITER_A", "WRITER_B", "WRITER_C", "WRITER_D"]

    for w in writers:
        writer_folder = base / w
        pages = list(writer_folder.glob("*.png"))

        lines_out = base / f"{w}_LINES"
        os.makedirs(lines_out, exist_ok=True)

        print(f"Processing {w} ...")
        for page in pages:
            page_out_dir = lines_out / page.stem
            segment_page_into_lines(page, page_out_dir)

if __name__ == "__main__":
    process_all_writers()
    print("DONE SEGMENTING ALL PAGES.")
