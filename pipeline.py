import os

from predict_writer import predict_writer
from ocr_trocr import extract_text
from enhance_notes import enhance_notes


# ----------------------------------------------------
# MAIN PIPELINE FUNCTION
# ----------------------------------------------------

def process_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError("Input image not found")

    print("\n===== STEP 1 : WRITER IDENTIFICATION =====")

    writer, confidence, vote_map = predict_writer(image_path)

    print("Predicted Writer:", writer)
    print("Confidence:", confidence, "%")

    print("\n===== STEP 2 : OCR TEXT EXTRACTION =====")

    raw_text = extract_text(image_path)

    print("\nRAW OCR TEXT:")
    print(raw_text)

    print("\n===== STEP 3 : NOTES ENHANCEMENT =====")

    enhanced_notes = enhance_notes(raw_text)

    print("\nFINAL ENHANCED NOTES:\n")
    print(enhanced_notes)

    result = {
        "writer": writer,
        "confidence": confidence,
        "ocr_text": raw_text,
        "enhanced_notes": enhanced_notes
    }

    return result


# ----------------------------------------------------
# TEST PIPELINE
# ----------------------------------------------------

if __name__ == "__main__":

    test_image = "test_page.png"

    output = process_image(test_image)

    with open("final_notes.txt", "w", encoding="utf-8") as f:
        f.write("Writer: " + output["writer"] + "\n\n")
        f.write("Confidence: " + str(output["confidence"]) + "%\n\n")
        f.write("===== OCR TEXT =====\n\n")
        f.write(output["ocr_text"] + "\n\n")
        f.write("===== ENHANCED NOTES =====\n\n")
        f.write(output["enhanced_notes"])

    print("\nResults saved to final_notes.txt")