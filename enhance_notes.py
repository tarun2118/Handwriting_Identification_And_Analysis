from google import genai
import re

# ------------------------------------------------
# CREATE CLIENT
# ------------------------------------------------

client = genai.Client(api_key="AIzaSyBspekisawJeBmPgVXbD6e4S-cxSOyKAO0")


# ------------------------------------------------
# FALLBACK LOCAL NOTE STRUCTURER
# ------------------------------------------------

import re

def fallback_notes(text):

    # remove symbols
    text = re.sub(r'[^a-zA-Z0-9., ]', ' ', text)

    # split sentences
    sentences = text.split('.')

    points = []

    for s in sentences:
        s = s.strip()

        # remove very short fragments
        if len(s) < 20:
            continue

        points.append("• " + s.capitalize())

    if not points:
        return "Could not generate structured notes from OCR text."

    return "\n".join(points)


# ------------------------------------------------
# NOTE ENHANCEMENT
# ------------------------------------------------

def enhance_notes(raw_text):

    prompt = f"""
Convert the following rough OCR notes into clean structured study notes.

Rules:
- Identify the topic
- Correct OCR errors
- Use bullet points
- Add short explanations if needed

Notes:
{raw_text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text

    except Exception as e:

        print("Gemini API failed. Using fallback structuring.")
        return fallback_notes(raw_text)