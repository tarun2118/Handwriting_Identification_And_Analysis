import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_PATH = "writer_id_best.pt"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# LOAD MODEL + LABELS
# ------------------------------------------------

ckpt = torch.load(MODEL_PATH, map_location=device)

id2label = ckpt["id2label"]
num_classes = len(id2label)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()


# ------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ------------------------------------------------
# REMOVE NOTEBOOK LINES
# ------------------------------------------------

def remove_ruled_lines(gray):

    th = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))

    detected = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    cleaned = cv2.subtract(th, detected)

    return 255 - cleaned


# ------------------------------------------------
# EXTRACT LINES
# ------------------------------------------------

def extract_lines(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cleaned = remove_ruled_lines(gray)

    th = cv2.threshold(
        cleaned,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,3))

    dil = cv2.dilate(th, kernel, iterations=1)

    cnts = cv2.findContours(
        dil,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[1])

    lines = []

    for c in cnts:

        x,y,w,h = cv2.boundingRect(c)

        if h < 12:
            continue

        line = gray[y:y+h, :]

        lines.append(line)

    return lines


# ------------------------------------------------
# WRITER PREDICTION
# ------------------------------------------------

def predict_writer(image_path):

    lines = extract_lines(image_path)

    votes = {}

    for line in lines:

        img = cv2.resize(line, (IMG_SIZE, IMG_SIZE))

        img = np.stack([img]*3, axis=-1)

        pil = Image.fromarray(img)

        tensor = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor)

        label_id = torch.argmax(pred).item()
        label = id2label[label_id]

        votes[label] = votes.get(label,0) + 1


    final_writer = max(votes, key=votes.get)

    confidence = votes[final_writer] / sum(votes.values())

    return final_writer, round(confidence*100,2), votes