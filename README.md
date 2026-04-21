# Handwriting_Identification_And_Analysis

This project is a machine learning-based system that analyzes handwritten documents and performs:

- ✍️ Writer Identification
- 🧾 OCR (Text Extraction)
- 📚 Notes Enhancement (Structured Output)

## 🚀 Features

- Detects **who wrote the document**
- Extracts handwritten text using **TrOCR**
- Converts raw text into **clean structured notes**
- Beautiful **Streamlit Web Interface**

## 🏗️ Project Pipeline

1. Image Upload
2. Preprocessing (noise removal, line cleaning)
3. Line Segmentation
4. Writer Identification (ResNet18)
5. OCR using TrOCR
6. Notes Enhancement using NLP

## 🖼️ Demo

Upload a handwritten image → get:

- Writer Prediction
- OCR Text
- Enhanced Notes

## 📂 Project Structure
HANDWRITING_AI/
├── app.py # Streamlit web app
├── pipeline.py # Main pipeline logic
├── segment_lines.py # Line segmentation
├── predict_writer.py # Writer identification
├── ocr_trocr.py # OCR using TrOCR
├── enhance_notes.py # Notes enhancement
├── train_writer_id.py # Model training
├── create_writer_id_csv.py # Dataset labeling
│
├── writer_id_best.pt # Trained model
├── writer_id.csv # Dataset labels
├── test_page.png # Sample input
│
├── Dataset/ # (Not uploaded - large size)
└── README.md

## ▶️ Run the Project
--on terminal run:
streamlit run app.py

📊 Dataset Section
- Multi-writer handwritten dataset
- Pages segmented into line-level images
- Dataset not uploaded due to large size (~776MB)

## Models Used
- ResNet18 → Writer Identification
- TrOCR → Handwriting OCR
- NLP Model / API → Notes Enhancement
