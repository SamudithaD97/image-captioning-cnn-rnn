Image Captioning Project (CNN + RNN with Attention)

Author: Samuditha Wijayasundara
Course: CSCI-P-558 Deep Learning

---

1. Project Overview

---

This project implements an image captioning system that generates textual descriptions for images using deep learning. The model combines a Convolutional Neural Network (CNN) encoder with a Recurrent Neural Network (RNN) decoder.

The encoder extracts visual features from images, while the decoder generates captions word-by-word. An attention mechanism is incorporated to improve performance by allowing the model to focus on relevant parts of the image during caption generation.

---

2. Project Structure

---

project/
│
├── data/
│   ├── Flickr8k_Dataset/
│   └── Flickr8k_text/
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── model.py
│
├── utils/
│   ├── dataset.py
│   ├── vocabulary.py
│   ├── config.py
│   └── bleu.py
│
├── train.py
├── generate.py
├── requirements.txt
└── README.txt

---

3. Requirements

---

Install required packages:

pip install -r requirements.txt

OR manually:

pip install torch torchvision pandas pillow nltk tqdm

Also run (once):

import nltk
nltk.download('punkt')

---

4. Dataset Setup

---

Download Flickr8k dataset and place files as:

data/Flickr8k_Dataset/        (images)
data/Flickr8k_text/           (captions)

Ensure folder names are exactly:

* Flickr8k_Dataset
* Flickr8k_text

---

5. How to Run

---

## Step 1: Train the model

python train.py

This will:

* Load dataset
* Train CNN + RNN model
* Print loss and BLEU score
* Save trained model as: model.pth

## Step 2: Generate captions

python generate.py

---

6. Model Details

---

* Encoder: ResNet-50 (pretrained)
* Decoder: LSTM
* Attention: Yes
* Loss: CrossEntropyLoss
* Optimizer: Adam

---

7. Results

---

* Training loss decreased from ~8.0 to ~2.1
* Model generates semantically meaningful captions
* BLEU score is low due to strict evaluation but improves with smoothing

---

8. Known Limitations

---

* Generates generic captions in some cases
* Limited fine-grained detail recognition
* BLEU score may not reflect semantic correctness

---

9. Notes

---

* Dataset filtering is applied to remove missing images
* Code is designed to run on both local machine and server
* Paths are dynamically handled for portability

---
