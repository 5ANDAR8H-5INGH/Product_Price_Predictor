# 📦 Product Price Predictor 

A multimodal ML pipeline for e-commerce product price prediction using text and image embeddings..
## 🌐 Live Demo

🚀 Deployed Application: 
[![Live-Demo Click-Here](https://img.shields.io/badge/Live-Demo-green?style=for-the-badge)](https://amazon-ml-challenge-2025.onrender.com)

## 🧠 Project Overview

This project predicts product prices using both:

Textual product descriptions (TF-IDF)

Product images (ResNet50 embeddings)

The task is formulated as a regression problem where the model learns to estimate prices based on multimodal features.
The evaluation metric used is Symmetric Mean Absolute Percentage Error (SMAPE), suitable for wide price ranges.


## 🚀 Features
🔹 Text Features

Cleaned and preprocessed catalog_content

TF-IDF vectored text with unigrams + bigrams

High-quality sparse matrix input for regression

🔹 Image Features

ResNet50 (ImageNet pretrained) as a fixed feature extractor

Extracted feature vectors (2048-dim) per image

Failed downloads handled with dummy embeddings


## 📊 Model Architecture
Component	Technique :

Text Encoder	TF-IDF Vectorizer (sklearn)

Image Encoder	ResNet50 + GlobalAvgPool

Regression Model	LightGBM Regressor

Loss Metric	SMAPE (Symmetric Mean Absolute Percentage Error)

## 🛠 How It Works
1️⃣ Data Preparation

Load Amazon product dataset

Clean and preprocess text fields

Extract numeric metadata (e.g. pack sizes)

Handle missing entries carefully

2️⃣ TF-IDF Vectorization

Fit TF-IDF on text

Limit vocabulary with min_df / max_df

Convert text to sparse representation

3️⃣ Image Embedding Extraction

Parse image link filenames

Download images (with error handling)

Process images with ResNet50

Save embeddings to disk

❗ If test images are unavailable (HTTP 429), dummy zero vectors are used.

4️⃣ Model Training

model = lightgbm

## Results:

Mean SMAPE: ~0.51

Std SMAPE: ~0.005

This shows a stable and reliable model.

## 📦 Inference Workflow

Load test dataset

Apply the same text cleaning

Transform text with saved TF-IDF

Extract or assign dummy image embeddings

Construct multimodal feature matrix

Predict and inverse log transform

Clip prices for SMAPE safety


## 📊 Evaluation

This problem uses SMAPE:

SMAPE = mean(|P − A| / ((|A| + |P|) / 2))

Where:

P = predicted price

A = actual price

Lower SMAPE is better.

## 💡Tips & Lessons Learned

✔ Log-transform prices before training

✔ Always clip predictions for SMAPE stability

✔ Preserve row ordering when extracting image embeddings

✔ Handle missing or broken images gracefully

## 🚩 Common Issues & Troubleshooting
❌ HTTP 429 during image download

Amazon will throttle requests — use dummy embeddings instead.

❌ Mismatched feature counts

Always reuse the same TF-IDF vectorizer and preprocessing objects.

❌ Mis-aligned embeddings

Generate image vectors in dataframe order — do not shuffle.


# 🧠 Final Note

This notebook is a complete end-to-end multimodal price predictor using classical ML + deep feature extraction — robust, interpretable, and suited for real competition needs.

Good luck! 🚀

## Authors : 
- Sandarbh Singh
- Rushabh Mowade
