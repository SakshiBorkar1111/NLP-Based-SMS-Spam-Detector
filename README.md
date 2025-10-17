# NLP-Based-SMS-Spam-Detector
Classifies SMS messages as spam or ham using NLP preprocessing and machine learning.

## Project Overview
This project demonstrates how to classify SMS messages as **spam** or **ham (normal)** using **Natural Language Processing (NLP)** techniques and the **Naive Bayes** machine learning algorithm. The pipeline includes text preprocessing, feature extraction, model training, evaluation, and prediction on new messages.

---

## Business Problem
The goal of this project is to classify incoming SMS messages as spam or ham (normal) using Natural Language Processing (NLP) and machine learning. This helps automate filtering of unwanted messages and improves communication efficiency.

---

## Dataset
The dataset used is the **SMS Spam Collection Dataset**, which contains labeled SMS messages as either `spam` or `ham`.  
- Format: Tab-separated (`\t`)  
- Columns: `label` (spam/ham), `message` (text content)  

---
## Technologies Used
-	Python 3
-	NLP: NLTK (Stopwords, Stemming)
-	Feature Extraction: CountVectorizer
-	Machine Learning: Multinomial Naive Bayes
-	Evaluation: Scikit-learn

  ---

## Project Workflow

1. **Data Understanding**
   - Load the dataset and explore label distribution.
   - Understand the characteristics of spam vs ham messages.

2. **Text Preprocessing**
   - Clean text: remove non-alphabetic characters, convert to lowercase.
   - Removed stopwords and applied **stemming**.
   - Build a cleaned corpus for modeling.

3. **Feature Extraction**
   - Convert text to numerical vectors using **CountVectorizer** (Bag of Words).

4. **Train-Test Split**
   - Split the dataset into training and testing sets (80%-20%).

5. **Modeling**
   - Train a **Multinomial Naive Bayes** classifier on the training data.
   - Predict labels for training and test sets.

6. **Evaluation**
   - Evaluate model using:
     - Accuracy
     - Confusion Matrix
     - Precision, Recall, F1-Score
     - Cross-validation

7. **Prediction on New Messages**
   - Input a new SMS message.
   - Preprocess and vectorize it.
   - Predict if it is spam or ham.

---

## Key Results
- High accuracy (~98%) on the SMS Spam Collection Dataset.
- Spam recall and precision are strong, ensuring reliable detection of spam messages.

---



