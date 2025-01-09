# Sentiment-Analysis
Sentiment Analysis Using NLP

Overview

This project implements a sentiment analysis pipeline to classify textual data (e.g., tweets or reviews) as positive or negative. Using natural language processing (NLP) techniques, the project preprocesses text data, applies vectorization, and trains a Logistic Regression model to predict sentiments.


# Features


Data Preprocessing: Includes text cleaning, tokenization, stopword removal, and normalization.

Vectorization: Uses TF-IDF (Term Frequency-Inverse Document Frequency) for text representation.

Model Training: Implements Logistic Regression for sentiment classification.

Evaluation: Provides accuracy, classification report, and confusion matrix for model performance assessment.

# Dataset


The project uses a small dataset of 10 sample reviews with corresponding sentiment labels:

Positive Sentiment: Labelled as 1

Negative Sentiment: Labelled as 0



# Project Workflow


Data Loading: Import a dataset containing text reviews and sentiment labels.

Preprocessing:

Convert text to lowercase.

Remove special characters and punctuation.

Tokenize and remove stopwords using NLTK.

Vectorization: Transform text into numerical features using TF-IDF.

Model Training: Train a Logistic Regression model on the processed data.

Evaluation: Evaluate model performance using:

Accuracy score.

Confusion matrix.

Classification report (precision, recall, F1-score).

Visualization: Display the confusion matrix using a heatmap.

# Results

Accuracy: Achieved approximately 80% accuracy (depending on the dataset split).

Confusion Matrix: Highlights true positives, true negatives, false positives, and false negatives.

# Insights:

The model performs well on a small dataset but requires a larger dataset for real-world applications.

Tools and Libraries

Programming Language: Python

# Libraries:

pandas for data manipulation.

nltk for natural language processing.

scikit-learn for machine learning.

matplotlib and seaborn for visualization.
