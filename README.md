# Fake News Detection

This project is a Machine Learning (ML) pipeline designed to detect fake news. It preprocesses text data, extracts features, and trains multiple classification models to classify news articles as fake or real. The project uses various models

---

## Features
- **Data Preprocessing**: 
  - Handles missing values by filling them with an empty string.
  - Combines title, author, and text into a single feature.
  - Cleans text data by removing punctuation and stopwords.
  - Tokenizes text and applies lemmatization for normalization.

- **Feature Extraction**:
  - Uses Count Vectorization and TF-IDF transformation to convert text into numerical features.

- **Model Training and Evaluation**:
  - Trains multiple models including Logistic Regression, Naive Bayes, Random Forest, Extra Trees, and AdaBoost classifiers.
  - Evaluates models using metrics such as Accuracy, Precision, Recall, and F1-score.
  - Visualizes performance with Confusion Matrices.

- **Pipeline Creation**:
  - Builds an end-to-end pipeline for vectorization, transformation, and classification.
  - Saves and loads the pipeline using joblib.

---

1. Clone the repository:
   ```bash
   git clone https://github.com/LavanyaBansod/-A-Systematic-Approach-to-Rumor-Detection.git
   ```

---

## Usage
1. **Train the Models**
   Run the script to preprocess the data, extract features, and train models:
   ```bash
   python train.py
   ```
2. **Test the Models**
   Use the trained pipeline to classify new text:
   ```python
   from joblib import load
   pipeline = load('./pipeline.sav')
   result = pipeline.predict(["Sample news article text here"])
   print(result)
   ```

---

## Project Workflow

1. **Data Preprocessing**:
   - Combines `title`, `author`, and `text` into a single feature `total`.
   - Cleans the text and removes noise like punctuation and stopwords.
   - Normalizes text using lemmatization.

2. **Feature Extraction**:
   - Converts text into a document-term matrix using CountVectorizer.
   - Applies TF-IDF transformation to scale term frequencies.

3. **Model Training**:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Random Forest
   - Extra Trees
   - AdaBoost

4. **Model Evaluation**:
   - Calculates metrics: Accuracy, Precision, Recall, and F1-score.
   - Plots Confusion Matrices for better insights.

5. **Pipeline Deployment**:
   - Saves the trained pipeline for future use.
   - Provides an interface to classify new text data.

---

## Results
- Trained multiple classifiers and compared their performance using evaluation metrics.
- Logistic Regression, Random Forest, and Extra Trees achieved competitive results with high accuracy and F1-scores.

---

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - nltk
  - scikit-learn
  - seaborn
  - matplotlib
  - joblib

---

## File Structure
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `train.py`: Main script for preprocessing, training, and evaluation.
- `pipeline.sav`: Saved pipeline for inference.
- `requirements.txt`: List of required Python libraries.

---

## Acknowledgments
- Inspired by the application of Machine Learning in natural language processing.
- Used publicly available datasets for training and testing.


