# AI-Powered Phishing Detection System

Machine learning-based phishing detection system designed to identify malicious Email and SMS messages through text analysis, feature engineering, and classification models.

## Overview

Phishing attacks remain one of the most common cybersecurity threats, targeting users through email and SMS communications. This project explores the effectiveness of machine learning techniques in detecting phishing attempts across multiple communication platforms.

The system utilizes text preprocessing, TF-IDF feature extraction, and supervised machine learning models to classify messages as legitimate or phishing.

---

## Live Demo

🚀 Hugging Face Space:

https://huggingface.co/spaces/christianj-cc/phishing-detection

---

## Features

- Email phishing detection
- SMS phishing detection
- Text preprocessing pipeline
- TF-IDF feature extraction
- Machine learning classification
- Interactive web interface
- Real-time prediction results

---

## Technologies Used

### Programming Languages

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- TF-IDF Vectorization

### Deployment

- Hugging Face Spaces
- Docker

### Data Processing

- Pandas
- NumPy

---

## Methodology

The project follows a machine learning pipeline consisting of:

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Extraction using TF-IDF
4. Model Training
5. Hyperparameter Tuning
6. Performance Evaluation
7. Deployment

---

## Models Evaluated

- Random Forest
- Support Vector Machine (SVM)
- Neural Network
- XGBoost
- Weighted Soft-Voting Ensemble

---

## Results

### Email Dataset

| Model         | Accuracy | F1 Score |
| ------------- | -------- | -------- |
| Random Forest | 98.62%   | 98.68%   |

### SMS Dataset

| Model | Accuracy | F1 Score |
| ----- | -------- | -------- |
| SVM   | 98.49%   | 95.98%   |

The project demonstrated strong phishing detection performance across both communication platforms while highlighting the strengths of different models under varying conditions.

---

## Project Structure

```text
app.py               Project code
requirements.txt     Deployment requirements
screenshots/         Project screenshots
docs/                Project documentation
```

## Installation

```bash
git clone <repository-url>
cd phishing-detection
pip install -r requirements.txt
python app.py
```

## Screenshots

### Application Interface
![Email Interface](screenshots/Screenshot_(968).png)
![SMS Interface](screenshots/Screenshot_(969).png)

### Prediction Results (Email)
![Phishing Email](screenshots/Screenshot_(973).png)
![Legitimate Email](screenshots/Screenshot_(974).png)

### Prediction Results (Email)
![Phishing SMS](screenshots/Screenshot_(975).png)
![Legitimate SMS](screenshots/Screenshot_(976).png)

---

## Contributors

This project was developed as part of an academic research project.

### My Contributions

- Machine learning pipeline development
- Data preprocessing and feature engineering
- Model training and evaluation
- Comparative analysis of classifiers
- Application deployment
- Documentation and reporting

---

## Learning Outcomes

- Machine Learning
- Cybersecurity
- Natural Language Processing
- Feature Engineering
- Model Evaluation
- Data Analysis
- Research Methodology
- AI Deployment

---

## Future Improvements

- Deep Learning models
- Explainable AI integration
- URL analysis features
- Browser extension deployment
- Real-time threat intelligence integration
