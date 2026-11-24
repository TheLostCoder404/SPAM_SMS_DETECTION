# 📩 Spam SMS Detection Using Machine Learning

## 📌 1. Project Overview

**Spam SMS Detection** is a Machine Learning project that classifies a text message as **spam** or **ham** (legitimate). It converts message text into numerical features with TF-IDF and compares three classifiers: Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM).

The best-performing model is saved and used by a Streamlit web application, where a user can enter a message and receive a prediction.

---

## 🎯 2. Problem Statement

Unwanted and fraudulent text messages can contain misleading offers, suspicious links, or requests for personal information. Reviewing every message manually can be difficult, so this project applies text classification to identify patterns associated with spam and legitimate messages.

### 🔍 Problems Addressed

- Prepare labeled SMS messages for analysis.
- Convert message text into useful numerical features.
- Compare different classification algorithms.
- Evaluate how well the models identify spam messages.
- Provide an interactive interface for trying a message.

The model provides an estimate based on patterns in its training data. It cannot guarantee that a message or link is safe.

---

## 🎯 3. Project Objectives

- Load and prepare the SMS message dataset.
- Remove duplicate or invalid records and retain the label and message columns.
- Split the data into training and testing sets.
- Extract text features using TF-IDF with unigram and bigram features.
- Train and compare three Machine Learning classifiers.
- Save the trained models and evaluation results.
- Build a Streamlit application for single-message predictions.

---

## 📊 4. Dataset Description

The project includes a raw SMS dataset at `data/raw/spam.csv` and a prepared dataset at `data/processed/cleaned_spam.csv`. The prepared file retains the label and message columns; extra mostly empty columns from the raw CSV are excluded.

### Dataset Information

- 📌 Records after cleaning: **5,158**
- 📌 Ham messages: **4,516**
- 📌 Spam messages: **642**
- 📌 Input: SMS message text
- 📌 Target: `label` (`ham` or `spam`)

The training script uses an 80:20 stratified train-test split with a fixed random state of 42. The source and license details for the supplied dataset should be checked in its original documentation before redistributing it.

---

## 🛠️ 5. Technologies Used

| Technology | Purpose |
|:---|:---|
| 🐍 Python | Programming language |
| 🐼 Pandas | Loading and preparing tabular data |
| 🔢 NumPy | Numerical operations and dependency support |
| 🤖 Scikit-learn | TF-IDF, model training, and evaluation |
| 📈 Matplotlib | Project visualizations |
| 💾 Joblib | Saving and loading trained models |
| 🌐 Streamlit | Interactive prediction application |

---

## 🔄 6. Project Workflow

1. 📂 Load the prepared SMS dataset.
2. 🧹 Use the message text and corresponding labels.
3. ✂️ Split the data into training and test sets, preserving class proportions.
4. 🔤 Fit a TF-IDF vectorizer on training messages using unigrams and bigrams.
5. 🤖 Train Naive Bayes, Logistic Regression, and Linear SVM classifiers.
6. 📈 Compare test-set accuracy, spam precision, spam recall, and spam F1-score.
7. 🏆 Select the model with the best spam F1-score (then recall and precision as tie-breakers).
8. 💾 Save model pipelines, model information, and evaluation reports.
9. 🌐 Load the selected model in the Streamlit app to classify entered messages.

---

## 🤖 7. Machine Learning Algorithms

The project compares three text-classification algorithms:

1. **Multinomial Naive Bayes** — a probabilistic classifier commonly used with word-count or TF-IDF text features.
2. **Logistic Regression** — a linear classifier that estimates class probabilities.
3. **Linear Support Vector Machine (SVM)** — finds a decision boundary for separating classes in the high-dimensional TF-IDF feature space.

Each model is trained in a pipeline with the same TF-IDF configuration so their results can be compared on the same held-out test set.

---

## 📈 8. Model Evaluation

The models are evaluated on the test set using accuracy, spam precision, spam recall, and spam F1-score.

### 🏆 Model Performance Comparison

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|:---|---:|---:|---:|---:|
| Linear SVM | 0.9845 | 0.9828 | 0.8906 | **0.9344** |
| Naive Bayes | 0.9806 | 0.9909 | 0.8516 | 0.9160 |
| Logistic Regression | 0.9622 | 0.9890 | 0.7031 | 0.8219 |

**Selected model:** Linear SVM. The selection uses spam F1-score, then spam recall and precision to break ties. These results are specific to the included train-test split and may change with different data or settings.

Detailed results are stored in `outputs/model_comparison.csv`; classification reports are stored in `outputs/classification_reports.txt`.

---

## 📊 9. Output Graphs

The project archive includes the following figures in `outputs/figures/`:

- `sms_class_distribution.png` — ham and spam class counts.
- `message_length_distribution.png` — distribution of message lengths.
- `frequent_spam_words.png` — frequent terms in spam messages.
- `model_performance_comparison.png` — comparison of model metrics.
- `model_confusion_matrices.png` — test-set classification outcomes.

---

## 🌐 10. Web Application

The Streamlit app lets a user enter one SMS message and view the model’s predicted class.

### ✨ Key Features

- 📝 Text box for entering an SMS message.
- 🤖 Classification as **SPAM** or **HAM (LEGITIMATE)**.
- 💾 Loads the saved best model from `models/best_model.pkl`.
- ⚡ Predicts without retraining the model during normal app use.
- ⚠️ Displays a reminder that predictions may be incorrect.

---

## 📁 11. Project Structure

```text
SPAM_SMS_DETECTION/
├── data/
│   ├── raw/
│   │   └── spam.csv
│   ├── processed/
│   │   └── cleaned_spam.csv
│   └── README.txt
├── models/
│   ├── best_model.pkl
│   ├── best_model_info.json
│   ├── linear_svm_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
├── outputs/
│   ├── figures/
│   ├── classification_reports.txt
│   └── model_comparison.csv
├── src/
│   └── train.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ 12. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/TheLostCoder404/SPAM_SMS_DETECTION.git
cd SPAM_SMS_DETECTION
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`). The included trained model lets the app make predictions without retraining.

### Optional: Train the Models Again

```bash
python src/train.py
```

The training script expects `data/processed/cleaned_spam.csv` to be present. It writes trained models to `models/` and evaluation tables and reports to `outputs/`.

---

## ⚠️ 13. Project Limitations

- The dataset is imbalanced, with fewer spam messages than ham messages.
- The model can miss new spam patterns or incorrectly flag a legitimate message.
- Evaluation results depend on the supplied dataset and the selected train-test split.
- A model prediction is not a guarantee that a sender, message, or link is safe.
- Check the original dataset documentation and license before sharing the dataset further.

---

## 🚀 14. Future Improvements

- Add cross-validation and hyperparameter tuning.
- Evaluate performance on newer and more varied SMS data.
- Add a batch prediction option for multiple messages.
- Add clearer explanations of which message features influence a prediction.
- Expand automated data preparation and visualization steps.
- Deploy the Streamlit app to a hosting service.

---

## 🏁 15. Conclusion

This project demonstrates an end-to-end SMS spam classification workflow, from preparing labeled text and building TF-IDF features to comparing classifiers and serving predictions in a Streamlit app. On the included test split, Linear SVM achieved the highest spam F1-score among the three evaluated models.

---

## 👨‍💻 Project Information

- **Project:** Spam SMS Detection Using Machine Learning
- **GitHub Repository:** [SPAM_SMS_DETECTION](https://github.com/TheLostCoder404/SPAM_SMS_DETECTION)

---

## ⭐ Support the Project

If this project is useful to you, consider starring the repository and sharing feedback.

`#MachineLearning` `#Python` `#SMSSpamDetection` `#NLP` `#ScikitLearn` `#Streamlit`
