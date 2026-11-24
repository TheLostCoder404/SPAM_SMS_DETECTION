# 📩 SPAM SMS DETECTION

## 📌 Project Overview
This machine learning project classifies SMS messages into two categories:
- **Ham:** legitimate messages
- **Spam:** unwanted or potentially fraudulent messages

The project uses **TF-IDF** to convert text into numerical features and compares three machine learning algorithms: Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM).

## 🎯 Objectives
- Prepare and analyze a labeled SMS dataset.
- Clean and preprocess message data.
- Extract text features using TF-IDF.
- Train and compare three classification algorithms.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Save the trained models and deploy a prediction interface using Streamlit.

## 📂 Dataset
The project uses the CSV contained in the ZIP file supplied for this project.

After selecting the two useful columns and removing duplicate records, the prepared dataset contains **5,158 messages**:
- Ham: **4,516**
- Spam: **642**

Original CSV columns `v1` and `v2` are interpreted as label and message. Extra mostly empty columns are excluded.

## 🧠 Machine Learning Algorithms
1. **Multinomial Naive Bayes** – probabilistic classifier commonly used for text classification.
2. **Logistic Regression** – linear binary classifier that estimates class probabilities.
3. **Linear SVM** – finds a separating hyperplane for high-dimensional TF-IDF features.

## 🔤 Feature Extraction
TF-IDF (Term Frequency–Inverse Document Frequency) converts SMS text into numerical features. The vectorizer uses unigram and bigram features. It is fitted only on the training data to avoid test-data leakage.

## 🔄 Workflow
1. Load the SMS dataset.
2. Keep the label and message columns.
3. Remove duplicate or invalid rows.
4. Split data into training (80%) and testing (20%) sets using stratification.
5. Fit TF-IDF on training messages.
6. Train Naive Bayes, Logistic Regression, and Linear SVM.
7. Compare their test-set metrics.
8. Save all trained pipelines and the best-performing pipeline.
9. Run the Streamlit application for new SMS predictions.

## 📊 Evaluation Results
The models were evaluated on a held-out test set. Results from this dataset run:

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-score |
|---|---:|---:|---:|---:|
| Linear SVM | 0.9845 | 0.9828 | 0.8906 | 0.9344 |\n| Naive Bayes | 0.9806 | 0.9909 | 0.8516 | 0.9160 |\n| Logistic Regression | 0.9622 | 0.9890 | 0.7031 | 0.8219 |\n
**Selected model:** Linear SVM

The best model is selected by Spam F1-score, with Spam recall and precision used to break ties. Results are specific to this train/test split and may differ if the data or settings change.

## 📈 Project Visualizations
Generated graphs are saved in `outputs/figures/`:
- `sms_class_distribution.png`
- `message_length_distribution.png`
- `frequent_spam_words.png`
- `model_performance_comparison.png`
- `model_confusion_matrices.png`

## 🗂️ Project Structure
```text
SPAM_SMS_DETECTION/
├── data/
│   ├── raw/
│   │   └── spam.csv
│   └── processed/
│       └── cleaned_spam.csv
├── models/
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── linear_svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── best_model.pkl
│   └── best_model_info.json
├── outputs/
│   ├── figures/
│   ├── model_comparison.csv
│   └── classification_reports.txt
├── src/
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SPAM_SMS_DETECTION
```

### 2. Create and activate a virtual environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

### Train the models again
```bash
python src/train.py
```

### Launch the web application
```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (usually `http://localhost:8501`).

## 🚀 Deploy on Streamlit Community Cloud
1. Push this project folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select the repository.
4. Set the main file path to `app.py`.
5. Deploy the application.

The `models/` folder is included in the repository, so the deployed app can load the trained model without retraining.

## 🧪 Example Messages
- **Possible spam:** `Congratulations! You have won a free prize. Call now!`
- **Possible ham:** `Hey, are we still meeting for lunch today?`

Predictions depend on the learned patterns in the training data; these examples are not guaranteed labels.

## ⚠️ Limitations
- The model learns from the dataset and may not recognize new spam styles.
- A legitimate message can be incorrectly classified as spam, and vice versa.
- Do not treat the output as a guarantee that a link or sender is safe.
- The dataset is imbalanced, so accuracy should be considered together with Spam precision, recall, and F1-score.

## 🛠️ Technologies Used
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Joblib | Streamlit

## 👨‍💻 Author
Add your name, GitHub profile, and contact details here.
