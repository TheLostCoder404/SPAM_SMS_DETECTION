"""Train and evaluate three SMS spam classifiers."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "cleaned_spam.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    data = pd.read_csv(DATA)
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"].astype(str), data["label"].astype(str),
        test_size=0.20, random_state=42, stratify=data["label"]
    )
    classifiers = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "Linear SVM": LinearSVC(random_state=42)
    }
    rows, report_text = [], []
    for name, classifier in classifiers.items():
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                min_df=2, max_df=0.98, sublinear_tf=True, stop_words="english"
            )),
            ("classifier", classifier)
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (Spam)": precision_score(y_test, pred, pos_label="spam", zero_division=0),
            "Recall (Spam)": recall_score(y_test, pred, pos_label="spam", zero_division=0),
            "F1-score (Spam)": f1_score(y_test, pred, pos_label="spam", zero_division=0)
        })
        report_text.append(f"\n{name}\n" + classification_report(y_test, pred, zero_division=0))
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}_model.pkl")
    results = pd.DataFrame(rows).sort_values(
        ["F1-score (Spam)", "Recall (Spam)", "Precision (Spam)"], ascending=False
    )
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["Model"]
    best_model = joblib.load(MODEL_DIR / f"{best_name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(best_model.named_steps["tfidf"], MODEL_DIR / "tfidf_vectorizer.pkl")
    (MODEL_DIR / "best_model_info.json").write_text(json.dumps({
        "best_model": best_name,
        "selection_criteria": "Highest Spam F1-score, then recall, then precision."
    }, indent=4), encoding="utf-8")
    (OUTPUT_DIR / "classification_reports.txt").write_text(
        "\n".join(report_text), encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")

if __name__ == "__main__":
    main()
