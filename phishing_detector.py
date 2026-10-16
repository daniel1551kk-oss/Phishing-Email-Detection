
import re
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# Demonstration dataset
data = [
    ("Your account is suspended. Verify now at http://secure-login.example", "Phishing"),
    ("Congratulations! You won a prize. Claim it immediately.", "Phishing"),
    ("Urgent: confirm your password to avoid account closure.", "Phishing"),
    ("Click this link to claim your free gift card.", "Phishing"),
    ("Your bank account is locked. Update your details now.", "Phishing"),
    ("You have won a lottery. Send your bank information.", "Phishing"),
    ("Verify your identity immediately using this link.", "Phishing"),
    ("Your payment failed. Sign in to restore your account.", "Phishing"),
    ("Act now! Your account will be deleted today.", "Phishing"),
    ("You are selected for a reward. Claim your cash prize.", "Phishing"),
    ("Please review the attached invoice and make payment.", "Phishing"),
    ("Your mailbox is full. Click here to increase storage.", "Phishing"),

    ("Hi team, the meeting is scheduled for Monday at 10 AM.", "Safe"),
    ("Please find the project report attached for review.", "Safe"),
    ("Thank you for attending today's training session.", "Safe"),
    ("Your appointment is confirmed for tomorrow morning.", "Safe"),
    ("The class assignment is due next Friday.", "Safe"),
    ("Can we discuss the project during our next meeting?", "Safe"),
    ("Your monthly newsletter is ready to read.", "Safe"),
    ("The delivery is scheduled to arrive tomorrow.", "Safe"),
    ("Please send me the presentation before the meeting.", "Safe"),
    ("Your registration confirmation is attached.", "Safe"),
    ("The office will be closed on the public holiday.", "Safe"),
    ("Here are the notes from our previous discussion.", "Safe"),
]

df = pd.DataFrame(data, columns=["email", "label"])


# Extract text, URL count, and suspicious keyword count
def extract_features(email):
    text = email.lower()

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text
    )

    keywords = [
        "urgent", "verify", "password", "prize", "claim",
        "bank", "suspended", "click", "immediately", "locked"
    ]

    keyword_count = sum(text.count(word) for word in keywords)

    return (
        f"{text} "
        f"URL_COUNT_{len(urls)} "
        f"KEYWORD_COUNT_{keyword_count}"
    )


df["processed_email"] = df["email"].apply(extract_features)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_email"],
    df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"],
)

# Build and train the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("       PHISHING EMAIL DETECTION MODEL")
print("=" * 50)
print(f"\nDataset size: {len(df)} emails")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

# Save the confusion matrix beside this Python file
BASE_DIR = Path(__file__).resolve().parent
matrix_path = BASE_DIR / "confusion_matrix.png"

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=["Safe", "Phishing"],
    cmap="Blues",
    values_format="d",
)

plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig(matrix_path)
plt.close()

print(f"Confusion matrix saved to: {matrix_path}")


# Terminal testing runs only when this file is executed directly.
# Streamlit can import model and extract_features without entering this loop.
def run_terminal_detector():
    print("\nEnter an email to test (or type 'exit' to stop).")

    while True:
        email = input("\nEmail text: ").strip()

        if email.lower() == "exit":
            print("Thank you for using the detector!")
            break

        if not email:
            print("Please enter some email text.")
            continue

        processed = extract_features(email)
        result = model.predict([processed])[0]

        probabilities = model.predict_proba([processed])[0]
        classes = list(model.named_steps["classifier"].classes_)
        confidence = probabilities[classes.index(result)] * 100

        print(f"Prediction: {result}")
        print(f"Model confidence: {confidence:.2f}%")

        if result == "Phishing":
            print("Warning: Check the sender and links carefully.")
        else:
            print("Caution: This prediction does not guarantee safety.")


if __name__ == "__main__":
    run_terminal_detector()
