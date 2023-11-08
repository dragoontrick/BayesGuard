# =============================================================================
# Project: Bayesian Spam Filter
# =============================================================================

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Function to read .eml files
def read_email_files(folder_path):
    emails = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".eml"):
            email_file_path = os.path.join(folder_path, filename)
            with open(email_file_path, "r", encoding="utf-8") as file:
                email_content = file.read()
                emails.append(email_content)
                # Label your emails as spam or legitimate based on their filenames
                if "spam" in filename:
                    labels.append("spam")
                else:
                    labels.append("legitimate")
    return emails, labels

# Specify the path to the folder containing your .eml files
folder_path = "emails"

# Read email data and labels
emails, labels = read_email_files(folder_path)

# Create a feature vector using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# ===== END OF FILE ===========================================================