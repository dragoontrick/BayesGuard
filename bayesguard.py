# =============================================================================
# Project: Bayesian Spam Filter
# =============================================================================

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Function to read .eml files
def read_email_files(folder_path):
    emails = []
    labels = []
    filenames = []  # To store the original filenames
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
                filenames.append(filename)
    return emails, labels, filenames  # Return the filenames along with emails and labels

# Specify the path to the folder containing your .eml files
folder_path = "path/to/your/email/folder"

# Read email data and labels
emails, labels, filenames = read_email_files(folder_path)

# Create a feature vector using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test, filenames_test = train_test_split(X, labels, filenames, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Output the filenames along with their classifications
for filename, prediction in zip(filenames_test, y_pred):
    print(f"File: {filename} - Classification: {prediction}")

# ===== END OF FILE ===========================================================