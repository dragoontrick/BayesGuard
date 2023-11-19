# =============================================================================
# Project Name: BayesGuard
# =============================================================================

import os
import email
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define a function to extract text from a .eml file
def extract_text_from_eml(eml_file):
    with open(eml_file, 'r', encoding='utf-8') as file:
        msg = email.message_from_file(file)
        text = ''
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                text += part.get_payload()
        return text

# Define the directory containing your .eml files
email_dir = 'emails'

# Create lists to store email text and labels (0 for legitimate, 1 for spam)
emails = []
labels = []

# Read and label the email content
for filename in os.listdir(email_dir):
    if filename.endswith('.eml'):
        email_path = os.path.join(email_dir, filename)
        email_text = extract_text_from_eml(email_path)
        if email_text:  # Check if email text is not empty
            emails.append(email_text)
            if '_spam' in email_path:
                labels.append(1)  # Spam
            else:
                labels.append(0)  # Legitimate

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.9, random_state=100)

# Check if there's any email text data to process
if not X_train:
    print("No valid email text data found.")
else:
    # Vectorize the email text using CountVectorizer
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)

    # Count how many emails that are being read through the program
    email_count = 0

    # Classify individual email files and output the results
    for filename, email_text in zip(os.listdir(email_dir), X_test):
        prediction = classifier.predict(email_text)
        result = 'spam email' if prediction[0] == 1 else 'legitimate source'
        email_count += 1
        print(f"{filename}: {result} ({email_count}/92)")

    # Print an output that shows the overall accuracy of the classifier
    print("=========================================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Print an output that shows how many emails were classified correctly
    print(f"Correctly Classified: {accuracy * len(y_test):.0f}")
    # and how many were classified incorrectly
    print(f"Incorrectly Classified: {(1 - accuracy) * len(y_test):.0f}")

    # Print a number for how many emails were read through the program
    print(f"Total Emails Read: {len(y_test)}")
    print("=========================================")

# ===== END OF FILE ===========================================================