# BayesGuard
This project demonstrates how to build a simple email spam classification system using Python, scikit-learn, and the Multinomial Naive Bayes classifier. The program reads email data from `.eml` files, creates a bag-of-words representation of the email text, trains a classifier, and evaluates its performance.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following libraries installed:

- scikit-learn
- numpy
- pandas (for data management, if needed)

You can install these libraries using `pip`:

```pip install scikit-learn numpy pandas```

### Dataset

Prepare your email dataset with labeled emails. The emails should be in `.eml` format, and the labels can be based on filenames (e.g., "spam" or "legitimate").

### Running the Program

1. Clone this repository:

```git clone https://github.com/your-username/email-spam-classification.git
cd email-spam-classification```

2. Specify the path to your email dataset folder in the `read_email_files` function within the Python script.

3. Run the Python script:

```python bayesguard.py```

The program will process your dataset, train a classifier, and evaluate its performance.
