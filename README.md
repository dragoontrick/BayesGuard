# BayesGuard
This project demonstrates how to build a simple email spam classification system using Python, scikit-learn, and the Multinomial Naive Bayes classifier. The program reads email data from `.eml` files, creates a bag-of-words representation of the email text, trains a classifier, and evaluates its performance.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following libraries installed:

- scikit-learn

You can install these libraries using `pip`:

```pip install scikit-learn```

### Dataset

Prepare your email dataset with labeled emails. The emails should be in `.eml` format, and the labels can be based on filenames with a tag including info if it is spam or legit. 

### Running the Program

Run the Python script:

```python bayesguard.py```

The program will process your dataset, train a classifier, and evaluate its performance.
