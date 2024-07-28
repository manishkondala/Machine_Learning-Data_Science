import os
import io
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def readFiles(path):
    for root, dir, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            inBody = False
            lines = []
            with io.open(file_path, 'r', encoding='latin1') as f:
                for line in f:
                    if inBody:
                        lines.append(line)
                    elif line == '\n':
                        inBody = True
            message = '\n'.join(lines)
            yield file_path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

spam_data = dataFrameFromDirectory(r'C:\Users\Lenovo\OneDrive\Desktop\Self\Udemy_genAI_ML_python\emails\spam', "spam")
ham_data = dataFrameFromDirectory(r'C:\Users\Lenovo\OneDrive\Desktop\Self\Udemy_genAI_ML_python\emails\ham', "ham")
data = pd.concat([spam_data, ham_data])

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)