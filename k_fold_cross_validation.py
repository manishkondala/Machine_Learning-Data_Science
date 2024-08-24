import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

clf = svm.SVC(kernel = 'linear', C = 1).fit(x_train, y_train)

# print(clf.score(x_test, y_test))

scores = cross_val_score(clf, iris.data, iris.target, cv=4)

# print(scores)

# print(scores.mean())

clf = svm.SVC(kernel='poly', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
print(scores.mean())