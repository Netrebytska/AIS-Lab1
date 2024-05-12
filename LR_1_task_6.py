import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('data_multivar_nb.txt', header=None)

y = data.iloc[:, -1]
X = data.drop(data.columns[-1], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC()
svm_model.fit(X_train, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)
nb_predictions = nb_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Accuracy for SVM:", svm_accuracy)
print("Accuracy for Naive Bayes:", nb_accuracy)

