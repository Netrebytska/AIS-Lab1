import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення наївного байєсовського класифікатора
classifier = GaussianNB()

# Тренування класифікатора
classifier.fit(X_train, y_train)

# Прогнозування значень для тестових даних
y_test_pred = classifier.predict(X_test)

# Обчислення точності класифікатора
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of Naive Bayes classifier on test data =", round(accuracy, 2), "%")

# Візуалізація результатів класифікації для тренувальних даних
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=50, cmap=plt.cm.Paired)
plt.title('Naive Bayes Classifier: Training Data')
plt.show()

# Візуалізація результатів класифікації для тестових даних
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', s=50, cmap=plt.cm.Paired)
plt.title('Naive Bayes Classifier: Test Data')
plt.show()
