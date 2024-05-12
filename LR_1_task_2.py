import numpy as np
from sklearn.preprocessing import Binarizer, scale, normalize

input_data = np.array([-3.3, -1.6, 6.1, 2.4, -1.2, 4.3, -3.2, 5.5, -6.1, -4.4, 1.4, -1.2])
threshold = 2.1

# Бінарізація
binarizer = Binarizer(threshold=threshold)
binary_data = binarizer.fit_transform(input_data.reshape(1, -1))
print("Binary Data:\n", binary_data)

# Виключення середнього
mean_excluded_data = input_data - np.mean(input_data)
print("\nMean Excluded Data:\n", mean_excluded_data)

# Масштабування
scaled_data = scale(input_data)
print("\nScaled Data:\n", scaled_data)

# Нормалізація
normalized_data = normalize(input_data.reshape(1, -1))
print("\nNormalized Data:\n", normalized_data)
