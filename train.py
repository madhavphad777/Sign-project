import pickle
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check the shape of each element
data_list = data_dict['data']
labels_list = data_dict['labels']

# Find the desired shape (assuming the first element has the correct shape)
desired_shape = np.shape(data_list[0])

# Pad or truncate inconsistent entries
def pad_or_truncate(data, desired_shape):
    current_shape = np.shape(data)
    if current_shape == desired_shape:
        return data
    elif current_shape[0] < desired_shape[0]:
        return np.pad(data, (0, desired_shape[0] - current_shape[0]), 'constant')
    else:
        return data[:desired_shape[0]]

processed_data = [pad_or_truncate(data, desired_shape) for data in data_list]

# Convert to NumPy arrays
data = np.array(processed_data)
labels = np.array(labels_list)

# Proceed with the rest of the code
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

