import numpy as np
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split  # splits database
from sklearn.preprocessing import StandardScaler  # standardize data
from sklearn.metrics import accuracy_score  # grade the results
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA

# read in data and get number of rows and columns
sonar_data = read_csv('sonar_all_data_2.csv')
# column 61 is a string but is already mapped to a value in the previous row 60
# Rock = 1 and Mine = 2, so last column is dropped
sonar_data.drop(sonar_data.columns[61], axis=1, inplace=True)
num_rows = len(sonar_data)  # number of rows spans 0 - 206
num_columns = len(sonar_data.columns)  # number of columns spans 0 - 60
# print(num_rows)  # debug statement
# print(num_columns)  # debug statement

# check for null item rows, none found
# print('Count of null item rows \n')  # debug statement
# print(sonar_data.isnull().sum(), '\n')  # debug statement

# split the training and test data and extract features (X) and classifications (y)
X = sonar_data.iloc[:, :-1]  # separate the features we want
y = sonar_data.iloc[:, -1]  # extract the classifications
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# transform the data
sc = StandardScaler()  # create the standard scalar
sc.fit(X_train)  # compute the required transformation
X_train_std = sc.transform(X_train)  # apply to the training data
X_test_std = sc.transform(X_test)  # and SAME transformation of test data

n = 1
accuracy_list = []
confusion_matrix_list = []
while n < num_columns:
    # perform Primary Component Analysis to find most effective number
    pca = PCA(n_components=n)                    # only keep "best" features!
    X_train_pca = pca.fit_transform(X_train_std) # apply to the train data
    X_test_pca = pca.transform(X_test_std)       # do the same to the test data

    # build the multi-level Perceptron
    model = MLPClassifier(hidden_layer_sizes=100, activation='logistic', max_iter=2000, alpha=0.00001, \
                          solver='adam', tol=0.0001, random_state=0)
    model.fit(X_train_pca, y_train)  # do the training
    print('\n')
    print('MCL Classifier Results using', n, 'components')
    print('Number in test ', len(y_test))
    y_pred = model.predict(X_test_pca)  # now try with the test data
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy_score_ = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy_score_)  # append accuracy score to list for sorting

    # Create and append confusion matrix to list for sorting
    cmat = confusion_matrix(y_test, y_pred)
    confusion_matrix_list.append(cmat)
    n += 1
# print(confusion_matrix_list)  # debug statement
# print(accuracy_list)  # debug statement
max_index = accuracy_list.index(max(accuracy_list))
# print(max_index)  # debug statement
print('\nMaximum accuracy found using', max_index + 1, 'components')
print('Accuracy percentage of test using', max_index + 1, 'components:', accuracy_list[7])
print('Confusion Matrix for test using', max_index + 1, 'coponents\n', confusion_matrix_list[7])
x_list = np.arange(1, 61)
plt.plot(x_list, accuracy_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Components Used')
plt.title('Accuracy Percentage per Number of Components Used')
plt.show()
