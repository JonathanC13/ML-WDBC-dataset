#2.1 Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

"""
   1. Define Problem.
   2. Prepare Data.
   3. Evaluate Algorithms.
   4. Improve Results.
   5. Present Results.
"""

pandas.set_option('display.max_columns', 40);

#<prep data>
#2.2 Load dataset
# before loading, deleted the ID column
# online source: http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/ file: WDBC.data
url = "dataset/WDBC.csv";   #source of the data
dataset = pandas.read_csv(url, header=None);    #Read a comma-separated values (csv) file into DataFrame (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html). In this case we use (filepath, names=column names);

# check value types
#print(dataset.dtypes); # shows that column 0 is not numerical, but no conversion needed since it will be used as the result

#print(dataset.head(5)); # peek top 5 instances

ResultDF = dataset[0];  # Copy a list of the result labelled attribute.

new_dataset = dataset.drop([0], axis=1); # copy the dataframe withouth the labelled attribute
cols_len = len(list(new_dataset.columns.values));
#print(cols_len);
# have to label attributes for plots to work, so just number
str_labels = [];
for i in range(0,cols_len):
    str_labels.append(str(i));
new_dataset.columns = str_labels;

#3.1 shape
print("The dimensions (row, col) of the dataset is {shape}" .format(shape=dataset.shape)); # returns the "shape/dimensions" of the dataset in terms of (rows, cols) and in this case rows = instances and col = attributes

#3.2 peek starting from the head
print("The first 20 entries.");
print(new_dataset.head(10))
print(ResultDF.head(10));
print("\n");

#3.3 descriptions
print("Describe:");
print(dataset.describe())
print("\n");

#3.4 class distribution
print("Class distribution of the labelled attribute");
print(dataset.groupby(dataset[0]).size()); # group the instances based on a specified column and get the size.
print("\n");

#4.1 box and whisker plots. Univariable plot
# Once you have made your plot, you need to tell matplotlib to show it. Since pandas already imports pyplot.
#kind is type of graph (box is a boxplot), subplots is seperate subplots for each column, layout is layout of subplots, share(x and y) is share the axis labels among the other subplots
"""
new_dataset.plot(kind='box', subplots=True, layout=(5,6), sharex=False, sharey=False);
plt.show(); #matplotlib.
"""

#4.1.1 histograms to visualize the distribution of the attribute columns
"""
dataset.hist()
plt.show()
"""
# notes:

#4.2 Multivariable plots to see the correlation of the attributes
# scatter plot matrix
"""
axs = scatter_matrix(dataset)
# rotating labels
n = len(dataset.columns) - 1
for x in range(n):
    for y in range(n):
        # to get the axis of subplots
        ax = axs[x, y]
        # to make x axis name vertical  
        #ax.xaxis.label.set_rotation(90)
        # to make y axis name horizontal 
        ax.yaxis.label.set_rotation(0)
        # to make sure y axis names are outside the plot area
        ax.yaxis.labelpad = 50
plt.show()
"""
# notes: Some display linear relation but majoirty of attribute names are unknown and as noted in the .names doc that some are ratioly related to other attributes.


# 5
# 5.1 Split-out validation dataset
array = new_dataset.values;
array_res = ResultDF.values;
attrLen = len(new_dataset.columns);
X = array[:,0:attrLen]  #list slicing for attributes. [start:stop:step], def step = 1. in this case [from start:until last instance (,0 until last first col):step = 4 (4 columns to copy and skip last column)]
Y = array_res      # list slice for class column
validation_size = 0.20 # 20% for the validation set
seed = 7
#print(len(X));
#print(X);
#print(len(Y));
#print(Y);
# returns tuple of values
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) # understand what thsis does mroe indepth and write about it. doing some fitting, look at leetcode intro to ML
# Split arrays or matrices into random train and test subsets. (numpy arr, numpy arr, testsize is the portion of the dataset to use for the test split, random_state is Pseudo-random number generator state used for random sampling.)
# X_train is for the intances used for training
# Y_train is for the expected outcome of each instance
# X_validation is the instances used for validating the model
# Y_validation is for the expected outcome of each corresponding instance in X_validation
#</prep data>

#<eval algorithms>
#5.2 Test harness
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# 5.3 build model
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed);  # sklearn. KFold is Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (>= 2 folds) (without shuffling by default). No shuffle to compare algorithms
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring); #Evaluate a score by cross validation (model is the algorithm to fit the data,x_train is the data to fit for the model,y_train is target variable to predict; result of the x_train instances,cv is the cross validation splitting strat,scoring is the accuracy of the test set)
	results.append(cv_results); #  store the scores (array) of each run of the cross validation in the result array
	names.append(name);         # Stores the name of the algorithm for the current result
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
#plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()

# notes: LDA comes out the highest
#</eval algorithms>

# 6  make predictions on dataset
print("Predicting on unseen data.");
lda = LinearDiscriminantAnalysis();
lda.fit(X_train, Y_train);  # build model

predictions = lda.predict(X_validation); #predict on unseen data
print(accuracy_score(Y_validation, predictions));   # compares the validation known answer with the predicted to determine accuracy
print(confusion_matrix(Y_validation, predictions)); # matrix of accuracy classification where C(0,0) is true negatives, C(1,0) is false negatives, C(1,1) true posivtes, C(0,1) false positvies.
# actual B and pred B = 74, actual B and pred M = 0.
# acutal M and pred M = 35, acutal M and pred B = 5.
print(classification_report(Y_validation, predictions)); #text report
#print("X_validation predict ===");
#print(X_validation);
#print("Y_validation predict ===")
#print(predictions); # array of predicted values

# show the incorrect indexes of the validation set
for row_index, (input, predictions, Y_validation) in enumerate(zip (X_validation, predictions, Y_validation)):
  if predictions != Y_validation:
    print('Row', row_index, 'has been classified as ', predictions, 'and should be ', Y_validation)
    print(X_validation[row_index]);

# =====

print("Validation set ===");
url = "dataset/validation_set.csv";   #source of the validation set data, subset of WDBC.csv
dataset_val = pandas.read_csv(url, header=None);

arrayV2 = dataset_val.values;
attrLenV2 = len(dataset_val.columns);
X_validationV2 = arrayV2[:,1:attrLenV2];
Y_validationV2 = dataset_val[0];

#print(X_validationV2);
print("--");
#print(Y_validationV2);

predictionsV2 = lda.predict(X_validationV2); #predict on unseen data
print(accuracy_score(Y_validationV2, predictionsV2));   # compares the validation known answer with the predicted to determine accuracy
print(confusion_matrix(Y_validationV2, predictionsV2)); # matrix of accuracy classification where C(0,0) is true negatives, C(1,0) is false negatives, C(1,1) true posivtes, C(0,1) false positvies
print(classification_report(Y_validationV2, predictionsV2)); #text report

# show the incorrect indexes of the validation set
for row_index, (input, predictionsV2, Y_validationV2) in enumerate(zip (X_validationV2, predictionsV2, Y_validationV2)):
  if predictionsV2 != Y_validationV2:
    print('Row', row_index, 'has been classified as ', predictionsV2, 'and should be ', Y_validationV2)
    print(X_validationV2[row_index]);





