import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix,classification_report,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


#Loading dataset
dataset = pd.read_csv("diabetes.csv")


#filtering dataset to remove outlier's and incomplete values
no_zero_values = ['Glucose', 'BloodPressure','SkinThickness','Insulin','BMI']

# replace 0 values with 'none'

for column in no_zero_values:
    dataset[column] = dataset[column].replace(0,np.NaN)
    #finds mean of the dataset of values that dont include 'none'
    mean = int(dataset[column].mean(skipna=True))
    # replace all of the 'none ' values with the means
    dataset[column] = dataset[column].replace(np.NaN,mean)


#Setting features and labels
y = dataset.Outcome

x = dataset.drop('Outcome',axis=1)


#Splitting data into train and testk

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=4)



rfModel = RandomForestClassifier(
    n_estimators = 1000,
    min_samples_split = 9,
    max_features = 'auto',
    max_depth = 80,
    bootstrap = True,
    min_samples_leaf = 1,
    random_state = 1
)

#Fitting random forest
rfModel.fit(x_train,y_train)

sc = StandardScaler()

scaledXtrain = sc.fit_transform(x_train)
scaledXtest = sc.transform(x_test)

# finding optimal k
k_range = range(1,30)
k_scores =[]

# mean of cross validation scores at k = 1 to k = 30
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, x,y,cv = 5, scoring = 'accuracy')
    k_scores.append(scores.mean())



# optimal k value
max_score = max(k_scores) 
#getting index of optimal k (+1 as index starts from 0)
optimal_k = k_scores.index(max_score) + 1
print(" optimal at k = ", optimal_k)


# plotting graph of k vs accuracy

plt.plot(k_range,k_scores)
plt.xlabel("K")
plt.ylabel("accuracy score")
plt.show()


#knn with optimal k value 
knn = KNeighborsClassifier(n_neighbors= optimal_k)
knn.fit(x_train,y_train)


lR_classifier = LogisticRegression(max_iter=500,random_state=42)
lR_classifier.fit(x_train,y_train)

# ypredictions

rf_Test_predictions = rfModel.predict(x_test)
rf_Train_predictions = rfModel.predict(x_train)

knn_Train_predictions = knn.predict(x_train)
knn_Test_predictions = knn.predict(x_test)


lR_Test_predictions = lR_classifier.predict(x_test)
lR_Train_predictions = lR_classifier.predict(x_train)

knn_c_matrix = pd.crosstab(y_test, knn_Test_predictions, rownames=['True'], colnames=['Predicted'], margins=True)
rf_c_matrix = pd.crosstab(y_test, rf_Test_predictions, rownames=['True'], colnames=['Predicted'], margins=True)
lr_c_matrix = pd.crosstab(y_test, lR_Test_predictions, rownames=['True'], colnames=['Predicted'], margins=True)




print("Confusion matrices")
print(knn_c_matrix)
print(rf_c_matrix)
print(lr_c_matrix)


print("-----Classification Report-----")
print(classification_report(y_test,knn_Test_predictions))
print(classification_report(y_test,rf_Test_predictions))
print(classification_report(y_test,lR_Test_predictions))


print("-----Accuracy Scores-----")

print("Train Accuracy for KNN: " + str(accuracy_score(y_train, knn_Train_predictions)*100) + "%")
print("Train Accuracy for RF: " + str(accuracy_score(y_train, rf_Train_predictions)*100) + "%")
print("Train Accuracy for LR: " + str(accuracy_score(y_train, lR_Train_predictions)*100) + "%")

print("Test Accuracy for KNN: " + str(accuracy_score(y_test, knn_Test_predictions)*100) + "%")
print("Test Accuracy for RF: " + str(accuracy_score(y_test, rf_Test_predictions)*100) + "%")
print("Test Accuracy for LR: " + str(accuracy_score(y_test, lR_Test_predictions)*100) + "%")


#Recall and Precision scores for the classifiers
print("-----Test Recall and Precision scores-----")

print("Recall score for KNN: " + str(recall_score(y_test,knn_Test_predictions)) + "%")
print("Recall score for RF: " + str(recall_score(y_test,rf_Test_predictions)) + "%")
print("Recall score for LR: " + str(recall_score(y_test,lR_Test_predictions)) + "%")

print("Precision score for KNN: " + str(precision_score(y_test,knn_Test_predictions)) + "%")
print("Precision score for RF: " + str(precision_score(y_test,rf_Test_predictions)) + "%")
print("Precision score for LR: " + str(precision_score(y_test,lR_Test_predictions)) + "%")
