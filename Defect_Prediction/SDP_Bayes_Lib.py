import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


dataset1 = pd.read_csv("jEdit_4.0_4.2.csv")
dataset2 = pd.read_csv("jEdit_4.2_4.3.csv")
dataset_target = pd.read_csv("petstore_metrics.csv")
dataset_target.drop("Module", axis=1, inplace=True)

# print("--------------------- Dataset Target ----------------------")
# print(dataset_target.head())
# print("--------------------- Dataset jEdit_4.0_4.2 ----------------------")
# print(dataset1.head())
# print("--------------------- Dataset jEdit_4.2_4.3 ----------------------")
# print(dataset2.head())

#calculate number of class
n_defect_dataset_1 = dataset1['Class'][dataset1['Class'] == True].count()
n_non_defect_dataset1 = dataset1['Class'][dataset1['Class'] == False].count()
total_dataset1 = dataset1['Class'].count()
P_defect_dataset1 = n_defect_dataset_1 / total_dataset1
P_non_defect_dataset1 = n_non_defect_dataset1 / total_dataset1


print("--------------------- Dataset jEdit_4.0_4.2 ----------------------")
print("Number of Defect Class: ", n_defect_dataset_1)
print("Number of Non-Defective Class: ", n_non_defect_dataset1)
print("Total Of Data On Dataset 1: ", total_dataset1)
print("Probabillity Of Defective Class On Dataset 1: ", P_defect_dataset1)
print("Probability Of Non-Defective Class On Dataset 1 : ", P_non_defect_dataset1)
print("All Attribut Mean:\n",dataset1.mean())
print("All Attribut Mean Based On Module Class:\n",dataset1.groupby("Class").mean())
print("All Attribut Variance Based On Module Class:\n",dataset1.groupby("Class").var())
n_defect_dataset_2 = dataset2['Class'][dataset2['Class'] == True].count()
n_non_defect_dataset2 = dataset2['Class'][dataset2['Class'] == False].count()
total_dataset2 = dataset2['Class'].count()
P_defect_dataset2 = n_defect_dataset_2 / total_dataset2
P_non_defect_dataset2 = n_non_defect_dataset2 / total_dataset2

print("--------------------- Dataset jEdit_4.2_4.3 ----------------------")
print("Number of Defect Class: ", n_defect_dataset_2)
print("Number of Non-Defective Class: ", n_non_defect_dataset2)
print("Total Of Data On Dataset 1: ", total_dataset2)
print("Probabillity Of Defective Class On Dataset 2: ", P_defect_dataset2)
print("Probability Of Non-Defective Class On Dataset 2 : ", P_non_defect_dataset2)
print("All Attribut Mean:\n",dataset2.mean())
print("All Attribut Mean Based On Module Class:\n",dataset2.groupby("Class").mean())
print("All Attribut Variance Based On Module Class:\n",dataset2.groupby("Class").var())


#Predict The Dataset Using Scikit-Learn
feature_dataset1 = dataset1[['WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'NPM', 'LOC']]
target_dataset1 = dataset1['Class']
feature_dataset2 = dataset2[['WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'NPM', 'LOC']]
target_dataset2 = dataset2['Class']
feature_dataset_target = dataset_target[['WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'NPM', 'LOC']]
target_dataset_target = dataset_target['Class']
# feature_dataset2_train, feature_dataset2_test, target_dataset2_train, target_dataset2_test = train_test_split(
#     feature_dataset2, target_dataset2, test_size=0.15, random_state=15)
gaussian_bayes_classifier = GaussianNB()
print("------------------------- Dataset jEdit 4.0-4.2 ------------------------")
fitting1 = gaussian_bayes_classifier.fit(feature_dataset1, target_dataset1)
print(fitting1)
classifier_models = gaussian_bayes_classifier.predict(feature_dataset_target)
print(classifier_models)
accuracy_performance = accuracy_score(target_dataset_target, classifier_models)
print(accuracy_performance)
print("------------------------- Dataset jEdit 4.2-4.3 ------------------------")
fitting2 = gaussian_bayes_classifier.fit(feature_dataset2, target_dataset2)
print(fitting2)
classifier_models = gaussian_bayes_classifier.predict(feature_dataset_target)
print(classifier_models)
accuracy_performance = accuracy_score(target_dataset_target, classifier_models)
print(accuracy_performance)