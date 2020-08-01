from __future__ import print_function
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn import svm

from pyfiglet import Figlet
from sklearn.model_selection import train_test_split

f = Figlet(font='slant')
print (f.renderText('Diabetes Disease Prediction'))

df=pd.read_csv('diabetes.csv')
#print(df.head())
print(df)

X = df.drop(columns=['class'])
Y = df['class']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print("x_train :",x_train)
print("y_train :",y_train)
print("x_test :",x_test)
print("y_test :",y_test)
clf = svm.SVC(kernel='linear')  # Linear Kernel
#print(x_train.value_counts())
# Train the model using the training sets
clf.fit(x_train, y_train)
from sklearn import metrics
# Predict the response for test dataset
y_pred = clf.predict(x_test)
print("supprot vector classcification model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
svm_acc=metrics.accuracy_score(y_test, y_pred) * 100
#m=svm_acc
from sklearn.metrics import confusion_matrix
print("confusion matrix ", confusion_matrix(y_test, y_pred))
m=svm_acc
#knn model accuracy
dft=pd.DataFrame()
dft=x_test
dft['actual']=y_test
dft["predictions"]=y_pred
print(dft["predictions"].value_counts())
dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                    title='SVM prediction distrubution in test data',
                                                    color='blue')
plt.show()
dft.to_csv(r'svm_valid_predections.csv')


from sklearn.neighbors import KNeighborsClassifier

x_train, x_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.2, random_state=1)
knn = KNeighborsClassifier()
knn_model = knn.fit(x_train, y_train)
y_true, y_pred1 = y_test, knn_model.predict(x_test1)
print('k-NN accuracy for test set: %f' % (knn_model.score(x_test1, y_test1)*100))
knn_acc=knn_model.score(x_test1, y_test1)*100
o=knn_acc    

dft=pd.DataFrame()
dft=x_test1
dft['actual']=y_test1
dft["predictions"]=y_pred1
print(dft["predictions"].value_counts())
dft['predictions'].value_counts().sort_index().plot(kind='bar', figsize=(20, 15),
                                                    title='KNN  prediction distrubution in test data',
                                                    color='orange')
plt.show()
dft.to_csv(r'Knn_valid_predections.csv')

#def plotm():
#acc_sc=model_building(ndf)
y=[o,m]
x=['knn','svm']
import matplotlib.pyplot as plt
plt.bar(x,y)
plt.title('comparison of accuracy_score of SVM,k-NN classifiers')
plt.savefig('comparison_graph.png')
plt.show()
print("knn acciracy:",o,"svm acc:",m)

#take inputs as features 
preg=input("pregnancies:")
glucose=input("glucose:")
bp=input("bp:")
skinthickness=input("skinthickness:")
insulin=input("insulin:")
bmi=input("bmi:")
diabetes_pedigree_funct=input("diab_predigree_function:")
age=input("age:")

userd=[preg,glucose,bp,skinthickness,insulin,bmi,diabetes_pedigree_funct,age]
userd=np.asarray(userd).astype(np.float64)
userd=userd.reshape(1,-1)
print("user input:",userd)
acc=clf.predict(userd)
print(acc[0])
if(acc[0]==1):
	print("patient of given user inputs is *****Diabetic*****")
	print("****Test Positive****")
else:
	print("Patient not Diabetic")
	print("***Test Negative***")
