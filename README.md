import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
labels = ['angular_leaf_spot','healthy']
img_size = 200
data = []
def get_training_data(data_dir):
  for label in labels:
    path=os.path.join (data_dir, label)
    class_num = labels.index(label)
    print(class_num)
    for img in os.listdir (path):
      try:
        img_arr = cv2.imread(os.path. join (path, img), cv2.COLOR_BAYER_GB2RGB)
        # print(img_arr.shape)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        data.append ([resized_arr, class_num])
      except Exception as e:
        print(e)
  return np.array(data)
train = get_training_data('/content/drive/MyDrive/Bean_Dataset')
print(data)

for label in labels:
  print(labels.index(label))

from google.colab import drive
drive.mount('/content/drive')

x=[]
y=[]
for i,j in data:
  x.append(i)
  y.append(j)
print(y)

y

x

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 47)

xtrain

import numpy as np

x=np.array(x).reshape(1320,120000)
x

print(np.array(xtrain).shape)

print(np.array(ytest).shape)

x1=np.array(x).shape

x1

x

y1=np.array(y).shape
y1

d=np.array(xtrain).reshape(990,120000)
d

e=np.array(xtest).reshape(330,120000)
e

print(np.asarray(d.shape))

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
d= sc_x.fit_transform(d)#normalizing
e = sc_x.transform(e)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)

model.fit(d, ytrain)

y_pred=model.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",cm)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(ytest,y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=12)#k value

knn.fit(d,ytrain)

y_pred = model.predict(e)
y_pred

print("predicted value for training value",knn.score(d,ytrain))
print("predicted value for testing value",knn.score(e,ytest))
print("Overall Accuracy:",knn.score(sc_x.transform(x),y))

y_pred=knn.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
knns=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",knns)

#this code is useful to find best k value using graphs
neighbors=np.arange(1,20)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))
overall_accuracy=np.empty(len(neighbors))
#loop over k values
for i,k in enumerate(neighbors):
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(d,ytrain)
  #compute the training and testing accuracy of ML model
  train_accuracy[i]=knn.score(d,ytrain)
  test_accuracy[i]=knn.score(e,ytest)

  #overall score
  overall_accuracy[i]=knn.score(sc_x.transform(x),y)

import matplotlib.pyplot as plt
plt.plot(neighbors,train_accuracy,label="training dataset accuracy") 
plt.plot(neighbors,test_accuracy,label="training dataset accuracy") 
plt.plot(neighbors,overall_accuracy,label="overall dataset accuracy")
plt.legend() 
plt.xlabel('k values-n_neigbors')
plt.ylabel('Accuracies')
plt.show()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(d,ytrain)

import numpy as np
print("Training Accuracy",nb.score(d,ytrain))
print("Testing Accuracy",nb.score(e,ytest))
print("Overall Accuracy:",nb.score(sc_x.transform(x),y))

y_pred=nb.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
nbb=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",nbb)

from sklearn import svm
SVM= svm.SVC()

SVM.fit(d, ytrain)

print("Training Accuracy",SVM.score(d,ytrain))
print("Testing Accuracy",SVM.score(e,ytest))
print("Overall Accuracy:",SVM.score(sc_x.transform(x),y))

y_pred=SVM.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
SVMS=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",SVMS)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()

dtc.fit(d,ytrain)

print("Training Accuracy",dtc.score(d,ytrain))
print("Testing Accuracy",dtc.score(e,ytest))
print("Overall Accuracy:",dtc.score(sc_x.transform(x),y))

y_pred=dtc.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
dtcs=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",dtcs)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

rfc.fit(d,ytrain)

print("Training Accuracy",rfc.score(d,ytrain))
print("Testing Accuracy",rfc.score(e,ytest))
print("Overall Accuracy:",rfc.score(sc_x.transform(x),y))

y_pred=rfc.predict(e)
y_pred

from sklearn.metrics import confusion_matrix
rfcs=confusion_matrix(ytest,y_pred)
print("Confusion matrix:\n",rfcs)

from sklearn.metrics import accuracy_score
accuracy_model = accuracy_score(y,model.predict(sc_x.transform(x)))
print("Logistic regression:",accuracy_model)
accuracy_nb = accuracy_score(y,nb.predict(sc_x.transform(x)))
print("navie bayes;",accuracy_nb)
accuracy_knn = accuracy_score(y,knn.predict(sc_x.transform(x)))
print("KNN:",accuracy_knn)
accuracy_SVM = accuracy_score(y,SVM.predict(sc_x.transform(x)))
print("Support vector machine:",accuracy_SVM)
accuracy_dtc = accuracy_score(y,dtc.predict(sc_x.transform(x)))
print("Descision tree:",accuracy_dtc)
accuracy_rfc = accuracy_score(y,rfc.predict(sc_x.transform(x)))
print("Random forest:",accuracy_rfc)

import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

algo = ['logistic reg','Naive bayes','knn clf','SVM','decision tree','random forest']

accuracy = [accuracy_model*100,accuracy_nb*100,accuracy_knn*100,accuracy_SVM*100,accuracy_dtc*100,accuracy_rfc*100]

ax.bar(algo[0],accuracy[0],color = 'b')

ax.bar(algo[1],accuracy[1],color = 'y')

ax.bar(algo[2],accuracy[2],color = 'pink')


ax.bar(algo[3],accuracy[3],color = 'green')

ax.bar(algo[4],accuracy[4],color = 'r')

ax.bar(algo[5],accuracy[5],color = 'orange')

plt.xlabel('Classifiers------------>')

plt.ylabel('Accuracies------------->')

plt.title('ACCURACIES RESULTED')

plt.show()





