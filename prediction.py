from pandas import read_csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset

data = read_csv('engineered_data.csv')

#Split the data
array = data.values
X = array[:, 0:15]
y = array[:, 15]
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

#Scale the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)

#Make prediction on test data
model = RandomForestClassifier(random_state=13)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

#Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))



