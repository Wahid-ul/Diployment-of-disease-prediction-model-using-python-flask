from cgi import test
from copyreg import pickle
from pickle import dump,load
from typing import Mapping
from unicodedata import category
import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode 

data=pd.read_csv('Prototype.csv')
#converting object datatype into numerical value
enco =LabelEncoder()
data["prognosis"]= enco.fit_transform(data['prognosis'])
#enc=data["prognosis"].to_dict()
X= data.iloc[:, :-1]
y=data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=24)
 
# Initializing Models
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
 
# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X, y)
#function for classification based on input
def classify(a,b,c,d):
    arr=np.array([a,b,c,d])#convert to numpy array
    arr=arr.astype(np.float64) #change the data type to float
    query=arr.reshape(1,-1) #reshape the array
    prediction=svm_model.predict(query)[0] #retrieving from dictonary
    return prediction

#saving model to disk
#dump(svm_model,open('model.pkl','wb'))
#model=load(open('model.pkl','rb'))
#print(model.predict([[2,9,6,1,4,5,2,3,3,3,3,3,4,5,6,7,8,9,3,5,7,9,8,2,3,4,5,6,7,8,2,4,7,8,9,3,4,8,9,0,8,9,0,3,4,5,6,7,8,9,2,9,6,1,4,5,2,3,3,3,3,3,4,5,6,7,8,9,3,5,7,9,8,2,3,4,5,6,7,8,2,4,7,8,9,3,4,8,9,0,8,9,0,3,4,5,6,7,8,9,2,9,6,1,4,5,2,3,3,3,3,3,4,5,6,7,8,9,3,5,7,9,8,2,3,4,5,6,7,8,2,4]]))
#print(model.predict([[1,3,4,5]]))
#print(PredictDisease("itching,diarrhoea,constipation"))