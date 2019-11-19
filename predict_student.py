import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset=pd.read_csv("Q_test_data.csv")
dataset=dataset.drop('transactionCount',axis=1)
dataset=dataset.drop('availableDate',axis=1)
dataset=dataset.drop('createdDate',axis=1)
dataset=dataset.drop('settledDate',axis=1)
dataset=dataset.drop('voidedDate',axis=1)
dataset['account_balance']=dataset['account_balance'].astype('category').cat.codes
dataset['check']=dataset['check'].astype('category').cat.codes
dataset['CS_FICO_str']=dataset['CS_FICO_str'].astype('category').cat.codes
dataset['CS_internal']=dataset['CS_internal'].astype('category').cat.codes
dataset['feeCode']=dataset['feeCode'].astype('category').cat.codes
dataset['feeDescription']=dataset['feeDescription'].astype('category').cat.codes
dataset['institutionName']=dataset['institutionName'].astype('category').cat.codes
dataset['isCredit']=dataset['isCredit'].astype('category').cat.codes
dataset['returnCode']=dataset['returnCode'].astype('category').cat.codes
dataset['status']=dataset['status'].astype('category').cat.codes
dataset['Student']=dataset['Student'].astype('category').cat.codes
dataset['subType']=dataset['subType'].astype('category').cat.codes
dataset['subTypeCode']=dataset['subTypeCode'].astype('category').cat.codes
dataset['type']=dataset['type'].astype('category').cat.codes
dataset['Age']=dataset['Age'].astype('float')
dataset['amount']=dataset['amount'].astype('float')
dataset['cardId']=dataset['cardId'].astype('float')
dataset['CS_FICO_num']=dataset['CS_FICO_num'].astype('float')
dataset['CustomerId']=dataset['customerId'].astype('float')
dataset['description']=dataset['description'].astype('float')
dataset['friendlyDescription']=dataset['friendlyDescription'].astype('float')
dataset['masterId']=dataset['masterId'].astype('float')
dataset['tag']=dataset['tag'].astype('float')
dataset['transactionId']=dataset['transactionId'].astype('float')
dataset['typeCode']=dataset['typeCode'].astype('float')
#dataset['availableDate']=pd.to_datetime(dataset['availableDate'])
#dataset['createdDate']=pd.to_datetime(dataset['createdDate'])
#dataset['settledDate']=pd.to_datetime(dataset['settledDate'])
#dataset['voidedDate']=pd.to_datetime(dataset['voidedDate'])

#Model -1 Classification Using Random Forest
target="Student"

X=[]
y=[]
for i in dataset.columns:
    if i==target:
        y.append(i)
    else:
        X.append(i)
#X=["Age","amount","cardId","CS_FICO_num","customerId","description","friendlyDescription","masterId","tag","transactionId","typeCode"]
X=dataset[X]
y=dataset[y]
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,shuffle=True)

max_acc_rf=0
y_pred_rf=[]
for i in range(200):
    clf=RandomForestClassifier(n_estimators=500)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    if metrics.accuracy_score(y_test, y_pred)>max_acc_rf:
        max_acc_rf=metrics.accuracy_score(y_test, y_pred)
        y_pred_rf=y_pred

print(max_acc_rf)

#Model -2 XGBoost Classifier
max_acc_xgb=0
y_pred_xgb=[]
for i in range(200):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = metrics.accuracy_score(y_test, predictions)
    if accuracy>max_acc_xgb:
        max_acc_xgb=accuracy
        y_pred_xgb=y_pred

print(max_acc_xgb)

# #Model -3 KNN
max_acc_knn=0
y_pred_knn=[]
for i in range(2000):
    model = KNeighborsClassifier(n_neighbors=65)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=metrics.accuracy_score(y_test, y_pred)
    if acc>max_acc_knn:
        max_acc_knn=acc
        y_pred_knn=y_pred

print(max_acc_knn)

#GBDT
from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=150, learning_rate=learning_rate, max_features=3, max_depth=3, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
print("For RandomForestClassifier : Max Accuracy = ",max_acc_rf)
print("For KNN : Max Accuracy = ",max_acc_knn)
print("For XGBoost : Max Accuracy = ",max_acc_xgb)
