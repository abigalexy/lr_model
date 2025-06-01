import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report

#数据处理
data = pd.read_csv('churn.csv')
print(data.info())
print(data.head())
data = pd.get_dummies(data)
print(data.head())
data = data.drop(['Churn_No','gender_Male'],axis=1)
print(data.head())
data = data.rename(columns = {'Churn_Yes':'flag'})
print(data.head())
print(data.flag.value_counts())

#特征工程
sns.countplot(data=data,y='Contract_Month',hue='flag')
plt.show()

x=data[['PaymentElectronic','Contract_Month','internet_other']]
y = data['flag']
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=22)

#模型训练
LR = LogisticRegression()
LR.fit(x_train,y_train)

#模型评估
y_predict = LR.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(roc_auc_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
