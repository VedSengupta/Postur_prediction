import pandas as pd
import numpy as np
from sklearn import model_selection, neighbors,tree
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler

d=['a','b','c','d','e','f','g','h']
df= pd.read_csv("C:/Users/VED/Documents/datasets/ConfLongDemo_JSI.csv",names=d)
df.drop(['d'],axis=1,inplace=True)
#print(df.head())

#df['d']=pd.to_datetime(pd.Series(df['d']))
##df['d'] =  pd.to_datetime(df['d'], format='%d%m%Y:%H:%M:%S:%s')

for col in df.columns.values:
    df[col]=pd.to_numeric(df[col],errors='ignore')
    
df.fillna(0, inplace=True)
#print(df.head())

text_digit_vals = {}
def convert_to_int(val):
    return text_digit_vals[val]

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 1
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df 

df = handle_non_numerical_data(df)

#print(df.corr()["h"])    
#print(df.head())

#df = df.sample(frac=1).reset_index(drop=True)

X=df.iloc[:,:-1].values
X = MinMaxScaler().fit_transform(X)
y=df.iloc[:,-1].values
#X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.025,random_state=40)

accuracy_list=[]
for i in range(1, 13):  
    knn = neighbors.KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test,pred_i))

k=accuracy_list.index(max(accuracy_list))+1    
print("Best K Value:",k)
print("knn accuracy score:",max(accuracy_list))
knn = neighbors.KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification report:")
print(classification_report(y_test,y_pred))



clf_entropy=tree.DecisionTreeClassifier(criterion="entropy", random_state=100,min_samples_leaf=2)
clf_entropy.fit(X_train,y_train)
y_pred=clf_entropy.predict(X_test)
print("Decision Tree Score:",accuracy_score(y_test,y_pred))

conf=confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf)
class_rep=classification_report(y_test,y_pred)
print("Classification report:")
print(class_rep)



