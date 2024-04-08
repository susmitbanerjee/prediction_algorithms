import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import basic_my

df = pd.read_csv('bank-additional-full.csv', delimiter=';')

print('Columns are: ', df.columns.values)
print(df.dtypes)

categorical_columns = ['job', 'marital','education','default', 'housing','loan','contact',
                       'month','day_of_week','poutcome']
for cat in categorical_columns:
    print("Column '{cat}': {values}".format(cat=cat, values = df[cat].unique()))

#Converting the target label y to 1(yes) or 0(no) as a new column named 'class'
df['class'] = df['y'].apply(lambda x:1 if x== 'yes' else 0)

#Creating indicator variables to replace the categorical columns
df_new = pd.get_dummies(df, columns=categorical_columns)
print('New dataset columns are: ', df_new.columns.values)
print("#total records = ", df.shape[0])
print('# success samples = ', sum(df['class'] == 1))
print('#failed samples = ', sum(df['class'] == 0))

x_all = df_new.drop(['y', 'class'], axis=1)
y_all = df['class']

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)
print("#training records = ", x_train.shape[0])
print("# testing records = ", x_test.shape[0])

#Train the model

classifier = LogisticRegression(penalty='l2', C=0.001, class_weight="balanced")
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)

print('\n Performance metrics for Logistic Regression Classifier: ')
basic_my.print_scores(y_test, y_predict)

basic_my.save_model(classifier)


