import pandas as pd  
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare']
X_test = test_data[features].copy()
X = train_data[features].copy()
y = train_data['Survived']

X['Sex'] = X['Sex'].map({'male': 1, 'female' : 0})
X_test['Sex'] = X_test['Sex'].map({'male': 1, 'female' : 0})

my_imputer = SimpleImputer(strategy='constant', fill_value=0)
X = pd.DataFrame(my_imputer.fit_transform(X))
X_test = pd.DataFrame(my_imputer.transform(X_test))

X.columns = features
X_test.columns = features

machine = RandomForestClassifier(n_estimators=500, random_state=1)
machine.fit(X, y)
predictions = machine.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Submission file created!")