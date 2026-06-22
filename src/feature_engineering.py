from sklearn.cluster import KMeans

def create_feature(X):
        X['Family'] = X['Parch'] + X['SibSp'] + 1
        X['IsAlone'] = (X['Family']==1).astype(int)
        X['HasCabin']= X['Cabin'].notna().astype(int)
        X['Pclass_Sex'] = X['Pclass'] * X['Sex']
        X['Age_Class'] = X['Age'] * X['Pclass']
        X['Fare_Family'] = X['Fare'] * X['Family']
        X['Family_Pclass'] = X['Family'] * X['Pclass']
        X['Age_Sex'] = X['Age'] * X['Sex']
        X['Fare_Sex'] = X['Fare'] * X['Sex']
        X['Fare_Class'] = X['Fare'] * X['Pclass']
        X['Age_Fare'] = X['Age'] * X['Fare']
    
        X.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'], inplace=True)

        return X

def feature_engineering(train_data, test_data):
    train_data = create_feature(train_data)
    test_data = create_feature(test_data)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    train_data['Cluster'] = kmeans.fit_predict(train_data)
    test_data['Cluster'] = kmeans.predict(test_data)

    return train_data, test_data, kmeans

def apply_feature_engineering(data, kmeans):
    data = create_feature(data)
    data['Cluster'] = kmeans.predict(data)
    return data