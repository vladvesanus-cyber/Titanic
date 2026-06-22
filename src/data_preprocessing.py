from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def basic_preprocessing(train_data, test_data):
    categorical = [col for col in train_data.columns if train_data[col].dtype == 'object' and col not in ['PassengerId', 'Name', 'Ticket', 'Cabin']]
    numerical = [col for col in train_data.columns if train_data[col].dtype != 'object' and col not in ['PassengerId', 'Name', 'Ticket', 'Cabin']]

    imp_num = SimpleImputer(strategy="median")
    imp_cat = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder()

    train_data[numerical] = imp_num.fit_transform(train_data[numerical])
    train_data[categorical] = imp_cat.fit_transform(train_data[categorical])
    train_data[categorical] = encoder.fit_transform(train_data[categorical])

    test_data[numerical] = imp_num.transform(test_data[numerical])
    test_data[categorical] = imp_cat.transform(test_data[categorical])
    test_data[categorical] = encoder.transform(test_data[categorical])

    preprocessor = {
        "imp_num": imp_num,
        "imp_cat": imp_cat,
        "encoder": encoder,
        "numerical": numerical,
        "categorical": categorical
    }

    return train_data, test_data, preprocessor

def apply_preprocessing(data, preprocessor):
    numerical = preprocessor["numerical"]
    categorical = preprocessor["categorical"]

    data[numerical] = preprocessor["imp_num"].transform(data[numerical])
    data[categorical]= preprocessor["imp_cat"].transform(data[categorical])
    data[categorical] = preprocessor["encoder"].transform(data[categorical])

    return data