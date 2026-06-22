from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 

def train_model(X, y):

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    best_model = None
    best_score = 0
    for name, model in models.items():
        score = cross_val_score(model, X, y, cv=5).mean()
        if score > best_score:
            best_score = score
            best_model = model
            name_of_best_model = name
    best_model.fit(X, y)
    print(f"Best model: {name_of_best_model} with accuracy: {best_score:.4f}")
    return best_model