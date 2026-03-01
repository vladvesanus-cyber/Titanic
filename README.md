# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using multiple classification algorithms with automated model selection.

## 📁 Project Structure

```
Titanic/
├── Data/
│   ├── Raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── Processed/
│       └── submission.csv
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── utils.py
├── main.py
└── README.md
```

## ⚙️ How It Works

The pipeline consists of three main stages:

**1. Data Preprocessing** (`src/data_preprocessing.py`)
- Fills missing numerical values using **median imputation**
- Fills missing categorical values using **most frequent** strategy
- Encodes categorical variables with **OrdinalEncoder**

**2. Feature Engineering** (`src/feature_engineering.py`)
- Creates `Family` size feature (`Parch + SibSp + 1`)
- Derives `IsAlone`, `HasCabin` binary flags
- Adds interaction features: `Pclass_Sex`, `Age_Class`, `Fare_Family`, `Age_Fare`, etc.
- Generates passenger **clusters** using KMeans (5 clusters)
- Drops irrelevant columns: `PassengerId`, `Name`, `Ticket`, `Cabin`

**3. Model Training & Selection** (`src/model.py`)
- Trains and evaluates 7 classification models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Naive Bayes
- Automatically selects the **best model** based on cross validation

## 🚀 Getting Started

### Requirements

```bash
pip install scikit-learn pandas numpy
```

### Run

```bash
python main.py
```

The script will:
1. Load `Data/Raw/train.csv`
2. Preprocess and engineer features
3. Train all models and print the best one with its accuracy

## 📊 Results

| Metric | Score |
|--------|-------|
| Local Validation Accuracy | ~84% |
| Kaggle Public Leaderboard | ~77% |

> The gap between local accuracy (~84%) and Kaggle submission (~77%) is a typical sign of **overfitting** to the training data. The model generalizes slightly less well to unseen test data.

## 💡 Possible Improvements

- Add cross-validation (e.g. `StratifiedKFold`) for more reliable evaluation
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Address overfitting with regularization or ensemble calibration
- Save the best model using `joblib` or `pickle` for reuse
- Add prediction export to `Data/Processed/submission.csv`

## 📄 Dataset

The dataset is from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

---

*Author: vladvesanus-cyber*
