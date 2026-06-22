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
│   └── model.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

**1. Data Preprocessing** (`src/data_preprocessing.py`)
- Fills missing numerical values using **median imputation**
- Fills missing categorical values using **most frequent** strategy
- Encodes categorical variables with **OrdinalEncoder**
- `fit` only on train data — no data leakage

**2. Feature Engineering** (`src/feature_engineering.py`)
- Creates `Family` size feature (`Parch + SibSp + 1`)
- Derives `IsAlone`, `HasCabin` binary flags
- Adds interaction features: `Pclass_Sex`, `Age_Class`, `Fare_Family`, `Age_Fare` etc.
- Generates passenger **clusters** using KMeans (5 clusters) — fitted only on train

**3. Model Training & Selection** (`src/model.py`)
- Evaluates 7 classification models using **StratifiedKFold cross-validation (cv=5)**:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine
  - Decision Tree
  - Naive Bayes
- Automatically selects the **best model** and fits it on full training data

## 🚀 Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The script will:
1. Load and preprocess `Data/Raw/train.csv` and `Data/Raw/test.csv`
2. Engineer features
3. Train and evaluate all models via cross-validation
4. Print the best model with its accuracy
5. Save predictions to `Data/Processed/submission.csv`

## 📊 Results

| Metric | Score |
|--------|-------|
| Local CV Accuracy | ~83% |
| Kaggle Public Leaderboard | ~76% |

> The gap between local CV (~83%) and Kaggle (~76%) is typical for this dataset due to its small size (891 rows). The model generalizes slightly less well to unseen data.

## 💡 Possible Improvements

- Extract `Title` from passenger name (`Mr`, `Mrs`, `Miss`, `Master`) as a feature
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Try ensemble methods (stacking, voting classifier)
- Save the best model using `joblib` for reuse

## 📄 Dataset

Dataset from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).

---
*Author: vladvesanus-cyber*