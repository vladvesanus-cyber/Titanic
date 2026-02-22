# Titanic Survival Prediction

A machine learning model to predict passenger survival on the Titanic using Random Forest Classifier.

## 🎯 Project Overview

This project uses historical data from the Titanic disaster to predict whether passengers survived based on features like passenger class, sex, age, and family size. The model achieved **76% accuracy** on the Kaggle test set.

## 📊 Dataset

- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- **Training data:** 891 passengers
- **Test data:** 418 passengers
- **Target variable:** Survived (0 = No, 1 = Yes)

## 🔧 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning (RandomForestClassifier, SimpleImputer)

## 🚀 Features

### Feature Engineering

**FamilySize:** Created a new feature combining siblings/spouses (SibSp) and parents/children (Parch)
```python
FamilySize = SibSp + Parch + 1
```

### Selected Features

The model uses 5 key features:
- `Pclass` - Passenger class (1st, 2nd, 3rd)
- `Sex` - Gender (encoded as 0=female, 1=male)
- `Age` - Passenger age
- `FamilySize` - Total family members aboard
- `Fare` - Ticket price

### Data Preprocessing

1. **Encoding categorical variables:** 
   - Sex: male=1, female=0

2. **Handling missing values:**
   - Imputation strategy: constant fill with 0
   - Applied to Age and Fare columns

## 🤖 Model

**Algorithm:** Random Forest Classifier
- **n_estimators:** 500 trees
- **random_state:** 1 (for reproducibility)

### Why Random Forest?

- Handles non-linear relationships well
- Robust to outliers
- Provides feature importance
- Good performance on Titanic dataset

## 📈 Results

- **Accuracy:** 76% on Kaggle leaderboard
- **Rank:** Top 50% of submissions

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── titanic_prediction.py     # Main script
├── my_submission.csv         # Output predictions
└── README.md                 # This file
```

## 🔄 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/vladvesanus-cyber/Titanic.git
cd titanic-survival-prediction
```

2. **Install dependencies**
```bash
pip install pandas scikit-learn
```

3. **Run the script**
```bash
python titanic_prediction.py
```

4. **Output**
- Creates `my_submission.csv` with predictions
- Format: PassengerId, Survived

## 💡 Key Insights

From the data analysis, we found:

1. **Gender matters most:** Women had significantly higher survival rates (~74%) compared to men (~19%)
2. **Class privilege:** 1st class passengers had 3x higher survival than 3rd class
3. **Age factor:** Children (under 10) had higher survival rates
4. **Family size:** Passengers with small families (2-4 members) survived more than solo travelers or very large families

## 🔮 Future Improvements

Potential enhancements to improve accuracy:

- [ ] Advanced feature engineering (Title extraction from Name, Cabin deck)
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Ensemble methods (combining multiple models)
- [ ] Better handling of missing Age values (predictive imputation)
- [ ] Feature interactions (e.g., Pclass * Sex)

## 📚 What I Learned

- Feature engineering significantly impacts model performance
- Importance of proper data preprocessing
- Random Forest as a strong baseline model
- Kaggle competition workflow (train, predict, submit)

## 🤝 Contributing

This is a learning project, but feedback is welcome! Feel free to:
- Open issues for suggestions
- Submit pull requests with improvements
- Share your approach to the problem

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Kaggle Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- **My Kaggle Profile:** [Vladyslav90](https://www.kaggle.com/vladyslav90)
- **GitHub Repository:** [This repo](https://github.com/vladvesanus-cyber/Titanic)

## 👤 Author

**Vlad**
- GitHub: [@vladvesanus-cyber](https://github.com/vladvesanus-cyber)
- Email: chetvlad9@gmail.com

---

⭐ If you found this project helpful, please give it a star!

**Last Updated:** January 2026
