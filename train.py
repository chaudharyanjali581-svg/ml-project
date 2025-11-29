import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def train_models(df):
    X = df.drop('booking_status', axis=1)
    y = df['booking_status'].apply(lambda x: 1 if x=='Canceled' else 0)

    X_res, y_res = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    rf = RandomForestClassifier()
    param = {'n_estimators':[100,200]}
    grid = GridSearchCV(rf, param, cv=3)
    grid.fit(X_train, y_train)

    joblib.dump(grid.best_estimator_, "best_model.pkl")
    return "best_model.pkl"
