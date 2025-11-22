import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


def build_xgb():
    return XGBRegressor(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.015,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1.0,
        reg_alpha=1.0,
        reg_lambda=1.0,
        tree_method="auto",
        n_jobs=-1,
    )


def train_kfold(X, Y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n[INFO] Fold {fold+1}/{k}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        reg = build_xgb()
        reg.fit(X_train_s, y_train)

        y_pred = reg.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)

        print(f"  Fold R² = {r2:.4f}")

    print(f"\n[INFO] Mean K-Fold R² = {np.mean(scores):.4f}")
    return np.mean(scores)


def train_final_model(X, Y, model_path, scaler_path):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    reg = build_xgb()
    reg.fit(X_s, Y)

    joblib.dump(scaler, scaler_path)
    joblib.dump(reg, model_path)

    print("[INFO] Final model saved.")
    return reg
