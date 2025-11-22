import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


def build_xgb():
    """
    Build the XGBoost regression model with optimized hyperparameters.

    These settings were chosen for strong performance on chemical
    property prediction tasks, balancing model complexity and
    regularization:

    - n_estimators: number of boosting rounds
    - max_depth: maximum tree depth (controls model capacity)
    - learning_rate: shrinkage factor for each boosting step
    - subsample, colsample_bytree: prevent overfitting via sampling
    - gamma: minimum loss reduction for split
    - reg_alpha/lambda: L1 / L2 regularization
    - n_jobs: parallel CPU usage

    Returns
    -------
    XGBRegressor
        Configured XGBoost regression model.
    """
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
    """
    Perform K-Fold cross validation for model evaluation.

    K-Fold is used to measure the model's stability and generalization by
    training/evaluating on multiple dataset splits.

    Workflow per fold:
    1. Split into train/test based on fold index
    2. Standardize features (fit on train only)
    3. Train XGBoost
    4. Evaluate using R² score

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    Y : np.ndarray
        Target regression values.
    k : int, default=5
        Number of folds.

    Returns
    -------
    float
        Mean R² score across folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n[INFO] Fold {fold+1}/{k}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train model
        reg = build_xgb()
        reg.fit(X_train_s, y_train)

        # Evaluate
        y_pred = reg.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)

        print(f"  Fold R² = {r2:.4f}")

    print(f"\n[INFO] Mean K-Fold R² = {np.mean(scores):.4f}")
    return np.mean(scores)


def train_final_model(X, Y, model_path, scaler_path):
    """
    Train the final XGBoost model using the entire dataset and save it.

    Steps:
    1. Fit StandardScaler on full dataset
    2. Train XGBoost on scaled data
    3. Save scaler & trained model as .pkl files

    This model is later used for inference in the CLI or app.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    Y : np.ndarray
        Full target values.
    model_path : str or Path
        Path to save the trained model.
    scaler_path : str or Path
        Path to save the fitted StandardScaler.

    Returns
    -------
    XGBRegressor
        Fully trained XGBoost model.
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    reg = build_xgb()
    reg.fit(X_s, Y)

    # Save components
    joblib.dump(scaler, scaler_path)
    joblib.dump(reg, model_path)

    print("[INFO] Final model saved.")
    return reg
