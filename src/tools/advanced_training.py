"""
Advanced Model Training Tools
Tools for hyperparameter tuning, ensemble methods, and cross-validation.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
import joblib
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from utils.polars_helpers import load_dataframe, get_numeric_columns, split_features_target
from utils.validation import (
    validate_file_exists, validate_file_format, validate_dataframe,
    validate_column_exists, validate_target_column
)


def hyperparameter_tuning(
    file_path: str,
    target_col: str,
    model_type: str = "random_forest",
    task_type: str = "auto",
    n_trials: int = 50,
    cv_folds: int = 5,
    optimization_metric: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform Bayesian hyperparameter optimization using Optuna.
    
    Args:
        file_path: Path to prepared dataset
        target_col: Target column name
        model_type: Model to tune ('random_forest', 'xgboost', 'logistic', 'ridge')
        task_type: 'classification', 'regression', or 'auto' (detect from target)
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        optimization_metric: Metric to optimize ('auto', 'accuracy', 'f1', 'roc_auc', 'rmse', 'r2')
        test_size: Test set size for final evaluation
        random_state: Random seed
        output_path: Path to save best model
        
    Returns:
        Dictionary with tuning results, best parameters, and performance
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Prepare data
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == "classification" else None
    )
    
    # Detect task type
    if task_type == "auto":
        unique_values = len(np.unique(y))
        task_type = "classification" if unique_values < 20 else "regression"
    
    # Set default metric
    if optimization_metric == "auto":
        optimization_metric = "accuracy" if task_type == "classification" else "rmse"
    
    # Define objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters based on model type
        if model_type == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': random_state
            }
            if task_type == "classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)
                
        elif model_type == "xgboost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': random_state
            }
            if task_type == "classification":
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            else:
                model = XGBRegressor(**params)
                
        elif model_type == "logistic":
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': 'saga',
                'max_iter': 1000,
                'random_state': random_state
            }
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            model = LogisticRegression(**params)
            
        elif model_type == "ridge":
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                'random_state': random_state
            }
            model = Ridge(**params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Cross-validation
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Select scoring metric
        if optimization_metric == "accuracy":
            scoring = 'accuracy'
        elif optimization_metric == "f1":
            scoring = 'f1_weighted'
        elif optimization_metric == "roc_auc":
            scoring = 'roc_auc_ovr_weighted'
        elif optimization_metric == "rmse":
            scoring = 'neg_root_mean_squared_error'
        elif optimization_metric == "r2":
            scoring = 'r2'
        else:
            scoring = 'accuracy' if task_type == "classification" else 'neg_root_mean_squared_error'
        
        # Cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        # Return mean score (Optuna maximizes by default)
        return scores.mean()
    
    # Run optimization
    print(f"🔧 Starting hyperparameter tuning with {n_trials} trials...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"✅ Best {optimization_metric}: {best_score:.4f}")
    print(f"📊 Best parameters: {best_params}")
    
    # Train final model with best parameters
    if model_type == "random_forest":
        if task_type == "classification":
            final_model = RandomForestClassifier(**best_params)
        else:
            final_model = RandomForestRegressor(**best_params)
    elif model_type == "xgboost":
        if task_type == "classification":
            final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        else:
            final_model = XGBRegressor(**best_params)
    elif model_type == "logistic":
        final_model = LogisticRegression(**best_params)
    elif model_type == "ridge":
        final_model = Ridge(**best_params)
    
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    
    if task_type == "classification":
        test_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        if len(np.unique(y)) == 2:
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            test_metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
    else:
        test_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }
    
    # Save model if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(final_model, output_path)
        print(f"💾 Model saved to: {output_path}")
    
    return {
        'status': 'success',
        'model_type': model_type,
        'task_type': task_type,
        'n_trials': n_trials,
        'best_params': best_params,
        'best_cv_score': float(best_score),
        'optimization_metric': optimization_metric,
        'test_metrics': test_metrics,
        'trials_summary': {
            'total_trials': len(study.trials),
            'best_trial': study.best_trial.number,
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        },
        'model_path': output_path if output_path else None
    }


def train_ensemble_models(
    file_path: str,
    target_col: str,
    ensemble_type: str = "voting",
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train ensemble models using stacking, blending, or voting.
    
    Args:
        file_path: Path to prepared dataset
        target_col: Target column name
        ensemble_type: 'voting', 'stacking', or 'blending'
        task_type: 'classification', 'regression', or 'auto'
        test_size: Test set size
        random_state: Random seed
        output_path: Path to save ensemble model
        
    Returns:
        Dictionary with ensemble performance and comparison
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Prepare data
    X, y = split_features_target(df, target_col)
    
    # Detect task type
    if task_type == "auto":
        unique_values = len(np.unique(y))
        task_type = "classification" if unique_values < 20 else "regression"
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None
    )
    
    # Define base models
    if task_type == "classification":
        base_models = [
            ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
            ('xgb', XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss'))
        ]
        meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        base_models = [
            ('ridge', Ridge(random_state=random_state)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=random_state)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=random_state))
        ]
        meta_model = Ridge(random_state=random_state)
    
    # Train individual models for comparison
    individual_results = {}
    for name, model in base_models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == "classification":
            individual_results[name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            individual_results[name] = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'r2': float(r2_score(y_test, y_pred))
            }
    
    # Create ensemble
    print(f"🎯 Building {ensemble_type} ensemble...")
    
    if ensemble_type == "voting":
        if task_type == "classification":
            ensemble = VotingClassifier(estimators=base_models, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=base_models)
            
    elif ensemble_type == "stacking":
        if task_type == "classification":
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5
            )
        else:
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5
            )
    
    elif ensemble_type == "blending":
        # Split training data for blending
        X_base_train, X_blend_train, y_base_train, y_blend_train = train_test_split(
            X_train, y_train, test_size=0.3, random_state=random_state,
            stratify=y_train if task_type == "classification" else None
        )
        
        # Train base models on base training set
        base_predictions_train = []
        base_predictions_test = []
        
        for name, model in base_models:
            model.fit(X_base_train, y_base_train)
            base_predictions_train.append(model.predict(X_blend_train))
            base_predictions_test.append(model.predict(X_test))
        
        # Stack predictions
        X_blend = np.column_stack(base_predictions_train)
        X_test_blend = np.column_stack(base_predictions_test)
        
        # Train meta-model
        meta_model.fit(X_blend, y_blend_train)
        y_pred = meta_model.predict(X_test_blend)
        
        # Calculate metrics
        if task_type == "classification":
            ensemble_metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        else:
            ensemble_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
        
        # Save for blending
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump({
                'base_models': dict(base_models),
                'meta_model': meta_model,
                'ensemble_type': 'blending'
            }, output_path)
        
        return {
            'status': 'success',
            'ensemble_type': ensemble_type,
            'task_type': task_type,
            'ensemble_metrics': ensemble_metrics,
            'individual_models': individual_results,
            'improvement': f"+{(ensemble_metrics.get('accuracy', ensemble_metrics.get('r2', 0)) - max([m.get('accuracy', m.get('r2', 0)) for m in individual_results.values()])) * 100:.2f}%",
            'model_path': output_path if output_path else None
        }
    
    else:
        raise ValueError(f"Unsupported ensemble_type: {ensemble_type}")
    
    # Train ensemble (voting or stacking)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    
    # Calculate ensemble metrics
    if task_type == "classification":
        ensemble_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        best_individual_metric = max([m['accuracy'] for m in individual_results.values()])
        improvement = ensemble_metrics['accuracy'] - best_individual_metric
    else:
        ensemble_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred))
        }
        best_individual_metric = max([m['r2'] for m in individual_results.values()])
        improvement = ensemble_metrics['r2'] - best_individual_metric
    
    # Save model
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(ensemble, output_path)
        print(f"💾 Ensemble model saved to: {output_path}")
    
    return {
        'status': 'success',
        'ensemble_type': ensemble_type,
        'task_type': task_type,
        'ensemble_metrics': ensemble_metrics,
        'individual_models': individual_results,
        'improvement': f"+{improvement * 100:.2f}%",
        'model_path': output_path if output_path else None
    }


def perform_cross_validation(
    file_path: str,
    target_col: str,
    model_type: str = "random_forest",
    task_type: str = "auto",
    cv_strategy: str = "kfold",
    n_splits: int = 5,
    random_state: int = 42,
    save_oof: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive cross-validation with out-of-fold predictions.
    
    Args:
        file_path: Path to prepared dataset
        target_col: Target column name
        model_type: 'random_forest', 'xgboost', 'logistic', 'ridge'
        task_type: 'classification', 'regression', or 'auto'
        cv_strategy: 'kfold', 'stratified', or 'timeseries'
        n_splits: Number of CV folds
        random_state: Random seed
        save_oof: Whether to save out-of-fold predictions
        output_path: Path to save OOF predictions
        
    Returns:
        Dictionary with CV scores, statistics, and OOF predictions
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Prepare data
    X, y = split_features_target(df, target_col)
    
    # Detect task type
    if task_type == "auto":
        unique_values = len(np.unique(y))
        task_type = "classification" if unique_values < 20 else "regression"
    
    # Create model
    if model_type == "random_forest":
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "xgboost":
        if task_type == "classification":
            model = XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        else:
            model = XGBRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "ridge":
        model = Ridge(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Create CV splitter
    if cv_strategy == "stratified" and task_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "timeseries":
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print(f"🔄 Performing {n_splits}-fold cross-validation ({cv_strategy})...")
    
    # Perform cross-validation with detailed tracking
    fold_scores = []
    oof_predictions = np.zeros(len(y))
    oof_indices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y if cv_strategy == "stratified" else None)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_pred_fold = model.predict(X_val_fold)
        
        # Store OOF predictions
        oof_predictions[val_idx] = y_pred_fold
        oof_indices.extend(val_idx.tolist())
        
        # Calculate fold metrics
        if task_type == "classification":
            fold_score = {
                'fold': fold_idx + 1,
                'accuracy': float(accuracy_score(y_val_fold, y_pred_fold)),
                'f1': float(f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0)),
                'samples': len(val_idx)
            }
        else:
            fold_score = {
                'fold': fold_idx + 1,
                'rmse': float(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))),
                'r2': float(r2_score(y_val_fold, y_pred_fold)),
                'samples': len(val_idx)
            }
        
        fold_scores.append(fold_score)
        print(f"  Fold {fold_idx + 1}: {fold_score}")
    
    # Calculate overall OOF metrics
    if task_type == "classification":
        oof_metrics = {
            'accuracy': float(accuracy_score(y, oof_predictions)),
            'precision': float(precision_score(y, oof_predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, oof_predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, oof_predictions, average='weighted', zero_division=0))
        }
        mean_fold_metric = np.mean([f['accuracy'] for f in fold_scores])
        std_fold_metric = np.std([f['accuracy'] for f in fold_scores])
        metric_name = "accuracy"
    else:
        oof_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y, oof_predictions))),
            'mae': float(mean_absolute_error(y, oof_predictions)),
            'r2': float(r2_score(y, oof_predictions))
        }
        mean_fold_metric = np.mean([f['rmse'] for f in fold_scores])
        std_fold_metric = np.std([f['rmse'] for f in fold_scores])
        metric_name = "rmse"
    
    print(f"\n✅ Overall OOF {metric_name}: {oof_metrics.get(metric_name):.4f} (±{std_fold_metric:.4f})")
    
    # Save OOF predictions if requested
    if save_oof and output_path:
        oof_df = pl.DataFrame({
            'index': list(range(len(y))),
            'true_values': y,
            'oof_predictions': oof_predictions
        })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        oof_df.write_csv(output_path)
        print(f"💾 OOF predictions saved to: {output_path}")
    
    return {
        'status': 'success',
        'model_type': model_type,
        'task_type': task_type,
        'cv_strategy': cv_strategy,
        'n_splits': n_splits,
        'fold_scores': fold_scores,
        'oof_metrics': oof_metrics,
        'mean_cv_score': float(mean_fold_metric),
        'std_cv_score': float(std_fold_metric),
        'confidence_interval_95': f"[{mean_fold_metric - 1.96 * std_fold_metric:.4f}, {mean_fold_metric + 1.96 * std_fold_metric:.4f}]",
        'oof_path': output_path if save_oof and output_path else None
    }
