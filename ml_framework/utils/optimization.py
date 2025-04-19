# utils/optimization.py
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from config import settings

# --- Define Hyperparameter Grids ---
# Structure: {'ModelName': {'param_name': [values] or distribution}}
HYPERPARAM_GRIDS = {
    # Regression
    'ElasticNet': {
        'model__alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    },
    'RandomForestRegressor': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 3, 5]
    },
    'XGBRegressor': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    'LGBMRegressor': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__num_leaves': [20, 31, 40, 50],
        'model__max_depth': [-1, 5, 10],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
         'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    'CatBoostRegressor': {
        'model__iterations': [50, 100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__depth': [4, 6, 8],
        'model__l2_leaf_reg': [1, 3, 5, 7] # L2 regularization
    },

    # Classification
    'LogisticRegression': {
        'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'model__penalty': ['l2'], # 'l1' needs solver='liblinear' or 'saga'
        'model__solver': ['lbfgs'] # Default, ok for l2
    },
    'SVC': {
        'model__C': [0.1, 1.0, 10.0, 100.0],
        'model__gamma': ['scale', 'auto', 0.01, 0.1],
        'model__kernel': ['rbf', 'linear'] # Can add 'poly', 'sigmoid'
    },
    'KNeighborsClassifier': {
        'model__n_neighbors': [3, 5, 7, 9, 11],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['minkowski', 'euclidean', 'manhattan']
    },
    'RandomForestClassifier': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 3, 5],
        'model__criterion': ['gini', 'entropy']
    },
     'XGBClassifier': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__gamma': [0, 0.1, 0.5] # Min loss reduction for split
    },
    'CatBoostClassifier': {
        'model__iterations': [50, 100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__depth': [4, 6, 8],
        'model__l2_leaf_reg': [1, 3, 5, 7]
    },
    # No tuning grids defined for Unsupervised models by default
}
# Note: Parameter names use 'model__' prefix assuming they are part of a scikit-learn Pipeline.


def tune_hyperparameters(pipeline: Pipeline, X_train, y_train, model_name, cv=settings.CROSS_VALIDATION_FOLDS, scoring='neg_root_mean_squared_error'):
    """Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV."""
    param_grid = HYPERPARAM_GRIDS.get(model_name)
    if not param_grid:
        logging.warning(f"No hyperparameter grid defined for {model_name} in optimization.py. Skipping tuning.")
        return pipeline, {} # Return original pipeline and empty params

    tuning_method = settings.TUNING_METHOD.lower()
    search = None

    logging.info(f"Starting {tuning_method} for {model_name} with scoring='{scoring}'...")

    try:
        if tuning_method == 'gridsearchcv':
            search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        elif tuning_method == 'randomizedsearchcv':
            search = RandomizedSearchCV(pipeline, param_grid, n_iter=settings.TUNING_ITERATIONS,
                                        cv=cv, scoring=scoring, n_jobs=-1, random_state=settings.RANDOM_STATE, verbose=1)
        else:
            logging.error(f"Unsupported tuning method: {settings.TUNING_METHOD}. Use GridSearchCV or RandomizedSearchCV.")
            return pipeline, {}

        search.fit(X_train, y_train)

        logging.info(f"Best score ({scoring}): {search.best_score_:.4f}")
        logging.info(f"Best params: {search.best_params_}")
        return search.best_estimator_, search.best_params_

    except Exception as e:
        logging.error(f"Hyperparameter tuning failed for {model_name}: {e}")
        logging.warning("Falling back to default parameters.")
        # Fit with default params if tuning fails
        pipeline.fit(X_train, y_train)
        # Attempt to get default params (might not reflect pipeline steps correctly)
        default_params = {}
        if hasattr(pipeline.named_steps['model'], 'get_params'):
            default_params = pipeline.named_steps['model'].get_params()
        return pipeline, default_params