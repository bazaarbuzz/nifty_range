# main.py
import argparse
import datetime
import logging
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Add encoders if needed
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from config import constants, settings
from utils import preprocessing, visualization, optimization # Import new optimization
from utils import evaluation # Import evaluation helpers
# Import model getters
from models.supervised.regression import get_regression_models
from models.supervised.classification import get_classification_models
from models.unsupervised.clustering import get_clustering_models
from models.unsupervised.dim_reduction import get_dim_reduction_models

logging.basicConfig(level=constants.LOG_LEVEL, format=constants.LOG_FORMAT)

# --- Helper to select features based on config/analysis ---
def get_feature_columns(df, feature_set_name=constants.DEFAULT_FEATURE_SET):
    # Define different feature sets
    all_cols = df.columns.tolist()
    base_mon = [c for c in all_cols if c.startswith(constants.MON_FIRST_HOUR_PREFIX)]
    base_exp = [c for c in all_cols if c.startswith(constants.EXP_PREFIX)]
    base_vix_mon = [c for c in all_cols if c.startswith(constants.VIX_PREFIX_MON)]
    base_vix_exp = [c for c in all_cols if c.startswith(constants.VIX_PREFIX_EXP)]
    engineered = [c for c in all_cols if c not in base_mon + base_exp + base_vix_mon + base_vix_exp + \
                  [constants.MON_DATE, constants.EXP_DATE, constants.WEEK_NUM, 'actual_start_date', 'actual_future_start_date', 'actual_future_end_date'] + \
                   settings.AVAILABLE_REGRESSION_TARGETS + settings.AVAILABLE_CLASSIFICATION_TARGETS] # Exclude identifiers & targets

    if feature_set_name == 'basic_ohlc_vix':
        # Combine base OHLC and VIX features (Mon & Exp)
        features = base_mon + base_vix_mon + base_vix_exp # Example: Only Mon VIX + Exp VIX
        # Ensure only existing columns are returned
        return [f for f in features if f in all_cols]
    elif feature_set_name == 'all_available':
         # Use all columns except identifiers and targets
         exclude_cols = [constants.MON_DATE, constants.EXP_DATE, constants.WEEK_NUM, 'actual_start_date', 'actual_future_start_date', 'actual_future_end_date'] + \
                         settings.AVAILABLE_REGRESSION_TARGETS + settings.AVAILABLE_CLASSIFICATION_TARGETS
         return [f for f in all_cols if f not in exclude_cols]
    else:
         logging.warning(f"Unknown feature_set_name: {feature_set_name}. Using default 'basic_ohlc_vix'.")
         return get_feature_columns(df, 'basic_ohlc_vix')


# --- Data Analysis Function (Minor updates if needed) ---
def perform_data_analysis(df, week_num):
    # ... (Keep existing logic, maybe add stationarity tests here) ...
    logging.info(f"--- Starting Data Analysis for Week {week_num} ---")
    if df is None or df.empty: return

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty: return

    # Stationarity example (on first target)
    first_target = settings.AVAILABLE_REGRESSION_TARGETS[0] if settings.AVAILABLE_REGRESSION_TARGETS else None
    if first_target and first_target in numeric_df.columns:
         preprocessing.perform_stationarity_test(numeric_df[first_target], first_target)
    # ... (rest of correlation, distribution plots) ...
    logging.info(f"--- Data Analysis Complete for Week {week_num} ---")


# --- Model Training and Evaluation (Refactored) ---
def train_evaluate_supervised(X_train, y_train, X_test, y_test, model_name, model_instance, model_type, target_variable, week_num, feature_names, skip_tuning=False):
    """Trains, tunes, evaluates a single supervised model, and saves it."""
    task_name = f"{target_variable}_Wk{week_num}_{model_name}"
    logging.info(f"--- Processing: {task_name} ({model_type}) ---")
    start_time = time.time()

    # --- Preprocessing Pipeline ---
    # Define numeric and categorical features (if any)
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
         # Use OneHotEncoder for nominal features, potentially others for ordinal
         transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    if not transformers:
         logging.error(f"[{task_name}] No numeric or categorical features found to build preprocessor.")
         return None # Cannot proceed without features

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # Keep other cols if needed? Or 'drop'?

    # --- Full Pipeline ---
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model_instance)])

    # --- Hyperparameter Tuning ---
    best_params = {}
    if not skip_tuning:
        scoring_metric = 'neg_root_mean_squared_error' if model_type == 'regression' else 'accuracy' # Choose appropriate default
        pipeline, best_params = optimization.tune_hyperparameters(
            pipeline, X_train, y_train, model_name, scoring=scoring_metric
        )
    else:
        logging.info(f"[{task_name}] Skipping hyperparameter tuning.")
        pipeline.fit(X_train, y_train)
        if hasattr(model_instance, 'get_params'): best_params = model_instance.get_params() # Get defaults


    # --- Evaluation ---
    logging.info(f"[{task_name}] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    metrics = {}
    cm_classes = None

    if model_type == 'regression':
        metrics = evaluation.calculate_regression_metrics(y_test, y_pred)
        # Plot regression results
        visualization.plot_predictions_vs_actual(
            y_test, y_pred,
            title=f"Predictions vs Actual - {task_name}",
            filename=f"{task_name}_pred_vs_actual.png"
        )
        # Optional: Plot residuals
        # residuals = y_test - y_pred
        # visualization.plot_distribution(residuals, title=f"Residuals Distribution - {task_name}", ...)
        # Optional: Plot time series
        if isinstance(y_test.index, pd.DatetimeIndex):
             visualization.plot_actual_vs_predicted_timeseries(
                  y_test, pd.Series(y_pred, index=y_test.index), # Ensure y_pred has index
                  title=f"Actual vs Predicted TS - {task_name}",
                  filename=f"{task_name}_actual_vs_pred_ts.png"
             )

    elif model_type == 'classification':
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
             try:
                 y_proba = pipeline.predict_proba(X_test)
             except Exception as e:
                 logging.warning(f"[{task_name}] Could not get predict_proba: {e}")
        metrics = evaluation.calculate_classification_metrics(y_test, y_pred, y_proba)
        # Plot classification results
        cm_classes = sorted(list(y_test.unique())) # Get class labels for confusion matrix
        visualization.plot_confusion_matrix(
            y_test, y_pred, classes=cm_classes,
            title=f"Confusion Matrix - {task_name}",
            filename=f"{task_name}_confusion_matrix.png"
        )
        # Add ROC curve plotting if needed

    logging.info(f"[{task_name}] Test Metrics: {metrics}")


    # --- Feature Importance ---
    feature_importances_dict = None
    try:
        # Get feature names after preprocessing (e.g., one-hot encoding)
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()

        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
        elif hasattr(pipeline.named_steps['model'], 'coef_'):
            # Use absolute coefficients for linear models (might need adjustment for multi-class)
            raw_coef = pipeline.named_steps['model'].coef_
            importances = np.abs(raw_coef[0] if len(raw_coef.shape) > 1 else raw_coef) # Handle potential multi-dim coef
        else:
             importances = None

        if importances is not None and len(importances) == len(feature_names_out):
             importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
             importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
             feature_importances_dict = importance_df.to_dict('records')
             visualization.plot_feature_importance(
                 importance_df,
                 title=f"Feature Importance - {task_name}",
                 filename=f"{task_name}_feature_importance.png"
             )
        elif importances is not None:
             logging.warning(f"[{task_name}] Feature importance length mismatch: {len(importances)} vs {len(feature_names_out)}. Skipping plot.")

    except Exception as fi_e:
        logging.warning(f"[{task_name}] Could not extract or plot feature importance: {fi_e}")


    # --- Save Model and Metadata ---
    # Determine save directory based on type
    save_dir = constants.REGRESSION_MODEL_DIR if model_type == 'regression' else constants.CLASSIFICATION_MODEL_DIR
    model_base_path = os.path.join(save_dir, task_name) # Uses target_WkNum_Model structure
    model_path = f"{model_base_path}.joblib"
    metadata_path = f"{model_base_path}_{constants.MODEL_METADATA_FILE}"

    try:
        joblib.dump(pipeline, model_path)
        logging.info(f"Saved trained pipeline to {model_path}")

        metadata = {
                constants.TRAINING_DATE: datetime.now().isoformat(),
                constants.MODEL_TYPE: model_type,
                'model_name': model_name,
                constants.TARGET_COL: target_variable,
                'week_num': week_num,
                constants.MODEL_PARAMS: pipeline.named_steps['model'].get_params() if hasattr(pipeline.named_steps['model'], 'get_params') else best_params,
                constants.HYPERPARAMS: best_params if best_params else "Default",
                constants.EVAL_METRICS: metrics,
                constants.FEATURE_COLS: feature_names,
                "feature_importances": feature_importances_dict,
                constants.INPUT_DATA_SOURCE: f"processed/week{week_num}_expiry_features.csv",
                # ----- ADD THIS FOR CLASSIFICATION -----
                "model_classes": pipeline.named_steps['model'].classes_.tolist() if model_type == 'classification' and hasattr(pipeline.named_steps['model'], 'classes_') else None
                # ----- END ADDITION -----
            }
        # Serialize metadata (handle numpy types)
        def default_serializer(o):
             if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(o)
             if isinstance(o, (np.float_, np.float16, np.float32, np.float64)): return float(o)
             if isinstance(o, (np.bool_)): return bool(o)
             if isinstance(o, (np.void)): return None
             if isinstance(o, np.ndarray): return o.tolist()
             return str(o) # Fallback for other types

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=default_serializer)
        logging.info(f"Saved model metadata to {metadata_path}")

    except Exception as e:
        logging.error(f"[{task_name}] Error saving model or metadata: {e}")

    end_time = time.time()
    logging.info(f"--- Completed: {task_name} in {end_time - start_time:.2f} seconds ---")
    return metrics # Return metrics for summary

# --- Unsupervised Task Function ---
def run_unsupervised_task(df, week_num, model_name, feature_names, model_args):
     """Runs clustering or dimensionality reduction."""
     task_name = f"Unsupervised_{model_name}_Wk{week_num}"
     logging.info(f"--- Processing: {task_name} ---")
     start_time = time.time()

     X = df[feature_names].copy()

     # Preprocessing (Scaling is crucial for distance-based methods like KMeans, DBSCAN, PCA)
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)

     results = {}
     pipeline = None # Store fitted object if needed

     if model_name in settings.AVAILABLE_CLUSTERING_MODELS:
         model_type = 'clustering'
         models_dict = get_clustering_models(**model_args) # Pass args like n_clusters
         if model_name not in models_dict:
             logging.error(f"Clustering model '{model_name}' not found.")
             return None
         model = models_dict[model_name]
         logging.info(f"Fitting {model_name}...")
         try:
             if hasattr(model, 'fit_predict'):
                 labels = model.fit_predict(X_scaled)
             else: # e.g., GMM uses fit().predict()
                 model.fit(X_scaled)
                 labels = model.predict(X_scaled)

             results['labels'] = labels.tolist()
             metrics = evaluation.calculate_clustering_metrics(X_scaled, labels)
             results['metrics'] = metrics
             logging.info(f"[{task_name}] Metrics: {metrics}")
             pipeline = model # Save the fitted clusterer

             # Visualization (requires dimensionality reduction first)
             if len(feature_names) > 3: # Need reduction for plotting
                  logging.info("Applying PCA for cluster visualization...")
                  pca = PCA(n_components=settings.CLUSTER_PLOT_DIMENSIONS, random_state=settings.RANDOM_STATE)
                  X_reduced = pca.fit_transform(X_scaled)
                  visualization.plot_clusters(
                      X_reduced, labels, method_name="PCA",
                      title=f"Clusters ({model_name}) - Week {week_num}",
                      filename=f"{task_name}_clusters_pca.png"
                  )
             elif len(feature_names) >= 2: # Can plot directly if 2 or 3 features
                  visualization.plot_clusters(
                      X_scaled, labels, method_name="Original Scaled",
                      title=f"Clusters ({model_name}) - Week {week_num}",
                      filename=f"{task_name}_clusters_original.png"
                  )

         except Exception as e:
             logging.error(f"[{task_name}] Clustering failed: {e}")
             return None


     elif model_name in settings.AVAILABLE_DIM_REDUCTION_MODELS:
         model_type = 'dim_reduction'
         models_dict = get_dim_reduction_models(**model_args) # Pass args like n_components
         if model_name not in models_dict:
              logging.error(f"Dim Reduction model '{model_name}' not found.")
              return None
         model = models_dict[model_name]
         logging.info(f"Fitting {model_name}...")
         try:
             X_reduced = model.fit_transform(X_scaled)
             results['transformed_data'] = X_reduced.tolist() # Save reduced data? Can be large
             # Add explained variance for PCA
             if isinstance(model, PCA):
                  results['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
                  logging.info(f"Explained variance ratio: {results['explained_variance_ratio']}")
             pipeline = model # Save the fitted transformer
             # Basic plot of first 2 components
             if X_reduced.shape[1] >= 2:
                  fig, ax = plt.subplots(figsize=settings.FIG_SIZE)
                  ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
                  ax.set_xlabel(f"{model_name} Component 1")
                  ax.set_ylabel(f"{model_name} Component 2")
                  ax.set_title(f"{model_name} Result - Week {week_num}")
                  visualization.save_plot(fig, f"{task_name}_reduction_2D.png")

         except Exception as e:
             logging.error(f"[{task_name}] Dimensionality reduction failed: {e}")
             return None

     else:
         logging.error(f"Unknown unsupervised model name: {model_name}")
         return None


     # --- Save Model/Results & Metadata ---
     save_dir = constants.CLUSTERING_MODEL_DIR if model_type == 'clustering' else constants.DIM_REDUCTION_MODEL_DIR
     model_base_path = os.path.join(save_dir, task_name)
     model_path = f"{model_base_path}.joblib"
     metadata_path = f"{model_base_path}_{constants.MODEL_METADATA_FILE}"

     try:
         if pipeline: # If model was fitted successfully
             joblib.dump(pipeline, model_path)
             logging.info(f"Saved fitted unsupervised object to {model_path}")

         metadata = {
             constants.TRAINING_DATE: datetime.now().isoformat(),
             constants.MODEL_TYPE: model_type,
             'model_name': model_name,
             'week_num': week_num,
             constants.MODEL_PARAMS: pipeline.get_params() if pipeline and hasattr(pipeline, 'get_params') else model_args,
             constants.EVAL_METRICS: results.get('metrics', {}),
             constants.FEATURE_COLS: feature_names,
             'results_summary': {k: v for k, v in results.items() if k not in ['labels', 'transformed_data']}, # Store metrics etc.
             constants.INPUT_DATA_SOURCE: f"processed/week{week_num}_expiry_features.csv",
         }
         # Save metadata
         def default_serializer(o): # Copy serializer
              if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(o)
              if isinstance(o, (np.float_, np.float16, np.float32, np.float64)): return float(o)
              if isinstance(o, (np.bool_)): return bool(o)
              if isinstance(o, (np.void)): return None
              if isinstance(o, np.ndarray): return o.tolist()
              return str(o)
         with open(metadata_path, 'w') as f:
             json.dump(metadata, f, indent=4, default=default_serializer)
         logging.info(f"Saved unsupervised task metadata to {metadata_path}")

         # Optionally save labels or reduced data to separate files if large
         # if 'labels' in results: pd.Series(results['labels']).to_csv(f"{model_base_path}_labels.csv")
         # if 'transformed_data' in results: pd.DataFrame(results['transformed_data']).to_csv(f"{model_base_path}_reduced_data.csv")

     except Exception as e:
         logging.error(f"[{task_name}] Error saving unsupervised results or metadata: {e}")

     end_time = time.time()
     logging.info(f"--- Completed: {task_name} in {end_time - start_time:.2f} seconds ---")
     return results.get('metrics', {})


# --- Main Execution Logic ---
def main(args):

    # --- 1. Preprocessing ---
    processed_files_available = True
    if args.preprocess:
        processed_files_dict = preprocessing.run_preprocessing()
        if not processed_files_dict:
            logging.error("Preprocessing failed. Exiting.")
            processed_files_available = False
            return # Exit if preprocessing fails and was requested

    # Check if processed files exist if training/analysis is requested without explicit preprocess flag
    if (args.task or args.analyze) and not processed_files_available:
         all_files_exist = True
         for week_num in args.expiry_week:
              fpath = os.path.join(constants.PROCESSED_DATA_DIR, f"week{week_num}_expiry_features.csv")
              if not os.path.exists(fpath):
                   logging.error(f"Processed file not found: {fpath}. Run with --preprocess first.")
                   all_files_exist = False
         if not all_files_exist: return # Exit if files needed are missing


    # --- Loop Through Weeks ---
    all_results_summary = {} # Store metrics across weeks/tasks

    for week_num in args.expiry_week:
        logging.info(f"========== Processing Week {week_num} ==========")
        week_results = {}
        processed_file = os.path.join(constants.PROCESSED_DATA_DIR, f"week{week_num}_expiry_features.csv")

        try:
            df_week = pd.read_csv(processed_file, index_col=constants.MON_DATE, parse_dates=True)
            if df_week.index.tz is None: df_week.index = df_week.index.tz_localize(constants.TIMEZONE)
            else: df_week.index = df_week.index.tz_convert(constants.TIMEZONE)
            df_week.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_week.dropna(inplace=True) # Ensure no NaNs before splitting/scaling
            if df_week.empty:
                logging.warning(f"DataFrame for Week {week_num} is empty after loading/dropna. Skipping.")
                continue
        except Exception as e:
            logging.error(f"Error loading data for Week {week_num}: {e}")
            continue

        # --- 2. Data Analysis ---
        if args.analyze:
            perform_data_analysis(df_week.copy(), week_num)

        # Determine features based on config or args
        feature_names = get_feature_columns(df_week, args.feature_set)
        if not feature_names:
             logging.error(f"No features identified for Week {week_num} with set '{args.feature_set}'. Skipping tasks.")
             continue
        logging.info(f"Using features for Week {week_num}: {feature_names}")
        X = df_week[feature_names]

        # --- 3. Supervised Task ---
        if args.task == 'supervised':
            # Determine targets: specified or all available for the model type
            targets_reg = args.target if args.target else settings.AVAILABLE_REGRESSION_TARGETS
            targets_cls = args.target if args.target else settings.AVAILABLE_CLASSIFICATION_TARGETS

            # Determine models: specified, all for type, or all available
            models_reg = args.model if args.model else settings.AVAILABLE_REGRESSION_MODELS
            models_cls = args.model if args.model else settings.AVAILABLE_CLASSIFICATION_MODELS
            if args.all_models: # Overrides specific models if requested
                models_reg = settings.AVAILABLE_REGRESSION_MODELS
                models_cls = settings.AVAILABLE_CLASSIFICATION_MODELS

            all_sup_models_dict = {**get_regression_models(), **get_classification_models()}


            # Loop through all applicable targets
            for target in targets_reg + targets_cls:
                if target not in df_week.columns:
                    logging.warning(f"Target '{target}' not found in Week {week_num} data. Skipping.")
                    continue

                y = df_week[target]
                model_type = 'regression' if target in settings.AVAILABLE_REGRESSION_TARGETS else 'classification'
                models_to_run = models_reg if model_type == 'regression' else models_cls

                target_results = {}
                logging.info(f"--- Processing Target: {target} (Week {week_num}) ---")

                if len(y.unique()) < 2:
                    logging.warning(f"Target '{target}' has only one unique value. Skipping training.")
                    continue

                # Split data
                try:
                     X_train, X_test, y_train, y_test = train_test_split(
                         X, y, test_size=settings.TEST_SPLIT_RATIO, random_state=settings.RANDOM_STATE#, stratify=y if model_type=='classification' else None # Stratify for classification
                     )
                except ValueError as e:
                     logging.error(f"Train/test split failed for {target} (check class balance?): {e}")
                     continue


                # Train/Evaluate selected models
                for model_name in models_to_run:
                    if model_name not in all_sup_models_dict:
                        logging.warning(f"Model '{model_name}' not found in model definitions. Skipping.")
                        continue
                    model_instance = all_sup_models_dict[model_name]

                    # --- Check if train/eval action is requested ---
                    if args.train or args.evaluate or args.all_actions:
                         metrics = train_evaluate_supervised(
                             X_train, y_train, X_test, y_test,
                             model_name, model_instance, model_type,
                             target, week_num, feature_names,
                             args.no_tune
                         )
                         if metrics: target_results[model_name] = metrics
                    else:
                         logging.info(f"Skipping train/evaluate for {model_name} (no --train or --evaluate flag).")


                if target_results: week_results[target] = target_results

        # --- 4. Unsupervised Task ---
        elif args.task == 'unsupervised':
            models_to_run = args.model if args.model else list(get_clustering_models().keys()) + list(get_dim_reduction_models().keys())
            if args.all_models:
                 models_to_run = list(get_clustering_models().keys()) + list(get_dim_reduction_models().keys())

            unsupervised_results = {}
            for model_name in models_to_run:
                 # Add model specific args (like n_clusters) if needed from argparse
                 model_args = {'n_clusters': args.n_clusters} if 'KMeans' in model_name or 'GaussianMixture' in model_name else {}
                 model_args.update({'n_components': args.n_components} if 'PCA' in model_name or 'UMAP' in model_name or 'TSNE' in model_name else {})

                 # --- Check if train/eval action is requested ---
                 if args.train or args.evaluate or args.all_actions:
                      metrics = run_unsupervised_task(df_week, week_num, model_name, feature_names, model_args)
                      if metrics: unsupervised_results[model_name] = metrics
                 else:
                      logging.info(f"Skipping unsupervised task for {model_name} (no --train or --evaluate flag).")

            if unsupervised_results: week_results['unsupervised'] = unsupervised_results

        # --- 5. Prediction Task (Delegated to predict.py) ---
        elif args.task == 'predict':
             logging.info("Prediction task should be run using predict.py. Use 'python predict.py --help' for options.")
             # Optionally, could add basic batch prediction here if needed.

        if week_results: all_results_summary[f"Week_{week_num}"] = week_results

    # --- Print Summary ---
    if all_results_summary:
        logging.info("========== Overall Results Summary ==========")
        try:
             print(json.dumps(all_results_summary, indent=4, default=str)) # Use default=str for basic numpy/date handling
             summary_path = os.path.join(settings.RESULTS_DIR, "training_summary.json")
             with open(summary_path, 'w') as f:
                  json.dump(all_results_summary, f, indent=4, default=str)
             logging.info(f"Overall summary saved to {summary_path}")
        except Exception as e:
             logging.error(f"Failed to print or save overall summary: {e}")

    logging.info("========== Main Script Finished ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Framework for Nifty/VIX Weekly Expiry Analysis")

    # --- Actions ---
    parser.add_argument('--preprocess', action='store_true', help='Run data loading and preprocessing.')
    parser.add_argument('--analyze', action='store_true', help='Run data analysis and generate visualizations.')
    # Task selection replaces separate train/evaluate flags for primary action
    parser.add_argument('--task', type=str, choices=['supervised', 'unsupervised', 'predict'], help='Specify the main task: supervised, unsupervised, or predict (use predict.py for prediction).')
    # Keep train/evaluate as modifiers for the chosen task
    parser.add_argument('--train', action='store_true', help='Train models for the specified task.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models for the specified task (often done implicitly with training).')
    parser.add_argument('--all_actions', action='store_true', help='Run preprocess, analyze, train, and evaluate.')

    # --- Model/Target Selection ---
    parser.add_argument('--model', type=str, nargs='+', help='Specify model(s) to run (e.g., RandomForestRegressor KMeans). Default depends on task.')
    parser.add_argument('--target', type=str, nargs='+', help='Specify target variable(s) for supervised task. Default uses all defined.')
    parser.add_argument('--all_models', action='store_true', help='Run all available models for the specified task.')

    # --- Data & Feature Options ---
    parser.add_argument('--expiry_week', type=int, nargs='+', default=[1, 2, 3, 4], help='Specify expiry week(s) to process (1-4). Default: all.')
    parser.add_argument('--feature_set', type=str, default=settings.DEFAULT_FEATURE_SET, help='Name of the feature set to use (defined in main.py).')

    # --- Tuning & Config ---
    parser.add_argument('--no_tune', action='store_true', help='Skip hyperparameter tuning.')
    # Add args for unsupervised params if needed
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters for KMeans/GMM.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of components for PCA/UMAP/TSNE.')

    # --- Prediction (Basic Placeholder, use predict.py) ---
    # parser.add_argument('--predict_data', type=str, help='Path to new data CSV for prediction (use predict.py instead).')


    args = parser.parse_args()

    # --- Argument Handling Logic ---
    # If --all_actions, set individual flags
    if args.all_actions:
        args.preprocess = True
        args.analyze = True
        args.train = True
        args.evaluate = True # Evaluation is tied to training here

    # Default action if only task is specified might be train+evaluate
    if args.task and not (args.train or args.evaluate):
        args.train = True
        args.evaluate = True
        logging.info(f"Task '{args.task}' specified without --train/--evaluate, defaulting to both.")

    # Basic validation
    if (args.train or args.evaluate) and not args.task:
         parser.error("--train or --evaluate requires --task (supervised/unsupervised) to be specified.")
    if args.task == 'predict':
         logging.warning("For predictions, please use the dedicated 'predict.py' script.")
         # Exit or just warn? Let's just warn and potentially do nothing else.
         # exit()

    if not (args.preprocess or args.analyze or args.task):
         parser.error("No action specified. Use --preprocess, --analyze, --task, or --all_actions.")

    main(args)