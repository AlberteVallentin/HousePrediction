# Model Export and Loading Guide

This guide shows how to export trained models from the notebook and load them in other notebooks for reuse.

## 1. Model Export Code

Add these code cells to your notebook to export trained models:

### Create Models Directory

```python
import os
import joblib
import json
from pathlib import Path

# Create models directory
models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)

print(f"Models directory created: {models_dir}")
```

### Export Individual Models

```python
def save_model_with_metadata(model, model_name, result_dict, models_dir='../models'):
    """
    Save a model with its performance metadata
    """
    models_path = Path(models_dir)

    # Save the model
    model_file = models_path / f"{model_name}.joblib"
    joblib.dump(model, model_file)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'cv_rmse_mean': result_dict['cv_rmse_mean'],
        'cv_rmse_std': result_dict['cv_rmse_std'],
        'val_rmse': result_dict.get('val_rmse', None),
        'best_params': result_dict.get('best_params', None),
        'model_type': type(model).__name__
    }

    metadata_file = models_path / f"{model_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {model_name}: {model_file}")
    print(f"  CV RMSE: {result_dict['cv_rmse_mean']:.4f}")

# Export all optimized individual models
individual_models = {
    'Ridge_Optimized': (best_ridge, ridge_result),
    'Lasso_Optimized': (best_lasso, lasso_result),
    'ElasticNet_Optimized': (best_elasticnet, elasticnet_result),
    'RandomForest_Optimized': (best_rf, rf_result),
    'GradientBoosting_Optimized': (best_gb, gb_result),
    'XGBoost_Optimized': (best_xgb, xgb_result),
    'CatBoost_Optimized': (best_catboost, catboost_result),
    'LightGBM_Optimized': (best_lightgbm, lightgbm_result)
}

print("Exporting individual models...")
for model_name, (model, result) in individual_models.items():
    save_model_with_metadata(model, model_name, result)
```

### Export Ensemble Models

```python
def save_ensemble_model(ensemble, ensemble_name, result_dict, component_models=None, models_dir='../models'):
    """
    Save an ensemble model with its components and metadata
    """
    models_path = Path(models_dir)

    # Save the ensemble model
    ensemble_file = models_path / f"{ensemble_name}.joblib"
    joblib.dump(ensemble, ensemble_file)

    # Save metadata
    metadata = {
        'ensemble_name': ensemble_name,
        'cv_rmse_mean': result_dict['cv_rmse_mean'],
        'cv_rmse_std': result_dict['cv_rmse_std'],
        'val_rmse': result_dict.get('val_rmse', None),
        'ensemble_type': type(ensemble).__name__,
        'component_models': component_models or []
    }

    metadata_file = models_path / f"{ensemble_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {ensemble_name}: {ensemble_file}")
    print(f"  CV RMSE: {result_dict['cv_rmse_mean']:.4f}")

# Export ensemble models
ensemble_models = {
    'Simple_Ensemble': (simple_ensemble, simple_result, ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso']),
    'Weighted_Ensemble': (weighted_ensemble, weighted_result, ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso']),
    'Stacking_Ensemble': (stacking_ensemble, stacking_result, ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'])
}

print("\nExporting ensemble models...")
for ensemble_name, (ensemble, result, components) in ensemble_models.items():
    save_ensemble_model(ensemble, ensemble_name, result, components)
```

### Export Final Model

```python
# Export the final selected model
final_model_metadata = {
    'final_model_name': final_model_name,
    'final_cv_rmse': final_model_rmse,
    'selection_criteria': 'Best cross-validation RMSE',
    'exported_at': str(pd.Timestamp.now())
}

# Save final model
final_model_file = models_path / "final_model.joblib"
joblib.dump(final_model, final_model_file)

# Save final model metadata
final_metadata_file = models_path / "final_model_metadata.json"
with open(final_metadata_file, 'w') as f:
    json.dump(final_model_metadata, f, indent=2)

print(f"\n✓ Final model saved: {final_model_file}")
print(f"  Model: {final_model_name}")
print(f"  CV RMSE: {final_model_rmse:.4f}")
```

### Create Model Summary

```python
# Create comprehensive model summary
model_summary = {
    'project_info': {
        'dataset': 'House Prices - Advanced Regression Techniques',
        'target': 'SalePrice (log-transformed)',
        'features': X_train.shape[1],
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    },
    'individual_models': {},
    'ensemble_models': {},
    'final_model': final_model_metadata,
    'performance_comparison': {}
}

# Add individual model performance
for model_name, (model, result) in individual_models.items():
    model_summary['individual_models'][model_name] = {
        'cv_rmse_mean': result['cv_rmse_mean'],
        'cv_rmse_std': result['cv_rmse_std'],
        'model_type': type(model).__name__
    }

# Add ensemble model performance
for ensemble_name, (ensemble, result, components) in ensemble_models.items():
    model_summary['ensemble_models'][ensemble_name] = {
        'cv_rmse_mean': result['cv_rmse_mean'],
        'cv_rmse_std': result['cv_rmse_std'],
        'components': components
    }

# Add performance comparison
all_results = {**{name: result for name, (model, result) in individual_models.items()},
               **{name: result for name, (ensemble, result, components) in ensemble_models.items()}}

for model_name, result in sorted(all_results.items(), key=lambda x: x[1]['cv_rmse_mean']):
    model_summary['performance_comparison'][model_name] = result['cv_rmse_mean']

# Save model summary
summary_file = models_path / "model_summary.json"
with open(summary_file, 'w') as f:
    json.dump(model_summary, f, indent=2)

print(f"\n✓ Model summary saved: {summary_file}")
```

## 2. Model Loading Code

Use this code in other notebooks to load the exported models:

### Load Individual Models

```python
import joblib
import json
from pathlib import Path

def load_model_with_metadata(model_name, models_dir='../models'):
    """
    Load a model with its metadata
    """
    models_path = Path(models_dir)

    # Load model
    model_file = models_path / f"{model_name}.joblib"
    model = joblib.load(model_file)

    # Load metadata
    metadata_file = models_path / f"{model_name}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return model, metadata

# Example: Load CatBoost model
catboost_model, catboost_metadata = load_model_with_metadata('CatBoost_Optimized')
print(f"Loaded {catboost_metadata['model_name']}")
print(f"CV RMSE: {catboost_metadata['cv_rmse_mean']:.4f}")
```

### Load Ensemble Models

```python
def load_ensemble_model(ensemble_name, models_dir='../models'):
    """
    Load an ensemble model with its metadata
    """
    models_path = Path(models_dir)

    # Load ensemble
    ensemble_file = models_path / f"{ensemble_name}.joblib"
    ensemble = joblib.load(ensemble_file)

    # Load metadata
    metadata_file = models_path / f"{ensemble_name}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return ensemble, metadata

# Example: Load stacking ensemble
stacking_model, stacking_metadata = load_ensemble_model('Stacking_Ensemble')
print(f"Loaded {stacking_metadata['ensemble_name']}")
print(f"Components: {stacking_metadata['component_models']}")
print(f"CV RMSE: {stacking_metadata['cv_rmse_mean']:.4f}")
```

### Load Final Model

```python
def load_final_model(models_dir='../models'):
    """
    Load the final selected model
    """
    models_path = Path(models_dir)

    # Load final model
    final_model_file = models_path / "final_model.joblib"
    final_model = joblib.load(final_model_file)

    # Load metadata
    final_metadata_file = models_path / "final_model_metadata.json"
    with open(final_metadata_file, 'r') as f:
        final_metadata = json.load(f)

    return final_model, final_metadata

# Load final model
final_model, final_metadata = load_final_model()
print(f"Final model: {final_metadata['final_model_name']}")
print(f"CV RMSE: {final_metadata['final_cv_rmse']:.4f}")
```

### Load Model Summary

```python
def load_model_summary(models_dir='../models'):
    """
    Load the complete model summary
    """
    models_path = Path(models_dir)
    summary_file = models_path / "model_summary.json"

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    return summary

# Load and display model summary
summary = load_model_summary()
print("Model Performance Ranking:")
for model_name, rmse in summary['performance_comparison'].items():
    print(f"  {model_name}: {rmse:.4f}")
```

## 3. Usage Examples

### Making Predictions with Loaded Models

```python
# Load data (example)
X_new = pd.read_csv('../data/processed/X_test_final.csv')

# Load final model
final_model, final_metadata = load_final_model()

# Make predictions
predictions_log = final_model.predict(X_new)
predictions_original = np.exp(predictions_log)

print(f"Generated {len(predictions_original)} predictions")
print(f"Price range: ${predictions_original.min():,.0f} - ${predictions_original.max():,.0f}")
```

### Compare Multiple Models

```python
# Load multiple models for comparison
models_to_compare = ['CatBoost_Optimized', 'XGBoost_Optimized', 'Stacking_Ensemble']

for model_name in models_to_compare:
    if 'Ensemble' in model_name:
        model, metadata = load_ensemble_model(model_name)
    else:
        model, metadata = load_model_with_metadata(model_name)

    # Make predictions
    predictions = model.predict(X_new)
    predictions_original = np.exp(predictions)

    print(f"{model_name}:")
    print(f"  CV RMSE: {metadata['cv_rmse_mean']:.4f}")
    print(f"  Mean prediction: ${predictions_original.mean():,.0f}")
```

## 4. Directory Structure

After running the export code, your models directory will look like:

```
models/
├── Ridge_Optimized.joblib
├── Ridge_Optimized_metadata.json
├── Lasso_Optimized.joblib
├── Lasso_Optimized_metadata.json
├── ElasticNet_Optimized.joblib
├── ElasticNet_Optimized_metadata.json
├── RandomForest_Optimized.joblib
├── RandomForest_Optimized_metadata.json
├── GradientBoosting_Optimized.joblib
├── GradientBoosting_Optimized_metadata.json
├── XGBoost_Optimized.joblib
├── XGBoost_Optimized_metadata.json
├── CatBoost_Optimized.joblib
├── CatBoost_Optimized_metadata.json
├── LightGBM_Optimized.joblib
├── LightGBM_Optimized_metadata.json
├── Simple_Ensemble.joblib
├── Simple_Ensemble_metadata.json
├── Weighted_Ensemble.joblib
├── Weighted_Ensemble_metadata.json
├── Stacking_Ensemble.joblib
├── Stacking_Ensemble_metadata.json
├── final_model.joblib
├── final_model_metadata.json
└── model_summary.json
```

## 5. Key Points

- **Model files (.joblib)**: Contains the trained model objects
- **Metadata files (.json)**: Contains performance metrics and model information
- **Final model**: The best performing model for easy access
- **Model summary**: Complete overview of all models and their performance
- **Consistent naming**: All files follow the same naming convention
- **Version control**: Consider adding timestamps or version numbers for model tracking

This structure allows you to easily load any trained model in other notebooks and continue your analysis or make predictions.

## 6. Flexible Model Loading with Fallback

Here's an improved loading function that tries to load optimized models first, and creates basic versions if they're not available:

```python
import joblib
import json
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def load_or_create_models(models_dir='../models', random_state=42):
    """
    Load optimized models from saved files or create basic versions as fallback
    """
    models = {}
    metadata = {}

    # Map model names to saved filenames
    model_files = {
        'Ridge': 'Ridge_Optimized.joblib',
        'Lasso': 'Lasso_Optimized.joblib',
        'ElasticNet': 'ElasticNet_Optimized.joblib',
        'RandomForest': 'RandomForest_Optimized.joblib',
        'GradientBoosting': 'GradientBoosting_Optimized.joblib',
        'XGBoost': 'XGBoost_Optimized.joblib',
        'CatBoost': 'CatBoost_Optimized.joblib',
        'LightGBM': 'LightGBM_Optimized.joblib'
    }

    models_path = Path(models_dir)
    loaded_count = 0

    print("Loading models...")

    for name, filename in model_files.items():
        try:
            # Try to load saved model
            model_file = models_path / filename
            models[name] = joblib.load(model_file)

            # Try to load metadata
            metadata_file = models_path / f"{filename.replace('.joblib', '_metadata.json')}"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata[name] = json.load(f)

            loaded_count += 1
            print(f"  ✓ Loaded optimized {name} from {filename}")

        except Exception as e:
            # Create basic model if loading fails
            if name == 'Ridge':
                models[name] = Ridge(alpha=1.0, random_state=random_state)
            elif name == 'Lasso':
                models[name] = Lasso(alpha=0.1, random_state=random_state, max_iter=1000)
            elif name == 'ElasticNet':
                models[name] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=1000)
            elif name == 'RandomForest':
                models[name] = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif name == 'GradientBoosting':
                models[name] = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
            elif name == 'XGBoost':
                models[name] = XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0)
            elif name == 'CatBoost':
                models[name] = CatBoostRegressor(iterations=100, random_state=random_state, verbose=False)
            elif name == 'LightGBM':
                models[name] = LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)

            # Add basic metadata
            metadata[name] = {
                'model_name': name,
                'model_type': type(models[name]).__name__,
                'optimized': False,
                'note': 'Basic model created (optimized version not found)'
            }

            print(f"  → Created basic {name} (optimized version not found)")

    print(f"\n✓ Model loading completed: {loaded_count} optimized models loaded, {len(models)-loaded_count} basic models created")

    return models, metadata

# Usage example
available_models, model_metadata = load_or_create_models()

# Check which models were loaded vs created
optimized_models = [name for name, meta in model_metadata.items() if meta.get('optimized', True)]
basic_models = [name for name, meta in model_metadata.items() if not meta.get('optimized', True)]

print(f"\nOptimized models loaded: {optimized_models}")
print(f"Basic models created: {basic_models}")
```

### Load Ensemble Models with Fallback

```python
def load_or_create_ensembles(models_dir='../models', base_models=None):
    """
    Load ensemble models or create basic versions
    """
    ensembles = {}
    ensemble_metadata = {}

    # Ensemble files to try loading
    ensemble_files = {
        'Simple_Ensemble': 'Simple_Ensemble.joblib',
        'Weighted_Ensemble': 'Weighted_Ensemble.joblib',
        'Stacking_Ensemble': 'Stacking_Ensemble.joblib'
    }

    models_path = Path(models_dir)
    loaded_count = 0

    print("Loading ensemble models...")

    for name, filename in ensemble_files.items():
        try:
            # Try to load saved ensemble
            ensemble_file = models_path / filename
            ensembles[name] = joblib.load(ensemble_file)

            # Try to load metadata
            metadata_file = models_path / f"{filename.replace('.joblib', '_metadata.json')}"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    ensemble_metadata[name] = json.load(f)

            loaded_count += 1
            print(f"  ✓ Loaded {name} from {filename}")

        except Exception as e:
            # Create basic ensemble if loading fails and base models are available
            if base_models and len(base_models) >= 2:
                from sklearn.ensemble import StackingRegressor

                if name == 'Simple_Ensemble':
                    # Create simple averaging ensemble
                    class SimpleEnsemble:
                        def __init__(self, models):
                            self.models = models

                        def fit(self, X, y):
                            self.fitted_models_ = []
                            for model in self.models:
                                fitted_model = model.fit(X, y)
                                self.fitted_models_.append(fitted_model)
                            return self

                        def predict(self, X):
                            predictions = np.array([model.predict(X) for model in self.fitted_models_])
                            return np.mean(predictions, axis=0)

                    ensembles[name] = SimpleEnsemble(list(base_models.values())[:4])

                elif name == 'Stacking_Ensemble':
                    # Create basic stacking ensemble
                    estimators = [(f"model_{i}", model) for i, model in enumerate(list(base_models.values())[:4])]
                    ensembles[name] = StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(alpha=1.0),
                        cv=3
                    )

                ensemble_metadata[name] = {
                    'ensemble_name': name,
                    'optimized': False,
                    'note': 'Basic ensemble created (optimized version not found)'
                }

                print(f"  → Created basic {name} (optimized version not found)")
            else:
                print(f"  ✗ Could not load or create {name}")

    print(f"\n✓ Ensemble loading completed: {loaded_count} optimized ensembles loaded")

    return ensembles, ensemble_metadata

# Usage example
available_ensembles, ensemble_metadata = load_or_create_ensembles(base_models=available_models)
```

### Complete Loading Function

```python
def load_all_models(models_dir='../models', random_state=42):
    """
    Load all available models (individual and ensemble) with fallback
    """
    print("="*60)
    print("LOADING ALL MODELS")
    print("="*60)

    # Load individual models
    individual_models, individual_metadata = load_or_create_models(models_dir, random_state)

    # Load ensemble models
    ensemble_models, ensemble_metadata = load_or_create_ensembles(models_dir, individual_models)

    # Try to load final model
    final_model = None
    final_metadata = None

    try:
        final_model, final_metadata = load_final_model(models_dir)
        print(f"\n✓ Final model loaded: {final_metadata['final_model_name']}")
    except:
        print("\n→ Final model not found")

    # Combine all models
    all_models = {**individual_models, **ensemble_models}
    all_metadata = {**individual_metadata, **ensemble_metadata}

    print(f"\n✓ Total models available: {len(all_models)}")

    return {
        'individual_models': individual_models,
        'ensemble_models': ensemble_models,
        'all_models': all_models,
        'final_model': final_model,
        'metadata': all_metadata,
        'final_metadata': final_metadata
    }

# Complete usage example
model_collection = load_all_models()

# Access different model types
ridge_model = model_collection['individual_models']['Ridge']
stacking_model = model_collection['ensemble_models'].get('Stacking_Ensemble')
final_model = model_collection['final_model']

# Check model performance info
for name, metadata in model_collection['metadata'].items():
    if metadata.get('cv_rmse_mean'):
        print(f"{name}: CV RMSE = {metadata['cv_rmse_mean']:.4f}")
    else:
        print(f"{name}: {metadata.get('note', 'Basic model')}")
```

### Model Validation Function

```python
def validate_loaded_models(models, X_sample, y_sample):
    """
    Validate that loaded models work correctly
    """
    print("\nValidating loaded models...")

    for name, model in models.items():
        try:
            # Test fit and predict
            model.fit(X_sample, y_sample)
            predictions = model.predict(X_sample)

            print(f"  ✓ {name}: Working correctly")

        except Exception as e:
            print(f"  ✗ {name}: Error - {str(e)}")

    print("Model validation completed.")

# Usage (assuming you have some sample data)
# validate_loaded_models(available_models, X_train_sample, y_train_sample)
```

This flexible approach allows you to:

- Load optimized models when available
- Fall back to basic models when optimized versions are missing
- Handle both individual models and ensembles
- Include metadata about model performance
- Validate that loaded models work correctly

## 7. Best Parameters Export and Loading System

This system saves all optimized hyperparameters in a structured format, allowing you to recreate optimized models even without saved model files.

### Export Best Parameters

```python
def export_best_parameters(models_dir='../models'):
    """
    Export all optimized hyperparameters to a structured JSON file
    """
    models_path = Path(models_dir)

    # Collect all best parameters from optimization results
    best_parameters = {
        'export_info': {
            'timestamp': str(pd.Timestamp.now()),
            'description': 'Optimized hyperparameters from model training',
            'random_state': RANDOM_STATE
        },
        'individual_models': {},
        'ensemble_configs': {},
        'optimization_info': {}
    }

    # Individual model parameters
    individual_params = {
        'Ridge': {
            'params': ridge_result.get('best_params', {'alpha': 1.0}),
            'cv_rmse': ridge_result['cv_rmse_mean'],
            'cv_std': ridge_result['cv_rmse_std']
        },
        'Lasso': {
            'params': lasso_result.get('best_params', {'alpha': 0.1}),
            'cv_rmse': lasso_result['cv_rmse_mean'],
            'cv_std': lasso_result['cv_rmse_std']
        },
        'ElasticNet': {
            'params': elasticnet_result.get('best_params', {'alpha': 0.1, 'l1_ratio': 0.5}),
            'cv_rmse': elasticnet_result['cv_rmse_mean'],
            'cv_std': elasticnet_result['cv_rmse_std']
        },
        'RandomForest': {
            'params': rf_result.get('best_params', {'n_estimators': 100, 'max_depth': 10}),
            'cv_rmse': rf_result['cv_rmse_mean'],
            'cv_std': rf_result['cv_rmse_std']
        },
        'GradientBoosting': {
            'params': gb_result.get('best_params', {'n_estimators': 100, 'learning_rate': 0.1}),
            'cv_rmse': gb_result['cv_rmse_mean'],
            'cv_std': gb_result['cv_rmse_std']
        },
        'XGBoost': {
            'params': xgb_result.get('best_params', {'n_estimators': 100, 'learning_rate': 0.1}),
            'cv_rmse': xgb_result['cv_rmse_mean'],
            'cv_std': xgb_result['cv_rmse_std']
        },
        'CatBoost': {
            'params': catboost_result.get('best_params', {'iterations': 100, 'learning_rate': 0.1}),
            'cv_rmse': catboost_result['cv_rmse_mean'],
            'cv_std': catboost_result['cv_rmse_std']
        },
        'LightGBM': {
            'params': lightgbm_result.get('best_params', {'n_estimators': 100, 'learning_rate': 0.1}),
            'cv_rmse': lightgbm_result['cv_rmse_mean'],
            'cv_std': lightgbm_result['cv_rmse_std']
        }
    }

    best_parameters['individual_models'] = individual_params

    # Ensemble configurations
    ensemble_configs = {
        'Simple_Ensemble': {
            'type': 'SimpleEnsemble',
            'base_models': ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'],
            'cv_rmse': simple_result['cv_rmse_mean'],
            'cv_std': simple_result['cv_rmse_std']
        },
        'Weighted_Ensemble': {
            'type': 'WeightedEnsemble',
            'base_models': ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'],
            'weights': weights.tolist(),  # Convert numpy array to list
            'cv_rmse': weighted_result['cv_rmse_mean'],
            'cv_std': weighted_result['cv_rmse_std']
        },
        'Stacking_Ensemble': {
            'type': 'StackingRegressor',
            'base_models': ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'],
            'final_estimator': {
                'type': 'Ridge',
                'params': {'alpha': 1.0}
            },
            'cv_folds': 3,
            'cv_rmse': stacking_result['cv_rmse_mean'],
            'cv_std': stacking_result['cv_rmse_std']
        }
    }

    best_parameters['ensemble_configs'] = ensemble_configs

    # Optimization information
    optimization_info = {
        'total_trials': {
            'Ridge': 30,
            'Lasso': 30,
            'ElasticNet': 30,
            'RandomForest': 50,
            'GradientBoosting': 60,
            'XGBoost': 75,
            'CatBoost': 75,
            'LightGBM': 75
        },
        'cv_strategy': {
            'type': 'KFold',
            'n_splits': cv_folds,
            'shuffle': True,
            'random_state': RANDOM_STATE
        },
        'optimization_framework': 'Optuna'
    }

    best_parameters['optimization_info'] = optimization_info

    # Save to file
    params_file = models_path / 'best_parameters.json'
    with open(params_file, 'w') as f:
        json.dump(best_parameters, f, indent=2)

    print(f"✓ Best parameters exported to: {params_file}")
    print(f"  Individual models: {len(individual_params)}")
    print(f"  Ensemble configs: {len(ensemble_configs)}")

    return best_parameters

# Export parameters
exported_params = export_best_parameters()
```

### Load Best Parameters

```python
def load_best_parameters(models_dir='../models'):
    """
    Load best parameters from the JSON file
    """
    models_path = Path(models_dir)
    params_file = models_path / 'best_parameters.json'

    try:
        with open(params_file, 'r') as f:
            best_parameters = json.load(f)

        print(f"✓ Best parameters loaded from: {params_file}")
        print(f"  Export date: {best_parameters['export_info']['timestamp']}")
        print(f"  Individual models: {len(best_parameters['individual_models'])}")
        print(f"  Ensemble configs: {len(best_parameters['ensemble_configs'])}")

        return best_parameters

    except FileNotFoundError:
        print(f"✗ Parameters file not found: {params_file}")
        return None
    except Exception as e:
        print(f"✗ Error loading parameters: {str(e)}")
        return None

# Load parameters
best_params = load_best_parameters()
```

### Create Models from Saved Parameters

```python
def create_model_from_params(model_name, params_dict, random_state=42):
    """
    Create a model instance using saved parameters
    """
    model_params = params_dict.copy()

    # Add random state if not present
    if 'random_state' not in model_params:
        model_params['random_state'] = random_state

    # Create model based on type
    if model_name == 'Ridge':
        return Ridge(**model_params)
    elif model_name == 'Lasso':
        model_params['max_iter'] = model_params.get('max_iter', 1000)
        return Lasso(**model_params)
    elif model_name == 'ElasticNet':
        model_params['max_iter'] = model_params.get('max_iter', 1000)
        return ElasticNet(**model_params)
    elif model_name == 'RandomForest':
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        return RandomForestRegressor(**model_params)
    elif model_name == 'GradientBoosting':
        return GradientBoostingRegressor(**model_params)
    elif model_name == 'XGBoost':
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        model_params['verbosity'] = model_params.get('verbosity', 0)
        return XGBRegressor(**model_params)
    elif model_name == 'CatBoost':
        model_params['verbose'] = model_params.get('verbose', False)
        return CatBoostRegressor(**model_params)
    elif model_name == 'LightGBM':
        model_params['n_jobs'] = model_params.get('n_jobs', -1)
        model_params['verbose'] = model_params.get('verbose', -1)
        return LGBMRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def load_or_create_models_with_params(models_dir='../models', random_state=42):
    """
    Load models or create them using saved parameters
    """
    models = {}
    metadata = {}

    # First try to load saved models
    models, metadata = load_or_create_models(models_dir, random_state)

    # For models that weren't loaded, try to create them with saved parameters
    best_params = load_best_parameters(models_dir)

    if best_params:
        print("\nCreating models from saved parameters...")

        for model_name, meta in metadata.items():
            if not meta.get('optimized', True):  # This is a basic model
                try:
                    # Get saved parameters
                    saved_params = best_params['individual_models'][model_name]['params']

                    # Create optimized model
                    optimized_model = create_model_from_params(model_name, saved_params, random_state)
                    models[model_name] = optimized_model

                    # Update metadata
                    metadata[model_name] = {
                        'model_name': model_name,
                        'model_type': type(optimized_model).__name__,
                        'optimized': True,
                        'cv_rmse_mean': best_params['individual_models'][model_name]['cv_rmse'],
                        'cv_rmse_std': best_params['individual_models'][model_name]['cv_std'],
                        'best_params': saved_params,
                        'note': 'Created from saved parameters'
                    }

                    print(f"  ✓ Created optimized {model_name} from saved parameters")
                    print(f"    CV RMSE: {metadata[model_name]['cv_rmse_mean']:.4f}")

                except Exception as e:
                    print(f"  ✗ Failed to create {model_name} from saved parameters: {str(e)}")

    return models, metadata

# Usage
optimized_models, model_metadata = load_or_create_models_with_params()
```

### Create Ensembles from Saved Configurations

```python
def create_ensemble_from_config(ensemble_name, config, base_models):
    """
    Create an ensemble from saved configuration
    """
    ensemble_type = config['type']

    if ensemble_type == 'SimpleEnsemble':
        # Create simple ensemble
        selected_models = [base_models[name] for name in config['base_models'] if name in base_models]

        class SimpleEnsemble:
            def __init__(self, models):
                self.models = models

            def fit(self, X, y):
                self.fitted_models_ = []
                for model in self.models:
                    fitted_model = clone(model)
                    fitted_model.fit(X, y)
                    self.fitted_models_.append(fitted_model)
                return self

            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.fitted_models_])
                return np.mean(predictions, axis=0)

        return SimpleEnsemble(selected_models)

    elif ensemble_type == 'WeightedEnsemble':
        # Create weighted ensemble
        selected_models = [base_models[name] for name in config['base_models'] if name in base_models]
        weights = np.array(config['weights'])

        class WeightedEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights

            def fit(self, X, y):
                self.fitted_models_ = []
                for model in self.models:
                    fitted_model = clone(model)
                    fitted_model.fit(X, y)
                    self.fitted_models_.append(fitted_model)
                return self

            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.fitted_models_])
                return np.average(predictions, axis=0, weights=self.weights)

        return WeightedEnsemble(selected_models, weights)

    elif ensemble_type == 'StackingRegressor':
        # Create stacking ensemble
        selected_models = [base_models[name] for name in config['base_models'] if name in base_models]
        estimators = [(f"model_{i}", model) for i, model in enumerate(selected_models)]

        # Create final estimator
        final_estimator_config = config['final_estimator']
        if final_estimator_config['type'] == 'Ridge':
            final_estimator = Ridge(**final_estimator_config['params'])
        else:
            final_estimator = Ridge(alpha=1.0)  # Default fallback

        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=config.get('cv_folds', 3),
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

def load_or_create_ensembles_with_params(models_dir='../models', base_models=None):
    """
    Load ensembles or create them from saved configurations
    """
    ensembles = {}
    ensemble_metadata = {}

    # First try to load saved ensembles
    ensembles, ensemble_metadata = load_or_create_ensembles(models_dir, base_models)

    # For ensembles that weren't loaded, try to create them from saved configs
    best_params = load_best_parameters(models_dir)

    if best_params and base_models:
        print("\nCreating ensembles from saved configurations...")

        for ensemble_name, meta in ensemble_metadata.items():
            if not meta.get('optimized', True):  # This is a basic ensemble
                try:
                    # Get saved configuration
                    saved_config = best_params['ensemble_configs'][ensemble_name]

                    # Create optimized ensemble
                    optimized_ensemble = create_ensemble_from_config(ensemble_name, saved_config, base_models)
                    ensembles[ensemble_name] = optimized_ensemble

                    # Update metadata
                    ensemble_metadata[ensemble_name] = {
                        'ensemble_name': ensemble_name,
                        'ensemble_type': saved_config['type'],
                        'optimized': True,
                        'cv_rmse_mean': saved_config['cv_rmse'],
                        'cv_rmse_std': saved_config['cv_std'],
                        'base_models': saved_config['base_models'],
                        'note': 'Created from saved configuration'
                    }

                    print(f"  ✓ Created optimized {ensemble_name} from saved configuration")
                    print(f"    CV RMSE: {ensemble_metadata[ensemble_name]['cv_rmse_mean']:.4f}")

                except Exception as e:
                    print(f"  ✗ Failed to create {ensemble_name} from saved configuration: {str(e)}")

    return ensembles, ensemble_metadata

# Usage
optimized_ensembles, ensemble_metadata = load_or_create_ensembles_with_params(base_models=optimized_models)
```

### Complete Parameter-Based Loading

```python
def load_all_models_with_params(models_dir='../models', random_state=42):
    """
    Complete model loading with parameter-based recreation
    """
    print("="*60)
    print("LOADING ALL MODELS WITH PARAMETER SUPPORT")
    print("="*60)

    # Load individual models (with parameter support)
    individual_models, individual_metadata = load_or_create_models_with_params(models_dir, random_state)

    # Load ensemble models (with parameter support)
    ensemble_models, ensemble_metadata = load_or_create_ensembles_with_params(models_dir, individual_models)

    # Try to load final model
    final_model = None
    final_metadata = None

    try:
        final_model, final_metadata = load_final_model(models_dir)
        print(f"\n✓ Final model loaded: {final_metadata['final_model_name']}")
    except:
        print("\n→ Final model not found")

    # Combine all models
    all_models = {**individual_models, **ensemble_models}
    all_metadata = {**individual_metadata, **ensemble_metadata}

    # Count optimized vs basic models
    optimized_count = sum(1 for meta in all_metadata.values() if meta.get('optimized', True))
    basic_count = len(all_models) - optimized_count

    print(f"\n✓ Total models loaded: {len(all_models)}")
    print(f"  Optimized models: {optimized_count}")
    print(f"  Basic models: {basic_count}")

    return {
        'individual_models': individual_models,
        'ensemble_models': ensemble_models,
        'all_models': all_models,
        'final_model': final_model,
        'metadata': all_metadata,
        'final_metadata': final_metadata
    }

# Complete usage
model_collection = load_all_models_with_params()
```

### Parameter Management Functions

```python
def view_saved_parameters(models_dir='../models'):
    """
    View all saved parameters in a readable format
    """
    best_params = load_best_parameters(models_dir)

    if not best_params:
        print("No saved parameters found")
        return

    print("\n" + "="*60)
    print("SAVED PARAMETERS SUMMARY")
    print("="*60)

    # Individual models
    print("\nIndividual Models:")
    for model_name, info in best_params['individual_models'].items():
        print(f"\n{model_name}:")
        print(f"  CV RMSE: {info['cv_rmse']:.4f} (±{info['cv_std']:.4f})")
        print(f"  Parameters: {info['params']}")

    # Ensemble configs
    print("\nEnsemble Configurations:")
    for ensemble_name, config in best_params['ensemble_configs'].items():
        print(f"\n{ensemble_name}:")
        print(f"  Type: {config['type']}")
        print(f"  CV RMSE: {config['cv_rmse']:.4f} (±{config['cv_std']:.4f})")
        print(f"  Base models: {config['base_models']}")
        if 'weights' in config:
            print(f"  Weights: {config['weights']}")

def compare_model_parameters(model_name, models_dir='../models'):
    """
    Compare saved parameters for a specific model
    """
    best_params = load_best_parameters(models_dir)

    if not best_params or model_name not in best_params['individual_models']:
        print(f"No saved parameters found for {model_name}")
        return

    model_info = best_params['individual_models'][model_name]

    print(f"\n{model_name} Parameters:")
    print(f"  Performance: {model_info['cv_rmse']:.4f} (±{model_info['cv_std']:.4f})")
    print(f"  Optimized parameters:")
    for param, value in model_info['params'].items():
        print(f"    {param}: {value}")

# Usage examples
view_saved_parameters()
compare_model_parameters('XGBoost')
```

This enhanced system provides:

1. **Complete parameter export** - All optimized hyperparameters saved in structured format
2. **Intelligent model recreation** - Create optimized models from saved parameters
3. **Ensemble configuration** - Save and recreate ensemble setups
4. **Fallback system** - Use saved parameters → basic models → error handling
5. **Parameter management** - View, compare, and manage saved parameters
6. **Performance tracking** - Keep CV scores with parameters for comparison

Now you can recreate your optimized models even without the actual model files, just using the saved parameters!
