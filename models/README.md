# Models Directory

This directory stores trained machine learning models:

- `random_forest_model.joblib` - Random Forest classifier
- `xgboost_model.joblib` - XGBoost classifier
- `*_scaler.joblib` - Feature scalers
- `*_metadata.json` - Model metadata
- `training_results.json` - Training performance results

Model files are excluded from Git due to size (see .gitignore) but the directory structure is preserved.

To generate models, run:
```bash
cd src
python train_baseline.py
```