# Project After Fixed - Change Summary

This folder is an isolated fixed copy of the original `Project ML` folder. The original files were not overwritten.

## Main Fixes

- Rebuilt `Feature Engineering + PreProcessing/preprocessing.ipynb` with relative paths and clear markdown.
- Regenerated `Models/processed_bank_data.csv` from the raw UCI zip.
- Removed 12 exact duplicate rows from the raw dataset.
- Added `is_contacted_before` to the processed dataset.
- Added `Models/preprocessing_summary.json`.
- Rebuilt `Models/Logistic_Regression.ipynb` with temporal train/validation/test evaluation.
- Rebuilt `Models/XGBoost_RF_LightGBM.ipynb` so `RandomizedSearchCV` no longer silently returns `nan`.
- Put SMOTE inside `imblearn.Pipeline` in model notebooks.
- Chose classification thresholds on validation data, not on test data.
- Refit final models on train+validation before final test evaluation.
- Fixed Web Demo data path handling.
- Updated Web Demo README/SETUP instructions.
- Updated Web Demo model hyperparameters based on the fixed tuning notebook.

## Final Modeling Rule

Use the no-duration scenario as the realistic result. The with-duration scenario is only a benchmark because `duration` is known only after a call has happened.

## How to Run Web Demo

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed"
python -m pip install -r "Web Demo\requirements.txt"
python -m streamlit run "Web Demo\app.py"
```

Open:

```text
http://localhost:8501
```
