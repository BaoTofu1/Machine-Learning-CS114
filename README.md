# Machine-Learning-CS114

Fixed submission version for the Bank Marketing Campaign Prediction project.

## Contents

- `EDA + Data/`: raw UCI Bank Marketing zip and EDA notebook.
- `Feature Engineering + PreProcessing/`: reproducible preprocessing notebook.
- `Models/`: processed dataset, preprocessing summary, Logistic Regression, XGBoost, Random Forest, and LightGBM notebooks.
- `Web Demo/`: Streamlit web demo and setup instructions.
- `FIXED_CHANGES.md`: concise list of fixes compared with the original project folder.

## Run Web Demo

```powershell
python -m pip install -r "Web Demo\requirements.txt"
python -m streamlit run "Web Demo\app.py"
```

Then open:

```text
http://localhost:8501
```

## Main Note

The no-duration scenario is the realistic result for pre-call prediction. The with-duration scenario is kept only as a leakage-aware benchmark because call duration is known after the call.
