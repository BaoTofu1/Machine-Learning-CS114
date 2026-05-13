# Setup Guide

This guide runs the fixed Streamlit Web Demo.

## Requirements

- Python 3.9 or newer
- pip
- 4 GB RAM minimum, 8 GB recommended

## Install Dependencies

Open PowerShell and run:

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed"
python -m pip install --upgrade pip
python -m pip install -r "Web Demo\requirements.txt"
```

Using a virtual environment is recommended:

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r "Web Demo\requirements.txt"
```

If PowerShell blocks activation, run this once for the current terminal:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Run the App

From the project root:

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed"
python -m streamlit run "Web Demo\app.py"
```

Open:

```text
http://localhost:8501
```

If port 8501 is busy:

```powershell
python -m streamlit run "Web Demo\app.py" --server.port 8502
```

## Run the Extended Web Demo

The copied extended version is still available:

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed\Web Demo\Machine-Learning-CS114-web_demo-custom_eda_page\Web_Demo"
python -m streamlit run app.py --server.port 8502
```

## Rebuild Processed Data

If you need to regenerate the processed CSV:

```powershell
cd "H:\Nam ba\Hoc ki 2\May Hoc\Project ML\Project after fixed"
jupyter notebook "Feature Engineering + PreProcessing\preprocessing.ipynb"
```

Run all cells. The notebook writes:

```text
Models/processed_bank_data.csv
Models/preprocessing_summary.json
```

## Common Issues

### ModuleNotFoundError

Install dependencies again:

```powershell
python -m pip install -r "Web Demo\requirements.txt"
```

### Data File Not Found

Check that this file exists:

```text
Project after fixed\Models\processed_bank_data.csv
```

Then launch the app again. The fixed app searches both the project-level `Models/` folder and a local `Web Demo/Models/` folder.

### Port Already in Use

Use another port:

```powershell
python -m streamlit run "Web Demo\app.py" --server.port 8502
```

Or find and stop the process:

```powershell
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

## Submission Notes

For an ML project report, treat the no-duration scenario as the main result. The with-duration scenario is only a leakage-aware benchmark.
