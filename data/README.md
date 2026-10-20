# Data

**Source:** UCI Machine Learning Repository - Heart Disease (id 45), doi:10.24432/C52P4X.
Principal investigators: A. Janosi (Hungarian Institute of Cardiology), W. Steinbrunn (University Hospital
Zurich), M. Pfisterer (University Hospital Basel), R. Detrano (V.A. Medical Center Long Beach and Cleveland
Clinic Foundation).

```bash
python -m heartrisk.data download   # data/raw/ + provenance.json (SHA-256 of every file)
python -m heartrisk.data prepare    # data/processed/*.csv + cleaning_report.json
```

| File | Rows | Use |
|---|---|---|
| `processed/heart_cleveland_clean.csv` | 303 | training (80%) and hold-out test (20%) |
| `processed/heart_hungarian_clean.csv` | 293 | external validation only (different hospital) |
| `processed/cleaning_report.json` | - | rows in/out, values nulled, missingness after cleaning |

Cleveland raw file SHA-256: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`.

## Data dictionary

| Column | Meaning | Type / values |
|---|---|---|
| age | age in years | numeric |
| sex | 1 = male, 0 = female | binary |
| cp | chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic | categorical |
| trestbps | resting blood pressure on admission (mm Hg) | numeric |
| chol | serum cholesterol (mg/dl) | numeric |
| fbs | fasting blood sugar > 120 mg/dl | binary |
| restecg | resting ECG: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy | categorical |
| thalach | maximum heart rate achieved | numeric |
| exang | exercise-induced angina | binary |
| oldpeak | ST depression induced by exercise relative to rest | numeric |
| slope | slope of peak exercise ST segment: 1 up, 2 flat, 3 down | categorical |
| ca | number of major vessels (0-3) coloured by fluoroscopy - 4 missing | numeric (ordinal) |
| thal | thallium scan: 3 normal, 6 fixed defect, 7 reversible defect - 2 missing | categorical |
| target | 1 if original `num` in 1-4 (>50% diameter narrowing), else 0 | binary label |

Missing values are kept in the CSV on purpose; they are imputed inside the model pipeline.
