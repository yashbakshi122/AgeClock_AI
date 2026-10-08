# AgeClock AI

Gene-expression based age prediction research project.

Dataset: GEO GSE74937

Model:
- SelectKBest
- 100 expression probes
- StandardScaler
- Ridge Regression
- alpha = 10
- Twin-aware GroupKFold validation

The application contains a Flask API and Streamlit interface.

This project is for research/educational purposes only and is
not a medical diagnostic system.