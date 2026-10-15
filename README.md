# AI Healthcare Triage (SDG 3)

An AI-powered dual-agent system designed for resource-constrained community clinics. It automates emergency triage priority scoring and translates dense clinical jargon into 5th-grade plain English discharge instructions.

## Features
- **Supervised ML Triage Model:** TF-IDF + Random Forest Classifier (90.8% CV Accuracy) mapping symptoms to ESI Levels 1–5.
- **Health Literacy Engine:** Regex word-boundary translation engine mapping medical jargon to plain English.
- **SDG 3 Alignment:** Promotes Good Health and Well-Being by reducing clinic wait times and boosting treatment compliance.

##  Repository Contents
- `main.py` - Core machine learning & translation pipeline
- `triage.csv` - 120 patient dataset across 5 triage levels
- `medical_jargon.csv` - 50 medical terms to plain English dictionary
- `Lean Canvas.png` - Strategic project Lean Canvas
