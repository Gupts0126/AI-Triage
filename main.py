import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class TriageClassifier:
    """Supervised Machine Learning Classifier for Patient Emergency Triage Assessment."""
    
    TRIAGE_METADATA = {
        1: {"name": "Level 1 - Resuscitation", "urgency": "CRITICAL / IMMEDIATE", "red_flag": True},
        2: {"name": "Level 2 - Emergent", "urgency": "HIGH PRIORITY", "red_flag": True},
        3: {"name": "Level 3 - Urgent", "urgency": "MODERATE PRIORITY", "red_flag": False},
        4: {"name": "Level 4 - Less Urgent", "urgency": "LOW PRIORITY", "red_flag": False},
        5: {"name": "Level 5 - Non-Urgent", "urgency": "ROUTINE", "red_flag": False}
    }

    def __init__(self, dataset_path="triage_dataset.csv"):
        self.dataset_path = dataset_path
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
        self.is_trained = False
        self.df = None

    def load_data(self):
        """Loads dataset from dynamic path with automated fallback for missing files."""
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            print(f"Loaded training dataset from '{self.dataset_path}' ({len(self.df)} records).")
        else:
            print(f"File '{self.dataset_path}' not found. Initializing built-in fallback dataset.")
            self.df = pd.DataFrame([
                {"symptom_text": "crushing chest pain shortness of breath sweating arm pain", "triage_level": 1},
                {"symptom_text": "sudden numbness face weakness difficulty speaking loss of vision", "triage_level": 1},
                {"symptom_text": "severe headache stiff neck high fever sensitivity to light nauseous", "triage_level": 2},
                {"symptom_text": "severe sudden abdominal pain high fever chills vomiting", "triage_level": 2},
                {"symptom_text": "sharp abdominal pain vomiting high fever localized tenderness", "triage_level": 3},
                {"symptom_text": "high fever chills body ache persistent cough fatigue", "triage_level": 3},
                {"symptom_text": "mild fever sore throat runny nose cough mild headache", "triage_level": 4},
                {"symptom_text": "twisted ankle minor swelling painful when walking", "triage_level": 4},
                {"symptom_text": "minor cut on finger bleeding stopped small bandaid needed", "triage_level": 5},
                {"symptom_text": "suture removal from old wound no redness or swelling", "triage_level": 5}
            ])

    def train(self):
        """Trains the Random Forest model and evaluates performance metrics."""
        self.load_data()
        X = self.vectorizer.fit_transform(self.df['symptom_text'])
        y = self.df['triage_level'].values

        self.model.fit(X, y)
        self.is_trained = True

        min_samples = pd.Series(y).value_counts().min()
        if min_samples >= 2 and len(y) >= 10:
            cv_folds = min(3, min_samples)
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds)
            print(f"Triage ML Model Trained Successfully. (Cross-Val Accuracy: {np.mean(cv_scores)*100:.1f}%)")
        else:
            print("Triage ML Model Trained Successfully.")

    def predict_triage(self, new_symptoms):
        """Predicts triage level, confidence, probability breakdown, and urgency flags."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please call `.train()` first.")

        input_vector = self.vectorizer.transform([new_symptoms])
        predicted_level = int(self.model.predict(input_vector)[0])
        probabilities = self.model.predict_proba(input_vector)[0]
        classes = self.model.classes_

        top_confidence = float(np.max(probabilities) * 100)
        prob_breakdown = {
            f"Level {cls}": f"{prob * 100:.1f}%" 
            for cls, prob in zip(classes, probabilities)
        }

        meta = self.TRIAGE_METADATA.get(predicted_level, {
            "name": "Unknown", "urgency": "UNKNOWN", "red_flag": False
        })

        return {
            "triage_score": predicted_level,
            "category_name": meta["name"],
            "urgency_level": meta["urgency"],
            "confidence": f"{top_confidence:.1f}%",
            "red_flag_alert": meta["red_flag"],
            "probability_breakdown": prob_breakdown
        }


class HealthLiteracyTranslator:
    """Fast plain-language translation engine using word-boundary regex and length sorting."""

    def __init__(self, dictionary_path="medical_jargon.csv"):
        self.dictionary_path = dictionary_path
        self.dict_df = self._load_dictionary()

    def _load_dictionary(self):
        if os.path.exists(self.dictionary_path):
            df = pd.read_csv(self.dictionary_path)
        else:
            df = pd.DataFrame([
                {"jargon": "cephalgia", "plain_english": "headache"},
                {"jargon": "nuchal rigidity", "plain_english": "stiff neck"},
                {"jargon": "photophobia", "plain_english": "sensitivity to light"},
                {"jargon": "pyrexia", "plain_english": "fever"},
                {"jargon": "nausea", "plain_english": "feeling sick / stomach upset"},
                {"jargon": "stat", "plain_english": "immediate emergency"},
                {"jargon": "CT head", "plain_english": "brain scan"},
                {"jargon": "LP", "plain_english": "spinal tap test"},
                {"jargon": "Empiric IV", "plain_english": "emergency IV medicine"},
                {"jargon": "admitted to step-down unit", "plain_english": "staying in hospital under close monitoring"}
            ])
        
       
        df['length'] = df['jargon'].astype(str).str.len()
        return df.sort_values(by='length', ascending=False).reset_index(drop=True)

    def simplify_notes(self, clinical_notes):
        """Replaces clinical jargon using exact word boundaries."""
        simplified = clinical_notes

        for _, row in self.dict_df.iterrows():
            jargon = str(row['jargon'])
            plain = str(row['plain_english'])
            pattern = re.compile(r'\b' + re.escape(jargon) + r'\b', re.IGNORECASE)
            simplified = pattern.sub(f"**{plain}**", simplified)

        return simplified

    def generate_patient_letter(self, clinical_notes, triage_info=None):
        """Formats discharge text into a structured, readable patient letter."""
        translated_text = self.simplify_notes(clinical_notes)

        lines = [
            "=" * 60,
            "             YOUR EASY-TO-READ CARE INSTRUCTIONS             ",
            "=" * 60
        ]
        
        if triage_info:
            lines.append(f"\n[Clinic Status]: {triage_info['category_name']}")
            lines.append(f"[Urgency Level]: {triage_info['urgency_level']}")
            if triage_info['red_flag_alert']:
                lines.append(" ATTENTION: Your condition requires close monitoring by clinic staff.")
            lines.append("-" * 60)

        lines.extend([
            "\n WHAT YOUR DOCTOR NOTED (SIMPLE WORDS):",
            translated_text,
            "\n NEXT STEPS & ACTION PLAN:",
            "1. Take all prescribed medications as instructed.",
            "2. Rest comfortably and drink plenty of fluids.",
            "3. Notify nursing staff immediately if symptoms change or worsen.",
            "=" * 60
        ])

        return "\n".join(lines)


if __name__ == "__main__":
  
    triage_system = TriageClassifier(dataset_path="triage.csv")
    triage_system.train()

    patient_input = "severe headache stiff neck high fever nauseous light hurts my eyes"
    print("\n--- WAITING ROOM PATIENT INPUT ---")
    print(patient_input)

    triage_result = triage_system.predict_triage(patient_input)
    print("\n--- TRIAGE AGENT OUTPUT (STRUCTURED JSON) ---")
    print(json.dumps(triage_result, indent=2))

    doctor_notes = (
        "Patient presents with severe cephalgia, nuchal rigidity, pyrexia, and photophobia. "
        "Accompanied by nausea. Plan: Stat CT head, followed by LP. "
        "Empiric IV started. Patient admitted to step-down unit."
    )
    
    translator = HealthLiteracyTranslator(dictionary_path="medical_jargon.csv")
    patient_letter = translator.generate_patient_letter(doctor_notes, triage_result)

    print("\n--- HEALTH LITERACY OUTPUT (PATIENT LETTER) ---")
    print(patient_letter)