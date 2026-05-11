# Resume Analyzer using TF-IDF + Cosine Similarity

A Python-based Resume Analyzer that matches candidate resumes with Job Descriptions (JDs) using:

- Skill Normalization
- Deduplication
- TF-IDF Vectorization
- Binary JD Vectors
- Cosine Similarity Ranking

The system simulates how ATS (Applicant Tracking Systems) and AI-based recruitment tools rank resumes against job requirements.

---

# Features

✅ Skill normalization using aliases  
✅ Typo handling (`Pyhton → python`, `kubernates → kubernetes`)  
✅ Duplicate skill removal  
✅ Vocabulary generation  
✅ TF-IDF vector computation  
✅ Binary vector representation for JDs  
✅ Cosine similarity matching  
✅ Top-3 candidate ranking for every JD  
✅ Fully implemented using pure Python (no external ML libraries)

---

# Project Workflow

The system works in 7 stages:

## Stage 1 — Skill Normalization

Raw resume skills are converted into standardized canonical skills.





###score###
JD1 = Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)
JD2 = Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)
JD3 = Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
