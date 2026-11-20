import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.sparse import hstack
from sklearn.exceptions import InconsistentVersionWarning

# Hide the scikit-learn version warning (results were checked and are the same)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent

# SETTINGS: change these to match your files
JOBS_FILE = project_folder / "data" / "jobs.csv"   # our 40 job postings
# MODEL_FILE = project_folder / "models" / "logreg_model.joblib" # Logistic Regression model file
# MODEL_FILE = project_folder / "models" / "decision_tree.joblib" # Decision Tree model file
MODEL_FILE = project_folder / "models" / "neural_network.joblib" # Neural Network model file

# The resumes to test (name shown in the output : file in the data folder)
RESUMES = {
    "Data Analyst": "Data analyst.txt",
    "Business Executive": "Business_executive.txt",
    "Energy Engineer": "Energy engineer.txt",
}

TITLE_COLUMN = "title"                       # column with the job title
COMPANY_COLUMN = "company"                   # column with the company name
DESCRIPTION_COLUMN = "job_description_text"  # column with the job description

TOP_N = 5  # how many top matches to show for each resume

# LOAD THE JOBS, MODEL, AND TF-IDF VECTORIZER
jobs = pd.read_csv(JOBS_FILE)
model = joblib.load(MODEL_FILE)
tfidf = joblib.load(project_folder / "models" / "tfidf_vectorizer.joblib")

# Stop early if a column name doesn't match the CSV
for column in [TITLE_COLUMN, COMPANY_COLUMN, DESCRIPTION_COLUMN]:
    if column not in jobs.columns:
        print(f"Column '{column}' not found. Your CSV has these columns:")
        print(list(jobs.columns))
        raise SystemExit

print("Job postings loaded:", len(jobs))

# The job features are the same for every resume, so build them once
X_jobs = tfidf.transform(jobs[DESCRIPTION_COLUMN].fillna("").astype(str))

# SCORE ONE RESUME AGAINST ALL JOBS
def match_resume(resume_text):
    # Pair the one resume with every job posting
    X_resume = tfidf.transform([resume_text] * len(jobs))

    # Words in both the resume and the job, plus overall similarity
    X_match = X_resume.multiply(X_jobs).tocsr()
    similarity = X_match.sum(axis=1).A1.reshape(-1, 1)

    # Same feature order as data_preprocessing.py
    X = hstack([X_resume, X_jobs, X_match, similarity]).tocsr()

    results = jobs[[TITLE_COLUMN, COMPANY_COLUMN]].copy()
    results["Prediction"] = model.predict(X)

    # One column of confidence per label
    probabilities = model.predict_proba(X)
    for i, label in enumerate(model.classes_):
        results[label + " %"] = (probabilities[:, i] * 100).round(1)

    results["Similarity"] = similarity.ravel().round(3)

    # Match score: full credit for Good Fit, half credit for Potential Fit
    results["Match Score"] = (results["Good Fit %"] + 0.5 * results["Potential Fit %"]).round(1)

    # Best matches first
    return results.sort_values("Match Score", ascending=False)

#RUN EVERY RESUME
all_results = []
summary_rows = []

for resume_name, resume_file in RESUMES.items():
    resume_path = project_folder / "data" / resume_file
    if not resume_path.exists():
        print(f"\nSkipping {resume_name}: {resume_file} not found in the data folder")
        continue

    results = match_resume(resume_path.read_text())

    print("\n" + "=" * 70)
    print(f"{resume_name.upper()}  ({resume_file})")
    print("=" * 70)

    for rank, (_, row) in enumerate(results.head(TOP_N).iterrows(), start=1):
        print(f"{rank}. {row[TITLE_COLUMN]} at {row[COMPANY_COLUMN]}")
        print(f"   Prediction: {row['Prediction']}  |  Match Score: {row['Match Score']}"
              f"  |  Good Fit: {row['Good Fit %']}%  |  Potential Fit: {row['Potential Fit %']}%"
              f"  |  No Fit: {row['No Fit %']}%")

    counts = results["Prediction"].value_counts()
    print(f"\nAcross all {len(jobs)} jobs: "
          f"{counts.get('Good Fit', 0)} Good Fit, "
          f"{counts.get('Potential Fit', 0)} Potential Fit, "
          f"{counts.get('No Fit', 0)} No Fit")

    # Keep the results for the combined file
    results.insert(0, "Resume", resume_name)
    results.insert(1, "Rank", range(1, len(results) + 1))
    all_results.append(results)

    # Top 3 jobs for the summary table
    top3 = [f"{r[TITLE_COLUMN]} ({r[COMPANY_COLUMN]})" for _, r in results.head(3).iterrows()]
    summary_rows.append({"Resume": resume_name, "Top Match": top3[0],
                         "2nd": top3[1], "3rd": top3[2]})

# SUMMARY AND SAVE
if all_results:
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("SUMMARY: TOP 3 JOBS FOR EACH RESUME")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"\n{row['Resume']}:")
        print(f"  1. {row['Top Match']}")
        print(f"  2. {row['2nd']}")
        print(f"  3. {row['3rd']}")

    # Every job for every resume in one file (for Excel or slides)
        model_name = MODEL_FILE.stem   # e.g. "neural_network", "logreg_model", "decision_tree"
    pd.concat(all_results).to_csv(project_folder / "models" / f"{model_name}_job_match_results.csv", index=False)
    summary.to_csv(project_folder / "models" / f"{model_name}_job_match_summary.csv", index=False)
    print(f"\nFull results saved to models/{model_name}_job_match_results.csv")
    print(f"Summary saved to models/{model_name}_job_match_summary.csv")