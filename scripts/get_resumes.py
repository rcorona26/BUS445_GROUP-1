import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
test_df = pd.read_csv(project_folder / "data" / "test_split.csv")

# Pick 5 different resumes from the test set
resumes = test_df["resume_text"].drop_duplicates().sample(5, random_state=1)

for i, text in enumerate(resumes, start=1):
    (project_folder / "data" / f"resume_{i}.txt").write_text(text)
    print(f"resume_{i}.txt:", text[:100].replace("\n", " "), "...")