import pandas as pd 
from pathlib import Path

#locate the main project folder 
project_folder = Path(__file__).resolve().parent.parent

#locate the CSV files 
train_path = project_folder / "data" / "train.csv"
test_path = project_folder / "data" / "test.csv"

#load both datasets 
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#display the dimensions of each dataset
print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

#combine then datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)

#Display the dimensions of the combined dataset
print("Combined dataset:", combined_df.shape)

# Check for missing values in each column 
print("\nMissing values:")
print(combined_df.isnull().sum())

# Count exact duplicate rows 
exact_duplicates = combined_df.duplicated().sum()

# Count repeated  resume-job pairs , ignoring their labels 
pair_duplicates = combined_df.duplicated(subset=['resume_text', 'job_description_text']).sum()

#Display the results 
print("\nExact duplicate rows:", exact_duplicates)
print("Repeated resume-job pairs:", pair_duplicates)

# Find all rows beloonging to repeated resume-job pairs 
duplicate_pairs = combined_df[combined_df.duplicated(subset=['resume_text', 'job_description_text'], keep=False)]

# Display the labels assigned to repeated pairs
print("\nLabels assigned to repeated resume-job pairs:")
print(duplicate_pairs['label'].value_counts())

# Count pairs that have more than one classification
conflicting_pairs = (duplicate_pairs.groupby(['resume_text', 'job_description_text'])['label'].nunique())

print("\nPairs with conflicting labels:", (conflicting_pairs > 1).sum())

# Identify pairs with conflicting classifications 
conflict_mask = (combined_df.groupby(['resume_text', 'job_description_text'])['label'].transform('nunique') > 1)

# Remove all observations belonging to conflicting pairs 
cleaned_df = combined_df.loc[~conflict_mask].copy()

# Remove exact duplicate observations, keeping only one copy
cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

#display cleaning results
print("\nOrigninal observations:", len(combined_df))
print ("Observations removed:", len(combined_df) - len(cleaned_df))
print("Cleaned observations:", len(cleaned_df))

# Define where to save the cleaned dataset
output_path = project_folder / "data" / "cleaned_resumes.csv"

# Save the cleaned dataset without the pandas index
cleaned_df.to_csv(output_path, index=False)

#confirm that the cleaned dataset has been saved
print("\nCleaned dataset saved to:", output_path)