import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import save_npz
import joblib

# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent

#Locate Cleaned dataset
data_path = project_folder / "data" / "cleaned_resumes.csv"

# Load the cleaned dataset
df = pd.read_csv(data_path)

# Display the dimensions of the cleaned dataset
print("Cleaned dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

#Check for blank or whitepace-only text
for column in ["resume_text", "job_description_text"]:
    blank_count = df[column].fillna("").str.strip().eq("").sum()
    print(f"Blank entries in {column}:{blank_count}")

# Check the distribution of our three classification labels
print("\nLabel distribution:")
print(df["label"].value_counts())

#check whether resumes appear more than once 
unique_resumes = df["resume_text"].nunique()

print("\nTotal observations:", len(df))
print("Unique resumes:", unique_resumes)
print("Repeated resume observations:", len(df) - unique_resumes)


# Create the first group-based split
splitter = GroupShuffleSplit(n_splits=1,test_size=0.15,random_state=42)

# Split using resume text as the grouping variable
train_val_idx, test_idx = next(splitter.split(df, groups=df["resume_text"]))

# Create the two datasets
train_val_df = df.iloc[train_val_idx].copy()
test_df = df.iloc[test_idx].copy()

# Display the results
print("\nTraining + Validation observations:", len(train_val_df))
print("Testing observations:", len(test_df))

print("\nUnique resumes in Training + Validation:",train_val_df["resume_text"].nunique())
print("Unique resumes in Testing:",test_df["resume_text"].nunique())


# Create the second group-based split
val_splitter = GroupShuffleSplit(n_splits=1,test_size=0.1765,random_state=42)

# Split the remaining data into training and validation
train_idx, val_idx = next(val_splitter.split(train_val_df, groups=train_val_df["resume_text"]))

# Create the final training and validation datasets
train_df = train_val_df.iloc[train_idx].copy()
val_df = train_val_df.iloc[val_idx].copy()

# Display the final dataset sizes
print("\nFinal dataset sizes:")
print("Training observations:", len(train_df))
print("Validation observations:", len(val_df))
print("Testing observations:", len(test_df))

# Display the number of unique resumes in each dataset
print("\nUnique resumes per dataset:")
print("Training:", train_df["resume_text"].nunique())
print("Validation:", val_df["resume_text"].nunique())
print("Testing:", test_df["resume_text"].nunique())


# Create sets containing the unique resumes in each dataset
train_resumes = set(train_df["resume_text"])
val_resumes = set(val_df["resume_text"])
test_resumes = set(test_df["resume_text"])

# Check whether any resumes appear in multiple datasets
train_val_overlap = train_resumes.intersection(val_resumes)
train_test_overlap = train_resumes.intersection(test_resumes)
val_test_overlap = val_resumes.intersection(test_resumes)

# Display the results
print("\nResume overlap checks:")
print("Training vs Validation:", len(train_val_overlap))
print("Training vs Testing:", len(train_test_overlap))
print("Validation vs Testing:", len(val_test_overlap))


# Check the class distribution in each dataset
for name, dataset in [("Training", train_df),("Validation", val_df),("Testing", test_df)]:   
 print(f"\n{name} label distribution:")
 print(dataset["label"].value_counts(normalize=True).round(3))


# Save the training, validation, and testing datasets
train_df.to_csv(project_folder / "data" / "train_split.csv", index=False)
val_df.to_csv(project_folder / "data" / "validation_split.csv", index=False)
test_df.to_csv(project_folder / "data" / "test_split.csv", index=False)

print("\nTraining, validation, and testing datasets saved successfully!")

# Create a shared TF-IDF vectorizer 
tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2),min_df=2)

# Combine training resumes and job descriptions 
training_text = pd.concat([train_df["resume_text"], train_df["job_description_text"]], ignore_index=True)

# Fit TF-IDF using training text only 
tfidf.fit(training_text) # pyright: ignore[reportUndefinedVariable]

#Display the number of features learned by the TF-IDF vectorizer
print("\nTF-IDF vocabulary size:",len(tfidf.vocabulary_))

#convert training resumes int TF-IDF features
X_train_resumes = tfidf.transform(train_df["resume_text"])

#convert training job descriptions into TF-IDF features
X_train_job_descriptions = tfidf.transform(train_df["job_description_text"])

#display the dimensions of both metrices 
print("\nTraining resume TF-IDF shape:", X_train_resumes.shape)
print("Training job TF-IDF shape:", X_train_job_descriptions.shape)


# Transform validation resumes and job descriptions
X_val_resumes = tfidf.transform(val_df["resume_text"])
X_val_job_descriptions = tfidf.transform(val_df["job_description_text"])

# Transform testing resumes and job descriptions
X_test_resumes = tfidf.transform(test_df["resume_text"])
X_test_job_descriptions = tfidf.transform(test_df["job_description_text"])

# Display the dimensions
print("\nValidation resume shape:", X_val_resumes.shape)
print("Validation job shape:", X_val_job_descriptions.shape)

print("\nTesting resume shape:", X_test_resumes.shape)
print("Testing job shape:", X_test_job_descriptions.shape)


# Calculate cosine similarity for each resume-job pair
train_similarity = X_train_resumes.multiply(X_train_job_descriptions).sum(axis=1).A1
val_similarity = X_val_resumes.multiply(X_val_job_descriptions).sum(axis=1).A1
test_similarity = X_test_resumes.multiply(X_test_job_descriptions).sum(axis=1).A1

# Display the first five training similarity scores
print("\nFirst five training similarity scores:")
print(train_similarity[:5])


# Calculate matching TF-IDF features for each resume-job pair
X_train_match = X_train_resumes.multiply(X_train_job_descriptions).tocsr()

X_val_match = X_val_resumes.multiply(X_val_job_descriptions).tocsr()

X_test_match = X_test_resumes.multiply(X_test_job_descriptions).tocsr()

# Display the dimensions of the matching feature matrices
print("\nTraining matching features:", X_train_match.shape)
print("Validation matching features:", X_val_match.shape)
print("Testing matching features:", X_test_match.shape)


# Combine matching TF-IDF features with cosine similarity
X_train = hstack([X_train_match,train_similarity.reshape(-1, 1)]).tocsr()
X_val = hstack([X_val_match,val_similarity.reshape(-1, 1)]).tocsr()
X_test = hstack([X_test_match,test_similarity.reshape(-1, 1)]).tocsr()

# Display the final feature matrix dimensions
print("\nFinal training features:", X_train.shape)
print("Final validation features:", X_val.shape)
print("Final testing features:", X_test.shape)


# Saving the final feature matrices
save_npz(project_folder / "data" / "X_train.npz", X_train)
save_npz(project_folder / "data" / "X_val.npz", X_val)
save_npz(project_folder / "data" / "X_test.npz", X_test)

print("\nAll feature matrices saved successfully!")


# Save the fitted TF-IDF vectorizer
vectorizer_path = project_folder / "models" / "tfidf_vectorizer.joblib"

joblib.dump(tfidf, vectorizer_path)

print("\nTF-IDF vectorizer saved successfully!")
