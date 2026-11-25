# Resume and Job Description Matching Classifier

## Project Overview

This project was developed for BUS 445 – Advanced Data Analytics. The objective was to develop a machine learning model capable of evaluating how well an applicant's resume matches a job description.

The models classify resume and job description pairs into three categories:

- Good Fit
- Potential Fit
- No Fit

The purpose of this project is to explore how machine learning can assist businesses with applicant screening and improve the efficiency of the recruitment process.

## Dataset

The dataset was obtained from Hugging Face:

[Resume Job Description Fit Dataset](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

The original dataset contained 8,000 observations, 643 unique resumes, and three variables:

- `resume_text`
- `job_description_text`
- `label`

## Data Cleaning and Preprocessing

The following steps were performed to prepare the dataset:

- Checked for missing values and duplicate observations.
- Removed one duplicate observation.
- Removed 12 observations associated with six conflicting-label pairs.
- Retained 7,987 cleaned observations.
- Split the dataset by unique resumes to prevent data leakage between training, validation, and testing sets.

| Dataset | Observations |

| Training | 5,611 |
| Validation | 1,092 |
| Testing | 1,284 |
| Total | 7,987 |

Term Frequency–Inverse Document Frequency (TF-IDF) was used to convert resume and job description text into numerical features. Cosine similarity was used to measure textual similarity between resumes and job descriptions.

## Analytical Approach

Three machine learning models were developed and evaluated:

- Logistic Regression
- Decision Tree
- Neural Network

All three models were developed using Python and scikit-learn.

Because approximately half of the observations were classified as No Fit, both accuracy and macro F1 were used to evaluate model performance.

The following performance goals were established:

- Outperform the majority-class baseline.
- Achieve at least 60% accuracy.
- Achieve a macro F1 score of at least 0.55.

Additionally, 40 real-world job postings were collected and evaluated against three sample resumes to demonstrate the practical application of the models. These job postings were not used during training or validation.

## Model Results

| Model | Test Accuracy | Macro F1 |

| Logistic Regression | 65.0% | 0.649 |
| Decision Tree | 59.2% | 0.554 |
| Neural Network | 68.1% | 0.663 |

The neural network achieved the highest overall accuracy and macro F1 score and was selected as the best-performing model.

### Final Neural Network Configuration

| Parameter | Value |

| Hidden Layers | 1 |
| Hidden Neurons | 15 |
| Activation Function | Logistic |
| Solver | LBFGS |
| Alpha | 0.2 |
| Maximum Iterations | 500 |
| Random State | 42 |

## Repository Structure

- `data/` – Project datasets and processed data.
- `models/` – Saved machine learning models.
- `scripts/` – Python scripts for data preparation, model development, and evaluation.
- `requirements.txt` – Required Python libraries.
- `BUS445presentation.pptx` – Project presentation.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib
- Matplotlib
- Visual Studio Code
- GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/rcorona26/BUS445_GROUP-1.git
```

Navigate to the project directory:

```bash
cd BUS445_GROUP-1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Limitations

- The original dataset labels were not independently verified.
- The dataset contained a limited number of unique resumes.
- Class imbalance affected model performance.
- The models had difficulty distinguishing between certain classification categories.
- The real-world job postings did not contain verified classification labels.

## Conclusion

The results demonstrate that machine learning can assist businesses with resume screening and job matching. Although the neural network achieved the strongest overall performance, human judgment remains necessary when making hiring decisions.

This project is intended for academic and educational purposes.
