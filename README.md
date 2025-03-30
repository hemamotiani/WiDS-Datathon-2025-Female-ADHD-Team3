### **👥 Team Members**

| Bhavya Agarwal | @bhavya632 | Model evaluation, README creation and documentation  |
| My Vo | @ | Model development and implementation, Model training using Random Forest Classifier |
| Jolie Liu| @ | Preprocessed training data and built the optimized model |
| Hema Motiani | @hemamtiani | Created GitHub repository, assisted with testing data cleaning |

---

## **🎯 Project Highlights**

**Example:**

* Built a \[multi-classifier model\ using \[Random Forest Classifier\] to solve \[Kaggle competition task\]
* Achieved an F1 score of \[insert score\] and a ranking of \[insert ranking out of participating teams\] on the final Kaggle Leaderboard
* Used \[explainability tool\] to interpret model decisions
* Implemented \[data preprocessing method\] to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)
🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## **🏗️ Project Overview**

**Describe:**

* The Kaggle competition and its connection to the Break Through Tech AI Program
* The objective of the challenge
* The real-world significance of the problem and the potential impact of your work

---

## **📊 Data Exploration**

**Describe:**

* The dataset(s) used (i.e., the data provided in Kaggle \+ any additional sources)
* Data exploration and preprocessing approaches
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## **🧠 Model Development**

**Describe (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**WiDS challenge:**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
2. How could your work help contribute to ADHD research and/or clinical care?

**AJL challenge:**

As Dr. Randi mentioned in her challenge overview, “Through poetry, art, and storytelling, you can reach others who might not know enough to understand what’s happening with the machine learning model or data visualizations, but might still be heavily impacted by this kind of work.”
As you answer the questions below, consider using not only text, but also illustrations, annotated visualizations, poetry, or other creative techniques to make your work accessible to a wider audience.
Check out [this guide](https://drive.google.com/file/d/1kYKaVNR\_l7Abx2kebs3AdDi6TlPviC3q/view) from the Algorithmic Justice League for inspiration!

1. What steps did you take to address [model fairness](https://haas.berkeley.edu/wp-content/uploads/What-is-fairness_-EGAL2.pdf)? (e.g., leveraging data augmentation techniques to account for training dataset imbalances; using a validation set to assess model performance across different skin tones)
2. What broader impact could your work have?

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---
# WiDS Datathon 2025: AI for Female Brain Health

#### Overview
Welcome to our GitHub repository for the WiDS Datathon 2025! Our team, placed together by the Break Through Tech AI Program, is participating in this global Kaggle competition to apply data science and machine learning techniques to real-world challenges in female brain health.

#### About the Competition
The WiDS Datathon (Women in Data Science Datathon) is an annual Kaggle competition designed to inspire and educate data scientists while addressing important societal challenges. Organized by Stanford’s Women in Data Science (WiDS) initiative, this year’s challenge focuses on the classification and analysis of ADHD, emphasizing sex-based differences in diagnosis and patterns.

As participants in the Break Through Tech AI Program, our team is leveraging this competition to enhance our machine learning skills, deepen our statistical analysis expertise, and contribute to meaningful research in healthcare.

#### Challenge Objective
Our mission is to analyze the WiDS Datathon 2025 ADHD dataset and uncover sex-based patterns in ADHD diagnosis. We aim to:
- Preprocess and clean the dataset to ensure high-quality inputs for model training
- Engineer relevant features to enhance the predictive power of our model
- Perform statistical analysis to identify potential sex-based differences in ADHD-related data
- Train and evaluate machine learning models, including multiclass classification algorithms, to predict ADHD patterns
- Optimize model performance through hyperparameter tuning and advanced ML techniques

#### Our Approach
Drawing on the skills we gained from the Break Through Tech AI Program, our strategy includes:
- Exploratory data analysis and preprocessing using pandas and NumPy
- Model development and evaluation with scikit-learn
- Application of deep learning techniques and hyperparameter tuning for improved performance
- Collaborative teamwork using GitHub, Kaggle Notebooks, Notion, and Google Collaboration


#### Real world Importance
The real-world significance of this problem lies in improving ADHD diagnosis, particularly for females, who are often underdiagnosed due to subtler symptoms. By leveraging fMRI data and socio-demographic factors, this model can help identify at-risk individuals early, leading to timely interventions and personalized treatments. Understanding sex-specific brain activity patterns associated with ADHD can contribute to more equitable and effective mental health care, reducing long-term challenges for undiagnosed individuals. This research also advances the broader field of neuroscience by shedding light on how neurodevelopmental disorders manifest differently across sexes, ultimately improving diagnostic tools and therapeutic strategies.

#### About the Data
- **Source:** Provided by the Healthy Brain Network (HBN) and the Reproducible Brain Charts (RBC) project, in collaboration with Ann S. Bowers Women’s Brain Health Initiative, Cornell University, and UC Santa Barbara.
- **Objective:** Predict ADHD diagnosis (0=Other/None, 1=ADHD) and sex (0=Male, 1=Female) using functional MRI and socio-demographic data.
- **Training Data (train_tsv) Includes:**
  - ADHD diagnosis and sex labels.
  - Functional MRI connectome matrices.
  - Socio-demographic, emotional, and parenting information.
- **Test Data (test_tsv) Includes:**
  - Functional MRI connectome matrices.
  - Socio-demographic, emotional, and parenting information (without labels).
- **Preprocessing Required:**
  - Handling categorical data (e.g., creating dummy variables).
  - Merging processed socio-demographic and functional MRI data for model training.
- **Data Characteristics:**
  - Size: 1.07 GB
  - Format: xlsx, csv
  - Columns: 110
  - Files: 13
- **Submission Requirement:** Predict ADHD diagnosis and sex for the test set and submit results for leaderboard ranking. 

#### Data Exploration and Preprocessing Approaches
- **Handling Missing Values:** Dropped columns with a significant percentage of missing values to maintain data quality and prevent model bias.
- **Categorical Data Processing:** Applied one-hot encoding to transform categorical features into numerical format for model compatibility.
- **Feature Scaling:** Used standardization or normalization to scale numerical features, ensuring uniform impact on the model.
- **Combining Datasets:** Merged socio-demographic, emotional, and parenting data with functional MRI connectome matrices for a comprehensive training dataset.
- **Data Splitting:** Reserved a portion of the training dataset for validation to assess model performance before final testing.



