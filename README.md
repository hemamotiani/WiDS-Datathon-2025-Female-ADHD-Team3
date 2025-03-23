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



