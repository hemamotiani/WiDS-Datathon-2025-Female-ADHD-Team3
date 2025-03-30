#### Welcome to our GitHub repository for the WiDS Datathon 2025! Our team, placed together by the Break Through Tech AI Program, is participating in this global Kaggle competition to apply data science and machine learning techniques to real-world challenges in female brain health.

---

### **👥 Team Members**

| Team Members     | GitHub Handle      | Contribution                                                                 |
|------------------|--------------------|-----------------------------------------------------------------------------|
| Bhavya Agarwal   | @bhavya632         | Model evaluation, README creation and documentation                         |
| My Vo            | @<GitHub Handle>   | Model development and implementation, Model training using Random Forest Classifier |
| Jolie Liu        | @<GitHub Handle>   | Preprocessed training data and built the optimized model                    |
| Hema Motiani     | @hemamtiani        | Created GitHub repository, assisted with testing data cleaning              |


---

## **🎯 Project Highlights**

**Example:**

* Built a _multi-classifier model_ using _Random Forest Classifier_ with the goal of _developing predictive models using brain imaging data to diagnose adolescent ADHD, aiming to shed light on how brain development differs between males and females._
* Achieved an F1 score of _0.69_ and a ranking of _#519_ on the final Kaggle Leaderboard
* Applied _data preprocessing techniques_ such as handling missing values, feature scaling, encoding categorical variables, and feature selection to optimize performance.
* Utilized _NumPy and Pandas_ for efficient data manipulation and analysis.

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

* Prerequisites: 
- Python 3.8+

* Clone the Repository:
  - First, you need to clone the repository to your local machine. Use the following command:
    ```bash
    git clone https://github.com/hemamotiani/WiDS-Datathon-2025-Female-ADHD-Team3.git
    ```
  - This will create a folder with the repository’s files on your local machine.

* Install dependencies:
   - Install Python: https://kinsta.com/knowledgebase/install-python/
   - Install pip: https://pip.pypa.io/en/stable/installation/
   - Install the following - 
     ``` bash
     pip install pandas numpy matplotlib seaborn joblib scikit-learn xgboost
     ```


* The Datasets: Place the sample datasets in the `/datasets` folder. Example files:
   - `Crash_data_LA_county.csv`

* How to run the notebook or scripts:
   1. _Open Jupyter Notebook:_ Launch Jupyter Notebook in the cloned project directory by running the following command:
      ```bash
      jupyter notebook
      ```
   2. _Run the notebook or Python scripts:_ Open the notebook (.ipynb) or Python scripts in Jupyter and run them in sequence. If you're using a Python script directly, you can run it from the command line like this:
     ```bash
     python model.py
     ```

---

## **🏗️ Project Overview**

**Describe:**

* The WiDS Datathon (Women in Data Science Datathon) is an annual Kaggle competition designed to inspire and educate data scientists while addressing important societal challenges. Organized by Stanford’s Women in Data Science (WiDS) initiative, this year’s challenge focuses on the classification and analysis of ADHD, emphasizing sex-based differences in diagnosis and patterns.
As participants in the Break Through Tech AI Program, our team is leveraging this competition to enhance our machine learning skills, deepen our statistical analysis expertise, and contribute to meaningful research in healthcare.

* Our mission is to analyze the WiDS Datathon 2025 ADHD dataset and uncover sex-based patterns in ADHD diagnosis. We aim to:
  - Preprocess and clean the dataset to ensure high-quality inputs for model training
  - Engineer relevant features to enhance the predictive power of our model
  - Perform statistical analysis to identify potential sex-based differences in ADHD-related data
  - Train and evaluate machine learning models, including multiclass classification algorithms, to predict ADHD patterns
  - Optimize model performance through hyperparameter tuning and advanced ML techniques
  
* The real-world significance of this problem lies in improving ADHD diagnosis, particularly for females, who are often underdiagnosed due to subtler symptoms. By leveraging fMRI data and socio-demographic factors, this model can help identify at-risk individuals early, leading to timely interventions and personalized treatments. Understanding sex-specific brain activity patterns associated with ADHD can contribute to more equitable and effective mental health care, reducing long-term challenges for undiagnosed individuals. This research also advances the broader field of neuroscience by shedding light on how neurodevelopmental disorders manifest differently across sexes, ultimately improving diagnostic tools and therapeutic strategies.

---

## **📊 Data Exploration**

**Describe:**

* About the Data
  - _Source:_ Provided by the Healthy Brain Network (HBN) and the Reproducible Brain Charts (RBC) project, in collaboration with Ann S. Bowers Women’s Brain Health Initiative, Cornell University, and UC Santa Barbara.
  - _Training Data (train_tsv) Includes:_
    - ADHD diagnosis and sex labels.
    - Functional MRI connectome matrices.
    - Socio-demographic, emotional, and parenting information.
  - _Test Data (test_tsv) Includes:_
    - Functional MRI connectome matrices.
    - Socio-demographic, emotional, and parenting information (without labels).
  - _Preprocessing Required:_
    - Handling categorical data (e.g., creating dummy variables).
    - Merging processed socio-demographic and functional MRI data for model training.
  - _Data Characteristics:_
    - Size: 1.07 GB
    - Format: xlsx, csv
    - Columns: 110
    - Files: 13
    
* Data Exploration and Preprocessing Approaches
  - _Handling Missing Values:_ Dropped columns with a significant percentage of missing values to maintain data quality and prevent model bias.
  - _Categorical Data Processing:_ Applied one-hot encoding to transform categorical features into numerical format for model compatibility.
  - _Feature Scaling:_ Used standardization or normalization to scale numerical features, ensuring uniform impact on the model.
  - _Combining Datasets:_ Merged socio-demographic, emotional, and parenting data with functional MRI connectome matrices for a comprehensive training dataset.
  - _Data Splitting:_ Reserved a portion of the training dataset for validation to assess model performance before final testing.
  
* Challenges and assumptions when working with the datasets:
  - _Large dataset:_ Managing computational constraints while processing 1.07 GB of data.
  - _Mixed data types:_ Had to preprocess both categorical and numerical data via one-hot encoding, scaling, and normalization.
  - _Handling missing values:_ Strategically dropped columns with excessive missing data while preserving essential information.

**Visualizations:**

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

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## **🖼️ Impact Narrative**

**Answer the relevant questions below based on your competition:**

**WiDS challenge:**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
2. How could your work help contribute to ADHD research and/or clinical care?

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

For any questions or contributions, please reach out to the project maintainers or open an issue in this repository.
