#### Welcome to our GitHub repository for the WiDS Datathon 2025! Our team, placed together by the Break Through Tech AI Program, is participating in this global Kaggle competition to apply data science and machine learning techniques to real-world challenges in female brain health.

---

### **👥 Team Members**

| Team Members     | GitHub Handle      | Contribution                                                                |
|------------------|--------------------|-----------------------------------------------------------------------------|
| Bhavya Agarwal   | [@bhavya632](https://github.com/bhavya632)         | Model evaluation, README creation and documentation                         |
| My Vo            | [@myvok](https://github.com/myvok)            | Model development and implementation, Model training using Random Forest Classifier |
| Jolie Liu        | [@Juwols1088](https://github.com/Juwols1088)        | Preprocessed training data and built the optimized model                    |
| Hema Motiani     | [@hemamotiani](https://github.com/hemamotiani)       | Created GitHub repository, assisted with testing data cleaning              |


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


* The Datasets: The datasets used are in the `/Datasets` folder. 

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

**Visualizations of the Data:**

* <img width="380" alt="Screenshot 2025-03-30 at 7 50 46 PM" src="https://github.com/user-attachments/assets/3c677990-c9eb-49e0-b451-96c19138d1e4" />
* <img width="690" alt="Screenshot 2025-03-30 at 7 49 55 PM" src="https://github.com/user-attachments/assets/96244cee-01c4-4691-afac-e1c59534aac3" />
* <img width="689" alt="Screenshot 2025-03-30 at 7 51 48 PM" src="https://github.com/user-attachments/assets/43fcd359-8e1c-4f33-a6e4-56b577f009a6" />

---

## **🧠 Model Development**

* In this competition, our team applied machine learning techniques to classify ADHD patterns using brain imaging and socio-demographic data. We used a Random Forest Classifier for this multi-class classification problem, as it has shown robustness in handling a mix of categorical and numerical features. The model was designed to predict ADHD diagnosis while considering gender-based differences in brain development. Additionally, we developed and optimized a XGBoost model, which is known for its performance and efficiency in classification tasks. XGBoost was applied as an alternative to Random Forest due to its ability to handle imbalances in the dataset and improve prediction accuracy through boosting techniques.

* Training setup:
  - _Data Split:_ We split the dataset into 80% training and 20% validation to ensure robust model performance. Cross-validation was also used to fine-tune the model's parameters and validate its generalization ability.
  - _Evaluation Metric:_ We used the F1-score as our evaluation metric to balance precision and recall, ensuring that the model’s performance was measured for both accuracy and reliability, especially given the class imbalance in ADHD diagnoses.
  - _Baseline Performance:_ The baseline performance was set with a simple logistic regression model to establish a starting point before experimenting with more complex models.



---

## **📈 Results & Key Findings**

* Performance Metrics:
   - F1-score: 0.69
   - Kaggle Leaderboard Rank: #519
   - The model showed promising performance in predicting ADHD patterns but there is room for further improvements in accuracy.
 
* Key Findings:
   - The Random Forest model showed the best balance between model complexity and performance, indicating that ensemble methods are effective in tackling this problem with mixed data types.
   - The importance of socio-demographic data such as age, gender, and emotional history was evident in improving the model’s ability to predict ADHD more accurately, especially in females, who are often underdiagnosed.

---

## **🖼️ Impact Narrative**

1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?
   - The model revealed notable brain connectivity differences between males and females diagnosed with ADHD, highlighting how male and female brains might develop differently in regions involved in attention and emotion regulation. Female ADHD patients often showed more subtle connectivity disruptions, which may contribute to the underdiagnosis of ADHD in women.
2. How could your work help contribute to ADHD research and/or clinical care?
   - By accurately predicting ADHD in both males and females using brain imaging and socio-demographic data, our model can aid in identifying at-risk individuals earlier. Early detection and personalized treatments based on sex-specific brain activity patterns could lead to more effective ADHD interventions, especially for females who often face challenges in receiving timely diagnoses and treatments.

---

## **🚀 Next Steps & Future Improvements**

**Address the following:**

* What are some of the limitations of your model?
  - _Data Quality:_ Some columns had excessive missing values, and although we dropped them, this could have potentially removed valuable information.
  - _Complexity of Features:_ Some features, particularly from the fMRI data, were difficult to interpret, which may have impacted the model’s ability to make accurate predictions.

* What would you do differently with more time/resources?
  - _Feature Engineering:_ We would explore more advanced feature engineering techniques to capture additional insights from the fMRI data, potentially involving more complex transformations or additional data sources.
  - _Deep Learning Models:_ With more time, we would experiment with deep learning models like Convolutional Neural Networks (CNNs), which could better capture the spatial relationships in brain connectivity data.
* What additional datasets or techniques would you explore?
  - _Time-series Data:_ Incorporating time related data to analyze how ADHD symptoms evolve over time would provide deeper insights.
  - _Neuroimaging Modalities:_ Exploring additional neuroimaging techniques (e.g., EEG, PET scans) could offer a more holistic view of brain activity patterns.
  - _Fairness and Explainability Tools:_ Implementing tools for model explainability (e.g., SHAP values) to better understand the decision-making process of the model and ensure fairness across different demographic groups.



---

For any questions or contributions, please reach out to the project maintainers or open an issue in this repository.
