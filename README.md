# RedBrick Hacks 2022 Submission Repository

## Walkthrough of the Repository



## Inspiration & Aim

As a very personal project for all of our team members, we worked hard in developing a realistic solution for our elderly. With personal experience and understanding how our own grandparents struggle with technology, we approached the project while keeping this in mind and understanding the actual needs of those who face technical difficulties.

A result of the RedBricks Hackathon 2022, this project aims to assist the elderly with their preliminary diagnosis and factors affecting Heart Diseases, Diabetes, Common Illnesses and Mental Illness through a simplistic, intuitive and easy to comprehend UI. Along with this, we aim to spread awareness about these diseases through the program. Furthermore, it also helps the rural population in recognising illnesses that might turn fatal if ignored or untreated. In such a way, the program guides those who don't have sufficient facilities for a checkup. Primarily, our intention is to help the users in case of any emergency or help them recogise these illnesses as a premptive measure to fight it. We accomplish this through a comprehensive yet simple questionnaire along with a straightforward chatbot trained with thousands of entries from official databases.


## Problem Definition 

Identifying key health indicators that significantly affect heart diseases so that we can detect and prevent the factors having the greatest impact. 
1) What risk factors are most predictive of heart disease risk?
2) Can we use a subset of the risk factors to accurately build a webapp that can predict whether an individual has heart disease?



## Datasets

1. The Behavioral Risk Factor Surveillance System (BRFSS) is the nation's premier system of health-related telephone surveys that collect state data about U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.

https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system

2. The Princeton University's Health Services center was a key factor in our diagnostics for our chatbot. The symptomps provided by the user help in the prediction of analysing the appropriate common illness. The Baymax chatbot, through its rigourous training, attempts its best to provide the most accurate diagnosis of the user. Utilizing the Python Chatbot Project from Data Flair, we were able to achieve our desired goal of an efficient and easy-to-use chatbot for the elderly.

https://uhs.princeton.edu/health-resources/common-illnesses
https://data-flair.training/blogs/python-chatbot-project/

3. Our ML model for diagnosing Mental Illness helps the user get a preliminary idea of the mental illness they might be facing. Again, trained by an extensive dataset from the Mental Health in Tech Survey, the state data collected helps diagnosing the user with verified and authentic previously obtained data. This is once again done through a skillfully selected questionnaire.

https://www.kaggle.com/code/kairosart/machine-learning-for-mental-health-1/data?select=survey.csv


## Cleaning the Dataset 
- The BRFSS Dataset initially had 441456 rows and 330 columns
- Shortlisted 22 columns most suited for heart diseases
- Dropped missing values. 97850 rows removed.
- Changing previously Ordinal variable to Categorical variable (Binary yes-no)
- Making ordering of Ordinal variable to become more precise by removing the unwanted statistical data
- Final 21 variables listed as Blood pressure, cholesterol, how recent is cholesterol check, BMI, smoke activity, stroke, diabetes, physical activity frequency, eat fruits, eat veggies, amount of alcohol consumption, registered healthcare insurance, financial problems for medical visits, general health, mental health, physical health, difficulty walking/climbing stairs, sex, age, education level, income level.

## Machine Learning Model
- Decided to use logistics regression after extensive research and consideration for other machine learning models
- Extracted response and predictors. Assigned Y as dependent variable (HeartDiseaseorAttack) and X as 21 independent variables dataframe
- Split dataset into random train and test with ratio of 0.25
- Imported modules such as preprocessing and pipeline from sklearn to scale the model
- Ran the model and dumped it into a PKL file using joblib. 
- Used spyder to create streamlit application where we use the ML Model (PKL file) to predict heart disease risk.

- The accuracy of the model came out to be around 89% which shows our model is not only ideal but also realistic.
- The model showed 1.4% false positives.

## Final Implementation & Combination of Model (FrontEnd)


## Scope

## What did we learn from this project?

1) Use of Streamlit for deploying webapp and incorporating a beautiful interface for our ML model pickled using joblib
2) Use of methods such as predict and predict_proba in logistic regression
3) Exploring Sklearn modules such as Pipeline , Standard scaler for scaling and further exploration of seaborn as a tool for visualization
4) Collaborating using GitHub
5) Cleaning and Curating datasets and binary and multiclass classification


## References
1) https://www.hindawi.com/journals/misy/2022/1410169/
2) https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system
3) https://slidesgo.com/theme/data-science-consulting
4) https://www.webmd.com/heart/heart-health-tips 
5) https://www.health.harvard.edu/healthbeat/10-small-steps-for-better-heart-health 
6) https://www.healthxchange.sg/heart-lungs/heart-disease/how-to-improve-heart-health-naturally
7) Buttar, H. S., Li, T., & Ravi, N. (2005). Prevention of cardiovascular diseases: Role of exercise, dietary interventions, obesity and smoking cessation. Experimental and clinical cardiology, 10(4), 229–249.
8) https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
9) https://www.cdc.gov/pcd/issues/2019/19_0109.html
10) https://docs.streamlit.io/library/api-reference


