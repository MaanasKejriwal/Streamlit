# RedBrick Hacks 2022 Submission Repository

## Walkthrough of the Repository

- main2.py: 
This is our main file. It is the python program containing all the functions, and is launched when the app is run. It extracts all the required details from the other supplementary file, and is the backbone of the app. 

- chatbot_train.py:
This file contains the training code for the chatbot. It does not need to be run again, and is attached for reference.

- logReg_model.pkl:
The trained ML model for heart diagnosis is contained within this pickled file.

- logReg3_model.pkl:
The trained ML model for diabetes diagnosis is contained within this pickled file.

- Mental_Health_model5.pkl:
The trained ML model for mental health diagnosis is contained within this pickled file.

- model.h5:
The trained ML model for the chatbot is contained within this pickled file.

- Ashoka_Final.ipynb:
It contains the visualisations, models, and cleaning of the datasets

## Inspiration & Aim

As a very personal project for all of our team members, we have worked hard in developing a realistic solution for the elderly members of our society along with the rural sections of India. With personal experience and understanding how our own grandparents struggle with technology, we approached the project while keeping this in mind and understanding the actual needs of those who face technical difficulties.

A result of the RedBricks Hackathon 2022, this project aims to assist the elderly with their preliminary diagnosis and factors affecting Heart Diseases, Diabetes, Common Illnesses and Mental Illness through a simplistic, intuitive and easy to comprehend UI.Furthermore, it also helps the rural population in recognising illnesses that might turn fatal if ignored or untreated and reach the nearest docotor in emergencies. In such a way, the program guides those who don't have sufficient facilities for a checkup. Primarily, our intention is to help the users in case of any emergency or help them recognise these illnesses as a premptive measure to fight it. We accomplish this through a comprehensive yet simple questionnaire along with a straightforward chatbot trained with thousands of entries from official databases.


## Problem Definition & Code Functionality 

Our Code aims to solve 4 main problems and provides the perfect streamlit interface for that 
1) Using an AI Trained Chatbot to converse with elderly patients and diagnose illnesses through symptoms provided  
2) Used a dataset of the risk factors to accurately build a webapp that can predict the probability of an individual getting heart disease?
3) Building a machine learning model to accurately build a webapp that can predict the probability of an individual getting diabetes?
4) Building a machine learning model using logistic regression that can predict the probability of an individual being susceptible to mental health problems.


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
- Similarly the diabetes and the mental health datasets were individually cleaned, curated, and finalized.

## Machine Learning Model
- Decided to use logistics regression after extensive research and consideration for other machine learning models
- Extracted response and predictors. Assigned Y as dependent variable (HeartDiseaseorAttack) and X as 21 independent variables dataframe while predicting heart disease probability.
- Split dataset into random train and test with ratio of 0.25
- Imported modules such as preprocessing and pipeline from sklearn to scale the model
- Ran the model and dumped it into a PKL file using joblib. 
- Used spyder to create streamlit application where we use the ML Model (PKL file) to predict heart disease risk.
- The accuracy of the models of heart disease, diabetes, and mental health ranged from 75% to 89% which shows our model is not only ideal but also realistic.

## Final Implementation & Future Scope

- The final implementation is done through streamlit where a webapp is successfully deployed that has all 3 of our ML Models and the chatbot.
- We would like to work on this project beyond the hackathon and would like to work on the full stack web development of this webapp so that we can connect databases which will allow elderly citizens and rural patients to send their data to their nearest doctor and contact emergency services if required.
- We would also like to further train our chatbot and work on different ML Model apart from logistic regressions for even greater accuracy.


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
11) https://data-flair.training/blogs/python-chatbot-project/


