import streamlit as st
import streamlit_chat
import nltk 
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from keras.models import load_model
model = load_model('model.h5')
import numpy as np
import json
import pickle
import random
import PIL
import pandas as pd
import joblib

intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        #breaks sentence into syllables
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        #lemmatizing = breaking into root word using inbuilt dictionary. e.g.: rocks- rock
        return(sentence_words)  

def matrix(sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = clean_up(sentence)
        # The bag contains words
        bag = [0]*len(words)
        #matrix contains number of elements = vocabulary, preset value=0
        for s in sentence_words:
            #traverses root words
            for i,w in enumerate(words):
                #i is roll no/dir no
                #w is unique word
                #makes directory, gives a 'roll no' to each word. If 'cramping' is entered, directory till cramping prints along w roll number, then matrix with 0s other than one 1 (one being element number=roll no of cramping)
                if w == s:
                    # assign 1 if the word is in the right position
                    bag[i] = 1
                    if show_details:
                        #will give name of bag of unique base word the entered word is found in
                        print ("found in bag: %s" % w)
        return(np.array(bag))

def predict_class(sentence, model):
        # filter out predictions below a threshold probability
        pred= matrix(sentence, words,show_details=False)
        res = model.predict(np.array([pred]))[0]
        filterthresh = 0.25
        global results
        results = [[i,r] for i,r in enumerate(res) if r>filterthresh]
        global results1
        results1 = [[i,r] for i,r in enumerate(res)]
        print(results)

        #for guesses above threshold
        f=open('r.txt','w')
        #for all guesses
        f1=open('s.txt','w')
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        results1.sort(key=lambda x: x[1], reverse=True)
        pr=results1[0]
        global pp
        pp=pr[1]
        print(pp)
        global return_list
        return_list = []
        global return_list1
        return_list1=[]
        for r in results1:
            return_list1.append({"intent": classes[r[0]], "probability": str(r[1])})
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        for x in return_list1:
            f1.write(str(x))
        for x in return_list:
            print(x)
            f.write(str(x))
        return return_list
def getResponse(ints, intents_json):
        global tag
        tag = ints[0]['intent']
        print(tag)
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result
def chatbot_response(msg):
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res


with st.sidebar:
    add_radio = st.radio(
        "What would you like to do?",
        ("Go to the home screen","Talk to Baymax", "Calculate heart risk","Calculate diabetes risk",'Calculate susceptibility to mental distress')
    )

if add_radio=='Go to the home screen':
    st.title('Title')
    st.subheader("A result of RedBricks Hacks 2022, this project aims to assist the elderly with their preliminary diagnosis and factors affecting Heart Diseases, Diabetes, Common Illnesses and Mental Illness through a simplistic, intuitive and easy to comprehend UI. We accomplish this through a comprehensive yet simple questionnaire along with a straightforward chatbot trained with thousands of entries from official databases.")

if add_radio=='Talk to Baymax':
    st.title('Baymax')
    st.subheader('A conversational chatbot that can help you identify common illnesses')
    st.write()
    chatbot_input=st.text_input('Enter message:')
    streamlit_chat.message('We can just talk, or ask me what I can do!')
    if chatbot_input:
        streamlit_chat.message(chatbot_input, is_user=True)
        res=chatbot_response(chatbot_input)
        streamlit_chat.message(res) 

if add_radio=="Calculate heart risk":
    st.title("Heart Disease Risk Prediction App")

    st.write(""" Welcome to our Heart Disease Risk Prediction App. Our aim is to use this platform to save lives by 
             alerting people of potential risks by analysing their lifestyle practices. We have collected/cleaned BRFSS 2015
             data which had inputs from over 400,000 people who were asked more than 330 questions. 
             We built a machine learning model to predict heart disease risk, and saved/exported the model for use in a Streamlit
             web app. Our model uses over 22 columns and more than 200,000 value rows after being cleaned.""")
      
    image1=PIL.Image.open('heart.jpg')
    st.image(image1,width=500)
    
    
    
    st.write("### Answer the following 21 Questions: ")
    
    # create the colums to hold user inputs
    col1, col2, col3 = st.columns((3,3,3))
    
    # gather user inputs
    
    # 1. HighBP
    highbp = col1.selectbox(
        "1. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that you have high Blood Pressure?",
        ('Yes', 'No'), index=0 )
    
    # 2. HighChol
    highchol = col2.selectbox(
        "2. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your Blood Cholesterol is high?",
        ('Yes', 'No'), index=1)
    
    # 3. CholCheck
    cholcheck = col3.selectbox(
        "3. About how long has it been since you last had your blood cholesterol checked? Put yes if less than 5 years, No if more than 5 years",
        ('Yes', 'No'), index=1)
    
    
    
    
    # 4.BMI 
    bmi = col1.number_input(
        '4. Enter your BMI : ', min_value=5, max_value=50, value=21)
    
    # 5. Smoke
    smoker = col2.selectbox(
        "5. Have you smoked atleast 100 cigarettes in your life?",
        ('Yes', 'No'), index=1)
    
    # 6. Stroke
    stroke = col3.selectbox(
        "6. Have you ever had stroke in your life?",
        ('No', 'Yes'), index=0)
    
    #7. Diabetes
    diabetes = col1.selectbox(
        "7. Have you ever had Diabetes in your life?",
        ('No Diabetes', 'Pre Diabetes', 'Diabetes'), index=0)
    
    #8.PhysActivity
    physActivity= col2.selectbox(
        "8. Are you physically active in your life?",
        ('No', 'Yes'), index=1)
    
    #9.Fruits
    fruits= col3.selectbox(
        "9. Do you have more than 1 fruit a day?",
        ('No', 'Yes'), index=1)
    
    #10.Veggies
    vegetables = col1.selectbox(
        "10. Do you have more than 1 vegetable a day?",
        ('No', 'Yes'), index=1)
    
    #11.Alcohol consumption
    Alcohol= col2.selectbox(
        "11. Are you a heavy drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)?",
        ('No', 'Yes'), index=0)
    
    #12.Alcohol consumption
    healthplan= col3.selectbox(
        "12. Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?",
        ('No', 'Yes'), index=1)
    
    #13.Alcohol consumption
    medcost= col1.selectbox(
        "13. Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
        ('No', 'Yes'), index=1)
    
    # 14. GenHlth
    genhealth = col2.selectbox("14. General Health: How would you rank your General Health on a scale from 1 = Excellent to 5 = Poor? Consider physical and mental health.",
                             ('Excellent', 'Very Good', 'Good', 'Fair', 'Poor'), index=3)
    
    #15.MentalHealth
    menhlth = col3.number_input(
        '15. Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? : '
            , min_value=0, max_value=30, value=10)
    
    #16.PhysicalHealth
    Physhlth = col1.number_input(
        '16. Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? : '
            , min_value=0, max_value=30, value=10)
    
    #17.Diff Walking
    Diffwalk= col2.selectbox(
        "17. Did you have trouble walking in last 30 days?",
        ('No', 'Yes'), index=0)
    
    
    #18.Sex
    Sex= col3.selectbox(
        "18. Sex M/F:",
        ('Female', 'Male'), index=1)
    
    # 19. Age
    age = col1.selectbox(
        '19. Select your Age:', ('Age 18 to 24',
                                'Age 25 to 29',
                                'Age 30 to 34',
                                'Age 35 to 39',
                                'Age 40 to 44',
                                'Age 45 to 49',
                                'Age 50 to 54',
                                'Age 55 to 59',
                                'Age 60 to 64',
                                'Age 65 to 69',
                                'Age 70 to 74',
                                'Age 75 to 79',
                                'Age 80 or older'), index=4)
    
    #20.education level
    educa = col2.number_input(
        '20. What is your education level with 1 being to never attended school , 6 being graduated college? : '
            , min_value=0, max_value=6, value=4)
    
    #21. Income
    Income= col3.number_input(
        '21. What is your income level with 1 being less than 10,000$ to 8 being more than 75000$? : '
            , min_value=1, max_value=8, value=4)
    
    
    
    
    # Create dataframe:
    df1 = pd.DataFrame([[highbp,highchol,cholcheck,bmi,smoker,stroke,diabetes,physActivity,fruits,
                         vegetables,Alcohol,healthplan,
                         medcost,genhealth,menhlth,Physhlth,Diffwalk,Sex,age,educa,Income]], 
                        columns=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","Diabetes","PhysActivity","Fruits","Veggies",
                                 "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlt","PhysHlth","DiffWalk"
                                 ,"Sex","Age","Education","Income"])
    
    
    
    
    def prepare_df (df):
        
        # Age
        df['Age'] = df['Age'].replace({'Age 18 to 24': 1, 'Age 25 to 29': 2, 'Age 30 to 34': 3, 'Age 35 to 39': 4, 'Age 40 to 44': 5, 'Age 45 to 49': 6,
                                   'Age 50 to 54': 7, 'Age 55 to 59': 8, 'Age 60 to 64': 9, 'Age 65 to 69': 10, 'Age 70 to 74': 11, 'Age 75 to 79': 12, 'Age 80 or older': 13})
    
        # HighChol
        df['HighChol'] = df['HighChol'].replace({'Yes': 1, 'No': 0})
        
        # HighBP
        df['HighBP'] = df['HighBP'].replace({'Yes': 1, 'No': 0})
        
        # GenHlth
        df['GenHlth'] = df['GenHlth'].replace(
        {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5})
        
        df['CholCheck'] = df['CholCheck'].replace({'Yes': 1, 'No': 0})
            
        df['Smoker'] = df['Smoker'].replace({'Yes': 1, 'No': 0})
        
        df['Stroke'] = df['Stroke'].replace({'Yes': 1, 'No': 0})
    
        df['Diabetes'] = df['Diabetes'].replace({'Diabetes':2,'Pre Diabetes': 1, 'No Diabetes': 0})
        
        df['Fruits'] = df['Fruits'].replace({'Yes': 1, 'No': 0})
    
        df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({'Yes': 1, 'No': 0})
    
        df['PhysActivity'] = df['PhysActivity'].replace({'Yes': 1, 'No': 0})
    
        df['Veggies'] = df['Veggies'].replace({'Yes': 1, 'No': 0})
    
        df['AnyHealthcare'] = df['AnyHealthcare'].replace({'Yes': 1, 'No': 0})
    
        df['NoDocbcCost'] = df['NoDocbcCost'].replace({'Yes': 1, 'No': 0})
    
        df['DiffWalk'] = df['DiffWalk'].replace({'Yes': 1, 'No': 0})
    
        df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})   
        
    
        return df
    
    
    
    df=prepare_df(df1)
    
    
    
    log_reg_model = joblib.load("logReg_model.pkl")
    
    if st.button('Click here to predict Heart Disease Risk'):
    
        # make the predictions
        prediction = log_reg_model.predict(df)
        prediction_probability = log_reg_model.predict_proba(df)
    
    
        if prediction == 0:
            st.balloons()
            st.success(f"**The probability that you'll have"
                    f" heart disease is {round(prediction_probability[0][1] * 100, 2)}%."
                    f" You are healthy!**")
       
        else:
            st.error(f"**The probability that you will have"
                    f" heart disease is {round(prediction_probability[0][1] * 100, 2)}%."
                    f" It sounds like you are not healthy. "
                    f" You should refer to these sites to work towards improving your health"
                    f" https://www.webmd.com/heart/heart-health-tips "
                    f" https://www.health.harvard.edu/healthbeat/10-small-steps-for-better-heart-health "
                    f" https://www.healthxchange.sg/heart-lungs/heart-disease/how-to-improve-heart-health-naturally**")
            image1= PIL.Image.open("BP.png")
            st.image(image1, caption='Risk vs. BP')
            image1= PIL.Image.open("GH.png")
            st.image(image1, caption='Risk vs. General Health')
            image1= PIL.Image.open("smoking.png")
            st.image(image1, caption='Risk vs. Smoking')
            image1= PIL.Image.open("PA.png")
            st.image(image1, caption='Risk vs. Physical Activity')
            image1= PIL.Image.open("AC.png")
            st.image(image1, caption='Risk vs. Age Category')
            image1= PIL.Image.open("cor.png")
            st.image(image1, caption='Correlation Matrix')
if add_radio=="Calculate diabetes risk":
        st.title('Diabetes Risk Prediction')
        st.write(""" Welcome to our Diabetes Risk Prediction App. Our aim is to use this platform to save lives by 
         alerting people of potential risks by analysing their lifestyle practices. We have collected/cleaned BRFSS 2015
         data which had inputs from over 400,000 people in the United States who were asked more than 330 questions. 
         We build a machine learning model to predict Diabetes risk, and saved/exported the model for use in a Streamlit
         web app. Our model uses over 22 columns and more than 200,000 value rows after being cleaned.""")
  
        image1=PIL.Image.open("diabetes.jpg")
        st.image(image1,width=500)
        
        
        
        st.write("### Answer the following 21 Questions: ")
        
        # create the colums to hold user inputs
        col1, col2, col3 = st.columns((3,3,3))
        
        # gather user inputs
        
        # 1. HighBP
        highbp = col1.selectbox(
            "1. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that you have high Blood Pressure?",
            ('Yes', 'No'), index=0 )
        
        # 2. HighChol
        highchol = col2.selectbox(
            "2. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your Blood Cholesterol is high?",
            ('Yes', 'No'), index=1)
        
        # 3. CholCheck
        cholcheck = col3.selectbox(
            "3. About how long has it been since you last had your blood cholesterol checked? Put yes if less than 5 years, No if more than 5 years",
            ('Yes', 'No'), index=1)
        
        
        
        
        # 4.BMI 
        bmi = col1.number_input(
            '4. Enter your BMI : ', min_value=5, max_value=50, value=21)
        
        # 5. Smoke
        smoker = col2.selectbox(
            "5. Have you smoked atleast 100 cigarettes in your life?",
            ('Yes', 'No'), index=1)
        
        # 6. Stroke
        stroke = col3.selectbox(
            "6. Have you ever had stroke in your life?",
            ('No', 'Yes'), index=0)
        
        
        #8.PhysActivity
        physActivity= col2.selectbox(
            "8. Are you physically active in your life?",
            ('No', 'Yes'), index=1)
        
        #9.Fruits
        fruits= col3.selectbox(
            "9. Do you have more than 1 fruit a day?",
            ('No', 'Yes'), index=1)
        
        #10.Veggies
        vegetables = col1.selectbox(
            "10. Do you have more than 1 vegetable a day?",
            ('No', 'Yes'), index=1)
        
        #11.Alcohol consumption
        Alcohol= col2.selectbox(
            "11. Are you a heavy drinker (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)?",
            ('No', 'Yes'), index=0)
        
        #12.Alcohol consumption
        healthplan= col3.selectbox(
            "12. Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?",
            ('No', 'Yes'), index=1)
        
        #13.Alcohol consumption
        medcost= col1.selectbox(
            "13. Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
            ('No', 'Yes'), index=1)
        
        # 14. GenHlth
        genhealth = col2.selectbox("14. General Health: How would you rank your General Health on a scale from 1 = Excellent to 5 = Poor? Consider physical and mental health.",
                                 ('Excellent', 'Very Good', 'Good', 'Fair', 'Poor'), index=3)
        
        #15.MentalHealth
        menhlth = col3.number_input(
            '15. Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? : '
                , min_value=0, max_value=30, value=10)
        
        #16.PhysicalHealth
        Physhlth = col1.number_input(
            '16. Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? : '
                , min_value=0, max_value=30, value=10)
        
        #17.Diff Walking
        Diffwalk= col2.selectbox(
            "17. Did you have trouble walking in last 30 days?",
            ('No', 'Yes'), index=0)
        
        
        #18.Sex
        Sex= col3.selectbox(
            "18. Sex M/F:",
            ('Female', 'Male'), index=1)
        
        # 19. Age
        age = col1.selectbox(
            '19. Select your Age:', ('Age 18 to 24',
                                    'Age 25 to 29',
                                    'Age 30 to 34',
                                    'Age 35 to 39',
                                    'Age 40 to 44',
                                    'Age 45 to 49',
                                    'Age 50 to 54',
                                    'Age 55 to 59',
                                    'Age 60 to 64',
                                    'Age 65 to 69',
                                    'Age 70 to 74',
                                    'Age 75 to 79',
                                    'Age 80 or older'), index=4)
        
        #20.education level
        educa = col2.number_input(
            '20. What is your education level with 1 being to never attended school , 6 being graduated college? : '
                , min_value=0, max_value=6, value=4)
        
        #21. Income
        Income= col3.number_input(
            '21. What is your income level with 1 being less than 10,000$ to 8 being more than 75000$? : '
                , min_value=1, max_value=8, value=4)
        
        
        
        
        # Create dataframe:
        df1 = pd.DataFrame([[highbp,highchol,cholcheck,bmi,smoker,stroke,physActivity,fruits,
                             vegetables,Alcohol,healthplan,
                             medcost,genhealth,menhlth,Physhlth,Diffwalk,Sex,age,educa,Income]], 
                            columns=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","PhysActivity","Fruits","Veggies",
                                     "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlt","PhysHlth","DiffWalk"
                                     ,"Sex","Age","Education","Income"])
        
        
        
        
        def prepare_df (df):
            
            # Age
            df['Age'] = df['Age'].replace({'Age 18 to 24': 1, 'Age 25 to 29': 2, 'Age 30 to 34': 3, 'Age 35 to 39': 4, 'Age 40 to 44': 5, 'Age 45 to 49': 6,
                                       'Age 50 to 54': 7, 'Age 55 to 59': 8, 'Age 60 to 64': 9, 'Age 65 to 69': 10, 'Age 70 to 74': 11, 'Age 75 to 79': 12, 'Age 80 or older': 13})
        
            # HighChol
            df['HighChol'] = df['HighChol'].replace({'Yes': 1, 'No': 0})
            
            # HighBP
            df['HighBP'] = df['HighBP'].replace({'Yes': 1, 'No': 0})
            
            # GenHlth
            df['GenHlth'] = df['GenHlth'].replace(
            {'Excellent': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4, 'Poor': 5})
            
            df['CholCheck'] = df['CholCheck'].replace({'Yes': 1, 'No': 0})
                
            df['Smoker'] = df['Smoker'].replace({'Yes': 1, 'No': 0})
            
            df['Stroke'] = df['Stroke'].replace({'Yes': 1, 'No': 0})
        
            #df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].replace({'Yes': 1, 'No': 0})
            
            df['Fruits'] = df['Fruits'].replace({'Yes': 1, 'No': 0})
        
            df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].replace({'Yes': 1, 'No': 0})
        
            df['PhysActivity'] = df['PhysActivity'].replace({'Yes': 1, 'No': 0})
        
            df['Veggies'] = df['Veggies'].replace({'Yes': 1, 'No': 0})
        
            df['AnyHealthcare'] = df['AnyHealthcare'].replace({'Yes': 1, 'No': 0})
        
            df['NoDocbcCost'] = df['NoDocbcCost'].replace({'Yes': 1, 'No': 0})
        
            df['DiffWalk'] = df['DiffWalk'].replace({'Yes': 1, 'No': 0})
        
            df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})   
            
        
            return df
        
        
        
        df=prepare_df(df1)
        
        
        
        log_reg_model = joblib.load("logReg3_model.pkl")
        
        if st.button('Click here to predict Diabetes Risk'):
        
            # make the predictions
            prediction = log_reg_model.predict(df)
            prediction_probability = log_reg_model.predict_proba(df)
        
        
            if prediction == 0:
                st.balloons()
                st.success(f"**The probability that you'll have"
                        f" Diabetes is {round(prediction_probability[0][1] * 100, 2)}%."
                        f" You are healthy!**")
           
            else:
                st.error(f"**The probability that you will have"
                        f" Diabetes is {round(prediction_probability[0][1] * 100, 2)}%."
                        f" It sounds like you are not healthy. ")
    
if add_radio=='Calculate susceptibility to mental distress':
    st.write("""# Mental distress Risk Prediction App """)
    
    st.write(""" Welcome to our Mental distress Risk Prediction App. Our aim is to use this platform to save lives by 
             alerting people of potential risks by analysing their lifestyle practices. """)
      
    image1=PIL.Image.open("mental.png")
    st.image(image1,width=500)
    
    
    
    st.write("### Answer the following 21 Questions: ")
    
    # create the colums to hold user inputs
    col1, col2, col3 = st.columns((3,3,3))
    
    # gather user inputs
    
    work_interfere = col1.selectbox(
        "1.If you have a mental health condition, do you feel that it interferes with your work?",
        ('Yes', 'No'), index=0 )
    
    
    family_history= col2.selectbox(
        "2.Do you have a family history of mental illness?",
        ('Yes', 'No'), index=1)
    
    care_options= col3.selectbox(
        "3.Do you know the options for mental health care your employer provides?",
        ('Not sure', 'No', 'Yes'), index=1)
    
    
    Gender= col1.selectbox(
        "4. Respondent gender:",
        ('Male', 'Female', 'Others'), index=1)
    
    
    anonymity = col2.selectbox(
        "5. Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment?",
        ('Yes', 'Dont Know', 'No'), index=0)
    
    
    benefits = col3.selectbox(
        "6. Does your employer provide mental health benefits?",
        ('Yes', 'Dont Know', 'No'), index=1)
    
    
    obs_consequence = col1.selectbox(
        "7. Have you observed negative consequences to people with mental health problems?",
        ('No', 'Yes'), index=0)
    
    
    coworkers = col2.selectbox(
        "8. Would you be willing to discuss mental health issues with your relatives? ",
        ('Some of them', 'No', 'Yes'), index=0)
    
    
    mental_health_consequence= col3.selectbox(
        "9. Do you think that discussing a mental health issue with your employer would have negative consequences?",
        ('No', 'Maybe','Yes'), index=1)
    
    wellness_program= col1.selectbox(
        "10.Has your employer/school ever discussed mental health as part of an employee wellness program?",
        ('No', 'Dont Know', 'Yes'), index=1)
    
    no_employees = col2.selectbox(
        "11. How many employees does your organization have?",
        ('1-25','26-500','500 and above'), index=1)
    
    
    seek_help = col3.selectbox(
        "12. Are you taught about how to seek help)?",
        ('Yes', 'Dont Know', 'No'),index=1)
    
    
    Age = col1.selectbox(
        '13. Select your Age:', ('Age 0 to 13',
                                'Age 14 to 25',
                                'Age 26 to 37',
                                'Age 38 to 49',
                                'Age 50 to 61',
                                'Age 62 to 72'), index=3)
    
    
    phys_health_interview= col2.selectbox(
        "14. Would you bring up a physical health issue with a potential employer? ",
        ('Maybe','No', 'Yes'), index=1)
    
    
    
    # Create dataframe:
    df1 = pd.DataFrame([[work_interfere,family_history,care_options,Gender,anonymity,benefits,obs_consequence,coworkers,
                         mental_health_consequence,wellness_program,no_employees,seek_help,Age,phys_health_interview]], 
                        columns=["work_interfere","family_history","care_options","Gender","anonymity","benefits","obs_consequence","coworkers",
                        "mental_health_consequence","wellness_program","no_employees","seek_help","Age","phys_health_interview"])
    
    
    def prepare_df (df):
        
        # Age
        df['Age'] = df['Age'].replace({'Age 0 to 13': 0,
                                'Age 14 to 25':0.125 ,
                                'Age 26 to 37': 0.25,
                                'Age 38 to 49': 0.375,
                                'Age 50 to 61': 0.5,
                                'Age 62 to 72': 0.625})
    
    
        df['work_interfere'] = df['work_interfere'].replace({'Yes': 1, 'No': 0})
        
    
        df['family_history'] = df['family_history'].replace({'Yes': 1, 'No': 0})
        
        df['care_options'] = df['care_options'].replace(
        {'Not sure':0.5, 'No':0, 'Yes':1})
        
        df['Gender'] = df['Gender'].replace(
        {'Male': 0, 'Female': 1, 'Others': 0.5})
            
        df['anonymity'] = df['anonymity'].replace({'Yes': 1, 'No': 0, 'Dont Know':0.5})
        
        df['benefits'] = df['benefits'].replace({'Yes': 1, 'No': 0, 'Dont Know':0.5})
    
        df['obs_consequence'] = df['obs_consequence'].replace({'Yes': 1, 'No': 0})
        
        df['coworkers'] = df['coworkers'].replace({'Some of them':0.5, 'No': 0, 'Yes': 1})
    
        df['mental_health_consequence'] = df['mental_health_consequence'].replace({'Yes': 1, 'No': 0, 'Maybe': 0.5 })
    
        df['wellness_program'] = df['wellness_program'].replace({'Yes': 1, 'No': 0, 'Dont Know':0.5})
    
        df['no_employees'] = df['no_employees'].replace({'1-25': 0,'26-500' : 0.5 ,'500 and above': 1})
    
        df['seek_help'] = df['seek_help'].replace({'Yes': 1, 'No': 0, 'Dont Know':0.5})
    
    
        df['phys_health_interview'] = df['phys_health_interview'].replace({'Yes': 1, 'No': 0, 'Maybe': 0.5})
    
    
        
    
        return df
    
    
    
    df=prepare_df(df1)
    
    
    
    log_reg_model = joblib.load("Mental_Health_model5.pkl")
    
    if st.button('Click here to predict Mental Health Problems'):
    
        # make the predictions
        prediction = log_reg_model.predict(df)
        prediction_probability = log_reg_model.predict_proba(df)
    
    
        if prediction == 0:
            st.balloons()
            st.success(f"**The probability that you'll be susceptible to"
                    f" Mental Health Problems is {abs(1- round(prediction_probability[0][1] * 100, 2))}%."
                    f" You are healthy!**")
       
        else:
            st.error(f"**The probability that you will be susceptible to"
                    f" Mental Health Problem is {abs(1 - round(prediction_probability[0][1] * 100, 2))}%."
                    f" It sounds like you are not healthy. "
                    f" You should refer to these sites to work towards improving your health")
