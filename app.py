import streamlit as st
import os
import pandas as pd
import joblib as jb

heading_style = '''
<div style="color:red;" align='center'>
<h1>Loan Amount Prediction System</h1>
</div>
'''
def return_df(Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
	ApplicantIncome,
	CoapplicantIncome,
	LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area):
    kbn={
    'Gender':[Gender],
    'Married':[Married],
    'Dependents':[Dependents],
    'Education':[Education],
    'Self_Employed':[Self_Employed],
	'ApplicantIncome':[ApplicantIncome],
	'CoapplicantIncome':[CoapplicantIncome],
    'LoanAmount':[LoanAmount],
    'Loan_Amount_Term':[Loan_Amount_Term],
    'Credit_History':[Credit_History],
    'Property_Area':[Property_Area]
    }   
    final_df=pd.DataFrame(kbn)
    return final_df


def base_model():
    bmodel=jb.load(os.path.join('finalised_rf_model.pkl'))
    return bmodel

st.markdown(heading_style, unsafe_allow_html=True)
Gender=st.selectbox('Select your gender...',['Male','Female'])
Married=st.selectbox('Are you Married or Not?',['Yes','No'])
Dependents=st.slider('How many dependents you have..',0,10,0)
Education=st.selectbox('Are you a graduate or not?',['Graduate','Not Graduate'])
Self_Employed=st.selectbox('Are you self employed or not?',['Yes','No'])
ApplicantIncome=st.number_input('Enter your income..', min_value=0)
CoapplicantIncome=st.number_input('Enter your co-applicant income..', min_value=0)
LoanAmount=st.number_input('Enter your loan amount',min_value=0)
Loan_Amount_Term=st.number_input('Enter your loan amount term..',min_value=0)
Credit_History=st.slider('Your credit history..',0,1,0)
Property_Area=st.selectbox('What is your property location area?',['Urban','Rural','Semiurban'])
df=return_df(Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
	ApplicantIncome,
    CoapplicantIncome,
	LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	if predictions=='Y':
		st.write('Loan Approved')
	elif predictions=='N':
		st.write('Loan Rejected')
