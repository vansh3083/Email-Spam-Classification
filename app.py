import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open('model/model.pkl','rb'))
cv = pd.read_csv('model/cv.csv')

def email_to_input(email):
    email = email.lower().split()

    for i in range(len(email)):
        if email[i] in cv['words'].to_list():
            cv.loc[cv['words'] == email[i], 'freq'] += 1
    return cv['freq'].to_list()

def predict(email):
    return model.predict([email_to_input(email)])[0]

def main():
    st.title(':sparkles: E-Mail Spam Classifier :sparkles:')
    st.subheader('Enter the E-Mail below to check if it is Spam or Ham')
    email = st.text_input('Email')
    if st.button('Predict'):
        result = predict(email)
        if result == 0:
            st.success('Ham :white_check_mark:')
        else:
            st.error('Spam :heavy_exclamation_mark:')

if __name__ == '__main__':
    main()