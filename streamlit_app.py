import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Load service account key from Streamlit secrets
key_dict = json.loads(st.secrets["gcp_service_account"]["key"])

# Google Sheets setup
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
client = gspread.authorize(creds)

sheet = client.open("SmartStep Intelligent Insole for Gait Abnormality Monitoring Dashboard").sheet1

st.title("User Input Form - Save to Google Sheets")

name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=1, max_value=120)
comment = st.text_area("Any comment?")

if st.button("Submit"):
    if name:
        sheet.append_row([name, age, comment])
        st.success("Thank you! Your data has been saved.")
    else:
        st.error("Please enter your name.")