import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets setup
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('healthy-skill-460806-r0-5a56fbb8ae3a.json', scope)  # nama file JSON kau
client = gspread.authorize(creds)

sheet = client.open("SmartStep Intelligent Insole for Gait Abnormality Monitoring Dashboard").sheet1  # ganti dengan nama Google Sheet kau

st.title("User Input Form - Save to Google Sheets")

name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=1, max_value=120)
comment = st.text_area("Any comment?")

if st.button("Submit"):
    if name:
        # Simpan data ke Google Sheets
        sheet.append_row([name, age, comment])
        st.success("Thank you! Your data has been saved.")
    else:
        st.error("Please enter your name.")