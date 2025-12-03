import streamlit as st
from agent import main


st.title("Welcome!")
msg = st.text_input("How can I help you? \n")
result = main(msg)
st.write(result)
