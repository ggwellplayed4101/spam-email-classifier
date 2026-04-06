import streamlit as st
import pickle
st.set_page_config(page_title="Spam E-Mail Classification")

# Use color and font themes

st.markdown("""
<style>

div[class*="stTextInput"] label p {
  font-size: 26px;
}
</style>
""", unsafe_allow_html=True)

saved = pickle.load(open("my_file.pkl", "rb"))
tfidf = saved["vectorizer"]
model = saved["model"]

st.title("Spam E-Mail Classifier")

input_mail = st.text_input("Enter the Message")

if st.button('Predict'):
    
    vector_input = tfidf.transform([input_mail])
    
    result = model.predict(vector_input)
    
    st.success("This is a " + ('Spam Mail' if result == 0 else 'Ham Mail'))