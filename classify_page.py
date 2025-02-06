import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')


def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

model = load_model()
vectorizer = load_vectorizer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    non_stop_words = []
    cleaned_sentence = " "
    for word in sentence.split():
        if word not in stop_words:
            non_stop_words.append(word)
    return cleaned_sentence.join(non_stop_words)

def show_classify_page():
    st.title("Fake Review Checker")
    st.write("""### Please input the review you want to check!""")
    text_input = st.text_area("Enter your review here:", "")    
    ok = st.button("Check Text!")
    if ok:
        X = text_input
        X = clean_text(X)
        X = remove_stop_words(X)
        X = word_tokenize(X)
        input_vectorized = vectorizer.transform(X)
        probs = model.predict_proba(input_vectorized)[0]  
        prediction = model.predict(input_vectorized)[0]
        probability = probs[prediction] * 100

        if probability < 60:
            st.subheader(f"🤔 Uncertain. The review seems balanced. (Confidence: {probability:.2f}%)")
        elif 60 <= probability < 80:
            st.subheader(f"🤔 ⚠️ Moderate chance of being fake. (Confidence: {probability:.2f}%)")
        else:
            st.subheader(f"🚨 Highly sus! Likely fake review. (Confidence: {probability:.2f}%)")                        
    else:
        st.subheader("⚠️ Please enter a review before checking.")        