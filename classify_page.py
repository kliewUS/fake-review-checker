import streamlit as st
import pickle
import pandas as pd
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

'''
    Removes unwanted characters such as numbers and html tags.
'''
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    
    return text
'''
    Removes unwanted stop words such as "the", "an", "in".
'''
def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    non_stop_words = []
    cleaned_sentence = " "
    for word in sentence.split():
        if word not in stop_words:
            non_stop_words.append(word)
    return cleaned_sentence.join(non_stop_words)


'''
    This method does the preprocessing of the input, which consists of removing unecessary characters and stop words and word tokenizing. 
    Once the review has been preprocessed, then the model will predict and determine if the review is real or fake along with a confidence percentage of its rating.
    It will return the classification as well as the confidence rating of the classification.
'''
def classify_text(text_input):
    X = text_input
    X = clean_text(X)
    X = remove_stop_words(X)
    X = word_tokenize(X)
    input_vectorized = vectorizer.transform(X)
    prediction = model.predict(input_vectorized)[0]       
    probs = model.predict_proba(input_vectorized)[0]
    probability = probs[prediction] * 100
    classification = "Fake" if prediction == 1 else "Real"

    return classification, probability
'''
    This method processes the file uploaded, checking if the file format is valid. If not, it will give out an error message.
    Afterwards, it will attempt to locate and parse the review column. If it is unable to find the review column, it will give an option to find the review column.
    The process is similar to the classify_text method. It will preprocess and the model will predict and determine each review as well as a confidence percentage.
    It will return the dataframe and the chosen_column name.
'''
def process_file(uploaded_file):
    COMMON_REVIEW_COLUMNS = ["review", "review_text", "text", "comment", "feedback", "message"]    
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1]

        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext == "json":
            df = pd.read_csv(uploaded_file)
        else:
            st.subheader("Unsupported file format.")

        chosen_column = None
        for col in df.columns:
            if col.lower() in COMMON_REVIEW_COLUMNS:
                chosen_column = col
                break
        
        if chosen_column is None:
            chosen_column = st.selectbox("Select the column that contains the reviews", df.columns)

        df[chosen_column] = df[chosen_column].apply(clean_text)
        df[chosen_column] = df[chosen_column].apply(remove_stop_words)
        df[chosen_column] = df[chosen_column].apply(word_tokenize)
        df[chosen_column] = df[chosen_column].apply(lambda x: ' '.join(x))        

        input_vectorized = vectorizer.transform(df[chosen_column])
        prediction = model.predict(input_vectorized)      
        probs = model.predict_proba(input_vectorized)
        df["Prediction"] = ["Fake" if p == 1 else "Real" for p in prediction]
        df["Confidence"] = [round(probs[i][pred] * 100, 2) for i, pred in enumerate(prediction)]

        return df, chosen_column

'''
    Shows the classify page. Gives an option to manually type in a review or a upload a csv or json file.
    Upon checking the review(s), it will determine if the review is real or fake along with a confidence percentage.
    Additional, if upload option was chosen, it will give an option to download the results.
'''
def show_classify_page():
    st.title("Fake Review Checker")
    input_option = st.radio("Select Input Method:", ["Type a Review", "Upload a File"])    

    if input_option == "Type a Review":
        text_input = st.text_area("Enter your review here:", "")
        ok = st.button("Check Text!")
        if ok:
            classification, probability = classify_text(text_input)
            st.subheader(f"Prediction: {classification} (Confidence: {probability:.2f}%)")                                 
        else:
            st.subheader("Please enter a review before checking.")          
    elif input_option == "Upload a File":
        uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
        if uploaded_file:
            df, chosen_column = process_file(uploaded_file)
            if df is not None:
                st.write(df[[chosen_column, "Prediction", "Confidence"]].head())
                st.write("Review classification breakdown", df["Prediction"].value_counts("Real"))
                st.download_button("Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")                                     