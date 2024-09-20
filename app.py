import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming the words
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit app title with custom style
st.markdown("<h1 style='text-align: center; color: blue;'>Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Try to load the model and vectorizer with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Input text area for user with placeholder
input_sms = st.text_area("Enter the message", placeholder="Type your message here...", height=150)

# Button to predict spam or not with custom design
if st.button('Predict'):
    # If input is empty, display an error message
    if input_sms.strip() == "":
        st.error("Please enter a message to classify!")
    else:
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the input message
        vector_input = tfidf.transform([transformed_sms])  # Wrapped in a list

        # 3. Predict using the model
        result = model.predict(vector_input)[0]

        # 4. Display the result with a colored header
        if result == 1:
            st.markdown("<h2 style='color: red;'>⚠️ Spam</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: green;'>✅ Not Spam</h2>", unsafe_allow_html=True)

