import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms]).toarray()

    result = model.predict(vector_input)[0]

    # Predict the probability
    prob = model.predict_proba(vector_input)[0]
    spam_prob = prob[1] * 100


    if result == 1:
        st.error("🚨 Spam Detected!")
    else:
        st.success("✅ Not Spam")

    st.write("### Spam Probability")
    st.progress(int(spam_prob))
    st.write(f"**{spam_prob:.2f}%**")