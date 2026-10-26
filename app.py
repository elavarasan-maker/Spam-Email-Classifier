"""
Spam Email Classifier - beginner version
Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------- 1. DATA ----------
# Tiny demo dataset so the app works instantly.
# For a real resume project, replace this with the free "SMS Spam Collection"
# dataset from Kaggle and load it with: pd.read_csv("spam.csv")
data = {
    "text": [
        "Win a free iPhone now, click here",
        "Congratulations! You won a $1000 gift card",
        "Claim your free prize today",
        "Urgent: your account is suspended, verify now",
        "Get cheap loans with no credit check",
        "Limited offer, buy now and save 70%",
        "You have been selected for a cash reward",
        "Free entry in a weekly contest, text WIN",
        "Hey, are we still meeting for lunch tomorrow?",
        "Please find the attached report for review",
        "Can you send me the notes from today's class?",
        "Happy birthday! Hope you have a great day",
        "The meeting has been moved to 3 PM",
        "Don't forget to submit your assignment by Friday",
        "Let's catch up this weekend, call me",
        "Thanks for your help with the project",
    ],
    "label": ["spam"] * 8 + ["ham"] * 8,
}
df = pd.DataFrame(data)

# ---------- 2. TRAIN THE MODEL ----------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    accuracy = accuracy_score(y_test, model.predict(vectorizer.transform(X_test)))
    return vectorizer, model, accuracy

vectorizer, model, accuracy = train_model()

# ---------- 3. THE APP ----------
st.title("📧 Spam Email Classifier")
st.write("Type or paste a message and the model will tell you if it's spam.")

message = st.text_area("Your message:", height=150)

if st.button("Check message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vec = vectorizer.transform([message])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max() * 100

        if prediction == "spam":
            st.error(f"🚨 SPAM ({confidence:.0f}% confident)")
        else:
            st.success(f"✅ Not spam ({confidence:.0f}% confident)")

st.caption(f"Model: TF-IDF + Naive Bayes | Test accuracy on demo data: {accuracy:.0%}")
