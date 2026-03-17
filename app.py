import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# LOAD MODEL
# ======================
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("X.pkl", "rb") as f:
    X = pickle.load(f)

with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

# ======================
# DEFINE FUNCTION (IMPORTANT 🔥)
# ======================
def get_answer(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, X)
    index = similarity.argmax()
    return answers[index]

def extract_entities(text):
    diseases = ["diabetes", "cancer", "asthma", "covid", "fever"]
    symptoms = ["pain", "cough", "headache", "fatigue"]

    found = []
    for word in diseases + symptoms:
        if word in text.lower():
            found.append(word)

    return list(set(found))

# ======================
# UI
# ======================
st.title("🩺 Medical Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    answer = get_answer(user_input)
    entities = extract_entities(user_input)

    st.write("### Answer:")
    st.write(answer)

    st.write("### Entities:")
    st.write(entities)