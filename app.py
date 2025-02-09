import streamlit as st
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_true, logits=y_pred)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the chatbot model
chatbot = pipeline("text-generation", model="distilgpt2", device=-1)


def healthcare_chatbot(user_input):
    if "symptom" in user_input:
        return "Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It is important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, max_length=50, num_return_sequences=1)
        return response[0]["generated_text"].strip()

def main():
    st.title("Healthcare Assistant Chatbot")
    st.write("Welcome to the Healthcare Assistant Chatbot! Ask any healthcare-related questions below.")

    user_input = st.text_input("How can I assist you today?")

    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("Health Assistant:", response)
        else:
            st.write("Please enter a message to get a response.")

if __name__ == "__main__":
    main()
