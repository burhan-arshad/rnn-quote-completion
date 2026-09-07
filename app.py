import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="AI Quote Generator",
    page_icon="✍️",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_resources():
    model = load_model(os.path.join(BASE_DIR, "lstm_model.h5"))

    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

def generate_quote(seed_text, num_words):
    generated_text = seed_text.lower().strip()

    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]

        if not sequence:
            break

        sequence = sequence[-max_len:]

        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="pre"
        )

        prediction = model.predict(
            padded_sequence,
            verbose=0
        )[0]

        predicted_id = np.argmax(prediction)
        predicted_word = index_to_word.get(predicted_id)

        if predicted_word is None:
            break

        generated_text += " " + predicted_word

    return generated_text

st.title("AI Quote Generator")
st.write("Generate quotes using an LSTM neural network.")

seed_text = st.text_input(
    "Enter the beginning of your quote",
    placeholder="Life is"
)

num_words = st.slider(
    "Number of words",
    min_value=5,
    max_value=50,
    value=20
)

if st.button("Generate Quote"):
    if seed_text.strip():
        quote = generate_quote(seed_text, num_words)
        st.subheader("Generated Quote")
        st.write(f"_{quote}_")
    else:
        st.warning("Please enter some starting words.")