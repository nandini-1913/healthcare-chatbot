import streamlit as st
import json
import pickle
import numpy as np
import os
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)


def ensure_nltk_resource(resource_paths, download_name):
    if isinstance(resource_paths, str):
        resource_paths = [resource_paths]

    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue

    nltk.download(download_name, download_dir=NLTK_DATA_DIR, quiet=True)

    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue

    raise LookupError(
        f"Missing NLTK resource '{download_name}'. Expected one of: {resource_paths}"
    )


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource(["corpora/wordnet", "corpora/wordnet.zip"], "wordnet")

model = load_model(os.path.join(BASE_DIR, "chatbot_model.keras"))
intents = json.load(open(os.path.join(BASE_DIR, "intents_medquad.json")))
words = pickle.load(open(os.path.join(BASE_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(BASE_DIR, "classes.pkl"), "rb"))

st.title("Healthcare Chatbot")

user_input = st.text_input("Ask your question:")

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

if user_input:
    bow = bag_of_words(user_input)
    res = model.predict(np.array([bow]))[0]
    idx = np.argmax(res)
    tag = classes[idx]

    for i in intents["intents"]:
        if i["tag"] == tag:
            response = np.random.choice(i["responses"])
            st.write(response)
