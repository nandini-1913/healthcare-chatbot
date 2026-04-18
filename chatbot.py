import random
import json
import os
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

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

# load files
intents = json.loads(open(os.path.join(BASE_DIR, "intents_medquad.json")).read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.keras")

# 🔹 Clean sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# 🔹 Convert to Bag of Words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                
    return np.array(bag)

# 🔹 Predict intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })
        
    return return_list

# 🔹 Get response
def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry, I didn't understand. Please describe your symptoms clearly."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# 🔹 Chat loop
print("🤖 Healthcare Chatbot is running! (type 'quit' to exit)\n")

while True:
    message = input("You: ")
    
    if message.lower() == "quit":
        print("Bot: Take care! Stay healthy 😊")
        break
    
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    
    print("Bot:", response)

