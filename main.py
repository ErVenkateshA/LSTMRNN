import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#load the model
model = load_model('predict_next_word_lstm.h5')

#load tokenizer
with open('tokenizer.pickel','rb') as handle:
    tokenizer = pickle.load(handle)

#Function of next Prediction word
def predict_next_word(model, tokenizer, text, max_len):
    token_lists = tokenizer.texts_to_sequences([text])[0]
    if len(token_lists) >= max_len:
        token_lists = token_lists[-(max_len-1):]
    token_lists = pad_sequences([token_lists], maxlen=max_len-1,padding='pre')
    predicted = model.predict(token_lists,verbose=1)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

#Streamlit app
st.title('Predict Next Word')
input_text = st.text_input('Enter a sequence')
if st.button('Click to predict'):
    max_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_len)
    st.write(f'Next word is:{next_word}')
