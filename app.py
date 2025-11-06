import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
#model = load_model('next_word_lstm.h5')
model = load_model('next_word_gru.h5')

# Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([text])[0] # Convertit text en liste d’indices (tokens)
    if len(token_list) >= max_sequence_len:  # Si la séquence est trop longue pour l’entrée du modèle, on garde seulement la fin
        token_list = token_list[-(max_sequence_len-1):]  
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #predicted = model.predict(token_list, verbose=0)  # Le modèle renvoie un vecteur de probabilités de taille vocab_size (shape (1, total_words))
    predicted = model.predict(token_list, verbose=0)[0]
    #predicted_word_index = np.argmax(predicted, axis=1)  # Prend l’indice du mot le plus probable
    #predicted_word_index = np.argmax(predicted)

    # Application de la température
    preds = np.log(predicted + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Échantillonnage pondéré
    predicted_index = np.random.choice(len(preds), p=preds)

    
    #predicted_word = tokenizer.index_word.get(predicted_word_index, None) # Convertit l’indice en mot avec tokenizer.index_word

    predicted_word = tokenizer.index_word.get(predicted_index, "")

    return predicted_word
    
    #for word, index in tokenizer.word_index.items():
        #if index == predicted_word_index:
            #return word
    #return None

# Streamlit app
st.title('Next word prediction')
input_text=st.text_input("Enter the sequence of words")

# Slider for temperature
temperature = st.slider("Creativity (temperature) : Logic -> creative", 0.3, 2.0, 1.0, 0.1)

if st.button('Predict Next Word'):
    max_sequence_len=model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')