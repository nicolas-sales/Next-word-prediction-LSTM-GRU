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

def predict_next_word2(model, tokenizer, text, max_sequences_len, temperature=1.0):
    # Tokenisation
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequences_len-1, padding='pre'
    )

    # Prédiction
    preds = model.predict(token_list, verbose=0)[0]

    # Appliquer la "température" -> paramètre pour contrôler la "créativité" du modèle
    preds = np.log(preds + 1e-9) / temperature  # plus la température est grabnde plus le modèle est créatif, délirant et plus elle est petite plus le modèle est structuré
    exp_preds = np.exp(preds)  # On revient dans l’espace des probabilités
    preds = exp_preds / np.sum(exp_preds)

    # Choisir un mot aléatoirement selon les probabilités
    predicted_index = np.random.choice(len(preds), p=preds)  # tire un mot au hasard selon les probabilités
    output_word = tokenizer.index_word.get(predicted_index, "")

    return output_word

# Streamlit app
st.title('Next words prediction')
st.write("Generate multiple words with adjustable creativity.")

# Entrée utilisateur
input_text = st.text_input("Enter a sequence of words:")
temperature = st.slider("Creativity (temperature) : Logic -> Creative", 0.3, 2.0, 1.0, 0.1)
num_words = st.number_input("Number of words to generate:", min_value=1, max_value=100, value=20, step=1)

if st.button("Generate Text"):
    if input_text.strip() == "":
        st.warning("Please enter a starting phrase.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        generated_text = input_text

        # 🔁 Boucle pour générer plusieurs mots
        for _ in range(num_words):
            next_word = predict_next_word2(model, tokenizer, generated_text, max_sequence_len, temperature)
            generated_text += " " + next_word

        st.success("### Generated Text:")
        st.write(generated_text)