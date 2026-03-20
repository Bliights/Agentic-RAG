import streamlit as st

st.set_page_config(page_title="Agentic-RAG", page_icon="💬")

st.title("Agentic-RAG")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrée utilisateur
user_input = st.chat_input("Écris ton message ici...")

if user_input:
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Réponse très simple du chatbot
    bot_response = f"Tu as écrit : {user_input}"

    # Ajouter la réponse du bot
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.write(bot_response)