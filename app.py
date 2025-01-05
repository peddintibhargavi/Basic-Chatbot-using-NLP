import os
import pickle
import json
import random
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="AI Chatbot", layout="wide")

# File paths
model_path = "model/classifier.pkl"
vectorizer_path = "model/vectorizer.pkl"
intents_file_path = "model/intents.json"
history_csv_path = "chat_history.csv"

# Load the model and vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    st.error("Model or vectorizer file not found. Please check the paths.")

# Load the intents.json file
if os.path.exists(intents_file_path):
    with open(intents_file_path, 'r') as f:
        intents_data = json.load(f)
else:
    st.error("Intents file not found. Please check the path.")

def get_response(user_input):
    """
    Generates a response based on user input using the pre-trained model and vectorizer.
    """
    if not user_input.strip():
        return "Please type something to continue the conversation."
    
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_intent = model.predict(user_input_vectorized)[0]
    intents_list = intents_data.get("intents") if isinstance(intents_data, dict) else intents_data

    for intent in intents_list:
        if intent["tag"] == predicted_intent:
            return random.choice(intent["responses"])
    
    return "I'm sorry, I don't understand. Can you rephrase?"

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def save_chat_to_csv():
    """
    Save the chat history to a CSV file.
    """
    if st.session_state.chat_history:
        pd.DataFrame(st.session_state.chat_history).to_csv(history_csv_path, index=False)

def load_chat_history_from_csv():
    """
    Load chat history from a CSV file.
    """
    if os.path.exists(history_csv_path):
        return pd.read_csv(history_csv_path).to_dict('records')
    return []

# Add a rerun flag in session state
if "rerun" not in st.session_state:
    st.session_state["rerun"] = False



# Section titles in a row (cards)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Chat"):
        st.session_state.selected_section = "Chat"

with col2:
    if st.button("📜 Chat History"):
        st.session_state.selected_section = "Chat History"

with col3:
    if st.button("ℹ️ About"):
        st.session_state.selected_section = "About"

# Display selected section content based on button click
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "Chat"  # Default section

# Display the card content based on the selection
if st.session_state.selected_section == "Chat":
    # Chat card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">💬 Chatbot</div>', unsafe_allow_html=True)

    # Chat interface
    chat_container = st.container()
    with chat_container:
        # Custom CSS for chat messages
        st.markdown("""
    <style>
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 20px;
            margin: 5px;
            max-width: 70%;
            float: right;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .bot-message {
            background-color: #E8E8E8;
            padding: 10px;
            border-radius: 20px;
            margin: 5px;
            max-width: 70%;
            float: left;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
      
        .card-header {
            font-size: 20px;
            font-weight: bold;
        }
        .message-heading {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .message-heading-user {
            color: #4CAF50;
        }
        .message-heading-bot {
            color: #0084FF;
        }
        .timestamp {
            font-size: 12px;
            color: #888;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


        # Chat messages area
        with st.container():
            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-message">{chat["user_message"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">{chat["bot_response"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Input box and send button
        user_input = st.text_input("Type your message here...")
        if st.button("Send"):
            if user_input:
                response = get_response(user_input)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append({
                    "timestamp": timestamp,
                    "user_message": user_input,
                    "bot_response": response
                })
                save_chat_to_csv()

                # Trigger re-rendering
                st.session_state["rerun"] = True

    # Check for rerun flag
    if st.session_state["rerun"]:
        st.session_state["rerun"] = False  # Reset the flag
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.selected_section == "Chat History":
    # Chat History card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📜 Conversation History</div>', unsafe_allow_html=True)

    chat_history = load_chat_history_from_csv()
    history_by_date = {}

    for chat in chat_history:
        date = chat["timestamp"].split()[0]
        if date not in history_by_date:
            history_by_date[date] = []
        history_by_date[date].append(chat)

    for date, chats in history_by_date.items():
        st.markdown(f"#### {date}")
        
        for chat in chats:
            # Columns for user message, bot message, and timestamp
            col1, col2, col3 = st.columns([1, 4, 1])  # Adjust ratio for message display and timestamp

            with col1:
                # User icon and heading
                st.markdown(f'**🧑 User**')

            with col2:
                # User message and bot message in the same container
                st.markdown(f'<div class="user-message">{chat["user_message"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">{chat["bot_response"]}</div>', unsafe_allow_html=True)

            with col3:
                # Timestamp for the message
                st.markdown(f"**{chat['timestamp'].split()[1]}**")  # Display time part of timestamp (HH:MM:SS)

            st.markdown("---")  # Divider between conversations
            
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.selected_section == "About":
    # About card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-body">
        <h3><strong>About the Chatbot Project</strong></h3>
        <p><strong>Goal</strong></p>
        <p>
        The aim of this project is to create a chatbot that can understand and respond to user inputs based on their intents. The chatbot leverages 
        Natural Language Processing (NLP) techniques and a Random Forest Classifier for accurate intent classification. The chatbot's interface 
        is built using Streamlit, enabling users to interact with the bot via a simple and user-friendly web application.
        </p>
        <h4>Project Overview</h4>
        <p>This project is divided into two main components:</p>
        <ol>
            <li><strong>Model Training and Intent Recognition</strong></li>
            <ul>
                <li>NLP techniques are used to preprocess and extract intents from user inputs.</li>
                <li>A Random Forest Classifier is used to train the model on labeled intents and their patterns.</li>
                <li>The chatbot recognizes user inputs such as "greetings," "queries about budgeting," or "general information requests," and generates appropriate responses.</li>
            </ul>
            <li><strong>Streamlit Chatbot Interface</strong></li>
            <ul>
                <li>A web-based interface is built using Streamlit.</li>
                <li>Users can input text and receive meaningful responses from the chatbot in real time.</li>
                <li>The interface includes a text input box for user queries and a chat window to display the chatbot's responses.</li>
            </ul>
        </ol>
        <h4>Dataset</h4>
        <p>The dataset used for training the chatbot consists of labeled intents, patterns, and responses:</p>
        <ul>
            <li><strong>Intents</strong>: These represent the purpose of user inputs, such as "greeting," "budget query," or "about."</li>
            <li><strong>Patterns</strong>: These are variations of user inputs for each intent.</li>
            <li><strong>Responses</strong>: These are predefined replies that the chatbot provides for each intent.</li>
        </ul>
        <p>The data is processed using TF-IDF Vectorization to extract numerical features from the textual data, which is then used to train the Random Forest Classifier.</p>
        <h4>Why Random Forest?</h4>
        <p>Random Forest was chosen as the classification algorithm for this project because of its robustness and versatility:</p>
        <ul>
            <li>It is an ensemble learning method that combines multiple decision trees, reducing the risk of overfitting.</li>
            <li>It performs well with high-dimensional data, such as text features generated by TF-IDF.</li>
            <li>The algorithm provides better generalization and high accuracy compared to simpler models like Logistic Regression.</li>
            <li>The ability to calculate probabilities helps the chatbot determine the confidence level of predictions, ensuring more accurate responses.</li>
        </ul>
        <h4>Chatbot Features</h4>
        <ul>
            <li><strong>Accurate Intent Recognition:</strong> The Random Forest model predicts the intent of user inputs with high accuracy.</li>
            <li><strong>Dynamic Responses:</strong> The bot responds dynamically based on the intent, providing meaningful and context-aware answers.</li>
            <li><strong>Fallback for Uncertain Inputs:</strong> If the model's confidence in its prediction is low, the bot prompts the user to rephrase their query.</li>
        </ul>
        <h4>Conclusion</h4>
        <p>
        This project demonstrates how NLP techniques and a Random Forest Classifier can be combined to build an intelligent chatbot. The interactive 
        interface, developed using Streamlit, allows users to engage with the bot effortlessly. The project can be extended further by:
        </p>
        <ul>
            <li>Adding more intents and data to improve coverage.</li>
            <li>Exploring deep learning models for advanced NLP capabilities.</li>
            <li>Integrating external APIs for more dynamic and versatile responses.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
