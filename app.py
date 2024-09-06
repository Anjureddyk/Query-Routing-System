import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

data = pd.read_csv('racoon_data.csv')
print("streamlit",st.__version__)

tokenizer_qa = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model_qa = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

text_generator = pipeline('text-generation', model='gpt2')  

keywords = {"why", "when", "where", "how", "what", "who", "which", "?"}

def rule_based_response(query):
    query = query.lower()
    for keyword in keywords:
        if keyword in query:
            response = data[data['Query'].str.lower() == query]
            if not response.empty:
                return response.iloc[0]['Answer']
            else:
                return "No data available for this query."
    return None

def generative_ai_based_response(query):
    generated_text = text_generator(query, max_length=150, num_return_sequences=1)[0]['generated_text']
    return generated_text.strip()

st.title("Chatbot")

user_query = st.text_input("Ask a question:")

if user_query:
    response = rule_based_response(user_query)
    
    if response is None:
        response = generative_ai_based_response(user_query)
    
    st.write(response)
