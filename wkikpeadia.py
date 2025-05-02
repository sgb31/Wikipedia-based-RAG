import streamlit as st
import wikipedia
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
load_dotenv()
api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

llm = HuggingFaceHub(
    repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    model_kwargs={"temperature": 0.9, "max_length": 30},
    huggingfacehub_api_token=api_token
)
def retrieve_wikipedia_content(question):
    try:
        page = wikipedia.page(question, auto_suggest=True)
        return page.content[:2000]
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No relevant Wikipedia page found."

def generate_answer(question, prompt):
    context = retrieve_wikipedia_content(question)
    if "Multiple results found:" in context or "No relevant Wikipedia page found." in context:
        return f'{{"Q": "{prompt}", "A": "{context}"}}'
    
    formatted_prompt = f'''Q: {prompt}'''

    return llm.invoke(formatted_prompt)

st.title("Wikipedia RAG")
st.write("Ask a question using Wikipedia as the only knowledge source.")

question = st.text_input("Enter the Wikipedia page topic:")
prompt = st.text_input("Enter your question about this topic:")

if st.button("Get Answer"):
    if question and prompt:
        with st.spinner("Searching answer..."):
            response = generate_answer(question, prompt)
            st.write("### JSON Output:")
            st.code(response, language="json")
    else:
        st.warning("Please enter both a topic and a question.")
