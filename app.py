import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from google import genai

load_dotenv('API.env')

def get_valid_gemini_model(api_key):
    try:
        client = genai.Client(api_key=api_key)
        models = client.list_models()
        candidate_models = [m.name for m in models if "gemini" in m.name.lower()]
        if not candidate_models:
            raise ValueError("No Gemini models available")
        candidate_models.sort(reverse=True)
        for candidate in candidate_models:
            return candidate
    except Exception as e:
        st.error(f"Error fetching Gemini models: {e}")
        return None

def summarize_pdf(pdf_file_path, custom_prompt_text, api_key):
    model_name = get_valid_gemini_model(api_key)
    if not model_name:
        st.error("No valid Gemini model found. Check API key and model availability.")
        return None

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key
    )

    loader = PyPDFLoader(pdf_file_path)
    docs_chunks = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    )

    prompt_template = custom_prompt_text + """


    {text}

    """
    prompt = PromptTemplate.from_template(prompt_template)

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    result = chain.invoke({"input_documents": docs_chunks})

    return result['output_text']

def main():
    st.set_page_config(page_title="Custom PDF Summarizer", page_icon="✍", layout="wide")

    st.title("✍ Custom PDF Summarizer")
    st.markdown("Upload a PDF, provide a custom prompt, and get a tailored summary.")

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        temp_file_path = os.path.join(".", "temp_uploaded_file.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"✅ Successfully uploaded {uploaded_file.name}")

        custom_prompt = st.text_area("Enter your custom prompt", height=150, placeholder="Summarize the key findings for a non-technical audience.")

        if st.button("Generate Summary"):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY not found. Please set it in API.env file.")
                return
            
            if not custom_prompt:
                st.warning("Please enter a prompt to guide the summary.")
            else:
                with st.spinner("🧠 Generating summary..."):
                    try:
                        summary = summarize_pdf(temp_file_path, custom_prompt, api_key)
                        if summary:
                            st.subheader("Your Custom Summary")
                            st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
