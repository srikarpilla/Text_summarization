import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def summarize_pdf(pdf_file_path, custom_prompt_text, api_key):
    """
    Summarizes a PDF file using the Gemini model with a map-reduce strategy.

    Args:
        pdf_file_path (str): The path to the PDF file.
        custom_prompt_text (str): The user's custom prompt for summarization.
        api_key (str): The Google Gemini API key.

    Returns:
        str: The generated summary.
    """
    try:
        # Use a specific, robust model known for large context and text generation.
        # gemini-1.5-flash-latest is efficient and cost-effective.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            google_api_key=api_key
        )

        # Load the PDF document
        loader = PyPDFLoader(pdf_file_path)
        # Split the document into smaller chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)

        # Define the prompt for the "map" step (summarizing each chunk)
        # This prompt incorporates the user's custom instructions.
        map_prompt_template = f"""
        You are a highly skilled summarization assistant.
        Analyze the following text chunk from a larger document.
        Your goal is to create a concise summary of this specific chunk, keeping in mind the user's overall objective: "{custom_prompt_text}"
        Extract only the most important points and main ideas.

        TEXT:
        {{text}}

        CONCISE SUMMARY:
        """
        map_prompt = PromptTemplate.from_template(map_prompt_template)

        # Define the prompt for the "combine" step (creating a final summary from chunk summaries)
        combine_prompt_template = f"""
        You are a highly skilled summarization assistant.
        You will be given a series of summaries from different parts of a single document.
        Your task is to synthesize these into a single, cohesive final summary that is well-structured and easy to read.
        The final summary should directly address the user's original request: "{custom_prompt_text}"
        Do not include any introductory phrases like "The document discusses" or "This text is about". Get straight to the point.

        SUMMARIES:
        {{text}}

        FINAL COHESIVE SUMMARY:
        """
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)

        # Use the map_reduce chain, which is ideal for large documents
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False # Set to True to see processing details in the console
        )

        # Run the chain on the input documents
        result = chain.invoke({"input_documents": docs}, return_only_outputs=True)
        return result['output_text']

    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        st.info("This could be due to an invalid API key, model access issues, or a corrupted PDF file. Please check your API key and the uploaded file.")
        return None

def main():
    """
    The main function for the Streamlit application.
    """
    st.set_page_config(page_title="Custom PDF Summarizer", page_icon="✍️", layout="wide")

    st.title("✍️ Custom PDF Summarizer with Gemini")
    st.markdown("Upload a PDF, provide a custom prompt, and get a tailored summary.")

    # Sidebar for API key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        st.markdown("[Get your Gemini API key](https://aistudio.google.com/app/apikey)")

    # Main area for file upload and interaction
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
            st.stop()

        # Create a temporary directory if it doesn't exist
        temp_dir = "temp_pdf_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the uploaded file to the temporary directory
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"✅ Successfully uploaded '{uploaded_file.name}'")

        custom_prompt = st.text_area(
            "Enter your custom prompt",
            height=150,
            placeholder="e.g., Summarize the key findings for a non-technical audience in bullet points."
        )

        if st.button("Generate Summary"):
            if not custom_prompt:
                st.warning("Please enter a prompt to guide the summary.")
            else:
                with st.spinner("🧠 Generating summary... This may take a moment for large documents."):
                    summary = summarize_pdf(temp_file_path, custom_prompt, api_key)
                    if summary:
                        st.subheader("Your Custom Summary")
                        st.success(summary)

        # Clean up the temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
