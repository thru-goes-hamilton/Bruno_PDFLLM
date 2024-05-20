import streamlit as st
import PyPDF2
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document

# Define initial values for variables
prompt = "Your prompt will show here"
answer = "Bruno will answer any questions you have about this pdf" 
uploaded_file = None
max_height = 300
upper_height = "<br>"
lower_height = "<br>"
explanation = """
            <div style="text-align: center">
                <p>Upload any PDF and chat with Bruno!!<br>Interact with PDFs by leveraging the power of RAG and LLMs.</p>
            </div>
            """

# Initialize session state variables if they don't exist
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "prompt_answer_html" not in st.session_state:
    st.session_state.prompt_answer_html = ""

# Define the processing function
def process_data(prompt, file):
    # Extract text from the uploaded PDF file
    if file is not None:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    else:
        text = ""

    # Set up LLM and embedding models
    llm = GradientBaseModelLLM(
        base_model_slug="llama2-7b-chat", max_tokens=400, 
        access_token=st.secrets["gradient_access_token"],
        workspace_id=st.secrets["gradient_workspace_id"],
    )
    embed_model = GradientEmbedding(
        gradient_access_token=st.secrets["gradient_access_token"],
        gradient_workspace_id=st.secrets["gradient_workspace_id"],
        gradient_model_slug="bge-large",
    )
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 1024

    # Load PDF text as a document
    documents = [Document(text=text, meta={"name": "PDF Document"})]

    # Create a VectorStoreIndex and query engine
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Query the engine with the provided prompt
    response = query_engine.query(prompt)
    return extract_refined_answer(response.response)

def extract_refined_answer(text):
    start_phrase = ':'
    start_index = text.find(start_phrase)

    # The length of the start_phrase is added to get the start of the actual refined answer.
    start_index += len(start_phrase)
    end_index1 = text.find('The refined answer', start_index + 1)
    end_index2 = text.find('The original answer', start_index + 1)
    if end_index1 != -1 and end_index2 != -1:
        end_index = min(end_index1, end_index2)
        refined_answer = text[start_index:end_index].strip()
    elif end_index1 != -1:
        refined_answer = text[start_index:end_index1].strip()
    elif end_index2 != -1:
        refined_answer = text[start_index:end_index2].strip()
    else:
        refined_answer = text[start_index:].strip()

    return refined_answer

# Placeholder for upper spacing
upper_height_placeholder = st.markdown(upper_height, unsafe_allow_html=True)

# Adding a new heading
st.markdown("<h1 style='text-align: center; color: #E0F0EA; font-size: 36px;font-family: Merriweather;'> BRUNO</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .container {
        padding: 18px;
        background-color: green;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Container for the main content
with st.container() as container:
    # Explanation and lower spacing placeholders
    explanation_placeholder = st.markdown(explanation, unsafe_allow_html=True)
    lower_height_placeholder = st.markdown("<br>", unsafe_allow_html=True)

    # Placeholder for displaying previous prompts and answers
    prompt_answer_placeholder = st.empty()
    if st.session_state.prompt_answer_html != "":
        prompt_answer_placeholder.markdown(st.session_state.prompt_answer_html, unsafe_allow_html=True)

    # Input field for new prompt
    new_prompt = st.text_input("", placeholder="Enter new prompt",key="text")

    st.markdown(
        """
        <style>
        div[data-baseweb="input"] > div {
            background-color: #574f7d !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Custom CSS for file uploader
    css = '''
    <style>
        [data-testid='stFileUploader'] {
            width: max-content;
        }
        [data-testid='stFileUploader'] section {
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid='stFileUploader'] section + div {
            float: right;
            padding-top: 0;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    # File uploader for PDF files
    uploaded_file = st.file_uploader("", type="pdf", accept_multiple_files=False, key="pdf")

    # Submit button to process the prompt and uploaded file
    submit_button = st.button("Submit", disabled=(uploaded_file is None or new_prompt == ""))
    if submit_button:
        upper_height_placeholder.empty()
        explanation_placeholder.empty()
        lower_height_placeholder.empty()
        if new_prompt:
            prompt = str(new_prompt)
            st.session_state.prompts.append(prompt)

        # Call the processing function
        with st.spinner("Processing..."):
            if uploaded_file is not None:
                # Process the uploaded file
                answer = process_data(prompt, uploaded_file)
                st.session_state.answers.append(answer)
                st.session_state["text"] = ""

            # Generate HTML for the new prompt and answer
            new_prompt_answer_html = ""
            new_prompt_answer_html += f'<p style="padding: 9px 12px; margin-left: 18px; margin-right: 18px; color: #E0F0EA; background-color:#574F7D;border-radius: 10px;"><span style="font-weight: bold; font-size: 20px; color: #E0F0EA ;">KU</span>&nbsp;&nbsp;{st.session_state.prompts[-1]}</p>'
            new_prompt_answer_html += f'<div style="max-height:500px; padding: 9px 12px; margin-left: 18px; margin-right: 18px; margin-bottom: 36px; overflow-y: auto; padding: 18px; color: #3C2A4D; background-color: #E0F0EA; border-radius: 10px; border: 3px solid #93ACBD">{st.session_state.answers[-1]}</div>'

            # Append the new content to the existing content
            st.session_state.prompt_answer_html += new_prompt_answer_html

            # Display the updated content
            prompt_answer_placeholder.markdown(st.session_state.prompt_answer_html, unsafe_allow_html=True)
