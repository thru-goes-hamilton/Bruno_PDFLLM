import streamlit as st
import fitz
from llama_index.llms.gradient import GradientBaseModelLLM
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.core import VectorStoreIndex, Settings, Document

# Define initial values for variables
prompt="Your prompt will show here"
answer="Bruno will answer any quesitons you have about this pdf" 
uploaded_file = None
max_height = 300

# Define the processing function
def process_data(prompt, file):

    # Extract text from the uploaded PDF file
    if file is not None:
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
            text = ""
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text += page.get_text()
    else:
        text = ""

    
    # Set up LLM and embedding models
    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400, access_token="ykT7u7KHiikryOLUPIZzP3YWLb5BQEPw", workspace_id= "9d431935-abb0-40b0-9905-3cf70173157d_workspace")
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
    
    return response.response

#Commenting out the title
# Define the Streamlit app layout
#st.title("BRUNO")

#Adding new heading here
st.markdown("<h1 style='text-align: center; color: #E0F0EA; font-size: 36px;font-family: Merriweather;'> BRUNO</h1>",unsafe_allow_html=True)

#Change here

# st.markdown(
#     """
#     <style>
#     .css-2trqyj-TitleContainer {
#         text-align: center;
#         font-size: 36px;
#         font-family: Merriweather;
#         padding-top: 10px;
#         padding-bottom: 10px;
#         margin: 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

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

with st.container() as container:    

    prompt_placeholder = st.empty()
    # prompt_placeholder.markdown(f'<p style="padding: 9px 12px; margin-left: 18px; margin-right: 18px; margin-bottom: 9; color: #E0F0EA; background-color: #574F7D; border-radius: 10px;"><span style="font-weight: bold; font-size: 20px; color: black;">KU</span>&nbsp;&nbsp;{prompt}</p>', unsafe_allow_html=True)
    # # Display the styled text using 

    answer_placeholder = st.empty()
    # answer_placeholder.markdown(f'<div style="max-height: {max_height}; overflow-y: auto; padding: 9px 12px; margin-left: 18px; margin-right: 18px; margin-bottom: 12; color: #3C2A4D; background-color: #E0F0EA; border-radius: 10px; border: 3px solid #93ACBD">{answer}</div>', unsafe_allow_html=True)
    # , unsafe_allow_html=True)

    prompt = st.chat_input("Say something")
    # new_prompt = st.text_input("",placeholder="Enter new prompt")

    # new_prompt = st.empty()
    # new_prompt.markdown(
    #     f'<input type="text" style="width: 100%; max-height: {max_height}; overflow-y: auto; padding: 9px 12px; margin-top: 12px; margin-bottom: 9; color: #3C2A4D; background-color: #E0F0EA; border-radius: 10px; border: 3px solid #93ACBD" />',
    #     unsafe_allow_html=True
    # )
    # Input text field to update the prompt variable
    # new_prompt = st.text_input("Enter new prompt")    
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

    uploaded_file = st.file_uploader("", type="pdf", accept_multiple_files=False, key="pdf")

# Button 1: Upload PDF
    if st.button("Submit"):
        if new_prompt:
            prompt = str(new_prompt)
            prompt_placeholder.markdown(f'<p style="padding: 9px 12px; margin-left: 18px; margin-right: 18px; color: #E0F0EA; background-color:#574F7D;border-radius: 10px;"><span style="font-weight: bold; font-size: 20px; color: #E0F0EA ;">KU</span>&nbsp;&nbsp;{prompt}</p>', unsafe_allow_html=True)

    # Call the processing function
        with st.spinner("Processing..."):
            
            # Update prompt if a new one is entered
        
        
            # Check if a file was uploaded
            if uploaded_file is not None:
                # Process the uploaded file
                answer = process_data(prompt, uploaded_file)
                answer_placeholder.markdown(f'<div style="max-height: {max_height};padding: 9px 12px; margin-left: 18px; margin-right: 18px; overflow-y: auto; padding: 18px; color: #3C2A4D; background-color: #E0F0EA; border-radius: 10px; border: 3px solid #93ACBD">{answer}</div>'
    , unsafe_allow_html=True)
            
                # del uploaded_file  # Delete the uploaded file object to clear memory
            else:
                # Process without a file            
                answer = "Please upload a PDF file to get started."
                answer_placeholder.markdown(f'<div style="max-height: {max_height}; padding: 9px 12px; margin-left: 18px; margin-right: 18px;  overflow-y: auto; padding: 18px; color: #3C2A4D; background-color: #E0F0EA; border-radius: 10px; border: 3px solid #93ACBD">{answer}</div>'
    , unsafe_allow_html=True)

    
        # Display the result
        st.success("Processing complete!")