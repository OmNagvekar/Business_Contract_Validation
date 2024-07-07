import streamlit as st
from query_processing import QueryProcessor
import os
from streamlit_pdf_viewer import pdf_viewer

def save_pd(uploaded_file):
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def load_pd(name="./new.pdf"):
    with open(name, "rb") as f:
        return f.read()

st.title("Business Contract Validation")
file = st.file_uploader("Pick a PDF file", type="pdf")

if file is not None:
    if "file" not in st.session_state:
        st.session_state.file = file
        save_pd(uploaded_file=file)

    st.write("Processing your file...")
    st.write(st.session_state.file.name)

    # Use the stored file from session state
    if "processed" not in st.session_state:
        obj = QueryProcessor(input_pdf=f"./{st.session_state.file.name}")
        obj.checking_alignment()
        obj.pdf_highlighter()
        st.session_state.processed = True

    pdf_data = load_pd()
    csv_data = load_pd(name="comment.csv")

    st.download_button(
        label="Download new PDF",
        data=pdf_data,
        file_name=f"processed_{st.session_state.file.name}"
    )

    st.download_button(
        label="Download Response CSV",
        data=csv_data,
        file_name="comment.csv"
    )

    pdf_viewer(input="./new.pdf", width=700)
