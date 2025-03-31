import streamlit as st
from query_processing import QueryProcessor
import os
from streamlit_pdf_viewer import pdf_viewer
import logging
import sys
import time
from datetime import datetime, timedelta
import threading

# Set up constants and log directory
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_DIR = 'Logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"logs_{TODAY_DATE}.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # Save logs locally
        logging.StreamHandler(sys.stdout),  # Also print logs in console
    ],
)

logger = logging.getLogger(__name__)
logger.info("Starting Streamlit application.")

def delete_old_logs():
    """Delete log files older than 1 day if they contain no errors (or only specific memory errors),
    and delete files older than 3 days regardless. Files in use are skipped.
    """
    while True:
        try:
            now = datetime.now()
            for filename in os.listdir(LOG_DIR):
                if filename.startswith("logs_") and filename.endswith(".log"):
                    file_path = os.path.join(LOG_DIR, filename)
                    # Extract date from filename (assumes format "logs_YYYY-MM-DD.log")
                    file_date_str = filename[5:15]
                    try:
                        file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    except Exception as e:
                        logger.error("Failed to parse date from filename %s: %s", filename, e)
                        continue

                    # Process files older than 1 day
                    if now - file_date > timedelta(days=1):
                        try:
                            with open(file_path, "r", encoding="utf-8") as log_file:
                                content = log_file.read()
                        except Exception as e:
                            logger.error("Error reading file %s: %s", filename, e)
                            continue

                        if "ERROR" not in content:
                            try:
                                os.remove(file_path)
                                logger.info("Deleted old log: %s", filename)
                            except Exception as e:
                                if hasattr(e, "winerror") and e.winerror == 32:
                                    logger.warning("File %s is in use; skipping deletion.", filename)
                                else:
                                    logger.error("Error deleting file %s: %s", filename, e)
                        else:
                            # Check if errors are only due to memory issues
                            error_lines = [line for line in content.splitlines() if "ERROR" in line]
                            if error_lines and all("model requires more system memory" in line for line in error_lines):
                                try:
                                    os.remove(file_path)
                                    logger.info("Deleted old log (only memory error present): %s", filename)
                                except Exception as e:
                                    if hasattr(e, "winerror") and e.winerror == 32:
                                        logger.warning("File %s is in use; skipping deletion.", filename)
                                    else:
                                        logger.error("Error deleting file %s: %s", filename, e)

                    # Delete files older than 3 days regardless of content
                    if now - file_date > timedelta(days=3):
                        try:
                            os.remove(file_path)
                            logger.info("Deleted old log (older than 3 days): %s", filename)
                        except Exception as e:
                            if hasattr(e, "winerror") and e.winerror == 32:
                                logger.warning("File %s is in use; skipping deletion.", filename)
                            else:
                                logger.error("Error deleting file %s: %s", filename, e)

        except Exception as e:
            logger.error("Error in log cleanup: %s", e, exc_info=True)

        time.sleep(3600)  # Run every hour

# Start cleanup thread for old logs
cleanup_thread = threading.Thread(target=delete_old_logs, daemon=True)
cleanup_thread.start()

@st.cache_data
def get_pdf_bytes(uploaded_file):
    """Return PDF file bytes from uploaded file."""
    try:
        file_bytes = bytes(uploaded_file.getbuffer())
        logger.info("Loaded uploaded file bytes: %s", uploaded_file.name)
        return file_bytes
    except Exception as e:
        logger.error("Failed to load uploaded file %s: %s", uploaded_file.name, e)
        return None


# Streamlit UI
st.title("Business Contract Validation")
uploaded_file = st.file_uploader("Pick a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if file is already in session state; if not, initialize session state keys.
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file.name != uploaded_file.name:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.processed = False  # flag to track processing
        st.session_state.result = None

    st.write("Processing your file...")
    st.write(st.session_state.uploaded_file.name)
    logger.info("Processing file: %s", st.session_state.uploaded_file.name)

    # Process the file only once if it hasn't been processed yet
    if not st.session_state.processed:
        with st.spinner("Please wait while we process your file...", show_time=True):
            pdf_bytes = get_pdf_bytes(st.session_state.uploaded_file)
            try:
                logger.info("Initializing QueryProcessor for file: %s", st.session_state.uploaded_file.name)
                qp = QueryProcessor(pdf_bytes, remote_llm=True)
                qp.checking_alignment()
                pdf_data, csv_data = qp.pdf_highlighter()
                st.session_state.processed = True
                st.session_state.result = (pdf_data, csv_data)
                logger.info("Finished processing file: %s", st.session_state.uploaded_file.name)
            except KeyboardInterrupt:
                logger.info("Execution interrupted by user (KeyboardInterrupt).")
                st.error("Execution interrupted by user.")
                if 'qp' in locals():
                    qp.stop = True
                st.stop()
                sys.exit(0)
            except Exception as e:
                logger.error("Error during file processing: %s", e, exc_info=True)
                if 'qp' in locals():
                    qp.stop = True
                st.error("An error occurred during processing. Please check the logs for details.")
    else:
        # If already processed, retrieve results from session state.
        pdf_data, csv_data = st.session_state.result
        st.success("File has been processed successfully.")

    if pdf_data:
        st.download_button(
            label="Download Highlighted PDF",
            data=pdf_data,
            file_name=f"Highlighted_{st.session_state.uploaded_file.name}",
            mime='application/pdf'
        )
    if csv_data:
        st.download_button(
            label="Download Response CSV",
            data=csv_data,
            file_name=f"{st.session_state.uploaded_file.name}_comments.csv",
            mime='text/csv'
        )
    if pdf_data:
        pdf_viewer(input=pdf_data, width=700)
