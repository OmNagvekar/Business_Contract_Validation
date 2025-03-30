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
                        logging.error("Failed to parse date from filename %s: %s", filename, e)
                        continue

                    # Process files older than 1 day
                    if now - file_date > timedelta(days=1):
                        try:
                            with open(file_path, "r", encoding="utf-8") as log_file:
                                content = log_file.read()
                        except Exception as e:
                            logging.error("Error reading file %s: %s", filename, e)
                            continue

                        if "ERROR" not in content:
                            try:
                                os.remove(file_path)
                                logging.info("Deleted old log: %s", filename)
                            except Exception as e:
                                if hasattr(e, "winerror") and e.winerror == 32:
                                    logging.warning("File %s is in use; skipping deletion.", filename)
                                else:
                                    logging.error("Error deleting file %s: %s", filename, e)
                        else:
                            # Check if errors are only due to memory issues
                            error_lines = [line for line in content.splitlines() if "ERROR" in line]
                            if error_lines and all("model requires more system memory" in line for line in error_lines):
                                try:
                                    os.remove(file_path)
                                    logging.info("Deleted old log (only memory error present): %s", filename)
                                except Exception as e:
                                    if hasattr(e, "winerror") and e.winerror == 32:
                                        logging.warning("File %s is in use; skipping deletion.", filename)
                                    else:
                                        logging.error("Error deleting file %s: %s", filename, e)

                    # Delete files older than 3 days regardless of content
                    if now - file_date > timedelta(days=3):
                        try:
                            os.remove(file_path)
                            logging.info("Deleted old log (older than 3 days): %s", filename)
                        except Exception as e:
                            if hasattr(e, "winerror") and e.winerror == 32:
                                logging.warning("File %s is in use; skipping deletion.", filename)
                            else:
                                logging.error("Error deleting file %s: %s", filename, e)

        except Exception as e:
            logging.error("Error in log cleanup: %s", e, exc_info=True)

        time.sleep(3600)  # Run every hour

# Start cleanup thread for old logs
cleanup_thread = threading.Thread(target=delete_old_logs, daemon=True)
cleanup_thread.start()

def save_pd(uploaded_file):
    """Save the uploaded PDF file."""
    file_path = os.path.join("./", uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info("Saved uploaded file: %s", uploaded_file.name)
    except Exception as e:
        logging.error("Failed to save uploaded file %s: %s", uploaded_file.name, e)

def load_pd(name="./new.pdf"):
    """Load a PDF or CSV file."""
    try:
        with open(name, "rb") as f:
            data = f.read()
        logging.info("Loaded file: %s", name)
        return data
    except Exception as e:
        logging.error("Error loading file %s: %s", name, e)
        return None

# Streamlit UI
st.title("Business Contract Validation")
file = st.file_uploader("Pick a PDF file", type="pdf")

if file is not None:
    if "file" not in st.session_state:
        st.session_state.file = file
        save_pd(uploaded_file=file)
    st.write("Processing your file...")
    st.write(st.session_state.file.name)
    logging.info("Processing file: %s", st.session_state.file.name)

    # Process the file only once per session
    if "processed" not in st.session_state:
        try:
            logging.info("Initializing QueryProcessor for file: %s", st.session_state.file.name)
            obj = QueryProcessor(input_pdf=f"./{st.session_state.file.name}",remote_llm=True)
            obj.checking_alignment()
            obj.pdf_highlighter()
            st.session_state.processed = True
            logging.info("Finished processing file: %s", st.session_state.file.name)
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user (KeyboardInterrupt). Exiting gracefully.")
            st.error("Execution interrupted by user.")
            STOP_EVENT.set()
            obj.stop =True
            sys.exit(0)
        except Exception as e:
            logging.error("Error during file processing: %s", e, exc_info=True)
            obj.stop =True
            st.error("An error occurred during processing. Please check the logs for details.")

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
