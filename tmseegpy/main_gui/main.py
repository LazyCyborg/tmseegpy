# main.py
import streamlit as st
from app import TEPApp

def main():
    # Set page config
    st.set_page_config(
        page_title="HePoTEP",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for app persistence
    if 'app' not in st.session_state:
        st.session_state.app = TEPApp()

    # Run the app using the persistent instance
    st.session_state.app.run()

if __name__ == "__main__":
    main()