import streamlit as st

# Configure the page (optional but recommended for a clean look)
st.set_page_config(page_title="Medico-AI", layout="wide")

# -----------------------------------------------------------------------------
# UI Header & Logos
# -----------------------------------------------------------------------------
try:
    # 1. The app tries to find and display the image here
    st.image("Medico-AI_header.png", use_container_width=True)
except FileNotFoundError:
    # 2. FALLBACK: If the image is NOT found, it adds the name as a header instead
    st.markdown("<h1 style='text-align: center;'>Medico-AI</h1>", unsafe_allow_html=True)

st.markdown("""
**Medico-AI** is an AI platform designed to assist in better medical diagnosis using advanced artificial intelligence. It is built upon a computational framework originally developed by <a href="https://www.linkedin.com/in/deeptarup-biswas-039825178/" target="_blank">Dr. Deeptarup Biswas</a>.
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Application Logic Goes Here
# -----------------------------------------------------------------------------

st.subheader("Diagnostic Tool Interface")
st.info("Upload patient data or enter symptoms below to begin the AI analysis.")

# Add your file uploaders, text inputs, and processing logic below this line
