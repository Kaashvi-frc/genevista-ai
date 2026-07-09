import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="SCD-Analytics Hub",
    page_icon="🩸",
    layout="centered"
)

# Main Header
st.title("SCD-Analytics Hub")
st.markdown("### Sickle Cell Disease Patient Registry")
st.markdown("Please fill out the clinical parameters below. Fields marked with an asterisk (*) are required.")

# Create the form
with st.form("patient_data_form"):
    
    # --- SECTION 1: Patient Information ---
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age *", min_value=0, max_value=120, step=1)
    with col2:
        gender = st.selectbox("Gender *", ["Male", "Female", "Other", "Prefer not to say"])
    
    genotype = st.selectbox(
        "Confirmed sickle cell genotype *", 
        ["HbSS", "HbSC", "HbSβ⁰ Thalassaemia", "HbSβ⁺ Thalassaemia", "Other", "Unknown"]
    )

    st.divider()

    # --- SECTION 2: Laboratory Results ---
    st.subheader("Laboratory Results")
    hb_level = st.number_input("Haemoglobin level (g/dL) *", min_value=0.0, max_value=30.0, format="%.1f")
    
    col3, col4 = st.columns(2)
    with col3:
        hbf_level = st.number_input("Foetal haemoglobin (HbF %) (Optional)", min_value=0.0, max_value=100.0, format="%.1f", value=None)
    with col4:
        wbc_count = st.number_input("White blood cell count (×10⁹/L) (Optional)", min_value=0.0, max_value=100.0, format="%.1f", value=None)

    st.divider()

    # --- SECTION 3: Clinical History ---
    st.subheader("Clinical History")
    col5, col6 = st.columns(2)
    with col5:
        voc_count = st.number_input("Vaso-occlusive crises (past 12 months) *", min_value=0, step=1)
    with col6:
        hosp_admissions = st.number_input("SCD hospital admissions (past 12 months) *", min_value=0, step=1)
    
    col7, col8 = st.columns(2)
    with col7:
        acs_history = st.radio("History of acute chest syndrome *", ["Yes", "No"])
        hydroxyurea = st.radio("Currently taking hydroxyurea *", ["Yes", "No"])
    with col8:
        stroke_history = st.radio("History of SCD-related stroke *", ["Yes", "No"])
        transfusions = st.selectbox("Blood transfusions (past 12 months) *", ["None", "1–2", "3–5", "More than 5"])

    st.divider()

    # --- SECTION 4: Supporting Documents ---
    st.subheader("Supporting Documents")
    lab_report = st.file_uploader("Upload latest laboratory report (Optional)", type=["pdf", "jpg", "png"])
    vcf_file = st.file_uploader("Upload genetic sequencing file (Optional)", type=["vcf", "vcf.gz"])

    st.divider()

    # --- SECTION 5: Additional Information ---
    st.subheader("Additional Information")
    clinical_notes = st.text_area("Additional clinical notes (Optional)", height=150)

    # Form Submission
    submitted = st.form_submit_button("Submit Patient Data", type="primary")

# Logic after submission
if submitted:
    # Here you would typically connect to a database or save the files
    st.success("Patient data successfully submitted!")
    st.balloons()
