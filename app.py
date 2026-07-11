"""Streamlit application entrypoint for GeneVista AI."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

from genevista.backend.analysis_orchestrator import AnalysisOrchestrator
from genevista.backend.document_parsing import DocumentParsingService
from genevista.backend.patient_processing import PatientValidationError, validate_patient_profile
from genevista.config import get_config
from genevista.domain import ClinicalHistory, DocumentBundle, LabResults, PatientProfile
from genevista.logging_config import configure_logging
from genevista.reporting.pdf import PDFReportGenerator, ReportGenerationError
from genevista.ui.components import card, digital_twin_radar, download_pdf_button, feature_importance_chart, risk_bar
from genevista.ui.styles import apply_global_styles


def init_state() -> None:
    defaults = {
        "stage": "landing",
        "question_step": 0,
        "patient_profile": None,
        "analysis_result": None,
        "report_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go(stage: str) -> None:
    st.session_state.stage = stage
    st.rerun()


def running_inside_streamlit() -> bool:
    """Return whether this file is being executed by Streamlit."""

    return get_script_run_ctx(suppress_warning=True) is not None


def landing_page() -> None:
    st.markdown(
        """
        <div class="gv-hero">
          <div class="gv-kicker">Biomedical intelligence for Sickle Cell Disease</div>
          <h1>GeneVista AI</h1>
          <p class="gv-subtitle">
          Evidence-grounded clinical and molecular interpretation for Sickle Cell Disease,
          combining patient information, laboratory signals, supervised prediction architecture,
          consensus biomedical evidence, and educational Digital Twins.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Analysis", use_container_width=True):
                go("questionnaire")
        with c2:
            if st.button("Learn More", use_container_width=True):
                go("learn")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Methodology", use_container_width=True):
                go("methodology")
        with c4:
            if st.button("About", use_container_width=True):
                go("about")
        st.markdown('<div class="gv-divider"></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            card("Consensus evidence", "Aggregates ClinVar, ClinGen, OMIM, NCBI Gene, UniProt, PubMed, AlphaFold, and future sources through extensible providers.")
            card("Prediction architecture", "Separates training, evaluation, registry, and inference so real supervised datasets can replace the bundled baseline.")
        with cols[1]:
            card("Clinical context", "Interprets genotype, haemoglobin, foetal haemoglobin, complications, admissions, and therapies together.")
            card("Educational Digital Twin", "Compares healthy reference biology with patient-specific disease biology without claiming exact physiological simulation.")
    with right:
        molecular_illustration()
        st.markdown(
            """
            <span class="gv-pill">Supported disease: Sickle Cell Disease</span>
            <span class="gv-pill">Primary gene: HBB</span>
            <span class="gv-pill">Evidence grounded</span>
            <span class="gv-pill">Clinician-readable</span>
            """,
            unsafe_allow_html=True,
        )


def molecular_illustration() -> None:
    st.markdown(
        """
        <svg viewBox="0 0 520 380" role="img" aria-label="Molecular biology illustration" style="width:100%; height:auto;">
          <rect x="8" y="8" width="504" height="364" rx="8" fill="#fffdf7" stroke="#d8d2c4"/>
          <path d="M91 274 C141 176, 195 196, 237 97 S358 112, 417 43" fill="none" stroke="#123047" stroke-width="2.5"/>
          <path d="M102 303 C152 205, 213 225, 255 126 S376 141, 435 72" fill="none" stroke="#7a2f39" stroke-width="2.5"/>
          <g fill="#31513c">
            <circle cx="91" cy="274" r="9"/><circle cx="153" cy="202" r="7"/><circle cx="237" cy="97" r="8"/>
            <circle cx="319" cy="105" r="7"/><circle cx="417" cy="43" r="9"/>
          </g>
          <g fill="#7a2f39">
            <circle cx="102" cy="303" r="8"/><circle cx="176" cy="229" r="6"/><circle cx="255" cy="126" r="7"/>
            <circle cx="349" cy="132" r="6"/><circle cx="435" cy="72" r="8"/>
          </g>
          <g fill="none" stroke="#536171" stroke-width="1.2" opacity="0.8">
            <ellipse cx="166" cy="82" rx="54" ry="24"/><ellipse cx="166" cy="82" rx="24" ry="54"/>
            <ellipse cx="362" cy="270" rx="58" ry="24"/><ellipse cx="362" cy="270" rx="24" ry="58"/>
          </g>
          <text x="56" y="56" fill="#123047" font-family="IBM Plex Sans" font-size="15" font-weight="700">HBB molecular context</text>
          <text x="56" y="82" fill="#536171" font-family="IBM Plex Sans" font-size="13">Variant, protein and red-cell biology are interpreted together.</text>
          <line x1="68" y1="320" x2="452" y2="320" stroke="#d8d2c4"/>
          <text x="68" y="346" fill="#31513c" font-family="IBM Plex Sans" font-size="13" font-weight="700">Healthy reference</text>
          <text x="330" y="346" fill="#7a2f39" font-family="IBM Plex Sans" font-size="13" font-weight="700">Disease biology</text>
        </svg>
        """,
        unsafe_allow_html=True,
    )


def informational_page(kind: str) -> None:
    labels = {
        "learn": ("Platform", "GeneVista works by connecting patient inputs to molecular, clinical, and literature evidence before producing explanations."),
        "methodology": ("Methodology", "The MVP uses modular retrieval, consensus scoring, supervised-ML-ready inference, and traceable explanation generation."),
        "about": ("About", "GeneVista AI is designed as premium biomedical software for Sickle Cell Disease research and clinical interpretation support."),
    }
    kicker, subtitle = labels[kind]
    st.markdown(f'<div class="gv-kicker">{kicker}</div><h1>{kicker}</h1><p class="gv-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        card("Why it exists", "Manual interpretation requires scattered literature, databases, clinical context, and careful uncertainty handling.")
    with cols[1]:
        card("How it works", "Inputs are validated, features are engineered, evidence is retrieved, model outputs are scored, and explanations are grounded.")
    with cols[2]:
        card("Scientific credibility", "The MVP exposes uncertainty, evidence completeness, source agreement, and supported prediction horizons.")
    if st.button("Start Analysis"):
        go("questionnaire")
    if st.button("Back"):
        go("landing")


def questionnaire() -> None:
    st.markdown('<div class="gv-kicker">Patient questionnaire</div><h1>New Sickle Cell Analysis</h1>', unsafe_allow_html=True)
    steps = ["Patient Information", "Laboratory Results", "Clinical History", "Supporting Documents", "Additional Information"]
    st.progress((st.session_state.question_step + 1) / len(steps), text=f"Section {st.session_state.question_step + 1} of {len(steps)}")
    step = st.session_state.question_step

    form_key = f"questionnaire_{step}"
    with st.form(form_key):
        if step == 0:
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("age", 28), step=1)
            sex = st.selectbox("Sex", ["Female", "Male", "Intersex", "Not specified"], index=["Female", "Male", "Intersex", "Not specified"].index(st.session_state.get("sex", "Not specified")))
            genotype = st.selectbox("Confirmed genotype", ["HbSS", "HbSC", "HbS beta+ thalassaemia", "HbS beta0 thalassaemia"], index=0)
        elif step == 1:
            hb = st.number_input("Haemoglobin (g/dL)", min_value=2.0, max_value=20.0, value=st.session_state.get("hb", 8.7), step=0.1)
            hbf = st.number_input("Foetal haemoglobin (%)", min_value=0.0, max_value=50.0, value=st.session_state.get("hbf", 9.5), step=0.1)
            wbc = st.number_input("White blood cell count (x10^9/L)", min_value=0.5, max_value=80.0, value=st.session_state.get("wbc", 12.0), step=0.1)
        elif step == 2:
            voc = st.number_input("Number of vaso-occlusive crises in the last year", min_value=0, max_value=100, value=st.session_state.get("voc", 2), step=1)
            admissions = st.number_input("Hospital admissions in the last year", min_value=0, max_value=100, value=st.session_state.get("admissions", 1), step=1)
            acute_chest = st.checkbox("History of acute chest syndrome", value=st.session_state.get("acute_chest", False))
            stroke = st.checkbox("Stroke history", value=st.session_state.get("stroke", False))
            hydroxyurea = st.checkbox("Currently on hydroxyurea", value=st.session_state.get("hydroxyurea", True))
            transfusions = st.checkbox("History of blood transfusions", value=st.session_state.get("transfusions", False))
        elif step == 3:
            lab_report = st.file_uploader("Laboratory report upload", type=["txt", "csv", "pdf"])
            vcf_file = st.file_uploader("VCF upload", type=["vcf", "txt"])
            notes = st.text_area("Clinical notes", value=st.session_state.get("notes", ""), height=150)
        else:
            additional = st.text_area("Additional information", value=st.session_state.get("additional", ""), height=160)

        prev_col, next_col = st.columns([0.35, 0.65])
        with prev_col:
            previous = st.form_submit_button("Back", use_container_width=True)
        with next_col:
            label = "Generate Analysis" if step == len(steps) - 1 else "Continue"
            submitted = st.form_submit_button(label, use_container_width=True)

    if previous:
        st.session_state.question_step = max(0, step - 1)
        st.rerun()
    if submitted:
        if step == 0:
            st.session_state.age, st.session_state.sex, st.session_state.genotype = age, sex, genotype
        elif step == 1:
            st.session_state.hb, st.session_state.hbf, st.session_state.wbc = hb, hbf, wbc
        elif step == 2:
            st.session_state.voc, st.session_state.admissions = voc, admissions
            st.session_state.acute_chest, st.session_state.stroke = acute_chest, stroke
            st.session_state.hydroxyurea, st.session_state.transfusions = hydroxyurea, transfusions
        elif step == 3:
            parser = DocumentParsingService()
            st.session_state.documents = parser.parse_uploads(lab_report, vcf_file, notes)
            st.session_state.notes = notes
        else:
            st.session_state.additional = additional
            profile = build_patient_profile()
            try:
                validate_patient_profile(profile)
            except PatientValidationError as exc:
                st.error(str(exc))
                return
            st.session_state.patient_profile = profile
            go("processing")
        if step < len(steps) - 1:
            st.session_state.question_step = step + 1
            st.rerun()


def build_patient_profile() -> PatientProfile:
    return PatientProfile(
        age=int(st.session_state.get("age", 0)),
        sex=st.session_state.get("sex", "Not specified"),
        confirmed_genotype=st.session_state.get("genotype", "HbSS"),
        labs=LabResults(
            haemoglobin_g_dl=float(st.session_state.get("hb", 0)),
            foetal_haemoglobin_pct=float(st.session_state.get("hbf", 0)),
            white_blood_cell_count=float(st.session_state.get("wbc", 0)),
        ),
        history=ClinicalHistory(
            vaso_occlusive_crises_year=int(st.session_state.get("voc", 0)),
            hospital_admissions_year=int(st.session_state.get("admissions", 0)),
            acute_chest_syndrome=bool(st.session_state.get("acute_chest", False)),
            stroke_history=bool(st.session_state.get("stroke", False)),
            hydroxyurea=bool(st.session_state.get("hydroxyurea", False)),
            blood_transfusions=bool(st.session_state.get("transfusions", False)),
        ),
        documents=st.session_state.get("documents") or DocumentBundle(),
        additional_information=st.session_state.get("additional", ""),
    )


def processing_screen() -> None:
    st.markdown('<div class="gv-kicker">Analysis in progress</div><h1>Generating GeneVista Intelligence</h1>', unsafe_allow_html=True)
    stages = [
        "Reading patient information",
        "Extracting uploaded documents",
        "Searching biomedical databases",
        "Retrieving literature",
        "Running prediction model",
        "Comparing evidence",
        "Generating confidence score",
        "Creating Digital Twin",
        "Preparing report",
        "Finalising dashboard",
    ]
    progress = st.progress(0, text=stages[0])
    status = st.empty()
    config = get_config()
    delay = config.processing_seconds / max(len(stages), 1)
    for index, stage in enumerate(stages, start=1):
        status.info(stage)
        progress.progress(index / len(stages), text=stage)
        if delay > 0:
            time.sleep(delay)

    orchestrator = AnalysisOrchestrator()
    result = orchestrator.run(st.session_state.patient_profile)
    st.session_state.analysis_result = result
    try:
        st.session_state.report_path = PDFReportGenerator(config.reports_path).generate(result)
    except ReportGenerationError as exc:
        st.session_state.report_path = None
        st.warning(f"Report generation failed: {exc}")
    go("dashboard")


def dashboard() -> None:
    result = st.session_state.analysis_result
    if result is None:
        go("landing")
        return
    st.markdown('<div class="gv-kicker">Prediction dashboard</div><h1>GeneVista Analysis Dashboard</h1>', unsafe_allow_html=True)
    patient = result.patient
    prediction = result.prediction
    summary_cols = st.columns(4)
    summary_cols[0].metric("Severity", prediction.severity_label)
    summary_cols[1].metric("Risk category", prediction.risk_category)
    summary_cols[2].metric("Model confidence", f"{prediction.model_confidence:.0%}")
    summary_cols[3].metric("Evidence confidence", f"{result.evidence_confidence:.0%}")

    tabs = st.tabs([
        "Patient Summary",
        "Prediction Summary",
        "Evidence",
        "Digital Twin",
        "Report",
    ])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Clinical inputs")
            st.write(f"Age: **{patient.age}**")
            st.write(f"Sex: **{patient.sex}**")
            st.write(f"Confirmed genotype: **{patient.confirmed_genotype}**")
            st.write(f"Haemoglobin: **{patient.labs.haemoglobin_g_dl} g/dL**")
            st.write(f"Foetal haemoglobin: **{patient.labs.foetal_haemoglobin_pct}%**")
            st.write(f"White blood cell count: **{patient.labs.white_blood_cell_count} x10^9/L**")
        with c2:
            st.subheader("History")
            st.write(f"Vaso-occlusive crises: **{patient.history.vaso_occlusive_crises_year}/year**")
            st.write(f"Hospital admissions: **{patient.history.hospital_admissions_year}/year**")
            st.write(f"Acute chest syndrome: **{yes_no(patient.history.acute_chest_syndrome)}**")
            st.write(f"Stroke history: **{yes_no(patient.history.stroke_history)}**")
            st.write(f"Hydroxyurea: **{yes_no(patient.history.hydroxyurea)}**")
            st.write(f"Blood transfusions: **{yes_no(patient.history.blood_transfusions)}**")
    with tabs[1]:
        st.subheader("Clinical interpretation")
        st.write(result.clinical_interpretation)
        st.subheader("Supported prediction horizon")
        st.info(prediction.progression_statement)
        st.subheader("Predicted likelihood of future complications")
        for label, value in prediction.complication_risks.items():
            risk_bar(label, value)
        st.subheader("Feature importance")
        st.plotly_chart(feature_importance_chart(result), use_container_width=True)
    with tabs[2]:
        st.subheader("Consensus evidence summary")
        for note in result.evidence_notes:
            st.info(note)
        for record in result.evidence:
            with st.expander(f"{record.source}: {record.title}"):
                st.write(record.summary)
                st.write(f"Quality weight: **{record.quality:.0%}**")
                if record.url:
                    st.link_button("Open source", record.url)
    with tabs[3]:
        digital_twin_view(result)
    with tabs[4]:
        st.subheader("Professional report")
        st.write("The report contains patient information, clinical inputs, prediction outputs, confidence score, evidence summary, feature importance, educational explanation, references, timestamp, and methodology summary.")
        report_path = st.session_state.get("report_path")
        if report_path:
            download_pdf_button(Path(report_path))
        else:
            st.warning("PDF report is unavailable for this run.")
        if st.button("Start a new analysis"):
            for key in ["analysis_result", "patient_profile", "report_path"]:
                st.session_state[key] = None
            st.session_state.question_step = 0
            go("questionnaire")


def digital_twin_view(result) -> None:
    st.subheader("Healthy reference biology vs patient-specific disease biology")
    st.caption(result.digital_twin["educational_boundary"])
    left, right = st.columns(2)
    healthy = result.digital_twin["healthy"]
    patient = result.digital_twin["patient"]
    with left:
        st.markdown("### Healthy reference")
        for key in ["gene", "protein", "red_cells", "pathway"]:
            st.write(f"**{key.replace('_', ' ').title()}**: {healthy[key]}")
    with right:
        st.markdown("### Patient-specific disease biology")
        for key in ["gene", "protein", "red_cells", "pathway"]:
            st.write(f"**{key.replace('_', ' ').title()}**: {patient[key]}")
    st.plotly_chart(digital_twin_radar(result), use_container_width=True)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def main() -> None:
    st.set_page_config(page_title="GeneVista AI", page_icon="GV", layout="wide")
    configure_logging()
    init_state()
    apply_global_styles()
    with st.sidebar:
        st.markdown("## GeneVista AI")
        st.caption("Sickle Cell Disease MVP")
        if st.button("Home", use_container_width=True):
            go("landing")
        if st.button("Start Analysis", use_container_width=True):
            go("questionnaire")
        st.markdown("---")
        st.caption("Educational and research-support software. Not a diagnostic device.")

    stage = st.session_state.stage
    if stage == "landing":
        landing_page()
    elif stage in {"learn", "methodology", "about"}:
        informational_page(stage)
    elif stage == "questionnaire":
        questionnaire()
    elif stage == "processing":
        processing_screen()
    elif stage == "dashboard":
        dashboard()
    else:
        landing_page()


if __name__ == "__main__":
    if not running_inside_streamlit():
        sys.exit(
            "GeneVista AI is a Streamlit application and cannot be launched with "
            "`python app.py` or ¯`python3 app.py`.\n\n"
            "Run it on macOS with:\n"
            "  python3 -m streamlit run app.py\n\n"
            "Or, after activating the virtual environment:\n"
            "  streamlit run app.py"
        )
    main()
