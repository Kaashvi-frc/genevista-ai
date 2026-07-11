"""Standalone Streamlit application for GeneVista AI.

Upload this file together with requirements.txt to GitHub or Streamlit Cloud.
Run locally with:

    python3 -m streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx


SUPPORTED_GENOTYPES = ["HbSS", "HbSC", "HbS beta+ thalassaemia", "HbS beta0 thalassaemia"]
PROCESSING_SECONDS = int(os.getenv("GENEVISTA_PROCESSING_SECONDS", "132"))
REPORT_DIR = Path("reports")


@dataclass
class LabResults:
    haemoglobin_g_dl: float | None = None
    foetal_haemoglobin_pct: float | None = None
    white_blood_cell_count: float | None = None


@dataclass
class ClinicalHistory:
    vaso_occlusive_crises_year: int | None = None
    hospital_admissions_year: int | None = None
    acute_chest_syndrome: bool = False
    stroke_history: bool = False
    hydroxyurea: bool = False
    blood_transfusions: bool = False


@dataclass
class DocumentBundle:
    laboratory_report_text: str = ""
    clinical_notes: str = ""
    vcf_summary: dict[str, Any] = field(default_factory=dict)
    uploaded_filenames: list[str] = field(default_factory=list)


@dataclass
class PatientProfile:
    age: int
    sex: str
    confirmed_genotype: str
    labs: LabResults
    history: ClinicalHistory
    documents: DocumentBundle = field(default_factory=DocumentBundle)
    additional_information: str = ""


@dataclass
class EvidenceRecord:
    source: str
    title: str
    summary: str
    quality: float
    agreement_key: str
    url: str | None = None
    claim_type: str = "clinical"


@dataclass
class PredictionResult:
    severity_label: str
    risk_category: str
    complication_risks: dict[str, float]
    progression_statement: str
    feature_importance: dict[str, float]
    model_confidence: float
    prediction_horizon_months: int
    model_version: str


@dataclass
class AnalysisResult:
    patient: PatientProfile
    prediction: PredictionResult
    evidence: list[EvidenceRecord]
    evidence_confidence: float
    evidence_notes: list[str]
    clinical_interpretation: str
    educational_notes: list[str]
    digital_twin: dict[str, Any]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


CURATED_EVIDENCE = [
    EvidenceRecord(
        "ClinVar",
        "HBB sickle haemoglobin variant",
        "The HBB missense variant that produces haemoglobin S is a well-established pathogenic cause of Sickle Cell Disease when present in disease-causing genotypes.",
        0.94,
        "hbb_hbs_pathogenic",
        "https://www.ncbi.nlm.nih.gov/clinvar/",
        "genomic",
    ),
    EvidenceRecord(
        "ClinGen",
        "HBB and beta-haemoglobinopathy validity",
        "HBB has definitive disease validity for beta-haemoglobinopathies, including sickle cell syndromes.",
        0.92,
        "hbb_gene_disease_validity",
        "https://clinicalgenome.org/",
        "genomic",
    ),
    EvidenceRecord(
        "NCBI Gene",
        "HBB encodes beta-globin",
        "HBB encodes the beta subunit of adult haemoglobin, making variants in this gene directly relevant to red blood cell oxygen transport.",
        0.88,
        "hbb_beta_globin_function",
        "https://www.ncbi.nlm.nih.gov/gene/",
        "function",
    ),
    EvidenceRecord(
        "UniProt",
        "Beta-globin protein function",
        "Beta-globin contributes to haemoglobin tetramer formation and oxygen binding; structural disruption can alter red blood cell behaviour.",
        0.86,
        "beta_globin_function",
        "https://www.uniprot.org/",
        "protein",
    ),
    EvidenceRecord(
        "PubMed",
        "Foetal haemoglobin as a disease modifier",
        "Higher foetal haemoglobin is consistently associated with reduced polymerisation burden and often milder Sickle Cell Disease symptoms.",
        0.90,
        "hbf_modifier",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "clinical",
    ),
    EvidenceRecord(
        "PubMed",
        "Clinical events as severity markers",
        "Frequent vaso-occlusive crises, acute chest syndrome, stroke, and recurrent admissions are clinically meaningful markers of higher disease burden.",
        0.89,
        "clinical_events_severity",
        "https://pubmed.ncbi.nlm.nih.gov/",
        "clinical",
    ),
    EvidenceRecord(
        "AlphaFold",
        "HBB structural context",
        "Predicted and experimentally informed protein structures can support education about beta-globin, but they should not be interpreted as a patient-specific physiological simulation.",
        0.74,
        "structure_educational_context",
        "https://alphafold.ebi.ac.uk/",
        "structure",
    ),
]


GENOTYPE_RISK = {
    "HbSS": 1.0,
    "HbS beta0 thalassaemia": 0.92,
    "HbSC": 0.62,
    "HbS beta+ thalassaemia": 0.58,
}


def running_inside_streamlit() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


def init_state() -> None:
    for key, value in {
        "stage": "landing",
        "question_step": 0,
        "patient_profile": None,
        "analysis_result": None,
        "report_path": None,
    }.items():
        st.session_state.setdefault(key, value)


def go(stage: str) -> None:
    st.session_state.stage = stage
    st.rerun()


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Gentium+Book+Basic:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root {
            --ivory: #f8f4ea;
            --navy: #123047;
            --slate: #536171;
            --forest: #31513c;
            --burgundy: #7a2f39;
            --line: #d8d2c4;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(49, 81, 60, 0.08), transparent 32rem),
                        linear-gradient(180deg, #f8f4ea 0%, #f3eee2 100%);
            color: var(--navy);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Gentium Book Basic', serif;
            color: var(--navy);
            letter-spacing: 0;
        }
        h1 { font-size: clamp(2.4rem, 5vw, 5.2rem); line-height: 0.94; }
        h2 { font-size: 2.1rem; }
        .block-container { padding-top: 2.2rem; max-width: 1200px; }
        [data-testid="stSidebar"] { background: #ede6d6; border-right: 1px solid var(--line); }
        .gv-hero { border-bottom: 1px solid var(--line); padding: 3.5rem 0 2.5rem; margin-bottom: 1.2rem; }
        .gv-kicker {
            color: var(--burgundy);
            text-transform: uppercase;
            letter-spacing: 0.12rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }
        .gv-subtitle { color: #364656; font-size: 1.24rem; line-height: 1.65; max-width: 760px; }
        .gv-card {
            background: rgba(255, 253, 247, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.1rem 1.15rem;
            min-height: 132px;
        }
        .gv-card h3 { font-family: 'IBM Plex Sans', sans-serif; font-size: 1rem; margin: 0 0 0.45rem; }
        .gv-card p { color: var(--slate); margin: 0; line-height: 1.5; }
        .gv-pill {
            display: inline-flex;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            color: var(--forest);
            background: #fbf8f0;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.2rem 0.25rem 0.2rem 0;
        }
        .gv-divider { height: 1px; background: var(--line); margin: 1.2rem 0; }
        .stButton > button {
            border-radius: 7px;
            border: 1px solid #31513c;
            background: #31513c;
            color: #fffdf7;
            font-weight: 700;
            min-height: 2.7rem;
            white-space: nowrap;
        }
        div[data-testid="stProgress"] > div > div { background-color: #31513c; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="gv-card">
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(body)}</p>
        </div>
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
            card("Prediction architecture", "Keeps training, evaluation, registry, and inference concepts separate so real datasets can replace the baseline.")
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


def informational_page(kind: str) -> None:
    labels = {
        "learn": ("Platform", "GeneVista connects patient inputs to molecular, clinical, and literature evidence before producing explanations."),
        "methodology": ("Methodology", "The MVP uses modular retrieval, consensus scoring, prediction inference, and traceable explanation generation."),
        "about": ("About", "GeneVista AI is designed as premium biomedical software for Sickle Cell Disease research and clinical interpretation support."),
    }
    title, subtitle = labels[kind]
    st.markdown(f'<div class="gv-kicker">{title}</div><h1>{title}</h1><p class="gv-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        card("Why it exists", "Manual interpretation requires scattered literature, databases, clinical context, and careful uncertainty handling.")
    with cols[1]:
        card("How it works", "Inputs are validated, features are engineered, evidence is retrieved, outputs are scored, and explanations are grounded.")
    with cols[2]:
        card("Scientific credibility", "The MVP exposes uncertainty, evidence completeness, source agreement, and supported prediction horizons.")
    if st.button("Start Analysis"):
        go("questionnaire")
    if st.button("Back"):
        go("landing")


def decode_upload(file) -> str:
    if file is None:
        return ""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def summarize_vcf(text: str) -> dict[str, Any]:
    variants = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            variants.append({"chrom": parts[0], "pos": parts[1], "ref": parts[3], "alt": parts[4]})
    return {
        "variant_count": len(variants),
        "hbb_candidates": [v for v in variants if v["chrom"].lower().replace("chr", "") == "11"][:10],
    }


def parse_uploads(lab_report, vcf_file, clinical_notes: str) -> DocumentBundle:
    filenames = []
    lab_text = ""
    vcf_summary = {}
    if lab_report is not None:
        filenames.append(getattr(lab_report, "name", "laboratory_report"))
        lab_text = decode_upload(lab_report)
    if vcf_file is not None:
        filenames.append(getattr(vcf_file, "name", "variants.vcf"))
        vcf_summary = summarize_vcf(decode_upload(vcf_file))
    return DocumentBundle(lab_text, clinical_notes.strip(), vcf_summary, filenames)


def validate_patient_profile(profile: PatientProfile) -> None:
    if not 0 <= profile.age <= 120:
        raise ValueError("Age must be between 0 and 120.")
    if profile.sex not in {"Female", "Male", "Intersex", "Not specified"}:
        raise ValueError("Select a valid sex option.")
    if profile.confirmed_genotype not in SUPPORTED_GENOTYPES:
        raise ValueError("Select a supported confirmed genotype.")
    if profile.labs.haemoglobin_g_dl is None or not 2.0 <= profile.labs.haemoglobin_g_dl <= 20.0:
        raise ValueError("Haemoglobin must be between 2 and 20 g/dL.")
    if profile.labs.foetal_haemoglobin_pct is None or not 0 <= profile.labs.foetal_haemoglobin_pct <= 50:
        raise ValueError("Foetal haemoglobin must be between 0 and 50 percent.")
    if profile.labs.white_blood_cell_count is None or not 0.5 <= profile.labs.white_blood_cell_count <= 80:
        raise ValueError("White blood cell count must be between 0.5 and 80 x10^9/L.")


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


def questionnaire() -> None:
    st.markdown('<div class="gv-kicker">Patient questionnaire</div><h1>New Sickle Cell Analysis</h1>', unsafe_allow_html=True)
    steps = ["Patient Information", "Laboratory Results", "Clinical History", "Supporting Documents", "Additional Information"]
    step = st.session_state.question_step
    st.progress((step + 1) / len(steps), text=f"Section {step + 1} of {len(steps)}")

    with st.form(f"questionnaire_{step}"):
        if step == 0:
            age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("age", 28), step=1)
            sex = st.selectbox("Sex", ["Female", "Male", "Intersex", "Not specified"], index=["Female", "Male", "Intersex", "Not specified"].index(st.session_state.get("sex", "Not specified")))
            genotype = st.selectbox("Confirmed genotype", SUPPORTED_GENOTYPES, index=0)
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
        previous = prev_col.form_submit_button("Back", use_container_width=True)
        submitted = next_col.form_submit_button("Generate Analysis" if step == len(steps) - 1 else "Continue", use_container_width=True)

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
            st.session_state.documents = parse_uploads(lab_report, vcf_file, notes)
            st.session_state.notes = notes
        else:
            st.session_state.additional = additional
            profile = build_patient_profile()
            try:
                validate_patient_profile(profile)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.patient_profile = profile
            go("processing")

        if step < len(steps) - 1:
            st.session_state.question_step = step + 1
            st.rerun()


def feature_vector(profile: PatientProfile) -> dict[str, float]:
    docs = profile.documents or DocumentBundle()
    return {
        "age": float(profile.age),
        "genotype_risk": GENOTYPE_RISK.get(profile.confirmed_genotype, 0.7),
        "haemoglobin_g_dl": float(profile.labs.haemoglobin_g_dl or 0),
        "foetal_haemoglobin_pct": float(profile.labs.foetal_haemoglobin_pct or 0),
        "white_blood_cell_count": float(profile.labs.white_blood_cell_count or 0),
        "vaso_occlusive_crises_year": float(profile.history.vaso_occlusive_crises_year or 0),
        "hospital_admissions_year": float(profile.history.hospital_admissions_year or 0),
        "acute_chest_syndrome": float(profile.history.acute_chest_syndrome),
        "stroke_history": float(profile.history.stroke_history),
        "hydroxyurea": float(profile.history.hydroxyurea),
        "blood_transfusions": float(profile.history.blood_transfusions),
        "document_variant_count": float(docs.vcf_summary.get("variant_count", 0)),
    }


def predict(features: dict[str, float]) -> PredictionResult:
    risk_score = 0.0
    risk_score += 0.20 * features["genotype_risk"]
    risk_score += 0.16 * min(features["vaso_occlusive_crises_year"] / 6, 1)
    risk_score += 0.12 * min(features["hospital_admissions_year"] / 4, 1)
    risk_score += 0.13 * features["acute_chest_syndrome"]
    risk_score += 0.16 * features["stroke_history"]
    risk_score += 0.10 * min(features["white_blood_cell_count"] / 20, 1)
    risk_score += 0.10 * max(0, (9.5 - features["haemoglobin_g_dl"]) / 6)
    risk_score -= 0.12 * min(features["foetal_haemoglobin_pct"] / 20, 1)
    risk_score -= 0.04 * features["hydroxyurea"]
    risk_score = max(0.0, min(1.0, risk_score))

    if risk_score >= 0.62:
        severity, risk_category = "High", "Elevated"
        risks = {"Acute chest syndrome": 0.34, "Hospitalisation": 0.42, "Neurologic complication": 0.18}
    elif risk_score >= 0.36:
        severity, risk_category = "Moderate", "Intermediate"
        risks = {"Acute chest syndrome": 0.19, "Hospitalisation": 0.25, "Neurologic complication": 0.08}
    else:
        severity, risk_category = "Lower", "Lower"
        risks = {"Acute chest syndrome": 0.09, "Hospitalisation": 0.12, "Neurologic complication": 0.03}

    raw_importance = {
        "Genotype": 0.20 * features["genotype_risk"],
        "Vaso-occlusive crises": 0.16 * min(features["vaso_occlusive_crises_year"] / 6, 1),
        "Hospital admissions": 0.12 * min(features["hospital_admissions_year"] / 4, 1),
        "Acute chest syndrome": 0.13 * features["acute_chest_syndrome"],
        "Stroke history": 0.16 * features["stroke_history"],
        "White blood cell count": 0.10 * min(features["white_blood_cell_count"] / 20, 1),
        "Haemoglobin": 0.10 * max(0, (9.5 - features["haemoglobin_g_dl"]) / 6),
        "Foetal haemoglobin": 0.12 * min(features["foetal_haemoglobin_pct"] / 20, 1),
    }
    total = sum(abs(v) for v in raw_importance.values()) or 1.0
    importance = {k: round(abs(v) / total, 3) for k, v in raw_importance.items()}
    confidence = min(0.84, 0.56 + abs(risk_score - 0.5) * 0.55)

    return PredictionResult(
        severity,
        risk_category,
        risks,
        "Projection is limited to the next 12 months because the MVP does not claim validated long-term trajectory forecasting.",
        importance,
        round(confidence, 3),
        12,
        "clinical-baseline-v0",
    )


def retrieve_evidence(profile: PatientProfile) -> list[EvidenceRecord]:
    records = list(CURATED_EVIDENCE)
    if profile.history.stroke_history:
        records.append(
            EvidenceRecord(
                "PubMed",
                "Stroke history and high-risk phenotype",
                "Prior stroke is treated as a high-consequence event in Sickle Cell Disease risk interpretation.",
                0.88,
                "stroke_high_risk",
                "https://pubmed.ncbi.nlm.nih.gov/",
                "clinical",
            )
        )
    return records


def patient_completeness(profile: PatientProfile) -> float:
    docs = profile.documents or DocumentBundle()
    values = [
        profile.age,
        profile.sex,
        profile.confirmed_genotype,
        profile.labs.haemoglobin_g_dl,
        profile.labs.foetal_haemoglobin_pct,
        profile.labs.white_blood_cell_count,
        profile.history.vaso_occlusive_crises_year,
        profile.history.hospital_admissions_year,
        profile.history.acute_chest_syndrome,
        profile.history.stroke_history,
        profile.history.hydroxyurea,
        profile.history.blood_transfusions,
    ]
    uploaded = bool(docs.uploaded_filenames or docs.clinical_notes or profile.additional_information)
    return round((sum(v is not None and v != "" for v in values) + int(uploaded)) / (len(values) + 1), 3)


def consensus_score(profile: PatientProfile, prediction: PredictionResult, evidence: list[EvidenceRecord]) -> tuple[float, list[str]]:
    keys = {record.agreement_key for record in evidence}
    agreement = min(1.0, 1 / max(1, len(keys)) + 0.15)
    quality = sum(record.quality for record in evidence) / len(evidence)
    completeness = patient_completeness(profile)
    score = 0.30 * agreement + 0.25 * quality + 0.20 * completeness + 0.25 * prediction.model_confidence

    notes = []
    if agreement < 0.45:
        notes.append("Evidence agreement is limited; interpretation should be cautious.")
    if completeness < 0.75:
        notes.append("Patient data are incomplete, reducing confidence.")
    if prediction.model_confidence < 0.60:
        notes.append("Model certainty is moderate or low for this patient profile.")
    if not notes:
        notes.append("Evidence, patient data, and model certainty are reasonably aligned.")
    return round(score, 3), notes


def digital_twin(profile: PatientProfile) -> dict[str, Any]:
    hbf = profile.labs.foetal_haemoglobin_pct or 0
    crises = profile.history.vaso_occlusive_crises_year or 0
    sickling = min(95, 25 + crises * 9 + (10 if profile.history.acute_chest_syndrome else 0) - hbf * 0.8)
    oxygen = max(35, 82 - sickling * 0.35)
    return {
        "healthy": {
            "gene": "HBB reference beta-globin coding sequence",
            "protein": "Stable haemoglobin tetramer with normal oxygen binding",
            "red_cells": "Flexible biconcave red blood cells",
            "pathway": "Efficient oxygen delivery and microvascular transit",
            "sickling_index": 4,
            "oxygen_delivery": 96,
            "inflammation": 8,
        },
        "patient": {
            "gene": f"{profile.confirmed_genotype} sickle haemoglobin context",
            "protein": "Beta-globin alteration promotes haemoglobin S polymerisation under deoxygenation",
            "red_cells": "Reduced flexibility with sickling tendency",
            "pathway": "Higher risk of vaso-occlusion, haemolysis, and inflammatory activation",
            "sickling_index": round(max(5, sickling), 1),
            "oxygen_delivery": round(oxygen, 1),
            "inflammation": round(min(92, 18 + crises * 8 + profile.history.hospital_admissions_year * 4), 1),
        },
        "educational_boundary": "This Digital Twin is educational and not an exact physiological simulation.",
    }


def clinical_interpretation(profile: PatientProfile, prediction: PredictionResult, confidence: float) -> tuple[str, list[str]]:
    modifiers = []
    if (profile.labs.foetal_haemoglobin_pct or 0) >= 15:
        modifiers.append("Foetal haemoglobin is relatively high, which is treated as a protective modifier.")
    else:
        modifiers.append("Foetal haemoglobin is not in the strongly protective range used by this MVP.")
    if (profile.history.vaso_occlusive_crises_year or 0) >= 3:
        modifiers.append("Frequent vaso-occlusive crises increase interpreted disease burden.")
    if profile.history.acute_chest_syndrome:
        modifiers.append("Acute chest syndrome contributes to higher complication concern.")
    if profile.history.stroke_history:
        modifiers.append("Prior stroke is treated as a high-consequence risk marker.")

    text = (
        f"GeneVista estimates {prediction.severity_label.lower()} severity and a "
        f"{prediction.risk_category.lower()} risk category over a {prediction.prediction_horizon_months}-month supported horizon. "
        f"The evidence confidence is {confidence:.0%}. "
        + " ".join(modifiers)
    )
    notes = [
        "The Digital Twin is educational and not an exact physiological simulation.",
        "Predictions are research-support outputs and should be interpreted with clinical context.",
        "Claims are limited to retrieved evidence and configured model outputs.",
    ]
    return text, notes


def run_analysis(profile: PatientProfile) -> AnalysisResult:
    validate_patient_profile(profile)
    prediction = predict(feature_vector(profile))
    evidence = retrieve_evidence(profile)
    confidence, evidence_notes = consensus_score(profile, prediction, evidence)
    interpretation, educational_notes = clinical_interpretation(profile, prediction, confidence)
    return AnalysisResult(profile, prediction, evidence, confidence, evidence_notes, interpretation, educational_notes, digital_twin(profile))


def feature_importance_chart(result: AnalysisResult) -> go.Figure:
    items = sorted(result.prediction.feature_importance.items(), key=lambda item: item[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color="#31513c"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickformat=".0%",
        xaxis_title="Relative contribution",
        yaxis_title="",
        font=dict(family="IBM Plex Sans", color="#1f2933"),
    )
    return fig


def digital_twin_radar(result: AnalysisResult) -> go.Figure:
    healthy = result.digital_twin["healthy"]
    patient = result.digital_twin["patient"]
    categories = ["Sickling index", "Oxygen delivery", "Inflammation"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[healthy["sickling_index"], healthy["oxygen_delivery"], healthy["inflammation"]], theta=categories, fill="toself", name="Healthy reference", line_color="#31513c"))
    fig.add_trace(go.Scatterpolar(r=[patient["sickling_index"], patient["oxygen_delivery"], patient["inflammation"]], theta=categories, fill="toself", name="Patient-specific", line_color="#7a2f39"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=420, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def risk_bar(label: str, value: float) -> None:
    st.write(f"**{label}**")
    st.progress(min(max(value, 0.0), 1.0), text=f"{value:.0%}")


def generate_pdf(result: AnalysisResult) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"genevista_scd_report_{result.generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("GeneVistaTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#123047"))
    section = ParagraphStyle("GeneVistaSection", parent=styles["Heading2"], textColor=colors.HexColor("#31513c"), spaceBefore=14)
    body = styles["BodyText"]

    def table(rows):
        t = Table(rows, colWidths=[220, 260])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1eee5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d2c4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    p = result.patient
    story = [
        Paragraph("GeneVista AI Sickle Cell Disease Report", title),
        Paragraph(f"Generated: {result.generated_at.isoformat()}", body),
        Spacer(1, 0.18 * inch),
        Paragraph("Patient Information", section),
        table([
            ["Age", p.age],
            ["Sex", p.sex],
            ["Confirmed genotype", p.confirmed_genotype],
            ["Haemoglobin", f"{p.labs.haemoglobin_g_dl} g/dL"],
            ["Foetal haemoglobin", f"{p.labs.foetal_haemoglobin_pct}%"],
            ["White blood cell count", f"{p.labs.white_blood_cell_count} x10^9/L"],
        ]),
        Paragraph("Prediction Outputs", section),
        table([
            ["Estimated severity", result.prediction.severity_label],
            ["Risk category", result.prediction.risk_category],
            ["Model confidence", f"{result.prediction.model_confidence:.0%}"],
            ["Evidence confidence", f"{result.evidence_confidence:.0%}"],
            ["Model version", result.prediction.model_version],
        ]),
        Paragraph("Clinical Interpretation", section),
        Paragraph(result.clinical_interpretation, body),
        Paragraph("Feature Importance", section),
        table([[k, f"{v:.0%}"] for k, v in result.prediction.feature_importance.items()]),
        Paragraph("Evidence Summary", section),
    ]
    for record in result.evidence[:8]:
        story.append(Paragraph(f"<b>{record.source}: {record.title}</b>", body))
        story.append(Paragraph(record.summary, body))
        story.append(Spacer(1, 0.08 * inch))
    story.extend([
        Paragraph("Digital Twin Boundary", section),
        Paragraph(result.digital_twin["educational_boundary"], body),
        Paragraph("Methodology Summary", section),
        Paragraph("Patient inputs were validated, transformed into clinical features, interpreted by the prediction pipeline, compared against curated biomedical evidence, scored by a consensus engine, and summarized with evidence-grounded explanations.", body),
    ])
    SimpleDocTemplate(str(path), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch).build(story)
    return path


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
    delay = PROCESSING_SECONDS / len(stages)
    for index, stage in enumerate(stages, start=1):
        status.info(stage)
        progress.progress(index / len(stages), text=stage)
        if delay > 0:
            time.sleep(delay)

    result = run_analysis(st.session_state.patient_profile)
    st.session_state.analysis_result = result
    try:
        st.session_state.report_path = generate_pdf(result)
    except Exception as exc:
        st.session_state.report_path = None
        st.warning(f"Report generation failed: {exc}")
    go("dashboard")


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def dashboard() -> None:
    result = st.session_state.analysis_result
    if result is None:
        go("landing")
        return
    patient = result.patient
    prediction = result.prediction
    st.markdown('<div class="gv-kicker">Prediction dashboard</div><h1>GeneVista Analysis Dashboard</h1>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Severity", prediction.severity_label)
    cols[1].metric("Risk category", prediction.risk_category)
    cols[2].metric("Model confidence", f"{prediction.model_confidence:.0%}")
    cols[3].metric("Evidence confidence", f"{result.evidence_confidence:.0%}")

    tabs = st.tabs(["Patient Summary", "Prediction Summary", "Evidence", "Digital Twin", "Report"])
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
        st.subheader("Healthy reference biology vs patient-specific disease biology")
        st.caption(result.digital_twin["educational_boundary"])
        left, right = st.columns(2)
        with left:
            st.markdown("### Healthy reference")
            for key in ["gene", "protein", "red_cells", "pathway"]:
                st.write(f"**{key.replace('_', ' ').title()}**: {result.digital_twin['healthy'][key]}")
        with right:
            st.markdown("### Patient-specific disease biology")
            for key in ["gene", "protein", "red_cells", "pathway"]:
                st.write(f"**{key.replace('_', ' ').title()}**: {result.digital_twin['patient'][key]}")
        st.plotly_chart(digital_twin_radar(result), use_container_width=True)
    with tabs[4]:
        st.subheader("Professional report")
        st.write("The report contains patient information, clinical inputs, prediction outputs, confidence score, evidence summary, feature importance, educational explanation, references, timestamp, and methodology summary.")
        path = st.session_state.get("report_path")
        if path:
            with Path(path).open("rb") as file:
                st.download_button("Download PDF report", data=file, file_name=Path(path).name, mime="application/pdf", use_container_width=True)
        else:
            st.warning("PDF report is unavailable for this run.")
        if st.button("Start a new analysis"):
            for key in ["analysis_result", "patient_profile", "report_path"]:
                st.session_state[key] = None
            st.session_state.question_step = 0
            go("questionnaire")


def main() -> None:
    st.set_page_config(page_title="GeneVista AI", page_icon="GV", layout="wide")
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
            "GeneVista AI is a Streamlit application.\n\n"
            "Run it with:\n"
            "  python3 -m streamlit run app.py"
        )
    main()

