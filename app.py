"""Standalone Streamlit application for GeneVista AI.

Upload this file together with requirements.txt to GitHub or Streamlit Cloud.
Run locally with:

    python3 -m streamlit run app.py
"""

from __future__ import annotations

import html
import math
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
TIMEPOINTS = [
    ("Current", 0),
    ("6 months", 0.5),
    ("1 year", 1),
    ("2 years", 2),
    ("5 years", 5),
    ("10 years", 10),
]
TWIN_STAGES = [
    "DNA / Gene Level",
    "Mutation Identification",
    "Protein Structure & Function",
    "Cellular Changes",
    "Tissue Effects",
    "Organ-Level Changes",
    "Whole Body Effects",
    "Disease Progression Over Time",
]
THEMES = {
    "Light": {
        "bg": "#f8f4ea",
        "surface": "#fffdf7",
        "surface_alt": "#f0eadc",
        "text": "#123047",
        "muted": "#536171",
        "line": "#d8d2c4",
        "forest": "#31513c",
        "burgundy": "#7a2f39",
        "gold": "#b98a2f",
        "orange": "#b85f2f",
        "red": "#9e2f3c",
        "chart_bg": "rgba(0,0,0,0)",
    },
    "Dark": {
        "bg": "#101820",
        "surface": "#182431",
        "surface_alt": "#223140",
        "text": "#f4efe4",
        "muted": "#b9c3c9",
        "line": "#3b4a55",
        "forest": "#7fb08a",
        "burgundy": "#e09aa3",
        "gold": "#d4b35f",
        "orange": "#e08a55",
        "red": "#ef6f7c",
        "chart_bg": "rgba(0,0,0,0)",
    },
}


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
    pain_severity: int = 4
    fatigue_severity: int = 4
    breathlessness_severity: int = 2
    medication_adherence: str = "Good"
    family_history: bool = False
    smoking_exposure: bool = False
    hydration_status: str = "Adequate"
    activity_level: str = "Moderate"


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
    progression_forecast: list[dict[str, Any]] = field(default_factory=list)
    organ_system_risks: dict[str, float] = field(default_factory=dict)
    symptom_timeline: dict[str, list[float]] = field(default_factory=dict)
    treatment_considerations: list[str] = field(default_factory=list)
    reasoning_factors: list[str] = field(default_factory=list)


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
        "patient_report_path": None,
        "theme_mode": "Light",
        "twin_playing": False,
        "twin_stage_index": 0,
        "twin_time_index": 0,
    }.items():
        st.session_state.setdefault(key, value)


def navigate(stage: str) -> None:
    st.session_state.stage = stage
    st.rerun()


def apply_global_styles(theme_name: str) -> None:
    theme = THEMES.get(theme_name, THEMES["Light"])
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Gentium+Book+Basic:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        :root {
            --bg: __BG__;
            --surface: __SURFACE__;
            --surface-alt: __SURFACE_ALT__;
            --text: __TEXT__;
            --muted: __MUTED__;
            --forest: __FOREST__;
            --burgundy: __BURGUNDY__;
            --gold: __GOLD__;
            --orange: __ORANGE__;
            --red: __RED__;
            --line: __LINE__;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(127, 176, 138, 0.12), transparent 32rem),
                        linear-gradient(180deg, var(--bg) 0%, var(--surface-alt) 100%);
            color: var(--text);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Gentium Book Basic', serif;
            color: var(--text);
            letter-spacing: 0;
        }
        h1 { font-size: clamp(2.4rem, 5vw, 5.2rem); line-height: 0.94; }
        h2 { font-size: 2.1rem; }
        .block-container { padding-top: 2.2rem; max-width: 1200px; }
        [data-testid="stSidebar"] { background: var(--surface-alt); border-right: 1px solid var(--line); }
        .gv-hero { border-bottom: 1px solid var(--line); padding: 3.5rem 0 2.5rem; margin-bottom: 1.2rem; }
        .gv-kicker {
            color: var(--burgundy);
            text-transform: uppercase;
            letter-spacing: 0.12rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }
        .gv-subtitle { color: var(--muted); font-size: 1.24rem; line-height: 1.65; max-width: 760px; }
        .gv-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.1rem 1.15rem;
            min-height: 132px;
        }
        .gv-card h3 { font-family: 'IBM Plex Sans', sans-serif; font-size: 1rem; margin: 0 0 0.45rem; }
        .gv-card p { color: var(--muted); margin: 0; line-height: 1.5; }
        .gv-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }
        .gv-metric-band {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.85rem;
            margin: 1rem 0;
        }
        .gv-metric-card {
            background: var(--surface);
            border-left: 4px solid var(--forest);
            border-radius: 8px;
            padding: 0.85rem;
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
        }
        .gv-metric-card span { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08rem; }
        .gv-metric-card strong { display: block; color: var(--text); font-size: 1.35rem; margin-top: 0.25rem; }
        .gv-pill {
            display: inline-flex;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            color: var(--forest);
            background: var(--surface);
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.2rem 0.25rem 0.2rem 0;
        }
        .gv-risk-low { color: var(--forest); font-weight: 700; }
        .gv-risk-moderate { color: var(--orange); font-weight: 700; }
        .gv-risk-high { color: var(--red); font-weight: 700; }
        .gv-disclaimer {
            border: 1px solid var(--line);
            background: var(--surface);
            border-left: 4px solid var(--gold);
            border-radius: 8px;
            padding: 0.85rem;
            color: var(--muted);
            font-size: 0.9rem;
        }
        .gv-divider { height: 1px; background: var(--line); margin: 1.2rem 0; }
        .stButton > button {
            border-radius: 7px;
            border: 1px solid #31513c;
            background: var(--forest);
            color: var(--bg);
            font-weight: 700;
            min-height: 2.7rem;
            white-space: nowrap;
        }
        div[data-testid="stTabs"] button { color: var(--text); }
        .stDataFrame, [data-testid="stMetric"] { color: var(--text); }
        div[data-testid="stProgress"] > div > div { background-color: #31513c; }
        </style>
        """
    for token, value in {
        "__BG__": theme["bg"],
        "__SURFACE__": theme["surface"],
        "__SURFACE_ALT__": theme["surface_alt"],
        "__TEXT__": theme["text"],
        "__MUTED__": theme["muted"],
        "__FOREST__": theme["forest"],
        "__BURGUNDY__": theme["burgundy"],
        "__GOLD__": theme["gold"],
        "__ORANGE__": theme["orange"],
        "__RED__": theme["red"],
        "__LINE__": theme["line"],
    }.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


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


def safety_privacy_notice() -> None:
    st.markdown(
        """
        <div class="gv-disclaimer">
        <strong>Clinical and privacy notice.</strong> GeneVista AI is clinical decision-support software.
        It is not a substitute for licensed medical judgment, emergency care, or a certified medical-device workflow.
        Do not enter identifiable patient information unless the deployment has approved access control, HTTPS,
        encryption at rest, audit logging, institutional review and applicable privacy agreements configured.
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_band(metrics: list[tuple[str, str, str | None]]) -> None:
    blocks = []
    for label, value, tone in metrics:
        class_name = f"gv-risk-{tone}" if tone else ""
        blocks.append(
            f"<div class='gv-metric-card'><span>{html.escape(label)}</span><strong class='{class_name}'>{html.escape(value)}</strong></div>"
        )
    st.markdown(f"<div class='gv-metric-band'>{''.join(blocks)}</div>", unsafe_allow_html=True)


def active_theme() -> dict[str, str]:
    if not running_inside_streamlit():
        return THEMES["Light"]
    return THEMES.get(st.session_state.get("theme_mode", "Light"), THEMES["Light"])


def themed_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    theme = active_theme()
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=24, b=34),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        font=dict(family="IBM Plex Sans", color=theme["text"]),
        xaxis=dict(gridcolor=theme["line"], zerolinecolor=theme["line"]),
        yaxis=dict(gridcolor=theme["line"], zerolinecolor=theme["line"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


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
          consensus biomedical evidence, and biological Digital Twins.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    safety_privacy_notice()
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Analysis", width="stretch"):
                navigate("questionnaire")
        with c2:
            if st.button("Learn More", width="stretch"):
                navigate("learn")
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Methodology", width="stretch"):
                navigate("methodology")
        with c4:
            if st.button("About", width="stretch"):
                navigate("about")
        st.markdown('<div class="gv-divider"></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            card("Consensus evidence", "Aggregates ClinVar, ClinGen, OMIM, NCBI Gene, UniProt, PubMed, AlphaFold, and future sources through extensible providers.")
            card("Prediction architecture", "Keeps training, evaluation, registry, and inference concepts separate so real datasets can replace the baseline.")
        with cols[1]:
            card("Clinical context", "Interprets genotype, haemoglobin, foetal haemoglobin, complications, admissions, and therapies together.")
            card("Biological Digital Twin", "Compares healthy reference biology with patient-specific disease biology while clearly separating simulation from measured physiology.")
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
        navigate("questionnaire")
    if st.button("Back"):
        navigate("landing")


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
            pain_severity=int(st.session_state.get("pain_severity", 4)),
            fatigue_severity=int(st.session_state.get("fatigue_severity", 4)),
            breathlessness_severity=int(st.session_state.get("breathlessness_severity", 2)),
            medication_adherence=st.session_state.get("medication_adherence", "Good"),
            family_history=bool(st.session_state.get("family_history", False)),
            smoking_exposure=bool(st.session_state.get("smoking_exposure", False)),
            hydration_status=st.session_state.get("hydration_status", "Adequate"),
            activity_level=st.session_state.get("activity_level", "Moderate"),
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
            pain_severity = st.slider("Typical pain severity", min_value=0, max_value=10, value=st.session_state.get("pain_severity", 4))
            fatigue_severity = st.slider("Typical fatigue severity", min_value=0, max_value=10, value=st.session_state.get("fatigue_severity", 4))
            breathlessness_severity = st.slider("Breathlessness severity", min_value=0, max_value=10, value=st.session_state.get("breathlessness_severity", 2))
            acute_chest = st.checkbox("History of acute chest syndrome", value=st.session_state.get("acute_chest", False))
            stroke = st.checkbox("Stroke history", value=st.session_state.get("stroke", False))
            hydroxyurea = st.checkbox("Currently on hydroxyurea", value=st.session_state.get("hydroxyurea", True))
            transfusions = st.checkbox("History of blood transfusions", value=st.session_state.get("transfusions", False))
            medication_adherence = st.selectbox("Medication adherence", ["Excellent", "Good", "Inconsistent", "Poor"], index=["Excellent", "Good", "Inconsistent", "Poor"].index(st.session_state.get("medication_adherence", "Good")))
            family_history = st.checkbox("Known family history of severe disease", value=st.session_state.get("family_history", False))
            smoking_exposure = st.checkbox("Smoking or significant second-hand smoke exposure", value=st.session_state.get("smoking_exposure", False))
            hydration_status = st.selectbox("Hydration pattern", ["Strong", "Adequate", "Inconsistent", "Poor"], index=["Strong", "Adequate", "Inconsistent", "Poor"].index(st.session_state.get("hydration_status", "Adequate")))
            activity_level = st.selectbox("Usual activity level", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(st.session_state.get("activity_level", "Moderate")))
        elif step == 3:
            lab_report = st.file_uploader("Laboratory report upload", type=["txt", "csv", "pdf"])
            vcf_file = st.file_uploader("VCF upload", type=["vcf", "txt"])
            notes = st.text_area("Clinical notes", value=st.session_state.get("notes", ""), height=150)
        else:
            additional = st.text_area("Additional information", value=st.session_state.get("additional", ""), height=160)

        prev_col, next_col = st.columns([0.35, 0.65])
        previous = prev_col.form_submit_button("Back", width="stretch")
        submitted = next_col.form_submit_button("Generate Analysis" if step == len(steps) - 1 else "Continue", width="stretch")

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
            st.session_state.pain_severity = pain_severity
            st.session_state.fatigue_severity = fatigue_severity
            st.session_state.breathlessness_severity = breathlessness_severity
            st.session_state.acute_chest, st.session_state.stroke = acute_chest, stroke
            st.session_state.hydroxyurea, st.session_state.transfusions = hydroxyurea, transfusions
            st.session_state.medication_adherence = medication_adherence
            st.session_state.family_history = family_history
            st.session_state.smoking_exposure = smoking_exposure
            st.session_state.hydration_status = hydration_status
            st.session_state.activity_level = activity_level
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
            navigate("processing")

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
        "pain_severity": float(profile.history.pain_severity),
        "fatigue_severity": float(profile.history.fatigue_severity),
        "breathlessness_severity": float(profile.history.breathlessness_severity),
        "adherence_risk": {"Excellent": 0.0, "Good": 0.1, "Inconsistent": 0.35, "Poor": 0.55}.get(profile.history.medication_adherence, 0.15),
        "family_history": float(profile.history.family_history),
        "smoking_exposure": float(profile.history.smoking_exposure),
        "hydration_risk": {"Strong": 0.0, "Adequate": 0.1, "Inconsistent": 0.3, "Poor": 0.5}.get(profile.history.hydration_status, 0.1),
        "activity_risk": {"Low": 0.18, "Moderate": 0.05, "High": 0.12}.get(profile.history.activity_level, 0.05),
    }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def risk_label(score: float) -> str:
    if score >= 0.66:
        return "High"
    if score >= 0.38:
        return "Moderate"
    return "Lower"


def predict(features: dict[str, float]) -> PredictionResult:
    protective_hbf = min(features["foetal_haemoglobin_pct"] / 25, 1)
    anaemia_burden = clamp((9.8 - features["haemoglobin_g_dl"]) / 6)
    inflammation_burden = clamp((features["white_blood_cell_count"] - 8) / 18)
    pain_burden = features["pain_severity"] / 10
    fatigue_burden = features["fatigue_severity"] / 10
    breath_burden = features["breathlessness_severity"] / 10
    crisis_burden = clamp(features["vaso_occlusive_crises_year"] / 6)
    admission_burden = clamp(features["hospital_admissions_year"] / 4)
    treatment_protection = 0.10 * features["hydroxyurea"] + 0.04 * features["blood_transfusions"]

    raw_contributors = {
        "Genotype severity": 0.18 * features["genotype_risk"],
        "Pain crisis frequency": 0.15 * crisis_burden,
        "Hospital admissions": 0.10 * admission_burden,
        "Acute chest syndrome": 0.11 * features["acute_chest_syndrome"],
        "Stroke history": 0.13 * features["stroke_history"],
        "Anaemia burden": 0.10 * anaemia_burden,
        "Inflammatory burden": 0.07 * inflammation_burden,
        "Current symptom burden": 0.09 * ((pain_burden + fatigue_burden + breath_burden) / 3),
        "Medication adherence": 0.06 * features["adherence_risk"],
        "Family history": 0.04 * features["family_history"],
        "Smoke exposure": 0.04 * features["smoking_exposure"],
        "Hydration/activity risk": 0.05 * ((features["hydration_risk"] + features["activity_risk"]) / 2),
    }
    risk_score = clamp(sum(raw_contributors.values()) - 0.13 * protective_hbf - treatment_protection)

    severity = risk_label(risk_score)
    risk_category = {"High": "Elevated", "Moderate": "Intermediate", "Lower": "Lower"}[severity]

    complication_risks = {
        "Pain crisis escalation": clamp(0.08 + risk_score * 0.55 + crisis_burden * 0.18),
        "Hospitalisation": clamp(0.06 + risk_score * 0.50 + admission_burden * 0.20),
        "Acute chest syndrome": clamp(0.05 + risk_score * 0.34 + features["acute_chest_syndrome"] * 0.18 + breath_burden * 0.10),
        "Neurologic complication": clamp(0.02 + risk_score * 0.18 + features["stroke_history"] * 0.22),
        "Transfusion requirement": clamp(0.05 + anaemia_burden * 0.32 + risk_score * 0.20),
    }
    organ_system_risks = {
        "Blood / haemolysis": clamp(0.20 + anaemia_burden * 0.45 + features["genotype_risk"] * 0.20),
        "Lung / acute chest": clamp(complication_risks["Acute chest syndrome"] + breath_burden * 0.12),
        "Brain / neurologic": complication_risks["Neurologic complication"],
        "Kidney": clamp(0.10 + risk_score * 0.25 + inflammation_burden * 0.12),
        "Heart / pulmonary pressure": clamp(0.12 + anaemia_burden * 0.20 + breath_burden * 0.20),
        "Quality of life": clamp(0.12 + pain_burden * 0.34 + fatigue_burden * 0.28 + risk_score * 0.18),
    }
    treatment_considerations = []
    if not features["hydroxyurea"]:
        treatment_considerations.append("Discuss hydroxyurea eligibility and expected benefit/risk with the treating clinician.")
    elif features["adherence_risk"] >= 0.3:
        treatment_considerations.append("Review hydroxyurea adherence barriers and monitoring plan.")
    if features["blood_transfusions"]:
        treatment_considerations.append("Review transfusion history, iron monitoring and alloimmunisation risk.")
    if complication_risks["Acute chest syndrome"] >= 0.30:
        treatment_considerations.append("Prioritise pulmonary risk mitigation and acute chest syndrome action planning.")
    if features["stroke_history"]:
        treatment_considerations.append("Maintain specialist neurologic/stroke prevention follow-up.")
    if features["hydration_risk"] >= 0.3:
        treatment_considerations.append("Address hydration strategy and heat/exertion triggers.")

    forecast = progression_forecast(risk_score, complication_risks, organ_system_risks, features)
    symptom_timeline = {
        "Pain crises/year": [point["pain_crises_per_year"] for point in forecast],
        "Hospitalisations/year": [point["hospitalisations_per_year"] for point in forecast],
        "Quality-of-life impact": [point["quality_of_life_impact"] for point in forecast],
        "Organ-risk burden": [point["organ_risk_burden"] for point in forecast],
    }
    total = sum(abs(v) for v in raw_contributors.values()) or 1.0
    importance = {k: round(abs(v) / total, 3) for k, v in raw_contributors.items()}
    confidence = clamp(0.58 + abs(risk_score - 0.50) * 0.35 + min(patient_signal_count(features) / 24, 0.18), 0.50, 0.91)
    reasoning = top_reasoning(raw_contributors, protective_hbf, treatment_protection)

    return PredictionResult(
        severity,
        risk_category,
        complication_risks,
        "Forecasts are personalised risk estimates across 6 months, 1 year, 2 years, 5 years and 10 years; they require clinical validation before operational use.",
        importance,
        round(confidence, 3),
        120,
        "genevista-rules-v2",
        forecast,
        organ_system_risks,
        symptom_timeline,
        treatment_considerations,
        reasoning,
    )


def patient_signal_count(features: dict[str, float]) -> int:
    return sum(value not in (None, "", 0.0) for value in features.values())


def top_reasoning(raw_contributors: dict[str, float], protective_hbf: float, treatment_protection: float) -> list[str]:
    ranked = sorted(raw_contributors.items(), key=lambda item: item[1], reverse=True)[:5]
    reasons = [f"{name} contributed materially to the risk estimate." for name, value in ranked if value > 0.025]
    if protective_hbf >= 0.45:
        reasons.append("Foetal haemoglobin appears protective in this profile.")
    if treatment_protection > 0:
        reasons.append("Current disease-modifying or transfusion therapy reduces the projected risk estimate.")
    return reasons or ["No single factor dominates; the estimate is driven by combined clinical and laboratory signals."]


def progression_forecast(
    baseline_risk: float,
    complication_risks: dict[str, float],
    organ_system_risks: dict[str, float],
    features: dict[str, float],
) -> list[dict[str, Any]]:
    forecast = []
    protective = 0.012 * features["hydroxyurea"] + 0.020 * min(features["foetal_haemoglobin_pct"] / 20, 1)
    drift = 0.018 + 0.026 * features["adherence_risk"] + 0.018 * features["smoking_exposure"] + 0.014 * features["hydration_risk"]
    for label, years in TIMEPOINTS:
        progression = clamp(baseline_risk + years * drift - years * protective, 0.02, 0.96)
        pain_crises = max(0.0, features["vaso_occlusive_crises_year"] * (1 + years * (0.04 + progression * 0.035 - protective)))
        admissions = max(0.0, features["hospital_admissions_year"] * (1 + years * (0.03 + progression * 0.025 - protective / 2)))
        organ_burden = clamp(sum(organ_system_risks.values()) / len(organ_system_risks) + years * (0.012 + progression * 0.018))
        quality_impact = clamp(0.20 + progression * 0.40 + features["fatigue_severity"] / 30 + features["pain_severity"] / 35)
        forecast.append(
            {
                "timepoint": label,
                "years": years,
                "severity_score": round(progression, 3),
                "severity": risk_label(progression),
                "pain_crises_per_year": round(pain_crises, 1),
                "hospitalisations_per_year": round(admissions, 1),
                "acute_chest_likelihood": round(clamp(complication_risks["Acute chest syndrome"] + years * 0.012 * progression), 3),
                "organ_risk_burden": round(organ_burden, 3),
                "quality_of_life_impact": round(quality_impact, 3),
                "confidence": round(clamp(0.86 - years * 0.045), 3),
                "explanation": forecast_explanation(label, progression, pain_crises, admissions, organ_burden),
            }
        )
    return forecast


def forecast_explanation(label: str, severity_score: float, pain_crises: float, admissions: float, organ_burden: float) -> str:
    return (
        f"At {label.lower()}, the projected severity is {risk_label(severity_score).lower()} with approximately "
        f"{pain_crises:.1f} pain crises/year, {admissions:.1f} hospitalisations/year and "
        f"{organ_burden:.0%} aggregate organ-system risk burden."
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
        "educational_boundary": "This Digital Twin is a clinical decision-support simulation and should be reviewed with the full medical record and clinician judgment.",
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
        f"{prediction.risk_category.lower()} risk category across a {prediction.prediction_horizon_months // 12}-year forecast horizon. "
        f"The evidence confidence is {confidence:.0%}. "
        + " ".join(modifiers)
    )
    notes = [
        "The Digital Twin is a clinical decision-support simulation, not a literal physiological duplicate.",
        "Predictions are decision-support outputs and require clinician review before use in care decisions.",
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
    theme = active_theme()
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h", marker_color=theme["forest"]))
    themed_layout(fig, 390)
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Relative contribution", yaxis_title="")
    return fig


def digital_twin_radar(result: AnalysisResult) -> go.Figure:
    healthy = result.digital_twin["healthy"]
    patient = result.digital_twin["patient"]
    categories = ["Sickling index", "Oxygen delivery", "Inflammation"]
    theme = active_theme()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[healthy["sickling_index"], healthy["oxygen_delivery"], healthy["inflammation"]], theta=categories, fill="toself", name="Healthy reference", line_color=theme["forest"]))
    fig.add_trace(go.Scatterpolar(r=[patient["sickling_index"], patient["oxygen_delivery"], patient["inflammation"]], theta=categories, fill="toself", name="Patient-specific", line_color=theme["burgundy"]))
    fig.update_layout(
        polar=dict(
            bgcolor=theme["chart_bg"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=theme["line"], tickfont=dict(color=theme["muted"])),
            angularaxis=dict(gridcolor=theme["line"], tickfont=dict(color=theme["text"])),
        ),
        showlegend=True,
        height=420,
        paper_bgcolor=theme["chart_bg"],
        font=dict(family="IBM Plex Sans", color=theme["text"]),
    )
    return fig


def progression_chart(result: AnalysisResult) -> go.Figure:
    points = result.prediction.progression_forecast
    theme = active_theme()
    labels = [p["timepoint"] for p in points]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=[p["severity_score"] for p in points], mode="lines+markers", name="Severity", line=dict(color=theme["burgundy"], width=3)))
    fig.add_trace(go.Scatter(x=labels, y=[p["organ_risk_burden"] for p in points], mode="lines+markers", name="Organ risk", line=dict(color=theme["orange"], width=3)))
    fig.add_trace(go.Scatter(x=labels, y=[p["quality_of_life_impact"] for p in points], mode="lines+markers", name="Quality-of-life impact", line=dict(color=theme["gold"], width=3)))
    themed_layout(fig, 390)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title="Prediction horizon", yaxis_title="Risk / burden")
    return fig


def organ_risk_chart(result: AnalysisResult) -> go.Figure:
    theme = active_theme()
    items = sorted(result.prediction.organ_system_risks.items(), key=lambda item: item[1])
    colors = [risk_color(value) for _, value in items]
    fig = go.Figure(go.Bar(x=[value for _, value in items], y=[key for key, _ in items], orientation="h", marker_color=colors))
    themed_layout(fig, 360)
    fig.update_layout(xaxis_tickformat=".0%", xaxis_range=[0, 1], xaxis_title="Estimated burden", yaxis_title="")
    return fig


def risk_color(value: float) -> str:
    theme = active_theme()
    if value >= 0.66:
        return theme["red"]
    if value >= 0.38:
        return theme["orange"]
    return theme["forest"]


def twin_scene(result: AnalysisResult, stage_index: int, time_index: int) -> go.Figure:
    theme = active_theme()
    point = result.prediction.progression_forecast[time_index]
    severity = point["severity_score"]
    stage = TWIN_STAGES[stage_index]
    fig = go.Figure()
    if stage_index <= 1:
        x = [i * 0.25 for i in range(48)]
        fig.add_trace(go.Scatter3d(x=x, y=[math.sin(i / 2) for i in x], z=[math.cos(i / 2) for i in x], mode="lines", name="Healthy DNA", line=dict(color=theme["forest"], width=6)))
        fig.add_trace(go.Scatter3d(x=x, y=[math.sin(i / 2) + 1.8 for i in x], z=[math.cos(i / 2) for i in x], mode="lines", name="Patient DNA", line=dict(color=theme["burgundy"], width=6)))
        fig.add_trace(go.Scatter3d(x=[4.0], y=[math.sin(4.0 / 2) + 1.8], z=[math.cos(4.0 / 2)], mode="markers+text", text=["HBB"], name="Mutation locus", marker=dict(size=8, color=theme["red"])))
    elif stage_index == 2:
        theta = [i * 0.25 for i in range(60)]
        fig.add_trace(go.Scatter3d(x=[math.cos(t) for t in theta], y=[math.sin(t) for t in theta], z=[t / 6 for t in theta], mode="lines", name="Stable beta-globin", line=dict(color=theme["forest"], width=8)))
        fig.add_trace(go.Scatter3d(x=[1.8 + math.cos(t) * (1 + severity / 3) for t in theta], y=[math.sin(t) * (1 - severity / 4) for t in theta], z=[t / 6 for t in theta], mode="lines", name="HbS polymer tendency", line=dict(color=theme["burgundy"], width=8)))
    elif stage_index <= 4:
        fig.add_trace(go.Scatter3d(x=[math.cos(i) for i in range(32)], y=[math.sin(i) for i in range(32)], z=[0 for _ in range(32)], mode="markers", name="Healthy flexible RBCs", marker=dict(size=9, color=theme["forest"], opacity=0.75)))
        fig.add_trace(go.Scatter3d(x=[2 + math.cos(i) * (1 + severity) for i in range(32)], y=[math.sin(i) * (1 - severity / 2) for i in range(32)], z=[severity * math.sin(i * 2) for i in range(32)], mode="markers", name="Patient sickling burden", marker=dict(size=9, color=risk_color(severity), opacity=0.82)))
    else:
        systems = list(result.prediction.organ_system_risks.keys())
        values = [clamp(v + severity * 0.10) for v in result.prediction.organ_system_risks.values()]
        fig.add_trace(go.Bar(x=systems, y=[0.12 for _ in systems], name="Healthy baseline", marker_color=theme["forest"]))
        fig.add_trace(go.Bar(x=systems, y=values, name="Patient projection", marker_color=[risk_color(v) for v in values]))
        themed_layout(fig, 430)
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_tickangle=-20, barmode="group")
        return fig
    fig.update_layout(
        height=430,
        paper_bgcolor=theme["chart_bg"],
        scene=dict(
            bgcolor=theme["chart_bg"],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        font=dict(color=theme["text"], family="IBM Plex Sans"),
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def risk_bar(label: str, value: float) -> None:
    st.write(f"**{label}**")
    st.progress(min(max(value, 0.0), 1.0), text=f"{value:.0%}")


def generate_pdf(result: AnalysisResult) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"genevista_scd_report_{result.generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("GeneVistaTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#123047"))
    section = ParagraphStyle("GeneVistaSection", parent=styles["Heading2"], textColor=colors.HexColor("#31513c"), spaceBefore=14)
    small_section = ParagraphStyle("GeneVistaSmallSection", parent=styles["Heading3"], textColor=colors.HexColor("#7a2f39"), spaceBefore=10)
    body = styles["BodyText"]
    small = styles["BodyText"]
    small.fontSize = 9

    def table(rows):
        column_count = max(len(row) for row in rows) if rows else 2
        if column_count == 2:
            widths = [220, 260]
        else:
            widths = [480 / column_count for _ in range(column_count)]
        t = Table(rows, colWidths=widths, repeatRows=1 if column_count > 2 else 0)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1eee5")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d8d2c4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def bullet(text: str):
        return Paragraph(f"• {html.escape(text)}", body)

    p = result.patient
    story = [
        Paragraph("GeneVista AI Sickle Cell Disease Report", title),
        Paragraph(f"Generated: {result.generated_at.isoformat()}", body),
        Paragraph("Clinical decision-support report for clinician/research review. Not a substitute for licensed medical judgment, emergency care, institutional policy, or certified medical-device workflow.", small),
        Spacer(1, 0.18 * inch),
        Paragraph("Executive Summary", section),
        Paragraph(result.clinical_interpretation, body),
        table([
            ["Estimated severity", result.prediction.severity_label],
            ["Risk category", result.prediction.risk_category],
            ["Model confidence", f"{result.prediction.model_confidence:.0%}"],
            ["Evidence confidence", f"{result.evidence_confidence:.0%}"],
            ["Prediction horizon", "10 years"],
        ]),
        Paragraph("Key AI Reasoning Factors", small_section),
        *[bullet(item) for item in result.prediction.reasoning_factors],
        PageBreak(),
        Paragraph("Patient Information", section),
        table([
            ["Age", p.age],
            ["Sex", p.sex],
            ["Confirmed genotype", p.confirmed_genotype],
            ["Haemoglobin", f"{p.labs.haemoglobin_g_dl} g/dL"],
            ["Foetal haemoglobin", f"{p.labs.foetal_haemoglobin_pct}%"],
            ["White blood cell count", f"{p.labs.white_blood_cell_count} x10^9/L"],
        ]),
        Paragraph("Clinical History", section),
        table([
            ["Vaso-occlusive crises/year", p.history.vaso_occlusive_crises_year],
            ["Hospital admissions/year", p.history.hospital_admissions_year],
            ["Pain severity", f"{p.history.pain_severity}/10"],
            ["Fatigue severity", f"{p.history.fatigue_severity}/10"],
            ["Breathlessness severity", f"{p.history.breathlessness_severity}/10"],
            ["Acute chest syndrome", yes_no(p.history.acute_chest_syndrome)],
            ["Stroke history", yes_no(p.history.stroke_history)],
            ["Hydroxyurea", yes_no(p.history.hydroxyurea)],
            ["Blood transfusions", yes_no(p.history.blood_transfusions)],
            ["Medication adherence", p.history.medication_adherence],
            ["Family history", yes_no(p.history.family_history)],
            ["Smoke exposure", yes_no(p.history.smoking_exposure)],
            ["Hydration pattern", p.history.hydration_status],
            ["Activity level", p.history.activity_level],
        ]),
        PageBreak(),
        Paragraph("Genetic Findings and Variant Interpretation", section),
        Paragraph(f"The selected genotype is {p.confirmed_genotype}. For sickle cell syndromes, HBB beta-globin variation affects haemoglobin structure and can promote haemoglobin S polymerisation under deoxygenated conditions.", body),
        Paragraph("Variant interpretation is grounded in curated HBB, haemoglobin and sickle-cell evidence sources represented in the application evidence layer.", body),
        Paragraph("Disease Severity Analysis", section),
        Paragraph(result.clinical_interpretation, body),
        Paragraph("Feature Importance", section),
        table([[k, f"{v:.0%}"] for k, v in result.prediction.feature_importance.items()]),
        PageBreak(),
        Paragraph("Current Risk Assessment", section),
        table([[k, f"{v:.0%}"] for k, v in result.prediction.complication_risks.items()]),
        Paragraph("Organ System Analysis", section),
        table([[k, f"{v:.0%}"] for k, v in result.prediction.organ_system_risks.items()]),
        Paragraph("Possible Future Complications", section),
        *[bullet(f"{name}: estimated current risk {value:.0%}") for name, value in result.prediction.complication_risks.items()],
        PageBreak(),
        Paragraph("Predicted Disease Progression", section),
        table([["Time point", "Severity", "Pain crises/year", "Hospitalisations/year", "Organ burden", "Confidence"]] + [
            [point["timepoint"], point["severity"], point["pain_crises_per_year"], point["hospitalisations_per_year"], f"{point['organ_risk_burden']:.0%}", f"{point['confidence']:.0%}"]
            for point in result.prediction.progression_forecast
        ]),
        Paragraph("Symptom Timeline", section),
        *[bullet(point["explanation"]) for point in result.prediction.progression_forecast],
        PageBreak(),
        Paragraph("Treatment Considerations", section),
        *[bullet(item) for item in (result.prediction.treatment_considerations or ["No specific treatment flags were triggered by the current rule engine. Continue routine specialist review."])],
        Paragraph("Confidence Scores", section),
        table([
            ["Model confidence", f"{result.prediction.model_confidence:.0%}"],
            ["Evidence confidence", f"{result.evidence_confidence:.0%}"],
            ["Patient-data completeness", f"{patient_completeness(p):.0%}"],
            ["Model version", result.prediction.model_version],
        ]),
        Paragraph("Biological Twin Comparison", section),
        Paragraph(result.digital_twin["educational_boundary"], body),
        table([
            ["Healthy biology", result.digital_twin["healthy"]["pathway"]],
            ["Patient biology", result.digital_twin["patient"]["pathway"]],
            ["Sickling index", f"{result.digital_twin['patient']['sickling_index']}/100"],
            ["Oxygen delivery", f"{result.digital_twin['patient']['oxygen_delivery']}/100"],
            ["Inflammation", f"{result.digital_twin['patient']['inflammation']}/100"],
        ]),
        PageBreak(),
        Paragraph("AI Reasoning Summary", section),
        *[bullet(item) for item in result.prediction.reasoning_factors],
        Paragraph("Evidence Summary", section),
    ]
    for record in result.evidence[:8]:
        story.append(Paragraph(f"<b>{record.source}: {record.title}</b>", body))
        story.append(Paragraph(record.summary, body))
        story.append(Spacer(1, 0.08 * inch))
    story.extend([
        PageBreak(),
        Paragraph("References to Established Clinical Knowledge", section),
        bullet("HBB beta-globin biology and haemoglobin S polymerisation are represented through curated ClinVar, ClinGen, NCBI Gene, UniProt, PubMed and AlphaFold-style evidence records."),
        bullet("Foetal haemoglobin is treated as a disease modifier that can reduce polymerisation burden in many sickle-cell contexts."),
        bullet("Frequent vaso-occlusive crises, acute chest syndrome, stroke history and recurrent admissions are treated as higher-burden clinical markers."),
        Paragraph("Privacy and Safety Waiver", section),
        Paragraph("This report may contain sensitive health information. It should be stored and transmitted only through institution-approved systems with appropriate access control, encryption, audit logging and privacy agreements. GeneVista outputs require review by qualified clinicians before clinical use.", body),
        Paragraph("Final Conclusion", section),
        Paragraph(f"The current GeneVista estimate places this patient in the {result.prediction.risk_category.lower()} risk category with {result.prediction.severity_label.lower()} predicted severity. Longitudinal projections show {result.prediction.progression_forecast[-1]['severity'].lower()} severity at 10 years under current assumptions.", body),
    ])
    SimpleDocTemplate(str(path), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch).build(story)
    return path


def generate_patient_pdf(result: AnalysisResult) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"genevista_patient_summary_{result.generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PatientTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#31513c"))
    section = ParagraphStyle("PatientSection", parent=styles["Heading2"], textColor=colors.HexColor("#123047"), spaceBefore=14)
    body = styles["BodyText"]

    def simple_table(rows):
        table = Table(rows, colWidths=[230, 250])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf4ed")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd8cf")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]))
        return table

    story = [
        Paragraph("GeneVista AI Patient & Family Summary", title),
        Paragraph("This plain-language report is designed to support discussion with your healthcare team. It should not be used to make urgent medical decisions without professional care.", body),
        Spacer(1, 0.18 * inch),
        Paragraph("What condition do I have?", section),
        Paragraph(f"Your entered genotype is {result.patient.confirmed_genotype}, which is a sickle-cell syndrome involving the HBB beta-globin gene.", body),
        Paragraph("What does my genetic result mean?", section),
        Paragraph("Changes in beta-globin can make haemoglobin more likely to form stiff fibres when oxygen is low. This can change red blood cell shape and make blood flow less smoothly.", body),
        Paragraph("What is happening inside my body?", section),
        simple_table([
            ["Healthy comparison", result.digital_twin["healthy"]["pathway"]],
            ["Patient-specific comparison", result.digital_twin["patient"]["pathway"]],
            ["Current risk category", result.prediction.risk_category],
        ]),
        PageBreak(),
        Paragraph("How may this change over time?", section),
        simple_table([["Time point", "What GeneVista estimates"]] + [[p["timepoint"], p["explanation"]] for p in result.prediction.progression_forecast]),
        Paragraph("Which symptoms should I watch for?", section),
        Paragraph("Watch for worsening pain crises, fever, chest pain, shortness of breath, weakness, severe headache, neurologic changes, dehydration, unusual fatigue or symptoms that feel different from your usual pattern.", body),
        Paragraph("What complications could develop?", section),
        simple_table([[k, f"{v:.0%} estimated risk signal"] for k, v in result.prediction.complication_risks.items()]),
        PageBreak(),
        Paragraph("What might my doctor recommend?", section),
        simple_table([[str(i + 1), item] for i, item in enumerate(result.prediction.treatment_considerations or ["Continue specialist follow-up and routine monitoring."])]),
        Paragraph("Lifestyle recommendations to discuss", section),
        Paragraph("Ask your care team about hydration, fever plans, pain-crisis plans, safe activity, vaccination, infection precautions, medication adherence and when to seek urgent care.", body),
        Paragraph("Questions to ask your healthcare provider", section),
        simple_table([
            ["1", "How should I interpret my predicted risk category?"],
            ["2", "Should my current treatment plan change?"],
            ["3", "What symptoms should trigger urgent care?"],
            ["4", "How often should labs and organ screening be repeated?"],
            ["5", "Are there disease-modifying therapies or clinical trials I should discuss?"],
        ]),
        Paragraph("Privacy and safety note", section),
        Paragraph("Keep this report private. Share it only with trusted healthcare professionals or caregivers. Confirm that any digital storage or messaging system is approved for health information.", body),
    ]
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
        st.session_state.patient_report_path = generate_patient_pdf(result)
    except Exception as exc:
        st.session_state.report_path = None
        st.session_state.patient_report_path = None
        st.warning(f"Report generation failed: {exc}")
    navigate("dashboard")


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def dashboard() -> None:
    result = st.session_state.analysis_result
    if result is None:
        navigate("landing")
        return
    patient = result.patient
    prediction = result.prediction
    st.markdown('<div class="gv-kicker">Prediction dashboard</div><h1>GeneVista Analysis Dashboard</h1>', unsafe_allow_html=True)
    metric_band(
        [
            ("Severity", prediction.severity_label, prediction.severity_label.lower() if prediction.severity_label != "Lower" else "low"),
            ("Risk category", prediction.risk_category, "high" if prediction.risk_category == "Elevated" else "moderate" if prediction.risk_category == "Intermediate" else "low"),
            ("Model confidence", f"{prediction.model_confidence:.0%}", None),
            ("Evidence confidence", f"{result.evidence_confidence:.0%}", None),
        ]
    )
    safety_privacy_notice()

    tabs = st.tabs(["Patient Summary", "Prediction", "Future Progression", "Evidence", "Digital Twin", "Reports"])
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
            st.write(f"Medication adherence: **{patient.history.medication_adherence}**")
            st.write(f"Hydration pattern: **{patient.history.hydration_status}**")
            st.write(f"Activity level: **{patient.history.activity_level}**")
    with tabs[1]:
        st.subheader("Clinical interpretation")
        st.write(result.clinical_interpretation)
        st.subheader("Supported prediction horizon")
        st.info(prediction.progression_statement)
        st.subheader("Predicted likelihood of future complications")
        for label, value in prediction.complication_risks.items():
            risk_bar(label, value)
        st.subheader("Feature importance")
        st.plotly_chart(feature_importance_chart(result), width="stretch")
    with tabs[2]:
        st.subheader("Personalised future progression")
        st.plotly_chart(progression_chart(result), width="stretch")
        st.dataframe(
            [
                {
                    "Time point": point["timepoint"],
                    "Severity": point["severity"],
                    "Pain crises/year": point["pain_crises_per_year"],
                    "Hospitalisations/year": point["hospitalisations_per_year"],
                    "Organ-risk burden": f"{point['organ_risk_burden']:.0%}",
                    "Confidence": f"{point['confidence']:.0%}",
                    "Explanation": point["explanation"],
                }
                for point in prediction.progression_forecast
            ],
            width="stretch",
            hide_index=True,
        )
        st.subheader("Organ system analysis")
        st.plotly_chart(organ_risk_chart(result), width="stretch")
        st.subheader("Treatment considerations")
        for item in prediction.treatment_considerations or ["No specific treatment flags were triggered by the current rule engine. Continue routine specialist review."]:
            st.info(item)
    with tabs[3]:
        st.subheader("Consensus evidence summary")
        for note in result.evidence_notes:
            st.info(note)
        for record in result.evidence:
            with st.expander(f"{record.source}: {record.title}"):
                st.write(record.summary)
                st.write(f"Quality weight: **{record.quality:.0%}**")
                if record.url:
                    st.link_button("Open source", record.url)
    with tabs[4]:
        st.subheader("Synchronized biological digital twin")
        st.caption(result.digital_twin["educational_boundary"])
        control_left, control_mid, control_right = st.columns([0.28, 0.44, 0.28])
        with control_left:
            if st.button("Previous stage", width="stretch"):
                st.session_state.twin_stage_index = max(0, st.session_state.twin_stage_index - 1)
                st.rerun()
        with control_mid:
            st.session_state.twin_stage_index = st.select_slider(
                "Biological level",
                options=list(range(len(TWIN_STAGES))),
                value=st.session_state.twin_stage_index,
                format_func=lambda idx: TWIN_STAGES[idx],
            )
            st.session_state.twin_time_index = st.select_slider(
                "Prediction time point",
                options=list(range(len(TIMEPOINTS))),
                value=st.session_state.twin_time_index,
                format_func=lambda idx: TIMEPOINTS[idx][0],
            )
        with control_right:
            if st.button("Next stage", width="stretch"):
                st.session_state.twin_stage_index = min(len(TWIN_STAGES) - 1, st.session_state.twin_stage_index + 1)
                st.rerun()
            if st.button("Restart twin", width="stretch"):
                st.session_state.twin_stage_index = 0
                st.session_state.twin_time_index = 0
                st.rerun()
        st.plotly_chart(twin_scene(result, st.session_state.twin_stage_index, st.session_state.twin_time_index), width="stretch")
        selected_stage = TWIN_STAGES[st.session_state.twin_stage_index]
        selected_point = prediction.progression_forecast[st.session_state.twin_time_index]
        st.markdown(
            f"""
            <div class="gv-panel">
            <strong>{html.escape(selected_stage)} at {html.escape(selected_point["timepoint"])}</strong><br>
            Healthy twin remains near baseline physiology. Patient twin shows {html.escape(selected_point["severity"].lower())}
            projected severity with {selected_point["pain_crises_per_year"]} estimated pain crises/year and
            {selected_point["organ_risk_burden"]:.0%} organ-system burden under current assumptions.
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            st.markdown("### Healthy reference")
            for key in ["gene", "protein", "red_cells", "pathway"]:
                st.write(f"**{key.replace('_', ' ').title()}**: {result.digital_twin['healthy'][key]}")
        with right:
            st.markdown("### Patient-specific disease biology")
            for key in ["gene", "protein", "red_cells", "pathway"]:
                st.write(f"**{key.replace('_', ' ').title()}**: {result.digital_twin['patient'][key]}")
        st.plotly_chart(digital_twin_radar(result), width="stretch")
    with tabs[5]:
        st.subheader("Professional reports")
        st.write("The clinician report contains patient information, clinical inputs, prediction outputs, confidence scores, evidence, feature importance, progression forecasts, organ-system analysis, twin comparison, reasoning, references and conclusion.")
        path = st.session_state.get("report_path")
        if path:
            with Path(path).open("rb") as file:
                st.download_button("Download clinician PDF report", data=file, file_name=Path(path).name, mime="application/pdf", width="stretch")
        patient_path = st.session_state.get("patient_report_path")
        if patient_path:
            with Path(patient_path).open("rb") as file:
                st.download_button("Download patient and family summary", data=file, file_name=Path(patient_path).name, mime="application/pdf", width="stretch")
        else:
            st.warning("One or more PDF reports are unavailable for this run.")
        if st.button("Start a new analysis"):
            for key in ["analysis_result", "patient_profile", "report_path", "patient_report_path"]:
                st.session_state[key] = None
            st.session_state.question_step = 0
            navigate("questionnaire")


def main() -> None:
    st.set_page_config(page_title="GeneVista AI", page_icon="GV", layout="wide")
    init_state()
    with st.sidebar:
        theme_choice = st.radio(
            "Theme",
            ["Light", "Dark"],
            index=["Light", "Dark"].index(st.session_state.get("theme_mode", "Light")),
            horizontal=True,
        )
    if theme_choice != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_choice
        st.rerun()
    apply_global_styles(st.session_state.theme_mode)
    with st.sidebar:
        st.markdown("## GeneVista AI")
        st.caption("Sickle Cell Disease MVP")
        if st.button("Home", width="stretch"):
            navigate("landing")
        if st.button("Start Analysis", width="stretch"):
            navigate("questionnaire")
        st.markdown("---")
        st.caption("Clinical decision-support software for validated deployments. Requires qualified clinician review.")

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
