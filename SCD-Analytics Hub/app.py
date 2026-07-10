"""GeneVista AI Streamlit application. Presentation layer only; business logic lives in src/genevista."""

from __future__ import annotations
import time
from pathlib import Path
import sys
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from genevista.core.config import settings
from genevista.core.logging import configure_logging
from genevista.domain.patient import Genotype, PatientInput, Sex
from genevista.ml.inference import ModelUnavailableError
from genevista.parsing.documents import inspect_upload
from genevista.reporting.pdf_report import generate_pdf
from genevista.services.analysis_service import AnalysisService
from genevista.twins.digital_twin import create_twin_pair

configure_logging()
st.set_page_config(
    page_title="GeneVista AI", page_icon="◈", layout="wide", initial_sidebar_state="collapsed"
)


def inject_css() -> None:
    st.markdown(
        """<style>
    @import url('https://fonts.googleapis.com/css2?family=Gentium+Book+Basic:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    :root { --ink:#17233a; --pine:#1d5a4c; --burgundy:#873d3d; --paper:#f7f3ea; --mist:#eee9de; --line:#d7d0c2; --muted:#637080; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'IBM Plex Sans',sans-serif; }
    #MainMenu, footer, header {visibility:hidden;} .block-container {max-width:1280px; padding:1.25rem 3.2rem 4rem;}
    h1,h2,h3 {font-family:'Gentium Book Basic',Georgia,serif!important; color:var(--ink); letter-spacing:-.025em;}
    h1 {font-size:clamp(3rem,5vw,5.4rem)!important; line-height:.96!important; font-weight:400!important;} h2 {font-size:2.15rem!important; font-weight:400!important;}
    p,li,label {color:var(--ink)!important;} .gv-nav{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding:0 0 1rem;margin-bottom:2.8rem}.gv-brand{font-family:'Gentium Book Basic';font-size:1.45rem;letter-spacing:-.03em}.gv-mark{color:var(--pine);margin-right:.4rem}.gv-nav-links{font-size:.76rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
    .eyebrow{color:var(--pine)!important;text-transform:uppercase;font-size:.72rem;letter-spacing:.14em;font-weight:600;margin-bottom:.65rem}.lede{font-size:1.18rem;line-height:1.65;color:#415067!important;max-width:34rem}.hero-wrap{min-height:520px;position:relative;overflow:hidden;border-bottom:1px solid var(--line);padding:3.2rem 0}.hero-art{position:absolute;right:-3rem;top:0;height:100%;width:68%;object-fit:cover;object-position:center;opacity:.96;mix-blend-mode:multiply;z-index:0}.hero-copy{width:54%;position:relative;z-index:1}.hero-copy .stButton{margin-top:1.5rem}.section{padding:4.5rem 0;border-bottom:1px solid var(--line)}
    .paper-card{background:rgba(255,255,255,.42);border:1px solid var(--line);border-radius:3px;padding:1.4rem;min-height:170px}.card-num{color:var(--pine);font-family:'Gentium Book Basic';font-size:1.6rem}.card-title{font-family:'Gentium Book Basic';font-size:1.45rem;margin:.5rem 0}.card-copy{font-size:.9rem;line-height:1.55;color:#526074!important}.method-strip{border-left:2px solid var(--pine);padding:.3rem 0 .3rem 1.2rem;color:#455166;font-size:.96rem;line-height:1.55}
    .stButton>button {background:var(--pine);color:#fff;border:1px solid var(--pine);border-radius:2px;padding:.62rem 1.1rem;font-family:'IBM Plex Sans';font-weight:500;transition:.15s ease}.stButton>button:hover{background:#15483d;border-color:#15483d;transform:translateY(-1px)}.stDownloadButton>button{background:transparent!important;color:var(--pine)!important;border:1px solid var(--pine)!important;border-radius:2px}.stTextInput input,.stNumberInput input,.stTextArea textarea,.stSelectbox select{border-radius:2px!important;border-color:var(--line)!important;background:#fffdf8!important}.stProgress>div>div>div>div{background:var(--pine)}
    .question-header{display:flex;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:.8rem;margin-bottom:1.4rem}.step-label{font-size:.72rem;letter-spacing:.12em;color:var(--pine);text-transform:uppercase}.metric-card{border-top:2px solid var(--pine);background:#fffdf8;padding:1.1rem 1.2rem}.metric-label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)!important}.metric-value{font-family:'Gentium Book Basic';font-size:2rem;margin-top:.25rem}.dashboard-title{padding:1.2rem 0 2rem;border-bottom:1px solid var(--line);margin-bottom:1.8rem}.evidence-row{border-top:1px solid var(--line);padding:.85rem 0}.source-tag{display:inline-block;color:var(--pine);font-weight:600;font-size:.78rem;margin-bottom:.25rem}.twin{border:1px solid var(--line);padding:1.5rem;min-height:360px;background:#fffdf8}.twin h3{margin-top:0}.twin-label{font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)!important;margin:1.2rem 0 .15rem}.cell{height:86px;width:150px;margin:1.2rem auto;border-radius:50%;background:#4f8a8a;position:relative}.cell.sickle{border-radius:70% 4% 70% 4%;transform:rotate(-25deg);background:#a64c48;width:135px}.cell:after{content:'';position:absolute;inset:22px 28px;border:2px solid rgba(255,255,255,.48);border-radius:50%}.cell.sickle:after{border-radius:70% 4% 70% 4%;inset:18px 23px}.notice{background:#e8f0ea;border-left:2px solid var(--pine);padding:1rem 1.15rem;font-size:.9rem;color:#405248}.warning{background:#f7ece5;border-left-color:var(--burgundy)}
    @media(max-width:800px){.block-container{padding:1rem 1.25rem 3rem}.hero-copy{width:75%}.hero-art{width:95%;opacity:.25;right:-8rem}.gv-nav-links{display:none}}
    </style>""",
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "page": "landing",
        "form_step": 0,
        "form_data": {},
        "patient": None,
        "analysis": None,
        "files": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def nav() -> None:
    st.markdown(
        '<div class="gv-nav"><div class="gv-brand"><span class="gv-mark">◈</span>GeneVista AI</div><div class="gv-nav-links">Sickle Cell Disease · Research Intelligence · v0.1</div></div>',
        unsafe_allow_html=True,
    )


def route(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def landing() -> None:
    nav()
    asset = ROOT / "assets/genevista-hero.png"
    st.markdown(
        f"""<section class="hero-wrap"><img class="hero-art" src="data:image/png;base64,{__import__("base64").b64encode(asset.read_bytes()).decode()}"><div class="hero-copy"><p class="eyebrow">Biomedical intelligence, carefully grounded</p><h1>See the biology<br>behind the burden.</h1><p class="lede">GeneVista AI brings patient context, evidence, supervised prediction, and educational biology into one considered workspace for sickle cell disease research.</p></div></section>""",
        unsafe_allow_html=True,
    )
    start, learn, method, about = st.columns([1.2, 1, 1, 0.8])
    with start:
        if st.button("Start analysis", use_container_width=True):
            route("questionnaire")
    with learn:
        if st.button("Learn more", use_container_width=True):
            st.session_state.anchor = "platform"
    with method:
        if st.button("Methodology", use_container_width=True):
            st.session_state.anchor = "methodology"
    with about:
        if st.button("About", use_container_width=True):
            st.session_state.anchor = "about"
    st.markdown(
        '<section class="section" id="platform"><p class="eyebrow">One workspace, five perspectives</p><h2>Evidence arrives before interpretation.</h2></section>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cards = [
        (
            "01",
            "Clinical context",
            "Structured inputs preserve the details that shape phenotype—laboratory values, crises, admissions, treatment, and history.",
        ),
        (
            "02",
            "Traceable evidence",
            "A retrieval layer normalises trusted source records, compares their agreement, and makes each major claim inspectable.",
        ),
        (
            "03",
            "Model-led prediction",
            "Registered supervised models quantify a supported prediction horizon, with versioned metrics and explicit uncertainty.",
        ),
        (
            "04",
            "Comparative biology",
            "A side-by-side Digital Twin turns molecular and cellular concepts into a clear educational model.",
        ),
        (
            "05",
            "Research-ready report",
            "A publication-minded PDF retains inputs, outputs, references, confidence, and methodology in one portable record.",
        ),
    ]
    for index, (num, title, copy) in enumerate(cards):
        with cols[index % 3]:
            st.markdown(
                f'<div class="paper-card"><div class="card-num">{num}</div><div class="card-title">{title}</div><div class="card-copy">{copy}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<section class="section" id="methodology"><p class="eyebrow">Methodology</p><h2>Built for scientific restraint.</h2><div class="method-strip">GeneVista is an evidence-grounded research and education tool. It presents a documented 12-month model horizon only, reports limitations when evidence is sparse, and does not generate diagnostic or treatment guidance.</div></section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<section class="section" id="about"><p class="eyebrow">Scope</p><h2>Beginning with sickle cell disease.</h2><p class="lede">The disease registry, feature schema, evidence adapters, and model registry are deliberately modular so additional inherited disorders can be added without rebuilding the platform.</p></section>',
        unsafe_allow_html=True,
    )


def progress_header(step: int) -> None:
    labels = ["Patient", "Laboratory", "History", "Documents", "Review"]
    st.progress(
        (step + 1) / len(labels), text=f"Section {step + 1} of {len(labels)} · {labels[step]}"
    )
    st.markdown(
        f'<div class="question-header"><h2>{labels[step]} information</h2><div class="step-label">Structured intake</div></div>',
        unsafe_allow_html=True,
    )


def questionnaire() -> None:
    nav()
    step = st.session_state.form_step
    st.markdown(
        '<p class="eyebrow">New analysis</p><h1 style="font-size:3.4rem!important">Patient intake</h1><p class="lede">Only de-identified information should be entered. Required fields are marked in each section.</p>',
        unsafe_allow_html=True,
    )
    progress_header(step)
    values = st.session_state.form_data
    if step == 0:
        c1, c2, c3 = st.columns(3)
        with c1:
            values["age"] = st.number_input(
                "Age (years) *", min_value=0, max_value=120, value=int(values.get("age", 18))
            )
        with c2:
            values["sex"] = st.selectbox(
                "Sex *",
                options=[x.value for x in Sex],
                index=[x.value for x in Sex].index(values.get("sex", Sex.undisclosed.value)),
            )
        with c3:
            values["genotype"] = st.selectbox(
                "Confirmed genotype *",
                options=[x.value for x in Genotype],
                index=[x.value for x in Genotype].index(
                    values.get("genotype", Genotype.hbs_hbs.value)
                ),
            )
    elif step == 1:
        c1, c2, c3 = st.columns(3)
        with c1:
            values["haemoglobin_g_dl"] = st.number_input(
                "Haemoglobin (g/dL) *", 2.0, 22.0, float(values.get("haemoglobin_g_dl", 8.5)), 0.1
            )
        with c2:
            values["fetal_haemoglobin_pct"] = st.number_input(
                "Foetal haemoglobin (%) *",
                0.0,
                100.0,
                float(values.get("fetal_haemoglobin_pct", 8.0)),
                0.1,
            )
        with c3:
            values["wbc_k_ul"] = st.number_input(
                "White blood cell count (K/μL) *",
                0.1,
                100.0,
                float(values.get("wbc_k_ul", 9.0)),
                0.1,
            )
    elif step == 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            values["voc_last_12m"] = st.number_input(
                "Vaso-occlusive crises, last 12 months *",
                0,
                100,
                int(values.get("voc_last_12m", 2)),
            )
        with c2:
            values["admissions_last_12m"] = st.number_input(
                "Hospital admissions, last 12 months *",
                0,
                100,
                int(values.get("admissions_last_12m", 1)),
            )
        with c3:
            values["transfusions_last_12m"] = st.number_input(
                "Blood transfusions, last 12 months *",
                0,
                100,
                int(values.get("transfusions_last_12m", 0)),
            )
        a, b, c = st.columns(3)
        with a:
            values["acute_chest_history"] = st.toggle(
                "Acute chest syndrome history", value=bool(values.get("acute_chest_history", False))
            )
        with b:
            values["stroke_history"] = st.toggle(
                "Stroke history", value=bool(values.get("stroke_history", False))
            )
        with c:
            values["hydroxyurea"] = st.toggle(
                "Currently taking hydroxyurea", value=bool(values.get("hydroxyurea", False))
            )
    elif step == 3:
        lab = st.file_uploader(
            "Laboratory report (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="lab"
        )
        vcf = st.file_uploader("VCF file (optional)", type=["vcf"], key="vcf")
        if lab:
            st.session_state.files["lab"] = inspect_upload(lab.name, lab.getvalue())
            values["laboratory_report_name"] = lab.name
        if vcf:
            st.session_state.files["vcf"] = inspect_upload(vcf.name, vcf.getvalue())
            values["vcf_name"] = vcf.name
        if st.session_state.files:
            for doc in st.session_state.files.values():
                st.caption(
                    f"{doc.filename} · {doc.kind.upper()} · {'; '.join(doc.warnings) if doc.warnings else 'ready for extraction'}"
                )
        st.info(
            "Document ingestion is separated from inference. This MVP retains upload metadata and preview-safe text; production deployment adds audited OCR, VCF normalization, malware scanning, and secure storage."
        )
    else:
        values["clinical_notes"] = st.text_area(
            "Additional clinical notes (optional)",
            value=values.get("clinical_notes", ""),
            max_chars=10000,
            placeholder="De-identified contextual notes only.",
        )
        st.markdown(
            '<div class="notice">Before generation, review the completeness of the structured clinical inputs. The prediction horizon is limited to 12 months and this platform is not a diagnostic or treatment decision tool.</div>',
            unsafe_allow_html=True,
        )
        st.json({k: v for k, v in values.items() if k != "clinical_notes"})
    back, _, next_col = st.columns([1, 4, 1.3])
    with back:
        if step and st.button("Back", use_container_width=True):
            st.session_state.form_step -= 1
            st.rerun()
    with next_col:
        if step < 4:
            if st.button("Continue", use_container_width=True):
                st.session_state.form_step += 1
                st.rerun()
        elif st.button("Generate analysis", use_container_width=True):
            try:
                st.session_state.patient = PatientInput(**values)
                route("processing")
            except Exception as exc:
                st.error(f"Please correct the intake: {exc}")


def processing() -> None:
    nav()
    st.markdown(
        '<div style="max-width:700px;margin:5rem auto;text-align:center"><p class="eyebrow">Analysis in progress</p><h1 style="font-size:3.5rem!important">Reading the clinical signal.</h1><p class="lede" style="margin:auto">GeneVista is assembling a traceable research view from the supplied context, registered model, and evidence records.</p></div>',
        unsafe_allow_html=True,
    )
    progress = st.progress(0, text="Preparing secure analysis workspace")
    status = st.empty()
    stages = [
        "Reading patient information",
        "Extracting uploaded document metadata",
        "Searching biomedical evidence adapters",
        "Retrieving literature context",
        "Running registered prediction model",
        "Comparing evidence",
        "Generating confidence score",
        "Creating Digital Twin",
        "Preparing report",
        "Finalising dashboard",
    ]
    started = time.monotonic()
    try:
        for i, stage in enumerate(stages):
            status.markdown(
                f"<div class='paper-card' style='min-height:auto;text-align:center'><span class='step-label'>{i + 1:02d} / {len(stages):02d}</span><div class='card-title'>{stage}</div></div>",
                unsafe_allow_html=True,
            )
            if stage == "Running registered prediction model":
                st.session_state.analysis = AnalysisService().run(st.session_state.patient)
            progress.progress((i + 1) / len(stages), text=stage)
            time.sleep(max(0.15, settings.min_processing_seconds / len(stages)))
        elapsed = time.monotonic() - started
        status.success(
            f"Analysis prepared in {elapsed:.1f} seconds. Opening the research dashboard…"
        )
        time.sleep(0.45)
        route("dashboard")
    except ModelUnavailableError as exc:
        st.error(str(exc))
        st.info(
            "For a local educational setup, run `python scripts/bootstrap_demo_data.py`. Replace its explicitly synthetic dataset with a governed dataset before research use."
        )
        if st.button("Return to intake"):
            route("questionnaire")
    except Exception as exc:
        st.error(f"Analysis stopped safely: {exc}")
        if st.button("Return to intake"):
            route("questionnaire")


def twin_card(twin) -> None:
    sickle = " sickle" if twin.cell_shape == "Sickle-like" else ""
    st.markdown(
        f"""<div class="twin"><p class="eyebrow">Educational model</p><h3>{twin.title}</h3><div class="cell{sickle}"></div><div class="twin-label">Gene</div><div>{twin.gene}</div><div class="twin-label">Protein</div><div>{twin.protein}</div><div class="twin-label">Red blood cell</div><div>{twin.red_cell}</div><div class="twin-label">Disease pathway</div><div>{twin.pathway}</div></div>""",
        unsafe_allow_html=True,
    )


def dashboard() -> None:
    nav()
    result = st.session_state.analysis
    if not result:
        route("landing")
    pred = result.prediction
    st.markdown(
        '<div class="dashboard-title"><p class="eyebrow">Sickle cell disease · Research analysis</p><h1 style="font-size:3.5rem!important">Prediction dashboard</h1><p class="lede">A structured view of model output, supporting evidence, and educational biology.</p></div>',
        unsafe_allow_html=True,
    )
    if pred.model_metadata.get("row_count", 0) == 1200:
        st.markdown(
            '<div class="notice warning"><b>Development data notice.</b> This result was produced from an explicitly synthetic training dataset. It is for interface and pipeline evaluation only—not clinical or research interpretation.</div>',
            unsafe_allow_html=True,
        )
    a, b, c, d = st.columns(4)
    values = [
        ("Estimated risk", f"{pred.probability:.0%}"),
        ("Risk category", pred.category),
        ("Prediction horizon", "12 months"),
        ("Evidence confidence", f"{result.evidence.confidence:.0%}"),
    ]
    for col, (label, value) in zip([a, b, c, d], values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section"><p class="eyebrow">Prediction summary</p><h2>Severity assessment</h2></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred.probability * 100,
                number={"suffix": "%", "font": {"size": 48, "color": "#17233A"}},
                title={
                    "text": "Estimated 12-month severe-outcome risk",
                    "font": {"size": 13, "color": "#637080"},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#637080"},
                    "bar": {"color": "#1D5A4C"},
                    "bgcolor": "#EEE9DE",
                    "steps": [
                        {"range": [0, 30], "color": "#EAF0EB"},
                        {"range": [30, 60], "color": "#EDE5D5"},
                        {"range": [60, 100], "color": "#F0DDDA"},
                    ],
                },
            )
        )
        gauge.update_layout(
            height=280, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<p class="eyebrow">Clinical interpretation</p>', unsafe_allow_html=True)
        st.write(result.interpretation)
        st.caption(
            f"Model {pred.model_metadata['version']} · trained {pred.model_metadata['trained_at'][:10]} · ROC-AUC {pred.model_metadata['metrics']['roc_auc']}"
        )
        if result.evidence.limitations:
            st.markdown(
                '<div class="notice warning"><b>Interpretation limits</b><br>'
                + "<br>".join(result.evidence.limitations)
                + "</div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="section"><p class="eyebrow">Model transparency</p><h2>Feature importance</h2></div>',
        unsafe_allow_html=True,
    )
    features = list(reversed(pred.feature_importance[:7]))
    figure = go.Figure(
        go.Bar(
            x=[v for _, v in features],
            y=[n for n, _ in features],
            orientation="h",
            marker_color="#1D5A4C",
        )
    )
    figure.update_layout(
        height=320,
        xaxis_title="Global model importance",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=40),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E2DCD0", zeroline=False)
    figure.update_yaxes(showgrid=False)
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "These are global model importances from the registered artifact, not individual causal contributions. Local explanation methods can be added behind the model adapter."
    )
    st.markdown(
        '<div class="section"><p class="eyebrow">Evidence ledger</p><h2>What supports this view</h2></div>',
        unsafe_allow_html=True,
    )
    ev_left, ev_right = st.columns([1.6, 0.8])
    with ev_left:
        for record in result.evidence.records:
            st.markdown(
                f'<div class="evidence-row"><div class="source-tag">{record.source} · quality {record.evidence_level:.0%}</div><div>{record.claim}</div><a href="{record.source_url}" target="_blank">View supporting source ↗</a></div>',
                unsafe_allow_html=True,
            )
    with ev_right:
        st.markdown(
            f'<div class="paper-card"><div class="metric-label">Agreement across records</div><div class="metric-value">{result.evidence.agreement:.0%}</div><div class="card-copy">Consensus confidence is configurable and combines source quality, agreement, data completeness, and model certainty.</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="section"><p class="eyebrow">Educational Digital Twin</p><h2>Compare the biological context</h2><p class="lede">A learning model rather than an exact physiological simulation. Use it to connect gene, protein, cell form, and pathway.</p></div>',
        unsafe_allow_html=True,
    )
    healthy, disease = create_twin_pair(result.patient)
    x, y = st.columns(2, gap="large")
    with x:
        twin_card(healthy)
    with y:
        twin_card(disease)
    st.markdown(
        '<div class="section"><p class="eyebrow">Exports</p><h2>Preserve the research record</h2></div>',
        unsafe_allow_html=True,
    )
    pdf = generate_pdf(result)
    st.download_button(
        "Download evidence report (PDF)",
        pdf,
        file_name="genevista-scd-analysis.pdf",
        mime="application/pdf",
    )
    st.caption(
        "Includes clinical inputs, prediction outputs, evidence references, confidence, methodology, and Digital Twin explanation. No direct identifiers are included by GeneVista."
    )


inject_css()
init_state()
{
    "landing": landing,
    "questionnaire": questionnaire,
    "processing": processing,
    "dashboard": dashboard,
}[st.session_state.page]()
