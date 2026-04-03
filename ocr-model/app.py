"""
Streamlit Web Interface for NassaQ OCR
Run with: streamlit run app.py --server.port 3001
"""

import streamlit as st
import pandas as pd
import time
import json

from ocr_pipeline import OCRPipeline, PipelineResult

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NassaQ OCR Engine",
    page_icon="",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
        line-height: 1.8;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">NassaQ OCR Engine</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Arabic + English document extraction '
    "powered by Azure AI Document Intelligence</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Session state ────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = [] 


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    model_choice = st.selectbox(
        "Analysis model",
        ["prebuilt-layout", "prebuilt-read"],
        index=0,
        help=(
            "layout: text + tables + structure (recommended).\n"
            "read: text only, faster and cheaper."
        ),
    )

    high_resolution = st.checkbox(
        "High Resolution OCR",
        value=True,
        help=(
            "Enable OCR_HIGH_RESOLUTION for scanned documents. "
            "Adds $10/1000 pages but dramatically improves text recall "
            "on low-DPI scans."
        ),
    )

    locale_options = {"Auto-detect": None, "Arabic (ar)": "ar", "English (en)": "en"}
    locale_label = st.selectbox(
        "Document locale hint",
        list(locale_options.keys()),
        index=0,
        help=(
            "Hint the OCR engine about the primary language. "
            "'Arabic' is recommended for Arabic-heavy documents to "
            "prevent Arabic glyphs from being skipped."
        ),
    )
    locale_value = locale_options[locale_label]

    output_format = st.selectbox(
        "Output format",
        ["markdown", "text"],
        index=0,
        help=(
            "Markdown preserves document structure (headers, tables, lists) "
            "and captures more text from complex layouts. "
            "Text is simpler but may drop content from multi-column pages."
        ),
    )

    strip_diacritics = st.checkbox(
        "Strip Arabic diacritics (tashkeel)",
        value=False,
        help="Remove fathah, dammah, kasrah, etc. Useful if downstream NLP doesn't need them.",
    )

    if st.button("Initialize Pipeline"):
        try:
            st.session_state.pipeline = OCRPipeline(
                model_id=model_choice,
                strip_diacritics=strip_diacritics,
                high_resolution=high_resolution,
                locale=locale_value,
                output_format=output_format,
            )
            opts = []
            opts.append(f"model: {model_choice}")
            opts.append(f"hi-res: {'on' if high_resolution else 'off'}")
            if locale_value:
                opts.append(f"locale: {locale_value}")
            opts.append(f"format: {output_format}")
            st.success(f"Ready  |  {' | '.join(opts)}")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.markdown("---")
    st.header("Supported Formats")
    st.markdown("PDF, JPEG, PNG, TIFF, BMP, HEIF, DOCX, XLSX, PPTX")

    st.markdown("---")
    st.header("Pricing")
    st.markdown(
        "- **prebuilt-read** : $1.50 / 1 000 pages\n"
        "- **prebuilt-layout** : $10 / 1 000 pages\n"
        "- **High Resolution add-on** : +$10 / 1 000 pages"
    )


# ── Tabs ─────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_history = st.tabs(
    ["Single Document", "Batch Upload", "History"]
)


# ── helpers ──────────────────────────────────────────────────────────────
def _display_result(res: PipelineResult) -> None:
    """Render a PipelineResult inside the current Streamlit container."""

    if not res.success:
        st.error(f"Error: {res.error}")
        return

    # ---- metrics row ----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Pages", res.page_count)
    c2.metric("Words", f"{res.word_count:,}")
    c3.metric("Confidence", f"{res.avg_confidence:.1%}")
    c4.metric("Cost", f"${res.cost_usd:.4f}")
    c5.metric("Time", f"{res.elapsed_seconds:.1f}s")
    c6.metric(
        "Chunks",
        res.chunks_used,
        help="How many API calls were needed (>1 means the file was split due to size)",
    )

    st.markdown("---")

    # ---- language badge ----
    lang_label = {"ar": "Arabic", "en": "English", "mixed": "Mixed (AR+EN)"}.get(
        res.primary_language, "Unknown"
    )
    st.info(f"Primary language detected: **{lang_label}**")

    # ---- extracted text ----
    st.subheader("Extracted Text")
    if res.primary_language in ("ar", "mixed"):
        st.markdown(
            f'<div class="rtl-text">{res.cleaned_text[:5000]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.text_area(
            "Full text", res.cleaned_text, height=300, label_visibility="collapsed"
        )

    if len(res.cleaned_text) > 5000:
        with st.expander("Show full text"):
            st.text(res.cleaned_text)

    # ---- tables ----
    if res.tables_markdown:
        st.subheader(f"Tables ({len(res.tables_markdown)} found)")
        for idx, md in enumerate(res.tables_markdown, 1):
            with st.expander(f"Table {idx}"):
                st.markdown(md, unsafe_allow_html=True)

                # also offer CSV download
                csv_data = (
                    res.tables_csv[idx - 1] if idx - 1 < len(res.tables_csv) else ""
                )
                if csv_data:
                    st.download_button(
                        f"Download Table {idx} as CSV",
                        data=csv_data,
                        file_name=f"table_{idx}.csv",
                        mime="text/csv",
                        key=f"dl_table_{idx}_{time.time()}",
                    )

    # ---- quality JSON ----
    with st.expander("Quality / metadata JSON"):
        st.json(res.quality)

    # ---- per-page diagnostics ----
    if res.per_page:
        with st.expander(f"Per-page diagnostics ({len(res.per_page)} pages)"):
            page_df = pd.DataFrame(res.per_page)

            # Highlight problematic pages
            problem_pages = [p for p in res.per_page if p["status"] != "good"]
            if problem_pages:
                st.warning(
                    f"{len(problem_pages)} page(s) may have issues: "
                    + ", ".join(f"p{p['page']} ({p['status']})" for p in problem_pages)
                )

            st.dataframe(page_df, use_container_width=True)


# ── Tab 1: Single Document ───────────────────────────────────────────────
with tab_single:
    st.header("Analyze Single Document")

    uploaded = st.file_uploader(
        "Upload a file (PDF, image, DOCX, ...)",
        type=[
            "pdf",
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp",
            "heif",
            "docx",
            "xlsx",
            "pptx",
        ],
        key="single_upload",
    )

    if uploaded is not None:
        st.write(
            f"**File:** {uploaded.name}  |  **Size:** {uploaded.size / 1024:.1f} KB"
        )

    run_disabled = st.session_state.pipeline is None or uploaded is None
    if st.button("Analyze", type="primary", disabled=run_disabled):
        with st.spinner("Running OCR pipeline ..."):
            data = uploaded.getvalue()
            result = st.session_state.pipeline.run_bytes(data, filename=uploaded.name)
            st.session_state.result = result
            st.session_state.history.append(result)

    if st.session_state.result is not None:
        _display_result(st.session_state.result)

        # download full text as markdown
        st.download_button(
            "Download extracted text",
            data=st.session_state.result.cleaned_text,
            file_name="extracted_text.md",
            mime="text/markdown",
        )


# ── Tab 2: Batch Upload ─────────────────────────────────────────────────
with tab_batch:
    st.header("Batch Analysis")
    st.markdown("Upload multiple files to process them sequentially.")

    batch_files = st.file_uploader(
        "Upload files",
        type=[
            "pdf",
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff",
            "bmp",
            "heif",
            "docx",
            "xlsx",
            "pptx",
        ],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_files:
        st.write(f"**{len(batch_files)} files** selected")

    batch_disabled = st.session_state.pipeline is None or not batch_files
    if st.button("Process Batch", type="primary", disabled=batch_disabled):
        progress = st.progress(0)
        status = st.empty()
        rows = []

        for i, f in enumerate(batch_files):
            status.text(f"Processing {i + 1}/{len(batch_files)}: {f.name}")
            res = st.session_state.pipeline.run_bytes(f.getvalue(), filename=f.name)
            st.session_state.history.append(res)

            rows.append(
                {
                    "file": f.name,
                    "pages": res.page_count,
                    "words": res.word_count,
                    "confidence": f"{res.avg_confidence:.1%}",
                    "language": res.primary_language,
                    "tables": len(res.tables_markdown),
                    "cost": f"${res.cost_usd:.4f}",
                    "time_s": res.elapsed_seconds,
                    "status": "ok" if res.success else res.error or "error",
                }
            )
            progress.progress((i + 1) / len(batch_files))

        status.text("Done!")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        total_cost = sum(
            r.cost_usd for r in st.session_state.history[-len(batch_files) :]
        )

        st.download_button(
            "Download batch report CSV",
            data=df.to_csv(index=False),
            file_name="ocr_batch_report.csv",
            mime="text/csv",
        )


# ── Tab 3: History ───────────────────────────────────────────────────────
with tab_history:
    st.header("Session History")

    if not st.session_state.history:
        st.info("No documents processed yet.")
    else:
        rows = []
        for idx, r in enumerate(st.session_state.history, 1):
            rows.append(
                {
                    "#": idx,
                    "file": r.source_file or "-",
                    "pages": r.page_count,
                    "words": r.word_count,
                    "confidence": f"{r.avg_confidence:.1%}",
                    "language": r.primary_language,
                    "tables": len(r.tables_markdown),
                    "cost": f"${r.cost_usd:.4f}",
                    "time_s": r.elapsed_seconds,
                    "status": "ok" if r.success else "error",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        total_cost = sum(r.cost_usd for r in st.session_state.history)
        total_pages = sum(r.page_count for r in st.session_state.history)
        st.markdown(
            f"**Totals:** {total_pages} pages  |  ${total_cost:.4f} spent this session"
        )


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;padding:1rem;">'
    "<b>NassaQ OCR Engine</b><br>"
    "Graduation Project 2026  |  Azure AI Document Intelligence"
    "</div>",
    unsafe_allow_html=True,
)
