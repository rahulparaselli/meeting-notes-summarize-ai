"""
Meeting Summariser — Streamlit App

Single app for:
  1. Ingesting transcripts (paste text, upload .txt, upload audio)
  2. Chatting with the AI agent pipeline
  3. Viewing agent traces (classify → retrieve → compress → specialist)

Run:
    streamlit run app.py
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Meeting Summariser AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LangSmith tracing (auto-enabled if key is set) ──────────────────────────

from dotenv import load_dotenv
load_dotenv()

if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "your_langsmith_api_key_here":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "meeting-summariser")

# ── Imports ──────────────────────────────────────────────────────────────────

from src.ingestion.pipeline import ingest_text, ingest_audio
from src.agents.graph import run_pipeline, run_pipeline_stream


# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp { font-family: 'Inter', sans-serif; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1118 0%, #161825 100%);
}

/* Chat message styling */
.stChatMessage { border-radius: 12px !important; }

/* Agent trace cards */
.trace-step {
    background: linear-gradient(135deg, #1a1d2e, #12141e);
    border: 1px solid #2d3154;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 13px;
}
.trace-step .step-name {
    color: #7c6bf5;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.trace-step .step-detail {
    color: #858aa4;
    margin-top: 4px;
}

/* Meeting card in sidebar */
.meeting-card {
    background: #1a1d2e;
    border: 1px solid #2d3154;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    cursor: pointer;
}
.meeting-card:hover { border-color: #7c6bf5; }
.meeting-card .title { font-weight: 600; font-size: 13px; }
.meeting-card .meta { color: #4a4f6a; font-size: 11px; font-family: monospace; }

/* Success/error boxes */
.success-box {
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #2ecc71;
}
.error-box {
    background: rgba(231,76,60,0.1);
    border: 1px solid rgba(231,76,60,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #e74c3c;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────

if "meetings" not in st.session_state:
    st.session_state.meetings = {}  # {id: {title, date, attendees}}
if "current_meeting_id" not in st.session_state:
    st.session_state.current_meeting_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{role, content, traces, query_type}]


# ── Helper: run async ────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from Streamlit's sync context.

    Streamlit runs scripts in a thread that has no event loop,
    so we create a fresh one each time.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Sidebar: Ingest + Meeting Selector ───────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎙️ Meeting Summariser")
    st.caption("AI-powered meeting analysis with RAG")

    st.divider()

    # ── Ingest section ──
    st.markdown("### 📥 Ingest Meeting")

    ingest_tab = st.radio(
        "Input method",
        ["📝 Paste text", "📄 Upload .txt", "🎵 Upload audio"],
        horizontal=True,
        label_visibility="collapsed",
    )

    meeting_title = st.text_input("Meeting title", placeholder="e.g. Q4 Planning · Oct 2024")
    attendees_str = st.text_input("Attendees (optional)", placeholder="Alice, Bob, Carol")

    if ingest_tab == "📝 Paste text":
        transcript_text = st.text_area(
            "Transcript",
            height=150,
            placeholder="Alice: Let's start the meeting...\nBob: First item is the budget...",
        )
        if st.button("⚡ Ingest Transcript", type="primary", use_container_width=True):
            if not transcript_text.strip():
                st.error("Please paste a transcript")
            else:
                with st.spinner("Ingesting transcript..."):
                    attendees = [a.strip() for a in attendees_str.split(",") if a.strip()]
                    title = meeting_title or "Untitled Meeting"
                    meta = run_async(ingest_text(
                        text=transcript_text,
                        title=title,
                        attendees=attendees,
                    ))
                    st.session_state.meetings[meta.id] = {
                        "title": meta.title,
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "attendees": meta.attendees,
                    }
                    st.session_state.current_meeting_id = meta.id
                    st.session_state.chat_history = []
                    st.success(f"Ingested! Meeting ID: `{meta.id}`")
                    st.rerun()

    elif ingest_tab == "📄 Upload .txt":
        uploaded_file = st.file_uploader(
            "Upload transcript file",
            type=["txt", "md", "csv"],
            help="Drag and drop or browse for a transcript file",
        )
        if st.button("⚡ Ingest File", type="primary", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a file")
            else:
                with st.spinner("Reading and ingesting file..."):
                    text = uploaded_file.read().decode("utf-8")
                    if not text.strip():
                        st.error("File is empty")
                    else:
                        attendees = [a.strip() for a in attendees_str.split(",") if a.strip()]
                        title = meeting_title or uploaded_file.name.replace(".txt", "")
                        meta = run_async(ingest_text(
                            text=text,
                            title=title,
                            attendees=attendees,
                        ))
                        st.session_state.meetings[meta.id] = {
                            "title": meta.title,
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "attendees": meta.attendees,
                        }
                        st.session_state.current_meeting_id = meta.id
                        st.session_state.chat_history = []
                        st.success(f"Ingested! Meeting ID: `{meta.id}`")
                        st.rerun()

    elif ingest_tab == "🎵 Upload audio":
        audio_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "mp4", "ogg", "flac"],
            help="Supports MP3, WAV, M4A, and more",
        )
        hf_token = st.text_input(
            "HuggingFace token (optional)",
            type="password",
            placeholder="For speaker diarization",
        )
        if st.button("⚡ Ingest Audio", type="primary", use_container_width=True):
            if not audio_file:
                st.error("Please upload an audio file")
            else:
                with st.spinner("Transcribing and ingesting audio... This may take a while."):
                    # Save to temp file
                    suffix = Path(audio_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = Path(tmp.name)
                    try:
                        attendees = [a.strip() for a in attendees_str.split(",") if a.strip()]
                        title = meeting_title or audio_file.name
                        meta = run_async(ingest_audio(
                            audio_path=tmp_path,
                            title=title,
                            attendees=attendees,
                            hf_token=hf_token,
                        ))
                        st.session_state.meetings[meta.id] = {
                            "title": meta.title,
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "attendees": meta.attendees,
                        }
                        st.session_state.current_meeting_id = meta.id
                        st.session_state.chat_history = []
                        st.success(f"Ingested! Meeting ID: `{meta.id}`")
                        st.rerun()
                    finally:
                        tmp_path.unlink(missing_ok=True)

    # ── Meeting selector ──
    st.divider()
    st.markdown("### 📋 Meetings")

    if st.session_state.meetings:
        for mid, mdata in st.session_state.meetings.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = mid == st.session_state.current_meeting_id
                label = f"{'✅ ' if is_selected else ''}{mdata['title']}"
                if st.button(label, key=f"sel_{mid}", use_container_width=True):
                    st.session_state.current_meeting_id = mid
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                st.caption(mid[:8])
    else:
        st.info("No meetings yet. Ingest one above.")

    # ── LangSmith status ──
    st.divider()
    ls_key = os.getenv("LANGSMITH_API_KEY", "")
    if ls_key and ls_key != "your_langsmith_api_key_here":
        st.success("🔗 LangSmith tracing: ON", icon="✅")
        st.caption(f"Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        st.warning("LangSmith tracing: OFF")
        st.caption("Set LANGSMITH_API_KEY in .env to enable")


# ── Main: Chat Interface ────────────────────────────────────────────────────

current_mid = st.session_state.current_meeting_id

if not current_mid:
    # Landing state
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px;">
        <div style="font-size: 64px; margin-bottom: 16px;">🎙️</div>
        <h1 style="font-size: 28px; font-weight: 700; margin-bottom: 8px;">Meeting Summariser AI</h1>
        <p style="color: #858aa4; font-size: 15px; max-width: 420px; margin: 0 auto;">
            Ingest a meeting transcript or audio file from the sidebar, then ask questions about it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Show current meeting info
mdata = st.session_state.meetings.get(current_mid, {})
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.markdown(f"### 💬 {mdata.get('title', 'Meeting')}")
with col2:
    st.caption(f"ID: `{current_mid[:12]}`")
with col3:
    st.caption(f"📅 {mdata.get('date', '')}")

# Quick action buttons
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
with qcol1:
    if st.button("📋 Summarise", use_container_width=True):
        st.session_state.pending_query = "Summarise this meeting"
with qcol2:
    if st.button("✅ Action Items", use_container_width=True):
        st.session_state.pending_query = "What are the action items?"
with qcol3:
    if st.button("🎯 Decisions", use_container_width=True):
        st.session_state.pending_query = "What decisions were made?"
with qcol4:
    if st.button("👤 Ownership", use_container_width=True):
        st.session_state.pending_query = "Who was responsible for what?"

st.divider()

# ── Display chat history ─────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show agent trace in expander
        if msg.get("traces"):
            with st.expander("🔍 Agent Trace", expanded=False):
                for trace in msg["traces"]:
                    step = trace.get("step", "")
                    detail = trace.get("detail", "")

                    # Color-coded icons for different steps
                    icons = {
                        "classify_query": "⚡", "classify_result": "⚡",
                        "expand_and_retrieve": "🔍", "retrieval_result": "🔍",
                        "compress_context": "🗜️", "compression_done": "🗜️",
                        "summary_agent": "📋", "summary_done": "📋",
                        "action_items_agent": "✅", "action_items_done": "✅",
                        "decisions_agent": "🎯", "decisions_done": "🎯",
                        "qa_agent": "💬", "qa_done": "💬",
                    }
                    icon = icons.get(step, "○")
                    st.markdown(f"**{icon} {step}**  \n{detail}")


# ── Handle new queries ───────────────────────────────────────────────────────

# Check for pending quick-action query
pending = st.session_state.pop("pending_query", None)

# Chat input
user_input = st.chat_input("Ask anything about this meeting...")
query = pending or user_input

if query:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the agent pipeline
    with st.chat_message("assistant"):
        # Show progress
        status_container = st.empty()
        trace_container = st.container()

        status_container.info("🔄 Running agent pipeline...")

        try:
            # Run pipeline
            result = run_async(run_pipeline(current_mid, query))

            output = result.get("output", {})
            query_type = result.get("query_type", "general")
            thinking = result.get("thinking", [])

            # Convert query_type enum to string if needed
            qt = query_type.value if hasattr(query_type, "value") else str(query_type)

            # Format the response nicely
            response_parts = []

            if qt == "summary":
                tldr = output.get("tldr", "")
                key_points = output.get("key_points", [])
                if tldr:
                    response_parts.append(f"## 📋 TL;DR\n{tldr}")
                if key_points:
                    response_parts.append("\n## Key Points")
                    for kp in key_points:
                        response_parts.append(f"- {kp}")

            elif qt == "action_items":
                items = output.get("action_items", [])
                if items:
                    response_parts.append(f"## ✅ Action Items ({len(items)})")
                    for item in items:
                        task = item.get("task", "")
                        owner = item.get("owner")
                        deadline = item.get("deadline")
                        line = f"- **{task}**"
                        if owner:
                            line += f"  →  {owner}"
                        if deadline:
                            line += f"  📅 {deadline}"
                        response_parts.append(line)
                else:
                    response_parts.append("No action items found in this meeting.")

            elif qt == "decisions":
                decs = output.get("decisions", [])
                if decs:
                    response_parts.append(f"## 🎯 Decisions ({len(decs)})")
                    for d in decs:
                        desc = d.get("description", "")
                        ctx = d.get("context", "")
                        response_parts.append(f"- **{desc}**")
                        if ctx:
                            response_parts.append(f"  > {ctx}")
                else:
                    response_parts.append("No decisions found in this meeting.")

            else:
                # QA / general
                answer = output.get("answer", output.get("tldr", ""))
                if not answer:
                    answer = str(output) if output else "I couldn't find an answer to that question."
                response_parts.append(answer)

                sources = output.get("sources", [])
                if sources:
                    response_parts.append("\n---\n**📚 Sources:**")
                    for s in sources:
                        speaker = s.get("speaker", "Unknown")
                        preview = s.get("text_preview", "")[:100]
                        response_parts.append(f"- **[{speaker}]** {preview}")

            response_text = "\n".join(response_parts) if response_parts else "No output generated."

            # Clear status and show response
            status_container.empty()
            st.markdown(response_text)

            # Show agent trace
            if thinking:
                with st.expander("🔍 Agent Trace", expanded=True):
                    for trace in thinking:
                        step = trace.get("step", "")
                        detail = trace.get("detail", "")
                        icons = {
                            "classify_query": "⚡", "classify_result": "⚡",
                            "expand_and_retrieve": "🔍", "retrieval_result": "🔍",
                            "compress_context": "🗜️", "compression_done": "🗜️",
                            "summary_agent": "📋", "summary_done": "📋",
                            "action_items_agent": "✅", "action_items_done": "✅",
                            "decisions_agent": "🎯", "decisions_done": "🎯",
                            "qa_agent": "💬", "qa_done": "💬",
                        }
                        icon = icons.get(step, "○")
                        st.markdown(f"**{icon} {step}**  \n{detail}")

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "traces": thinking,
                "query_type": qt,
            })

        except Exception as e:
            status_container.empty()
            error_msg = str(e)
            st.error(f"Pipeline error: {error_msg}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {error_msg}",
                "traces": [],
            })
