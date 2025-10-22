"""
PodScribe - Production RAG Chatbot
"""
import streamlit as st
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from src.rag import search_transcripts, generate_answer
from src.rag.summarization import generate_episode_summary
from src.database import get_collection_stats, initialize_collection
from src.utils import Config
from src.audio_processing import transcribe_audio, diarize_audio, merge_transcription_and_diarization, convert_to_mp3
from src.database import index_transcript
from src.utils.helpers import get_episode_name, save_json

st.set_page_config(
    page_title="PodScribe",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MASSIVE HEADER CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 4rem 0 2rem 0 !important;
        letter-spacing: -4px;
        line-height: 0.9;
        text-transform: uppercase;
    }
    .subtitle {
        text-align: center;
        color: #999;
        font-size: 1.8rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: white !important;
    }
    .stat-label {
        font-size: 0.9rem;
        color: white !important;
    }
    .summary-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_episodes" not in st.session_state:
    st.session_state.selected_episodes = set()
if "process_query" not in st.session_state:
    st.session_state.process_query = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"  # "chat" or "summaries"

def get_available_transcripts():
    """Get available episodes from transcripts directory"""
    try:
        transcript_files = list(Config.TRANSCRIPTS_DIR.glob("*.json"))
        episodes = sorted([get_episode_name(f) for f in transcript_files])
        return episodes
    except Exception as e:
        return []

def get_available_summaries():
    """Get available summaries from summaries directory"""
    try:
        summaries_dir = Path("D:/RAG/data/summaries")
        summary_files = list(summaries_dir.glob("*_summary.md"))
        # Extract episode names from summary filenames
        summaries = {}
        for f in summary_files:
            episode_name = f.stem.replace("_summary", "")
            summaries[episode_name] = f
        return summaries
    except Exception as e:
        return {}

def load_summary(summary_path):
    """Load markdown summary content"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error loading summary: {str(e)}"

def is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

def is_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']

def process_user_query(user_q, top_k, speaker_filter):
    """Process a user query and return response"""
    try:
        if st.session_state.selected_episodes:
            sel = list(st.session_state.selected_episodes)

            if len(sel) == 1:
                res = generate_answer(user_q, top_k=top_k, episode_filter=f"{sel[0]}.mp3", speaker_filter=speaker_filter)
            else:
                all_res = []
                for ep in sel:
                    all_res.extend(search_transcripts(user_q, top_k=top_k, episode_filter=f"{ep}.mp3", speaker_filter=speaker_filter))

                all_res.sort(key=lambda x: x['similarity_score'], reverse=True)
                top_res = all_res[:top_k]

                ctx = "\n\n".join([f"[Source {i}]\nEpisode: {r['episode']}\nTimestamp: {r['timestamp_start']} - {r['timestamp_end']}\nContent: {r['text']}" for i, r in enumerate(top_res, 1)])
                srcs = [{'episode': r['episode'], 'timestamp': f"{r['timestamp_start']} - {r['timestamp_end']}", 'speakers': ', '.join(r['speakers'])} for r in top_res]

                res = generate_answer(user_q, context=ctx, top_k=top_k)
                res["sources"] = srcs
        else:
            res = generate_answer(user_q, top_k=top_k, speaker_filter=speaker_filter)

        return res
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}

# MASSIVE HEADER
st.markdown('<p class="main-header">PODSCRIBE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Multi-Source Podcast Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Podcast Management")

    # Stats
    with st.expander("Database Statistics", expanded=True):
        try:
            stats = get_collection_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{stats["total_chunks"]}</p>
                    <p class="stat-label">Chunks</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                episode_count = len(stats["sample_episodes"]) if stats["sample_episodes"] else 0
                st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{episode_count}</p>
                    <p class="stat-label">Episodes</p>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.error("Database not initialized")

    st.markdown("---")

    # VIEW MODE SELECTOR
    st.subheader("View Mode")
    view_mode = st.radio(
        "Select view",
        ["💬 Chat", "📄 Summaries"],
        label_visibility="collapsed",
        horizontal=True
    )

    if "💬 Chat" in view_mode:
        st.session_state.view_mode = "chat"
    else:
        st.session_state.view_mode = "summaries"

    st.markdown("---")

    # Upload
    st.subheader("Upload Podcasts")
    st.caption("MP3, MP4, WAV, M4A, AVI, MOV")

    uploaded_files = st.file_uploader(
        "Select files",
        type=["mp3", "wav", "m4a", "mp4", "avi", "mov", "mkv", "flac"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"**{len(uploaded_files)} selected**")

        col1, col2 = st.columns(2)
        with col1:
            gen_summary = st.checkbox("Summaries", value=True)
        with col2:
            save_trans = st.checkbox("Transcripts", value=True)

        for f in uploaded_files:
            st.write(f"• {f.name}")
    else:
        gen_summary = True
        save_trans = True

    process_btn = st.button("Process All Files", type="primary", disabled=not uploaded_files)

    # Processing
    if process_btn and uploaded_files:
        progress = st.progress(0)
        status = st.empty()

        total = len(uploaded_files)
        steps = 5 if gen_summary else 4
        done = 0

        for idx, file in enumerate(uploaded_files):
            try:
                status.info(f"Processing {idx+1}/{total}: {file.name}")

                # Save
                path = Config.RAW_AUDIO_DIR / file.name
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                # Convert if video
                audio_path = path
                if is_video_file(file.name):
                    audio_path = convert_to_mp3(path, Config.CONVERTED_AUDIO_DIR / f"{path.stem}.mp3")

                # Transcribe
                progress.progress((idx * steps + 1) / (total * steps))
                trans = transcribe_audio(audio_path)

                # Diarize
                progress.progress((idx * steps + 2) / (total * steps))
                diar = diarize_audio(audio_path)

                # Merge
                progress.progress((idx * steps + 3) / (total * steps))
                merged = merge_transcription_and_diarization(trans, diar)

                # Save transcript
                if save_trans:
                    trans_file = Config.TRANSCRIPTS_DIR / f"{get_episode_name(path)}.json"
                    save_json(merged, trans_file)
                    st.success(f"✓ Saved: {trans_file.name}")


                progress.progress((idx * steps + 4) / (total * steps))
                index_transcript(merged)

                # Summary
                if gen_summary:
                    progress.progress((idx * steps + 5) / (total * steps))
                    try:
                        sum_file = Config.SUMMARIES_DIR / f"{get_episode_name(path)}_summary.md"
                        generate_episode_summary(merged, output_path=sum_file)
                        st.success(f"✓ Summary: {sum_file.name}")
                    except Exception as e:
                        st.warning(f"Summary failed: {str(e)}")

                done += 1

            except Exception as e:
                st.error(f"Failed: {file.name} - {str(e)}")

        progress.progress(1.0)
        status.success(f"Processed {done}/{total} files")

        if done > 0:
            st.balloons()
            st.cache_resource.clear()
            time.sleep(2)
            st.rerun()

    st.markdown("---")


    st.subheader("Episode Filter")


    transcript_files = list(Config.TRANSCRIPTS_DIR.glob("*.json"))

    if transcript_files:
        available_episodes = get_available_transcripts()

        st.caption(f"Found {len(available_episodes)} processed episode(s)")
        st.caption(f"{len(st.session_state.selected_episodes)} selected")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_episodes = set(available_episodes)
                st.rerun()
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.selected_episodes.clear()
                st.rerun()

        st.markdown("---")


        for idx, ep in enumerate(available_episodes):

            is_selected = st.checkbox(
                ep,
                value=ep in st.session_state.selected_episodes,
                key=f"ep_checkbox_{idx}_{ep}"
            )


            if is_selected:
                st.session_state.selected_episodes.add(ep)
            else:
                st.session_state.selected_episodes.discard(ep)

        st.markdown("---")

        if st.session_state.selected_episodes:
            st.success(f"**Active:** {len(st.session_state.selected_episodes)} episode(s)")
        else:
            st.info("**Active:** All episodes")
    else:
        st.warning("**No episodes available**")
        st.info("👆 Click 'Process All Files' above to process uploaded podcasts first")


        with st.expander("Debug Info"):
            st.write(f"**Transcripts dir:** {Config.TRANSCRIPTS_DIR}")
            st.write(f"**Files in dir:** {len(list(Config.TRANSCRIPTS_DIR.glob('*')))}")

            raw_files = list(Config.RAW_AUDIO_DIR.glob("*"))
            if raw_files:
                st.write("**Raw audio files (not processed):**")
                for f in raw_files:
                    st.write(f"• {f.name}")

    st.markdown("---")


    if st.session_state.view_mode == "chat":
        st.subheader("Additional Filters")
        speaker_filter = st.text_input("Speaker", placeholder="SPEAKER_00")
        speaker_filter = speaker_filter if speaker_filter else None
        top_k = st.slider("Sources", 1, 15, 5)
    else:
        speaker_filter = None
        top_k = 5

    st.markdown("---")

    with st.expander("AI Config"):
        st.write(f"**LLM:** {Config.LLM_MODEL}")
        st.write(f"**Device:** {Config.DEVICE}")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area - CHAT MODE
if st.session_state.view_mode == "chat":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander(f"Sources ({len(msg['sources'])})"):
                        for i, s in enumerate(msg["sources"], 1):
                            st.markdown(f"""
**{i}.** {s.get('episode', 'Unknown')}  
`{s.get('timestamp', '00:00:00')}` • {s.get('speakers', 'Unknown')}
                            """)


        user_q = st.chat_input("Ask about your podcasts...")


        if st.session_state.process_query:
            user_q = st.session_state.process_query
            st.session_state.process_query = None

        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})

            with st.spinner("..."):
                res = process_user_query(user_q, top_k, speaker_filter)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res.get("sources", [])
                })
                st.rerun()

    with col2:
        st.subheader("Quick Search")

        sq = st.text_input("Keywords", placeholder="...")

        if sq:
            try:
                if st.session_state.selected_episodes:
                    all_r = []
                    for ep in st.session_state.selected_episodes:
                        all_r.extend(search_transcripts(sq, top_k=5, episode_filter=f"{ep}.mp3", speaker_filter=speaker_filter))
                    all_r.sort(key=lambda x: x['similarity_score'], reverse=True)
                    results = all_r[:5]
                else:
                    results = search_transcripts(sq, top_k=5, speaker_filter=speaker_filter)

                if results:
                    st.success(f"{len(results)} results")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"{i}. {r['episode'][:25]}...", expanded=(i==1)):
                            st.markdown(f"`{r['timestamp_start']}` • {r['similarity_score']:.0%}\n\n{r['text']}")
                else:
                    st.warning("No results")
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        st.subheader("Examples")

        # Fixed example buttons
        for q in ["What are the topics?", "Who speaks?", "Summarize"]:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.process_query = q
                st.rerun()


else:
    st.subheader("📄 Episode Summaries")

    available_summaries = get_available_summaries()

    if not available_summaries:
        st.warning("No summaries available yet. Upload and process podcasts with summaries enabled.")
    else:
        # Filter summaries based on selected episodes
        if st.session_state.selected_episodes:
            filtered_summaries = {k: v for k, v in available_summaries.items()
                                 if k in st.session_state.selected_episodes}
            if not filtered_summaries:
                st.info("No summaries available for selected episodes.")
                filtered_summaries = available_summaries
        else:
            filtered_summaries = available_summaries

        st.caption(f"Showing {len(filtered_summaries)} summary/summaries")
        st.markdown("---")

        # Display summaries
        for episode_name, summary_path in sorted(filtered_summaries.items()):
            with st.expander(f"📝 {episode_name}", expanded=len(filtered_summaries)==1):
                summary_content = load_summary(summary_path)

                # Display in a styled container
                st.markdown(f'<div class="summary-container">', unsafe_allow_html=True)
                st.markdown(summary_content)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary_content,
                    file_name=f"{episode_name}_summary.md",
                    mime="text/markdown",
                    key=f"download_{episode_name}"
                )

st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;padding:1rem'><p><b>PodScribe</b></p><p style='font-size:0.9rem'>WhisperX • Pyannote • ChromaDB • Groq</p></div>", unsafe_allow_html=True)
