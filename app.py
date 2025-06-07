import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_audiorecorder import audiorecorder
import openai
import tempfile
import os
from PyPDF2 import PdfReader
import weasyprint
from utils.video_analytics import analyze_frame

# Theme
st.set_page_config(page_title="SP Jain Mock Interview Tool", page_icon=":mortar_board:", layout="wide")
hide_streamlit_style = """
    <style>
    .css-1d391kg {background-color: #321911;} /* dark brown */
    .stApp {background-color: #321911;}
    .css-1v0mbdj {color: #FFD600;}
    .css-1cpxqw2 {color: #FFD600;}
    .css-10trblm {color: #FFD600;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Logo/Header
st.image("static/spjlogo.png", width=200)
st.title("SP Jain Mock Interview Tool")
st.markdown("AI-powered mock interviews with real-time video analytics and feedback.")

# --- 1. Upload JD and Generate Questions ---
st.header("1. Upload Job Description")
uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if uploaded_jd:
    reader = PdfReader(uploaded_jd)
    jd_text = ""
    for page in reader.pages:
        jd_text += page.extract_text() or ""
    st.success("Job Description Uploaded!")

    # --- 2. Generate Questions with OpenAI ---
    st.header("2. AI-Generated Interview Questions")
    with st.spinner("Generating questions..."):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        prompt = f"""
        You are an HR interviewer. Based on the following job description, generate 6 interview questions (not generic; job-specific).
        Return them as a numbered list.

        Job Description:
        {jd_text}
        """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
    st.write("### Questions:")
    for idx, q in enumerate(questions):
        st.markdown(f"**{idx+1}. {q}**")

    # --- 3. Interview Loop ---
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.responses = []
        st.session_state.analytics = []

    st.header("3. Live Interview & Analytics")
    st.image("static/demoimage.png", width=150)

    # Live video analytics
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = analyze_frame(img)
            st.session_state.last_video_analytics = result
            return frame
    ctx = webrtc_streamer(
        key="analytics",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Display analytics
    analytics = st.session_state.get("last_video_analytics", {})
    st.markdown("**Video Analytics:**")
    st.json(analytics)

    # --- 4. Audio Recorder & AI Feedback ---
    if st.session_state.current_question < len(questions):
        st.subheader(f"Q{st.session_state.current_question+1}: {questions[st.session_state.current_question]}")
        audio = audiorecorder("🎤 Record your answer", "Recording...")
        if audio is not None:
            # Save answer for later
            if st.button("Submit Answer"):
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio)
                temp_audio.close()
                # AI feedback
                with st.spinner("Analyzing..."):
                    audio_feedback = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an HR interview feedback assistant."},
                            {"role": "user", "content": f"Listen to this answer (audio not available, but the question was: '{questions[st.session_state.current_question]}'). Give a short, actionable feedback, focusing on content, clarity, and confidence."}
                        ],
                        max_tokens=150
                    )
                    feedback = audio_feedback.choices[0].message.content.strip()
                st.session_state.responses.append({
                    "question": questions[st.session_state.current_question],
                    "audio_file": temp_audio.name,
                    "feedback": feedback,
                    "video_analytics": analytics
                })
                st.session_state.current_question += 1
                st.experimental_rerun()
    else:
        st.success("Interview complete!")
        st.write("See your analytics and download your report below.")

        # --- 5. PDF Export ---
        st.header("Download PDF Report")
        report_html = f"""
        <h2>SP Jain Mock Interview Report</h2>
        <h3>Questions & Feedback</h3>
        <ol>
        {''.join([f"<li><b>{r['question']}</b><br>Feedback: {r['feedback']}<br>Analytics: {r['video_analytics']}</li>" for r in st.session_state.responses])}
        </ol>
        """
        pdf_file = weasyprint.HTML(string=report_html).write_pdf()
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="spjmock_report.pdf",
            mime="application/pdf"
        )

# --- Footer ---
st.markdown("""
<div style='text-align:center; color:#FFD600; margin-top:2em;'>
<b>Powered by SP Jain School of Global Management</b>
</div>
""", unsafe_allow_html=True)
