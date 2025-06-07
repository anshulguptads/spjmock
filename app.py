#app.py for mock interview spj
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import numpy as np
import openai
import requests
import tempfile
import PyPDF2
import re
import pandas as pd
from textblob import TextBlob
from weasyprint import HTML
import base64

# --- CONFIG ---
st.set_page_config(page_title="SPJMock AI Interview", page_icon=":microphone:")
openai.api_key = st.secrets["OPENAI_API_KEY"]
ELEVEN_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
ELEVEN_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

FILLERS = ["um", "uh", "like", "so", "actually", "basically", "you know", "I mean", "okay"]

st.image("static/spjlogo.png", width=220)
st.title("SPJMock: JD-Aware AI Interview Coach")

# --- Live Webcam Analytics Setup ---
mp_face_detection = mp.solutions.face_detection

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.centered = False
        self.face_found = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(img_rgb)
        h, w, _ = img.shape
        self.face_found = False
        self.centered = False

        if results.detections:
            detection = results.detections[0]  # Use first face detected
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            bbox_w = int(bbox.width * w)
            bbox_h = int(bbox.height * h)
            center_x = x_min + bbox_w // 2
            center_y = y_min + bbox_h // 2
            img_center_x, img_center_y = w // 2, h // 2
            margin_x = w * 0.2
            margin_y = h * 0.2
            self.centered = (abs(center_x - img_center_x) < margin_x) and (abs(center_y - img_center_y) < margin_y)
            self.face_found = True

            # Draw bounding box and status
            box_color = (0, 255, 0) if self.centered else (0, 255, 255)
            cv2.rectangle(img, (x_min, y_min), (x_min + bbox_w, y_min + bbox_h), box_color, 2)
            label = "Centered" if self.centered else "Not Centered"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        else:
            cv2.putText(img, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- SESSION STATE ---
if "questions" not in st.session_state:
    st.session_state.questions = []
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = []
if "analytics" not in st.session_state:
    st.session_state.analytics = []
if "video_metrics" not in st.session_state:
    st.session_state.video_metrics = []
if "done" not in st.session_state:
    st.session_state.done = False
if "summary" not in st.session_state:
    st.session_state.summary = ""

# --- JD Upload and Question Generation ---
def extract_jd_text(uploaded_file):
    if uploaded_file is None:
        return ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return uploaded_file.read().decode("utf-8")

def generate_questions_from_jd(jd_text, n_questions=6):
    prompt = (
        "Given the following job description, generate " +
        f"{n_questions} specific, high-quality, non-generic, role- and company-relevant interview questions. "
        "Do NOT ask generic questions like 'Tell me about yourself' or 'Why do you want this job.' "
        "Focus on domain knowledge, role requirements, technical/functional skills, and fit.\n\n"
        f"Job Description:\n{jd_text}\n\n"
        "Questions:"
    )
    chat_resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_questions = chat_resp.choices[0].message.content
    questions = []
    for line in raw_questions.split("\n"):
        line = line.strip()
        if re.match(r"^\d+\.", line):
            q = line.split(".", 1)[1].strip()
            if len(q) > 10:
                questions.append(q)
        elif line and not questions:
            questions.append(line)
    return questions[:7] if len(questions) >= 5 else questions

def analyze_transcript(text):
    words = text.split()
    num_words = len(words)
    text_lower = text.lower()
    filler_counts = {fw: len(re.findall(rf"\b{fw}\b", text_lower)) for fw in FILLERS}
    total_fillers = sum(filler_counts.values())
    unique_words = len(set(words))
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0.1 else ("Negative" if sentiment < -0.1 else "Neutral")
    return {
        "Word count": num_words,
        "Unique words": unique_words,
        "Filler words": total_fillers,
        "Filler breakdown": str({fw: count for fw, count in filler_counts.items() if count}),
        "Sentiment": sentiment_label,
        "Sentiment score": round(sentiment, 2)
    }

def analytics_to_html(df, summary):
    html = "<h1>SPJMock Interview Report</h1>"
    html += "<h2>Summary</h2>"
    html += f"<p>{summary}</p>"
    html += "<h2>Interview Details</h2>"
    html += "<table border='1' cellspacing='0' cellpadding='3'>"
    html += "<tr>" + "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr>"
    for _, row in df.iterrows():
        html += "<tr>" + "".join([f"<td>{row[col]}</td>" for col in df.columns]) + "</tr>"
    html += "</table>"
    return html

# --- 1. JD Upload/Questions ---
if not st.session_state.questions:
    st.subheader("Step 1: Upload a Job Description (PDF or Text File)")
    uploaded_file = st.file_uploader("Upload JD", type=["pdf", "txt"])
    if uploaded_file:
        jd_text = extract_jd_text(uploaded_file)
        st.markdown("**Extracted JD Text (preview):**")
        st.info(jd_text[:1000] + ("..." if len(jd_text) > 1000 else ""))
        with st.spinner("Analyzing JD and generating interview questions..."):
            questions = generate_questions_from_jd(jd_text, n_questions=7)
            st.session_state.questions = questions if len(questions) >= 5 else []
            if st.session_state.questions:
                st.success("Generated questions successfully! Ready to start the mock interview.")
            else:
                st.error("Failed to generate enough specific questions from JD. Please try another JD file.")
        st.stop()

# --- 2. Interview Loop ---
if not st.session_state.done:
    st.subheader(f"Question {st.session_state.q_idx + 1} of {len(st.session_state.questions)}")
    q = st.session_state.questions[st.session_state.q_idx]
    st.markdown(f"**{q}**")

    st.markdown("### Live Video Analytics")
    webrtc_ctx = webrtc_streamer(
        key=f"face-analytics-demo-{st.session_state.q_idx}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Show the real-time result
    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        face_found = webrtc_ctx.video_processor.face_found
        centered = webrtc_ctx.video_processor.centered
        st.info("Face detected. Centered." if (face_found and centered) else
                "Face detected, but not centered." if face_found else
                "No face detected or looking away.")
    else:
        st.info("Click 'Start' above to activate your webcam.")

    st.markdown("### Now record your answer:")

    audio_bytes = st.audio_recorder("🎤 Record Your Answer", sample_rate=16000, key=f"recorder{st.session_state.q_idx}")

    if audio_bytes and st.button("Submit Answer", key=f"submit{st.session_state.q_idx}"):
        st.audio(audio_bytes, format="audio/wav")

        # Capture current video analytics (face/camera)
        face_found = False
        centered = False
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            face_found = webrtc_ctx.video_processor.face_found
            centered = webrtc_ctx.video_processor.centered

        # 1. Transcribe
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            temp_audio_path = temp_audio.name
        with st.spinner("Transcribing..."):
            audio_file = open(temp_audio_path, "rb")
            transcript_resp = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            transcript = transcript_resp.text
            audio_file.close()
        st.markdown("**Transcript:**")
        st.info(transcript)

        # --- Analytics ---
        analytics = analyze_transcript(transcript)
        with st.expander("Show Analytics"):
            st.write(f"- **Word count:** {analytics['Word count']}")
            st.write(f"- **Unique words:** {analytics['Unique words']}")
            st.write(f"- **Filler words:** {analytics['Filler words']} ({analytics['Filler breakdown']})")
            st.write(f"- **Sentiment:** {analytics['Sentiment']} ({analytics['Sentiment score']})")
            st.write(f"- **Face detected (looking at camera):** {'Yes' if face_found else 'No'}")
            st.write(f"- **Centered:** {'Yes' if centered else 'No'}")

        # --- Feedback (Yoodli style) ---
        feedback_prompt = f"""You are an expert AI interview coach. Give feedback for this answer in the following Yoodli-like structure:
Content: (comment on clarity, relevance, completeness)
Delivery: (voice, pace, confidence, energy)
Filler Words: (mention if any used, e.g. um, uh)
Summary & Tip: (1-line summary and a specific tip for improvement)
Confidence Score: (rate 1-10 how confidently this answer would be received in an interview)

Here is the interview answer:
\"\"\"{transcript}\"\"\"
"""
        with st.spinner("AI is analyzing your answer..."):
            chat_resp = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": feedback_prompt}]
            )
            feedback = chat_resp.choices[0].message.content
        st.markdown("**AI Coach Feedback (Text):**")
        st.success(feedback)

        # --- ElevenLabs TTS ---
        with st.spinner("AI is speaking..."):
            tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
            headers = {
                "xi-api-key": ELEVEN_API_KEY,
                "Content-Type": "application/json"
            }
            tts_payload = {
                "text": feedback,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            }
            tts_resp = requests.post(tts_url, headers=headers, json=tts_payload)
            if tts_resp.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_audio:
                    tts_audio.write(tts_resp.content)
                    tts_audio.flush()
                    tts_audio_path = tts_audio.name
                st.markdown("**AI Coach Feedback (Audio):**")
                audio_file = open(tts_audio_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                audio_file.close()
            else:
                st.error("ElevenLabs TTS failed! Check API key or try again.")

        # Save all metrics
        st.session_state.answers.append(transcript)
        st.session_state.feedbacks.append(feedback)
        st.session_state.analytics.append(analytics)
        st.session_state.video_metrics.append({
            "Face Detected": "Yes" if face_found else "No",
            "Centered": "Yes" if centered else "No"
        })
        st.session_state.q_idx += 1
        if st.session_state.q_idx >= len(st.session_state.questions):
            st.session_state.done = True
        st.experimental_rerun()

# --- 3. Completion: Summary, Downloads ---
else:
    st.success("Interview complete! Here’s your overall summary and improvement plan:")
    # Final summary prompt to GPT-4o
    full_feedback_prompt = "You are an AI interview coach. Give a summary of the candidate's strengths and top 3 areas to improve, based on their answers and your previous feedback.\n\n"
    for idx, (q, a, f) in enumerate(zip(st.session_state.questions, st.session_state.answers, st.session_state.feedbacks)):
        full_feedback_prompt += f"Q{idx+1}: {q}\nAnswer: {a}\nFeedback: {f}\n\n"
    full_feedback_prompt += "Now summarize as a friendly interview coach."
    with st.spinner("AI is preparing your summary..."):
        chat_resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_feedback_prompt}]
        )
        summary = chat_resp.choices[0].message.content
        st.session_state.summary = summary
    st.info(summary)

    # --- TTS for summary ---
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }
    tts_payload = {
        "text": summary,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    tts_resp = requests.post(tts_url, headers=headers, json=tts_payload)
    if tts_resp.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_audio:
            tts_audio.write(tts_resp.content)
            tts_audio.flush()
            tts_audio_path = tts_audio.name
        st.markdown("**Summary (Audio):**")
        audio_file = open(tts_audio_path, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
        audio_file.close()
    else:
        st.error("ElevenLabs TTS failed for summary! Check API key or try again.")

    # --- Download as CSV ---
    df = pd.DataFrame({
        "Question": st.session_state.questions,
        "Answer": st.session_state.answers,
        "AI Feedback": st.session_state.feedbacks,
        "Word count": [a["Word count"] for a in st.session_state.analytics],
        "Unique words": [a["Unique words"] for a in st.session_state.analytics],
        "Filler words": [a["Filler words"] for a in st.session_state.analytics],
        "Filler breakdown": [a["Filler breakdown"] for a in st.session_state.analytics],
        "Sentiment": [a["Sentiment"] for a in st.session_state.analytics],
        "Sentiment score": [a["Sentiment score"] for a in st.session_state.analytics],
        "Face Detected": [v["Face Detected"] for v in st.session_state.video_metrics],
        "Centered": [v["Centered"] for v in st.session_state.video_metrics],
    })
    st.download_button("Download your interview report (CSV)", df.to_csv(index=False), file_name="interview_report.csv")

    # --- Download as PDF ---
    html_report = analytics_to_html(df, st.session_state.summary)
    pdf_bytes = HTML(string=html_report).write_pdf()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="interview_report.pdf">Download your interview report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

st.caption("Powered by OpenAI GPT-4o, Whisper, ElevenLabs TTS, Streamlit UI, MediaPipe, and streamlit-webrtc. © SP Jain School of Global Management  ")
