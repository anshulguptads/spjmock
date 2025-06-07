#app.py for mock interview spj
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
#import mediapipe as mp
import numpy as np
#from streamlit_audiorecorder import audiorecorder
from audiorecorder import audiorecorder
import openai
import requests
import tempfile
import PyPDF2
import re
import pandas as pd
from textblob import TextBlob
# from weasyprint import HTML   # <-- COMMENT OUT/REMOVE FOR CLOUD
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

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Load OpenCV's default face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.centered = False
        self.face_found = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        h, w = img.shape[:2]
        self.face_found = False
        self.centered = False

        for (x, y, fw, fh) in faces:
            center_x = x + fw // 2
            center_y = y + fh // 2
            img_center_x, img_center_y = w // 2, h // 2
            margin_x = w * 0.2
            margin_y = h * 0.2
            self.centered = (abs(center_x - img_center_x) < margin_x) and (abs(center_y - img_center_y) < margin_y)
            self.face_found = True

            color = (0, 255, 0) if self.centered else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + fw, y + fh), color, 2)
            label = "Centered" if self.centered else "Not Centered"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            break  # Only consider first detected face

        if not self.face_found:
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

def generate_questions_from_jd(jd_text, n_questions=7):
    prompt = (
        f"""Given the following job description, generate at least 5 specific, high-quality, non-generic, role- and company-relevant interview questions.
If you cannot generate 5 targeted questions, rephrase or vary questions as needed to ensure you return 5.
Output ONLY the questions, numbered 1. to 5., one per line, with no explanations, greetings, or extra text.

Job Description:
{jd_text}

Questions:"""
    )
    chat_resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    raw_questions = chat_resp.choices[0].message.content

    # Parse numbered lines: 1. ... 2. ... 3. ...
    questions = []
    for line in raw_questions.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() and '.' in line[:3]):
            q = line.split(".", 1)[1].strip()
            if len(q) > 10:
                questions.append(q)
    # If less than 5, try again ONCE (rare case, but can help)
    if len(questions) < 5:
        # Retry with slightly stronger instruction
        prompt2 = (
            f"""IMPORTANT: Return exactly 5 interview questions. If needed, repeat or rephrase to make 5. Output only numbered questions (1. ... 2. ...).
Job Description:
{jd_text}
Questions:"""
        )
        chat_resp2 = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt2}]
        )
        raw_questions2 = chat_resp2.choices[0].message.content
        questions = []
        for line in raw_questions2.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() and '.' in line[:3]):
                q = line.split(".", 1)[1].strip()
                if len(q) > 10:
                    questions.append(q)
    # Return only first 5 if extra
    return questions[:5]


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
# ---- AUDIO RECORDING (streamlit-audiorecorder) ----
st.markdown("### Record your answer:")

audio_bytes = audiorecorder("Click to record your answer", "Recording...")

if audio_bytes is not None and len(audio_bytes) > 0:
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Submit Answer", key=f"submit{st.session_state.q_idx}"):
        # Save the audio as a temp file for Whisper
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

        # --- Analytics, feedback, and metrics (as before) ---
        # (Insert your analytics/feedback code here, unchanged)


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

st.caption("Powered by OpenAI GPT-4o, Whisper, ElevenLabs TTS, Streamlit UI, OpenCV, and streamlit-webrtc. © SP Jain School of Global Management  ")
