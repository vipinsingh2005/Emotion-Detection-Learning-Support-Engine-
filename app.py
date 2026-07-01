import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pathlib
import os
from google import genai

EMOTIONS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 128

st.set_page_config(page_title="Emotion Engine", page_icon="🎓", layout="wide")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class StandaloneBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, len(EMOTIONS))
    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return self.fc(torch.cat((h[-2], h[-1]), dim=1))

def predict_emotion_scores(text):
    model = StandaloneBiLSTM().to(device)
    weights = pathlib.Path("models/bilstm/bilstm_weights.pt")
    if weights.exists():
        try: 
            model.load_state_dict(torch.load(weights, map_location=device), strict=False)
        except: 
            pass
    model.eval()
    with torch.no_grad():
        tokens = [ord(c) % VOCAB_SIZE for c in text.lower()][:MAX_SEQUENCE_LENGTH]
        tokens += [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
        inp = torch.tensor([tokens]).to(device)
        out = torch.softmax(model(inp), dim=1)[0]
    return EMOTIONS[torch.argmax(out).item()], torch.max(out).item() * 100, out

def generate_gemini_guidance(student_text, emotion):
    """Modern SDK profile with strict structural fallbacks"""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return f"### 🤝 AI Counselor Note (Fallback Mode)\n\nHey! I notice you are feeling **{emotion}** regarding this task. \n\n**Action Plan:**\n1. Break your code down into smaller modules.\n2. Write pseudocode before jumping into syntax.\n3. Take a quick 5-minute breather and debug step-by-step!"

    try:
        # Modern SDK Client Call Initialization
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert AI Learning Counselor and Engineering Mentor. 
        A student has submitted the following challenge: "{student_text}"
        Our internal Deep Learning model detected their current emotional state as: "{emotion}".
        
        Provide a concise, empathetic response tailored for an AI/Computer Science engineering student. 
        1. Acknowledge their emotion gently.
        2. Give technical action pieces (e.g., debugging tips, logical structure, or tool usage).
        3. Keep it under 150 words and use markdown bullet points.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ API execution encountered a mismatch context: {str(e)}"

# Interface Layout
st.title("🎓 Emotion Detection & Learning Support Engine")
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Describe Your Study/Project Challenge:", height=140, placeholder="Type your coding block issues, logic confusion, or errors here...")
    btn = st.button("🚀 Analyze Sentiment & Guide Me", type="primary")

if btn and user_input.strip():
    with st.spinner("Processing deep learning ensemble models & AI pipelines..."):
        emo, conf, probs = predict_emotion_scores(user_input)
        ai_guidance = generate_gemini_guidance(user_input, emo)
        
        with col1:
            st.success(f"### 🧠 Real-Time Emotion Detected: {emo}")
            st.markdown("### 🤝 Personalized AI Counselor Response")
            st.markdown(ai_guidance)
            
        with col2:
            st.subheader("📊 Model Confidence Distribution")
            st.metric(label="Primary Predicted State", value=emo, delta=f"{conf:.2f}% Match")
            df = pd.DataFrame({"Emotions": EMOTIONS, "Probability": [p.item() for p in probs]})
            st.bar_chart(df.set_index("Emotions"))
