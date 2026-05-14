import streamlit as st
import joblib
import pandas as pd

model    = joblib.load('E:/For_Work/Portfolio/Python/Spotify Tracks Dataset/models/model.pkl')
features = joblib.load('E:/For_Work/Portfolio/Python/Spotify Tracks Dataset/models/features.pkl')

st.title("🎵 Spotify Hit Prediction")
st.write("Predict whether a song will become a hit based on its audio features.")

st.sidebar.header("Audio Features")

# Fixed: = not - on all sliders
danceability     = st.sidebar.slider("Danceability",     0.0, 1.0,   0.5)
energy           = st.sidebar.slider("Energy",           0.0, 1.0,   0.5)
loudness         = st.sidebar.slider("Loudness (dB)",  -30.0, 0.0, -10.0)
speechiness      = st.sidebar.slider("Speechiness",      0.0, 1.0,   0.05)
acousticness     = st.sidebar.slider("Acousticness",     0.0, 1.0,   0.5)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0,   0.0)
liveness         = st.sidebar.slider("Liveness",         0.0, 1.0,   0.1)
valence          = st.sidebar.slider("Valence",          0.0, 1.0,   0.5)
tempo            = st.sidebar.slider("Tempo (BPM)",     50.0, 220.0, 120.0)
duration_ms      = st.sidebar.number_input("Duration (ms)", 30000, 600000, 200000)
key              = st.sidebar.slider("Key",   0, 11, 5)       # fixed: = not -
mode             = st.sidebar.selectbox("Mode", [0, 1])
time_signature   = st.sidebar.slider("Time Signature", 1, 5, 4)  # fixed: 1-5 range
explicit         = st.sidebar.selectbox("Explicit", [0, 1])

# Genre selection — pulls valid genres from feature list automatically
genre_cols       = [f for f in features if f.startswith("track_genre_")]
genre_options    = [g.replace("track_genre_", "") for g in genre_cols]
genre            = st.sidebar.selectbox("Genre", sorted(genre_options))

# Build input row
input_data = {
    "danceability":     danceability,
    "energy":           energy,
    "loudness":         loudness,
    "speechiness":      speechiness,
    "acousticness":     acousticness,
    "instrumentalness": instrumentalness,
    "liveness":         liveness,
    "valence":          valence,
    "tempo":            tempo,
    "duration_ms":      duration_ms,
    "key":              key,
    "mode":             mode,
    "time_signature":   time_signature,
    "explicit":         explicit,
}

df_input = pd.DataFrame([input_data])

# Apply genre one-hot encoding properly
for col in genre_cols:
    df_input[col] = 0
selected_genre_col = f"track_genre_{genre}"
if selected_genre_col in df_input.columns:
    df_input[selected_genre_col] = 1

# Align to training feature order — fill any missing with 0
for col in features:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[features]

# Prediction
st.subheader("Prediction")
prob = model.predict_proba(df_input)[0][1]
pred = model.predict(df_input)[0]

st.metric("Hit Probability", f"{prob:.1%}")
st.progress(float(prob))

if pred == 1:
    st.success(f"✅ Predicted HIT  (probability: {prob:.2f})")
else:
    st.error(f"❌ Not a hit  (probability: {prob:.2f})")

with st.expander("Show input sent to model"):
    st.dataframe(df_input)