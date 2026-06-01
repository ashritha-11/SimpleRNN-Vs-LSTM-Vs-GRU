import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align:center;
    color:#FF4B4B;
    font-size:42px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:20px;
    margin-bottom:20px;
}

.metric-card {
    padding:15px;
    border-radius:10px;
    background:#f5f5f5;
    text-align:center;
}

.stButton button {
    width:100%;
    height:55px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='title'>🎬 Movie Review Sentiment Analysis System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning Based Sentiment Classification using SimpleRNN, LSTM and GRU</div>",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# FILE CHECKING
# =====================================================

required_files = [
    "simple_rnn_model.h5",
    "lstm_model.h5",
    "gru_model.h5",
    "tokenizer.pkl"
]

missing_files = [
    f for f in required_files
    if not os.path.exists(f)
]

if missing_files:
    st.error(
        f"Missing files: {', '.join(missing_files)}"
    )
    st.stop()

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    model_rnn = tf.keras.models.load_model(
        "simple_rnn_model.h5"
    )

    model_lstm = tf.keras.models.load_model(
        "lstm_model.h5"
    )

    model_gru = tf.keras.models.load_model(
        "gru_model.h5"
    )

    return model_rnn, model_lstm, model_gru


@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer


with st.spinner("Loading Deep Learning Models..."):

    model_rnn, model_lstm, model_gru = load_models()

    tokenizer = load_tokenizer()

# =====================================================
# PARAMETERS
# =====================================================

MAX_LENGTH = 200

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Sample Reviews")

sample_positive = """
This movie was absolutely fantastic and I enjoyed every minute.
"""

sample_negative = """
The movie was boring and a complete waste of time.
"""

sample_neutral = """
The acting was good but the story was average.
"""

if st.sidebar.button("Positive Review"):
    st.session_state.review = sample_positive

if st.sidebar.button("Negative Review"):
    st.session_state.review = sample_negative

if st.sidebar.button("Neutral Review"):
    st.session_state.review = sample_neutral

# =====================================================
# INPUT AREA
# =====================================================

review = st.text_area(
    "Enter your movie review here...",
    value=st.session_state.get("review", ""),
    height=200
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_review(model, text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence, probability

# =====================================================
# BUTTON
# =====================================================

if st.button("🔍 Analyze Review"):

    if len(review.strip()) == 0:

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":
            current_model = model_rnn

        elif selected_model == "LSTM":
            current_model = model_lstm

        else:
            current_model = model_gru

        sentiment, confidence, prob = predict_review(
            current_model,
            review
        )

        # =================================================
        # OUTPUT METRICS
        # =================================================

        col1, col2 = st.columns(2)

        with col1:

            if sentiment == "Positive":
                st.success(
                    f"😊 Sentiment: {sentiment}"
                )
            else:
                st.error(
                    f"😞 Sentiment: {sentiment}"
                )

        with col2:

            st.info(
                f"📊 Confidence: {confidence*100:.2f}%"
            )

        st.divider()

        # =================================================
        # PROBABILITY CHART
        # =================================================

        positive_prob = prob * 100
        negative_prob = (1 - prob) * 100

        chart_df = pd.DataFrame(
            {
                "Probability (%)":
                [
                    positive_prob,
                    negative_prob
                ]
            },
            index=[
                "Positive",
                "Negative"
            ]
        )

        st.subheader("📈 Probability Distribution")

        st.bar_chart(chart_df)

        # =================================================
        # COMPARE ALL MODELS
        # =================================================

        st.subheader(
            "🤖 Compare Predictions from All Models"
        )

        results = []

        for name, model in [

            ("SimpleRNN", model_rnn),
            ("LSTM", model_lstm),
            ("GRU", model_gru)

        ]:

            sent, conf, _ = predict_review(
                model,
                review
            )

            results.append([
                name,
                sent,
                round(conf * 100, 2)
            ])

        comparison_df = pd.DataFrame(

            results,

            columns=[
                "Model",
                "Prediction",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # =================================================
        # BEST MODEL
        # =================================================

        best = comparison_df.loc[
            comparison_df["Confidence (%)"].idxmax()
        ]

        st.success(
            f"🏆 Highest Confidence: "
            f"{best['Model']} "
            f"({best['Confidence (%)']}%)"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <center>
    Built using TensorFlow, Streamlit, SimpleRNN, LSTM and GRU
    </center>
    """,
    unsafe_allow_html=True
)
