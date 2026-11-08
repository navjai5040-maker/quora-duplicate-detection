import streamlit as st

from model_utils import load_artifacts, predict_pair


st.set_page_config(
    page_title="Quora Duplicate Detector",
    page_icon="🔎",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 850px; padding-top: 2.2rem;}
    .hero {
        padding: 1.6rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #1e3a8a);
        color: white;
        margin-bottom: 1.4rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem;}
    .hero p {color: #dbeafe; margin-bottom: 0;}
    .result {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.35);
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔎 Quora Duplicate Detector</h1>
      <p>Compare two questions using a Siamese Bidirectional LSTM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter two questions below. The model estimates whether they express "
    "the same underlying question."
)


@st.cache_resource(show_spinner="Loading the trained model and supporting files...")
def get_artifacts():
    return load_artifacts("artifacts")


try:
    artifacts = get_artifacts()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.exception(exc)
    st.info(
        "Check that all five exported files are inside the project's "
        "artifacts folder and that requirements.txt installs the dependencies. "
        "Then reboot the app."
    )
    st.stop()


with st.form("duplicate_check"):
    question1 = st.text_area(
        "Question 1",
        placeholder="e.g. What is the capital of India?",
        height=110,
    )
    question2 = st.text_area(
        "Question 2",
        placeholder="e.g. Which city is India's capital?",
        height=110,
    )
    submitted = st.form_submit_button(
        "Check for duplicates",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not question1.strip() or not question2.strip():
        st.warning("Please enter both questions before checking.")
    else:
        with st.spinner("Comparing the questions..."):
            try:
                result = predict_pair(question1, question2, artifacts)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                probability = result["probability"]
                percentage = probability * 100

                st.divider()
                st.subheader("Prediction")

                if result["prediction"] == 1:
                    st.success(
                        "The model predicts that these questions are duplicates."
                    )
                else:
                    st.info(
                        "The model predicts that these questions are not duplicates."
                    )

                st.metric("Duplicate probability", f"{percentage:.2f}%")
                st.progress(float(max(0.0, min(1.0, probability))))

                left, right = st.columns(2)
                with left:
                    st.caption("Question 1")
                    st.write(question1)
                with right:
                    st.caption("Question 2")
                    st.write(question2)

                with st.expander("How should I interpret this result?"):
                    st.write(
                        "The probability is the model's sigmoid output. "
                        "The application currently uses the same 0.5 decision "
                        "threshold as the notebook: values at or above 0.5 "
                        "are classified as duplicates."
                    )
                    st.caption(
                        "This is an experimental ML model, not a guarantee "
                        "that two questions have identical meaning."
                    )


st.divider()
st.caption(
    "Built with Streamlit and a trained Siamese BiLSTM. "
    "The app uses the saved tokenizer, numerical scaler, question-frequency "
    "dictionary, and feature-column order from the training notebook."
)
