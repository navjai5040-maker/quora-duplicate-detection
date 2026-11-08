# Quora Duplicate Detector — Streamlit

This project wraps the trained Siamese Bidirectional LSTM from the Quora
Question Pairs notebook in a small Streamlit application.

## 1. Project structure

```text
quora_duplicate_streamlit/
│
├── app.py
├── model_utils.py
├── requirements.txt
├── README.md
├── .gitignore
└── artifacts/
    ├── quora_siamese_bilstm.keras
    ├── quora_tokenizer.pkl
    ├── quora_numeric_scaler.pkl
    ├── quora_dl_feature_columns.pkl
    └── question_frequency.pkl
```

The ZIP includes the source code and an empty `artifacts/` folder. Copy your
five downloaded model files into that folder. The model files are personal
training outputs and are not bundled in this source ZIP.

## 2. Set up the environment

Use a Python environment with a TensorFlow build compatible with your Python
version. Python 3.11 or 3.12 is a practical starting point if you run into
TensorFlow wheel compatibility issues.

Windows PowerShell:

```powershell
cd quora_duplicate_streamlit
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the model artifacts

Copy these exact files into `artifacts/`:

- `quora_siamese_bilstm.keras`
- `quora_tokenizer.pkl`
- `quora_numeric_scaler.pkl`
- `quora_dl_feature_columns.pkl`
- `question_frequency.pkl`

The app intentionally stops with a clear error if any file is missing.

## 4. Run the website locally

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.
Open it in your browser.

## 5. How the application works

1. `app.py` displays the two text areas and the prediction result.
2. `model_utils.py` applies the same `preprocess()` function used in the
   notebook.
3. The saved tokenizer converts each cleaned question to integer tokens.
4. `pad_sequences` makes both inputs length 40, with post-padding and
   post-truncation.
5. The app recreates the 24 numerical features in the same order as the
   notebook's `dl_feature_columns`.
6. The saved `StandardScaler` transforms those 24 values.
7. The `.keras` model receives three inputs:
   - `q1_input`
   - `q2_input`
   - `engineered_input`
8. The sigmoid output is treated as duplicate probability. The current
   decision threshold remains 0.5, matching the notebook.

## 6. Important model details

- The app uses the **Siamese BiLSTM**, not the classical Word2Vec model.
- The BiLSTM's Embedding layer is already part of the saved neural model.
- The Word2Vec experiment in the classical ML section does not need to be
  loaded by this app.
- Question frequencies are looked up using the preprocessed question text.
  The dictionary must come from the same training split and preprocessing
  used to train the model.
- The app preserves the notebook's current preprocessing behavior. In the
  notebook, `preprocess()` cleans text but does not actually stem it, even
  though Porter-stemmed stopwords are used for the stopword-overlap features.
  Do not add stemming here unless you retrain the model with that change.
- The app does not retrain the model. It only loads saved artifacts and runs
  inference.

## 7. Troubleshooting

### Missing artifact

Check that all five files are in the `artifacts` directory and that their
names match exactly.

### TensorFlow cannot load the `.keras` model

Install a TensorFlow version compatible with the Python version in your
environment. The app registers and supplies the custom `AbsoluteDifference`
layer used by the notebook.

### Feature-column mismatch

Do not reorder, rename, or regenerate the saved feature-column file using a
different notebook. The scaler and neural network expect the same 24 columns
in the same order.

### Prediction differs from Colab

Use the same pair of raw question strings in both places. Confirm that the
five artifacts came from the same trained run. Compare the cleaned questions,
the 24 unscaled features, the scaled feature row, and then the probability.

## 8. Suggested learning order

Read `model_utils.py` first:
- `preprocess_question`
- `token_features`
- `length_features`
- `fuzzy_features`
- `build_engineered_features`
- `load_artifacts`
- `predict_pair`

Then read `app.py` to understand how Streamlit collects input, calls the
prediction function, and displays the result.
