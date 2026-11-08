"""
Inference utilities for the Quora Question Pairs Siamese BiLSTM.

IMPORTANT:
- These features mirror the training notebook's 24 engineered features.
- The neural network has its own trainable Embedding layer.
- The Word2Vec experiment in the classical-ML section is not used here.
"""

import pickle
import re
from pathlib import Path

import distance
import numpy as np
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences


MAX_SEQUENCE_LENGTH = 40


# This custom layer is part of the saved Siamese model architecture.
@keras.utils.register_keras_serializable()
class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        return __import__("tensorflow").abs(inputs[0] - inputs[1])


# Same text-cleaning function copied from the training notebook.
# Note: the notebook's actual preprocess() function does NOT apply Porter stemming.
def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q



def preprocess_question(question):
    """Apply exactly the text cleaning used before the model's tokenizer."""
    return preprocess(question)


def common_words(q1, q2):
    w1 = {word.lower().strip() for word in q1.split(" ")}
    w2 = {word.lower().strip() for word in q2.split(" ")}
    return len(w1 & w2)


def total_words(q1, q2):
    w1 = {word.lower().strip() for word in q1.split(" ")}
    w2 = {word.lower().strip() for word in q2.split(" ")}
    return len(w1) + len(w2)


def token_features(q1, q2, stemmed_stop_words):
    safe_div = 0.0001
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    features = [0.0] * 8

    if not q1_tokens or not q2_tokens:
        return features

    q1_words = {w for w in q1_tokens if w not in stemmed_stop_words}
    q2_words = {w for w in q2_tokens if w not in stemmed_stop_words}
    q1_stops = {w for w in q1_tokens if w in stemmed_stop_words}
    q2_stops = {w for w in q2_tokens if w in stemmed_stop_words}

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + safe_div)
    features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + safe_div)
    features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + safe_div)
    features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + safe_div)
    features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
    features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + safe_div)
    features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    features[7] = int(q1_tokens[0] == q2_tokens[0])
    return features


def length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    features = [0.0] * 3

    if not q1_tokens or not q2_tokens:
        return features

    features[0] = abs(len(q1_tokens) - len(q2_tokens))
    features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    common_substrings = list(distance.lcsubstrings(q1, q2))
    longest = len(common_substrings[0]) if common_substrings else 0
    features[2] = longest / (min(len(q1), len(q2)) + 1)
    return features


def fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


# These names describe the values generated below. The saved artifact may
# store them in a different order; we always reorder by the saved column list.
CANONICAL_FEATURE_COLUMNS = [
    "q1_len", "q2_len", "q1_num_words", "q2_num_words",
    "word_common", "word_total", "word_share", "q1_freq", "q2_freq",
    "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max",
    "last_word_eq", "first_word_eq", "abs_len_diff", "mean_len",
    "longest_substr_ratio", "fuzz_ratio", "fuzz_partial_ratio",
    "token_sort_ratio", "token_set_ratio",
]


def build_engineered_features(
    question1,
    question2,
    question_frequency,
    stemmed_stop_words,
    feature_columns,
):
    """Build the 24 engineered features in the exact saved scaler order."""
    q1 = preprocess_question(question1)
    q2 = preprocess_question(question2)

    common = common_words(q1, q2)
    total = total_words(q1, q2)

    token_values = token_features(q1, q2, stemmed_stop_words)
    length_values = length_features(q1, q2)
    fuzzy_values = fuzzy_features(q1, q2)

    # Use named values rather than relying on positional assumptions.
    feature_values = {
        "q1_len": len(q1),
        "q2_len": len(q2),
        "q1_num_words": len(q1.split(" ")),
        "q2_num_words": len(q2.split(" ")),
        "word_common": common,
        "word_total": total,
        "word_share": round(common / (total + 0.0001), 2),
        "q1_freq": question_frequency.get(q1, 0),
        "q2_freq": question_frequency.get(q2, 0),
        "cwc_min": token_values[0],
        "cwc_max": token_values[1],
        "csc_min": token_values[2],
        "csc_max": token_values[3],
        "ctc_min": token_values[4],
        "ctc_max": token_values[5],
        "last_word_eq": token_values[6],
        "first_word_eq": token_values[7],
        "abs_len_diff": length_values[0],
        "mean_len": length_values[1],
        "longest_substr_ratio": length_values[2],
        "fuzz_ratio": fuzzy_values[0],
        "fuzz_partial_ratio": fuzzy_values[1],
        "token_sort_ratio": fuzzy_values[2],
        "token_set_ratio": fuzzy_values[3],
    }

    # This is the critical fix: scaler.transform receives columns in the
    # precise order recorded when the scaler was fitted in the notebook.
    ordered_values = [feature_values[column] for column in feature_columns]
    numeric_row = np.asarray(ordered_values, dtype=np.float32).reshape(1, -1)
    return numeric_row, q1, q2


def load_artifacts(artifacts_dir="artifacts"):
    """Load the five files exported from the training notebook."""
    folder = Path(artifacts_dir)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder

    required = {
        "model": folder / "quora_siamese_bilstm.keras",
        "tokenizer": folder / "quora_tokenizer.pkl",
        "scaler": folder / "quora_numeric_scaler.pkl",
        "feature_columns": folder / "quora_dl_feature_columns.pkl",
        "frequency": folder / "question_frequency.pkl",
    }

    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s):\\n- " + "\\n- ".join(missing)
            + "\\n\\nPlace all five exported files inside the 'artifacts' folder."
        )

    with required["tokenizer"].open("rb") as f:
        tokenizer = pickle.load(f)
    with required["scaler"].open("rb") as f:
        scaler = pickle.load(f)
    with required["feature_columns"].open("rb") as f:
        feature_columns = pickle.load(f)
    with required["frequency"].open("rb") as f:
        question_frequency = pickle.load(f)

    if not isinstance(feature_columns, (list, tuple)):
        raise TypeError("The feature-column artifact must contain a list or tuple.")

    feature_columns = list(feature_columns)

    # Validate names and count, but do NOT require a particular order.
    # The saved order is authoritative because it matches the fitted scaler.
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("The saved feature-column list contains duplicate names.")

    missing_columns = [c for c in CANONICAL_FEATURE_COLUMNS if c not in feature_columns]
    extra_columns = [c for c in feature_columns if c not in CANONICAL_FEATURE_COLUMNS]
    if missing_columns or extra_columns or len(feature_columns) != len(CANONICAL_FEATURE_COLUMNS):
        raise ValueError(
            "The saved feature names do not match the 24 features generated by the app.\\n"
            f"Missing: {missing_columns}\\nExtra: {extra_columns}\\n"
            f"Saved: {feature_columns}"
        )

    scaler_feature_count = getattr(scaler, "n_features_in_", len(feature_columns))
    if scaler_feature_count != len(feature_columns):
        raise ValueError(
            "The scaler's feature count does not match the saved feature list: "
            f"scaler={scaler_feature_count}, columns={len(feature_columns)}."
        )

    # The notebook creates STEMMED_STOP_WORDS for token features, although
    # preprocess() itself does not stem the questions.
    from nltk.corpus import stopwords
    import nltk

    try:
        raw_stopwords = stopwords.words("english")
    except LookupError:
        downloaded = nltk.download("stopwords", quiet=True)
        if not downloaded:
            raise RuntimeError(
                "NLTK English stopwords are unavailable. Check the app's internet access "
                "or install the stopwords corpus before deployment."
            )
        raw_stopwords = stopwords.words("english")

    stemmer = PorterStemmer()
    stemmed_stop_words = {stemmer.stem(word) for word in raw_stopwords}

    model = keras.models.load_model(
        required["model"],
        custom_objects={"AbsoluteDifference": AbsoluteDifference},
        compile=False,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "question_frequency": question_frequency,
        "stemmed_stop_words": stemmed_stop_words,
    }


def predict_pair(question1, question2, artifacts):
    """Return duplicate probability and class using the notebook's 0.5 threshold."""
    if not isinstance(question1, str) or not isinstance(question2, str):
        raise TypeError("Both questions must be text strings.")
    if not question1.strip() or not question2.strip():
        raise ValueError("Please enter both questions.")

    q1_clean = preprocess_question(question1)
    q2_clean = preprocess_question(question2)

    q1_seq = artifacts["tokenizer"].texts_to_sequences([q1_clean])
    q2_seq = artifacts["tokenizer"].texts_to_sequences([q2_clean])

    q1_pad = pad_sequences(
        q1_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        dtype="int32",
    )
    q2_pad = pad_sequences(
        q2_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        dtype="int32",
    )

    numeric_row, _, _ = build_engineered_features(
        question1,
        question2,
        artifacts["question_frequency"],
        artifacts["stemmed_stop_words"],
        artifacts["feature_columns"],
    )

    expected_count = len(artifacts["feature_columns"])
    if numeric_row.shape[1] != expected_count:
        raise ValueError(
            f"Generated {numeric_row.shape[1]} features, but the scaler expects "
            f"{expected_count}."
        )

    scaled_row = artifacts["scaler"].transform(numeric_row).astype(np.float32)

    prediction_output = artifacts["model"].predict(
        {
            "q1_input": q1_pad,
            "q2_input": q2_pad,
            "engineered_input": scaled_row,
        },
        verbose=0,
    )
    probability = float(np.asarray(prediction_output).reshape(-1)[0])

    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "clean_question1": q1_clean,
        "clean_question2": q2_clean,
    }
