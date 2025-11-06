import gradio as gr
import joblib

# --- Load trained artifact ---
artifacts = joblib.load("sent140_logreg_consistent.pkl")
model = artifacts["model"]
tfidf = artifacts["tfidf"]

# --- Define prediction function ---
def predict_sentiment(text):
    if not text.strip():
        return "⚠️ Please enter text"
    X = tfidf.transform([text])
    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    conf = round(max(probs) * 100, 2)
    emoji = "😊" if pred == "positive" else "😞"
    return f"{emoji} Predicted Sentiment: **{pred}** ({conf}% confidence)"

# --- Ready-to-try examples ---
examples = [
    ["I absolutely love this new phone! It's so fast and sleek."],
    ["Had a great time with my friends today! 😄"],
    ["What a wonderful experience, I’d recommend it to everyone!"],
    ["My new laptop works perfectly, very satisfied with the purchase."],
    ["The update improved everything, great job by the developers!"],
    ["This is the worst service I’ve ever experienced."],
    ["Ugh, my phone keeps crashing after the update. Terrible!"],
    ["Completely disappointed with the customer support 😡"],
    ["The movie was boring and way too long."],
    ["I wasted my money on this product, it doesn’t even work."]
]

# --- Build Gradio Interface ---
gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type or paste any tweet here..."),
    outputs="markdown",
    title="💬 Twitter Sentiment Analyzer (Sentiment140)",
    description=(
        "Predict whether a tweet expresses **positive** or **negative** sentiment. "
        "Model trained on 100,000 tweets from the Sentiment140 dataset using TF-IDF + Logistic Regression."
    ),
    examples=examples,
    allow_flagging="never",
    theme="gradio/soft"
).launch()
