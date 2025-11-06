# 💬 Twitter Sentiment Analyzer (Sentiment140)

This interactive web app predicts whether a tweet expresses **positive 😊** or **negative 😞** sentiment.  
It was trained on 100,000 tweets from the **Sentiment140** dataset using **TF-IDF features** and a **Logistic Regression** model.

### 🚀 Demo
Try it live here 👉 [Sentiment Analyzer](https://huggingface.co/spaces/cookingML/sentiment140-analyzer)

### 🧠 Model Info
| Property | Details |
|-----------|----------|
| **Dataset** | Sentiment140 (sample of 100k tweets) |
| **Algorithm** | Logistic Regression |
| **Features** | TF-IDF (1–2 grams) |
| **Accuracy** | ~77% (macro F1 ≈ 0.77) |

### 🧾 Example Inputs
#### ✅ Positive
- I absolutely love this new phone!  
- The update improved everything, great job by the developers!  
- Had a great time with my friends today! 😄  
- What a wonderful experience — I’d recommend it to everyone!  
- My new laptop works perfectly, very satisfied with the purchase.

#### ❌ Negative
- This is the worst service I’ve ever experienced.  
- My phone keeps crashing after the update, terrible!  
- Completely disappointed with the customer support 😡  
- The movie was boring and way too long.  
- I wasted my money on this product; it doesn’t even work.

### 🧰 Tech Stack
- Python, scikit-learn, Gradio  
- Deployed on Hugging Face Spaces (CPU)

---

👨‍💻 **Author:** [CookingML](https://huggingface.co/cookingML)
