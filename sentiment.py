from transformers import pipeline
import gradio as gr

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label'], result[0]['score']

def sentiment_analysis_app():
    interface = gr.Interface(
        fn=predict_sentiment,
        inputs="text",
        outputs=["text", "number"],
        title="+-?",
        description="Enter any sentence to analyze its sentimetn (Positive/Negative).",
        examples=["I love you", "This is the worst experience ever."]
    )
    return interface

if __name__ == "__main__":
    app = sentiment_analysis_app()
    app.launch()