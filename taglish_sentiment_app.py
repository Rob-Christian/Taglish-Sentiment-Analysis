import streamlit as st
from transformers import pipeline

def classify_sentiment(label):
    """Map the model's label to positive, neutral, or negative."""
    if label in ["4 stars", "5 stars"]:
        return "Positive"
    elif label == "3 stars":
        return "Neutral"
    else:
        return "Negative"

def main():
    st.title("Taglish Sentiment Analyzer")
    st.write("This app evaluates the sentiment of a Taglish (Tagalog-English) phrase or sentence.")

    # Display model definition information
    st.subheader("About the Model")
    st.write("""
        The model used in this app is **BERT-based Multilingual Sentiment Model** (`nlptown/bert-base-multilingual-uncased-sentiment`).
        This model is a variant of BERT (Bidirectional Encoder Representations from Transformers) that has been trained to perform sentiment analysis.
        It can process text in multiple languages, including Taglish (a combination of Tagalog and English).
        The original model classifies sentiment into five categories based on a scale from 1 to 5 stars (from very negative to very positive).

        The revised model classifies sentiment only into three categories for ease of subjectiveness (positive, neutral, negative).
        Also, confidence score is adjusted for better granularity
    """)

    st.info("""
        **Sentiment Categories:**
        - Positive: Indicates praise, relief, or admiration.
        - Neutral: Indicates mixed or indifferent sentiment.
        - Negative: Indicates frustration, stress, or criticism.
    """)

    # Input text
    user_input = st.text_area("Enter a Taglish sentence:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                # Load a sentiment analysis pipeline (multilingual model)
                sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

                # Analyze sentiment
                result = sentiment_analyzer(user_input)

                # Extract result details
                raw_label = result[0]['label']
                adjusted_score = (result[0]['score'] ** 0.5) * 100  # Adjust confidence for better granularity
                sentiment = classify_sentiment(raw_label)

                # Display result
                st.success(f"The sentiment for the given Taglish phrase/sentence is **{sentiment}** with a confidence level of **{adjusted_score:.2f}%**.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a sentence to analyze.")

if __name__ == "__main__":
    main()
