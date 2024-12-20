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
                score = result[0]['score'] * 100  # Convert to percentage
                sentiment = classify_sentiment(raw_label)

                # Display result
                st.success(f"The sentiment for the given Taglish phrase/sentence is **{sentiment}** with a confidence level of **{score:.2f}%**.")

                # Add analysis for debugging or explanation
                st.write("### Analysis of Model Output")
                st.write(f"**Raw Label:** {raw_label}")
                st.write(f"**Confidence Score:** {score:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a sentence to analyze.")

if __name__ == "__main__":
    main()
