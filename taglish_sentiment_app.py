import streamlit as st
from transformers import pipeline

def main():
    st.title("Taglish Sentiment Analyzer")
    st.write("This app evaluates the sentiment of a Taglish (Tagalog-English) phrase or sentence.")

    # Define sentiment values and their meanings
    sentiment_info = {
        "1 star": "Very negative sentiment",
        "2 stars": "Negative sentiment",
        "3 stars": "Neutral sentiment",
        "4 stars": "Positive sentiment",
        "5 stars": "Very positive sentiment"
    }

    st.info("""
        **Sentiment Values:**
        - 1 Star: Very Negative
        - 2 Stars: Negative
        - 3 Stars: Neutral
        - 4 Stars: Positive
        - 5 Stars: Very Positive
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
                sentiment = result[0]['label']
                score = result[0]['score'] * 100  # Convert to percentage
                sentiment_description = sentiment_info.get(sentiment, "Unknown sentiment")

                # Display result
                st.success(f"The sentiment for the given Taglish phrase/sentence is **{sentiment_description}** with a confidence level of **{score:.2f}%**.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a sentence to analyze.")

if __name__ == "__main__":
    main()
