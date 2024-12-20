import streamlit as st
from transformers import pipeline

def main():
    st.title("Taglish Sentiment Analyzer")
    st.write("This app evaluates the sentiment of a Taglish (Tagalog-English) phrase or sentence.")

    # Input text
    user_input = st.text_area("Enter a Taglish sentence:", "")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            try:
                # Load a sentiment analysis pipeline (multilingual model)
                sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

                # Analyze sentiment
                result = sentiment_analyzer(user_input)

                # Display result
                sentiment = result[0]['label']
                score = result[0]['score']
                st.success(f"Sentiment: {sentiment}")
                st.info(f"Confidence Score: {score:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a sentence to analyze.")

if __name__ == "__main__":
    main()
