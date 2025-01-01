# Install Hugging Face transformers if you haven't already
# pip install transformers

from transformers import pipeline

# Initialize the sentiment-analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Analyze sentiment of a sentence
sentence = "I love using AI, it's amazing!"
result = sentiment_analyzer(sentence)

# Print result
print(result)
