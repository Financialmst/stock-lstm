from agents.sentiment_agent.sentiment import SentimentAgent


agent = SentimentAgent()

result = agent.analyze_sentiment("AAPL")

print(result)