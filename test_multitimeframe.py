from agents.multitimeframe_agent.multitimeframe import (
    MultiTimeframeAgent
)

agent = MultiTimeframeAgent()

result = agent.analyze_timeframes(
    "AAPL"
)

print(result)