from agents.prediction_agent.predictor import (
    PredictionAgent
)

from agents.sentiment_agent.sentiment import (
    SentimentAgent
)

from agents.risk_agent.risk import (
    RiskAgent
)

from agents.technical_agent.technical import (
    TechnicalAgent
)

from agents.regime_agent.regime import (
    RegimeAgent
)

from agents.multitimeframe_agent.multitimeframe import (
    MultiTimeframeAgent
)

from agents.probability_agent.probability import (
    ProbabilityAgent
)

# --------------------------------
# LOAD AGENTS
# --------------------------------
prediction_agent = PredictionAgent()

sentiment_agent = SentimentAgent()

risk_agent = RiskAgent()

technical_agent = TechnicalAgent()

regime_agent = RegimeAgent()

multitimeframe_agent = (
    MultiTimeframeAgent()
)

probability_agent = ProbabilityAgent()

# --------------------------------
# TICKER
# --------------------------------
ticker = "AAPL"

# --------------------------------
# RUN AGENTS
# --------------------------------
prediction = prediction_agent.predict(
    ticker
)

sentiment = (
    sentiment_agent.analyze_sentiment(
        ticker
    )
)

risk = risk_agent.calculate_risk(
    ticker
)

technical = (
    technical_agent.analyze_technical(
        ticker
    )
)

regime = regime_agent.detect_regime(
    ticker
)

timeframe = (

    multitimeframe_agent
    .analyze_timeframes(ticker)

)

# --------------------------------
# PROBABILITY
# --------------------------------
probability = (

    probability_agent
    .calculate_probability(

        prediction,

        sentiment,

        technical,

        risk,

        regime,

        timeframe

    )

)

# --------------------------------
# RESULT
# --------------------------------
print("\n")

print("MARKET DIRECTION:")
print(
    probability["market_direction"]
)

print("\n")

print("BULLISH PROBABILITY:")
print(
    probability["bullish_probability"]
)

print("\n")

print("BEARISH PROBABILITY:")
print(
    probability["bearish_probability"]
)

print("\n")

print("CONFIDENCE:")
print(
    probability["confidence"]
)

print("\n")