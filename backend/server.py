from flask import Flask
from flask import jsonify

from flask_cors import CORS

# --------------------------------
# AGENTS
# --------------------------------
from agents.prediction_agent.predictor import (
    PredictionAgent
)

from agents.sentiment_agent.sentiment import (
    SentimentAgent
)

from agents.decision_agent.decision import (
    DecisionAgent
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
from agents.llm_agent.llm_reasoning import (
    LLMReasoningAgent
)

# --------------------------------
# APP
# --------------------------------
app = Flask(__name__)

CORS(app)

# --------------------------------
# LOAD AGENTS
# --------------------------------
prediction_agent = PredictionAgent()

sentiment_agent = SentimentAgent()

decision_agent = DecisionAgent()

risk_agent = RiskAgent()

technical_agent = TechnicalAgent()

regime_agent = RegimeAgent()

multitimeframe_agent = (
    MultiTimeframeAgent()
)

probability_agent = ProbabilityAgent()

llm_agent = LLMReasoningAgent()

# --------------------------------
# ROOT ROUTE
# --------------------------------
@app.route("/")
def home():

    return {
        "message":
        "Global AI Market Intelligence Running"
    }

# --------------------------------
# ANALYZE STOCK
# --------------------------------
@app.route("/analyze/<ticker>")
def analyze_stock(ticker):

    ticker = ticker.upper()

    # --------------------------------
    # PREDICTION
    # --------------------------------
    prediction = prediction_agent.predict(
        ticker
    )

    # --------------------------------
    # SENTIMENT
    # --------------------------------
    sentiment = (
        sentiment_agent.analyze_sentiment(
            ticker
        )
    )

    # --------------------------------
    # DECISION
    # --------------------------------
    decision = (
        decision_agent.make_decision(
            prediction,
            sentiment
        )
    )

    # --------------------------------
    # RISK
    # --------------------------------
    risk = risk_agent.calculate_risk(
        ticker
    )

    # --------------------------------
    # TECHNICALS
    # --------------------------------
    technical = (
        technical_agent.analyze_technical(
            ticker
        )
    )

    # --------------------------------
    # MARKET REGIME
    # --------------------------------
    regime = regime_agent.detect_regime(
        ticker
    )

    # --------------------------------
    # MULTI-TIMEFRAME
    # --------------------------------
    timeframe = (

        multitimeframe_agent
        .analyze_timeframes(ticker)

    )

    # --------------------------------
    # PROBABILITY ENGINE
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
    llm_reasoning = (

    llm_agent.generate_reasoning(

        sentiment,

        technical,

        regime,

        probability

    )

)

    # --------------------------------
    # FINAL RESPONSE
    # --------------------------------
    result = {

        # -----------------------------
        # BASIC
        # -----------------------------
        "ticker":
        ticker,

        "current_price":
        prediction["current_price"],

        "predicted_price":
        prediction["predicted_price"],

        "trend":
        prediction["trend"],

        # -----------------------------
        # SENTIMENT
        # -----------------------------
        "sentiment":
        sentiment["sentiment"],

        "sentiment_score":
        sentiment["score"],

        # -----------------------------
        # DECISION
        # -----------------------------
        "decision":
        decision["decision"],

        "percent_change":
        decision["percent_change"],

        "confidence":
        decision["confidence"],

        "explanation":
        decision["explanation"],

        # -----------------------------
        # RISK
        # -----------------------------
        "risk_level":
        risk["risk_level"],

        "volatility_percent":
        risk["volatility_percent"],

        # -----------------------------
        # TECHNICALS
        # -----------------------------
        "rsi":
        technical["rsi"],

        "rsi_signal":
        technical["rsi_signal"],

        "macd":
        technical["macd"],

        "macd_signal":
        technical["macd_signal"],

        "sma_20":
        technical["sma_20"],

        "sma_50":
        technical["sma_50"],

        "technical_trend":
        technical["technical_trend"],

        # -----------------------------
        # MARKET REGIME
        # -----------------------------
        "market_regime":
        regime["market_regime"],

        "regime_volatility":
        regime["volatility"],

        # -----------------------------
        # MULTI-TIMEFRAME
        # -----------------------------
        "timeframes":
        timeframe["timeframes"],

        # -----------------------------
        # PROBABILITY ENGINE
        # -----------------------------
        "market_direction":
        probability["market_direction"],

        "bullish_probability":
        probability["bullish_probability"],

        "bearish_probability":
        probability["bearish_probability"],

        "overall_confidence":
        probability["confidence"],

        "llm_reasoning":
         llm_reasoning,

    }

    return jsonify(result)

# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == "__main__":

    app.run(

        debug=True,

        port=5000

    )