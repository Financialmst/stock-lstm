class ProbabilityAgent:

    def __init__(self):

        print("Loading Probability Agent...")

        print("Probability Agent Ready")


    def calculate_probability(

        self,

        prediction,

        sentiment,

        technical,

        risk,

        regime,

        timeframe

    ):

        bullish_score = 0

        bearish_score = 0


        # --------------------------------
        # PREDICTION
        # --------------------------------
        if (

            prediction["predicted_price"]

            >

            prediction["current_price"]

        ):

            bullish_score += 20

        else:

            bearish_score += 20


        # --------------------------------
        # SENTIMENT
        # --------------------------------
        if sentiment["sentiment"] == "positive":

            bullish_score += 15

        else:

            bearish_score += 15


        # --------------------------------
        # TECHNICAL TREND
        # --------------------------------
        if (

            technical["technical_trend"]

            ==

            "BULLISH"

        ):

            bullish_score += 20

        else:

            bearish_score += 20


        # --------------------------------
        # RSI
        # --------------------------------
        if technical["rsi_signal"] == "OVERSOLD":

            bullish_score += 10

        elif technical["rsi_signal"] == "OVERBOUGHT":

            bearish_score += 10


        # --------------------------------
        # MACD
        # --------------------------------
        if technical["macd_signal"] == "BULLISH":

            bullish_score += 10

        else:

            bearish_score += 10


        # --------------------------------
        # MARKET REGIME
        # --------------------------------
        if regime["market_regime"] == "TRENDING_BULL":

            bullish_score += 15

        elif regime["market_regime"] == "TRENDING_BEAR":

            bearish_score += 15


        # --------------------------------
        # MULTI-TIMEFRAME
        # --------------------------------
        bullish_tf = 0

        bearish_tf = 0

        for tf in timeframe["timeframes"].values():

            if tf["trend"] == "BULLISH":

                bullish_tf += 1

            elif tf["trend"] == "BEARISH":

                bearish_tf += 1

        bullish_score += bullish_tf * 5

        bearish_score += bearish_tf * 5


        # --------------------------------
        # RISK PENALTY
        # --------------------------------
        if risk["risk_level"] == "HIGH":

            bullish_score -= 10

            bearish_score -= 10


        # --------------------------------
        # FINAL PROBABILITIES
        # --------------------------------
        total = bullish_score + bearish_score

        bullish_probability = (

            bullish_score / total

        ) * 100

        bearish_probability = (

            bearish_score / total

        ) * 100


        # --------------------------------
        # FINAL DIRECTION
        # --------------------------------
        if bullish_probability > bearish_probability:

            direction = "BULLISH"

        else:

            direction = "BEARISH"


        # --------------------------------
        # CONFIDENCE
        # --------------------------------
        confidence = max(

            bullish_probability,

            bearish_probability

        )


        # --------------------------------
        # RETURN
        # --------------------------------
        return {

            "bullish_probability":
            round(float(bullish_probability), 2),

            "bearish_probability":
            round(float(bearish_probability), 2),

            "market_direction":
            direction,

            "confidence":
            round(float(confidence), 2)

        }