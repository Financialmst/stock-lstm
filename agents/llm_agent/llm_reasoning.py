class LLMReasoningAgent:

    def __init__(self):

        print("Loading LLM Reasoning Agent...")

        print("LLM Reasoning Agent Ready")


    def generate_reasoning(

        self,

        sentiment,

        technical,

        regime,

        probability

    ):

        reasoning = []


        # --------------------------------
        # SENTIMENT
        # --------------------------------
        if sentiment["sentiment"] == "positive":

            reasoning.append(

                "Market sentiment is positive."

            )

        else:

            reasoning.append(

                "Market sentiment is negative."

            )


        # --------------------------------
        # TECHNICAL
        # --------------------------------
        if (

            technical["technical_trend"]

            ==

            "BULLISH"

        ):

            reasoning.append(

                "Technical trend remains bullish."

            )

        else:

            reasoning.append(

                "Technical trend remains bearish."

            )


        # --------------------------------
        # RSI
        # --------------------------------
        if technical["rsi_signal"] == "OVERBOUGHT":

            reasoning.append(

                "RSI indicates overbought conditions."

            )

        elif technical["rsi_signal"] == "OVERSOLD":

            reasoning.append(

                "RSI indicates oversold conditions."

            )


        # --------------------------------
        # MARKET REGIME
        # --------------------------------
        reasoning.append(

            f"Current market regime is "
            f"{regime['market_regime']}."

        )


        # --------------------------------
        # PROBABILITY
        # --------------------------------
        reasoning.append(

            f"Bullish probability is "
            f"{probability['bullish_probability']}%."

        )


        # --------------------------------
        # FINAL RESPONSE
        # --------------------------------
        return " ".join(reasoning)