class DecisionAgent:

    def __init__(self):

        print("Loading Decision Agent...")

        print("Decision Agent Ready")


    def make_decision(

        self,

        prediction,

        sentiment

    ):

        current_price = (
            prediction["current_price"]
        )

        predicted_price = (
            prediction["predicted_price"]
        )

        sentiment_score = (
            sentiment["score"]
        )

        # --------------------------------
        # PERCENT CHANGE
        # --------------------------------
        percent_change = (

            (
                predicted_price
                - current_price
            )

            / current_price

        ) * 100


        # --------------------------------
        # DECISION LOGIC
        # --------------------------------
        if (

            predicted_price > current_price

            and

            sentiment_score > 0

        ):

            decision = "BUY"


        elif (

            predicted_price < current_price

            and

            sentiment_score < 0

        ):

            decision = "SELL"


        else:

            decision = "HOLD"


        # --------------------------------
        # CONFIDENCE
        # --------------------------------
        confidence = abs(
            percent_change
        ) * 10

        if confidence > 100:

            confidence = 100


        # --------------------------------
        # AI EXPLANATION
        # --------------------------------
        if decision == "BUY":

            explanation = (

                "AI predicts upward movement "
                "with positive market sentiment."

            )


        elif decision == "SELL":

            explanation = (

                "AI predicts downside risk "
                "with negative market sentiment."

            )


        else:

            explanation = (

                "Market signals are mixed. "
                "Holding may be safer."

            )


        # --------------------------------
        # RETURN
        # --------------------------------
        return {

            "decision": decision,

            "percent_change": round(
                float(percent_change),
                2
            ),

            "confidence": round(
                float(confidence),
                2
            ),

            "explanation": explanation

        }