from agents.prediction_agent.predictor import PredictionAgent

from agents.sentiment_agent.sentiment import SentimentAgent

from agents.decision_agent.decision import DecisionAgent

from agents.risk_agent.risk import RiskAgent


class MasterOrchestrator:

    def __init__(self):

        print("Initializing Master Orchestrator...")

        self.prediction_agent = PredictionAgent()

        self.sentiment_agent = SentimentAgent()

        self.decision_agent = DecisionAgent()

        self.risk_agent = RiskAgent()

        print("Master Orchestrator Ready")

    def run(self, ticker):

        # --------------------------------
        # PREDICTION
        # --------------------------------
        prediction_result = (
            self.prediction_agent.predict(
                ticker
            )
        )

        # --------------------------------
        # SENTIMENT
        # --------------------------------
        sentiment_result = (
            self.sentiment_agent.analyze_sentiment(
                ticker
            )
        )

        # --------------------------------
        # DECISION
        # --------------------------------
        decision_result = (
            self.decision_agent.make_decision(
                ticker
            )
        )

        # --------------------------------
        # RISK
        # --------------------------------
        risk_result = (
            self.risk_agent.calculate_risk(
                ticker
            )
        )

        # --------------------------------
        # FINAL OUTPUT
        # --------------------------------
        final_result = {

            "ticker": ticker,

            "prediction": prediction_result,

            "sentiment": sentiment_result,

            "decision": decision_result,

            "risk": risk_result

        }

        return final_result