from agents.portfolio_agent.portfolio import (
    PortfolioAgent
)

agent = PortfolioAgent()

portfolio = [

    "AAPL",

    "NVDA",

    "TSLA",

    "RELIANCE.NS"

]

result = agent.analyze_portfolio(
    portfolio
)

print(result)