from orchestrator.master_orchestrator import MasterOrchestrator

agent = MasterOrchestrator()

result = agent.run("AAPL")

print(result)