from src.RAG import run_tactical_query
import asyncio

result = asyncio.run(run_tactical_query("Give tactical suggestions for Pakistan vs Myanmar."))
print(result)
