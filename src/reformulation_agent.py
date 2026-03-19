#from langchain_ollama import ChatOllama
import ollama
from ollama import chat
import workspace as ws
#from langchain_core.messages import HumanMessage, AIMessage

class ReformulationAgent:
    def __init__(self):
        """
        Initializes the ReformulationAgent with the specified model.
        Args:
            model_name (str): The name of the Ollama model to use for reformulation.
        """
        self.model_name = ws.model_name
        # self.message = """Your role is to break down queries into subqueries to be more specific. The goal of this reformulation is to use information retrieval system. 
        # Your goal is not to answer the query or help the user finding the response but only to reformulate.
        # Each subqueries must be separated by '|'. 
        # Only respond with the subqueries without any additional explanation or formatting.
        # For example, if the query is : What should I do to get a job in the field of data science?
        # You should respond with 'What are the required skills for a job in data science? | What are the best resources to learn data science? | How can I gain practical experience in data science?'."""
        # For example, if the input query is 'What is the capital of France and who is the president?', you should respond with 'What is the capital of France? | Who is the president of France?'"""

        self.message = """You are an expert Query Reformulation Agent for a RAG (Retrieval-Augmented Generation) system.
        Your ONLY task is to break down complex user queries into simpler, specific, and independent sub-queries optimized for a search engine.

        ### STRICT RULES:
        1. DO NOT answer the user's original query.
        2. DO NOT provide any conversational filler, greetings, or explanations (e.g., never say "Here are the subqueries:").
        3. Separate each sub-query using ONLY the pipe character '|'.
        4. Output NOTHING ELSE but the reformulated sub-queries.

        ### EXAMPLE:
        User: What should I do to get a job in the field of data science?
        Output: What are the required skills for a data science job? | What are the best resources to learn data science? | How can I gain practical experience in data science?

        ### TASK:
        Reformulate and organize by order of importance the following query strictly following the rules above."""

    def invoke(self, query: str) -> list[str]:
        """
        Reformulates the input query to be more specific and clear.
        Args:
            query (str): The original query to reformulate.
        Returns:
            list[str]: A list of reformulated query parts.
        """

        prompt = f"The query is: '{query}'"
        
        response = chat(model=self.model_name, 
                        messages=[{"role": "system", "content": self.message}, {"role": "user", "content": prompt}], 
                        options={
                "temperature": 0.0  # Paramétrage de la température ici !
            })

        return response["message"]["content"].split('|')

if __name__ == "__main__":
    import time
    start = time.time()
    agent = ReformulationAgent()
    #query = "What is the capital of France and who is the president?"
    query = "What should I do to prepare a trip to Paris?"
    reformulated_parts = agent.invoke(query)
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
    print(reformulated_parts)

