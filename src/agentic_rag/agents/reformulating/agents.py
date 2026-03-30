from agentic_rag.agents.core.base import BaseAgent
from agentic_rag.agents.core.config import REFORMULATING_AGENT_PROMPT
from agentic_rag.agents.core.types import AgentState, ReformulatingOutput


class ReformulationAgent(BaseAgent):
    prompt = REFORMULATING_AGENT_PROMPT

    @property
    def name(self) -> str:
        """
        Return the unique name of the agent

        Returns
        -------
        str
            The name
        """
        return "reformulation"

    def invoke(self, state: AgentState) -> dict:
        """
        Execute the agent logic on the given state

        Parameters
        ----------
        state : AgentState
            The current state of the agent pipeline

        Returns
        -------
        dict
            A dictionary containing state updates produced by the agent
        """
        query = state["query"]

        return self._chat(
            system_prompt=self.prompt,
            user_prompt=f"User query: {query}",
            output_format=ReformulatingOutput,
            temperature=0.0,
        )
