from agentic_rag.agents.core.base import BaseAgent
from agentic_rag.agents.core.config import (
    ANSWER_GENERATION_PROMPT,
)
from agentic_rag.agents.core.types import (
    AgentState,
    AnswerGenerationOutput,
)
from agentic_rag.utils.utils import collect_context


class AnswerGenerationAgent(BaseAgent):
    prompt = ANSWER_GENERATION_PROMPT

    @property
    def name(self) -> str:
        """
        Return the unique name of the agent

        Returns
        -------
        str
            The name
        """
        return "answer_generation"

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
        visual_context, textual_context = collect_context(state, self.nb_docs)

        return self._chat(
            system_prompt=self.prompt,
            user_prompt=f"User query: {query}\n\nRetrieved textual context:\n{textual_context}",
            images=visual_context,
            output_format=AnswerGenerationOutput,
            temperature=0.0,
        )
