from agentic_rag.agents.core.base import BaseAgent
from agentic_rag.agents.core.config import (
    LOGIC_VERIF_AGENT_PROMPT,
    RAG_VERIF_AGENT_PROMPT,
)
from agentic_rag.agents.core.types import (
    AgentState,
    LogicVerifOutput,
    RAGVerifOutput,
)
from agentic_rag.utils.utils import collect_context


class RetrievalVerificationAgent(BaseAgent):
    prompt = RAG_VERIF_AGENT_PROMPT

    @property
    def name(self) -> str:
        """
        Return the unique name of the agent

        Returns
        -------
        str
            The name
        """
        return "retrieval_verification"

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
            output_format=RAGVerifOutput,
            temperature=0.0,
        )


class LogicCheckAgent(BaseAgent):
    prompt = LOGIC_VERIF_AGENT_PROMPT

    @property
    def name(self) -> str:
        """
        Return the unique name of the agent

        Returns
        -------
        str
            The name
        """
        return "logic_check"

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
        draft_answer = state["draft_answer"]

        return self._chat(
            system_prompt=self.prompt,
            user_prompt=f"User query: {query}\n\nRetrieved textual context:\n{textual_context}\n\nDraft answer:\n{draft_answer}",
            images=visual_context,
            output_format=LogicVerifOutput,
            temperature=0.0,
        )
