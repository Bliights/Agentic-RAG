from abc import ABC, abstractmethod

from ollama import chat
from pydantic import BaseModel

from agentic_rag.agents.core.types import AgentState
from agentic_rag.utils.utils import encode_image


class BaseAgent(ABC):
    def __init__(self, model_name: str, nb_docs: int) -> None:
        """
        Initialize the base agent

        Parameters
        ----------
        model_name : str
            Name of the language model used for inference
        nb_docs : int
            Number of documents to consider when building context
        """
        self.model_name = model_name
        self.nb_docs = nb_docs

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        images: list[str] | None = None,
        output_format: type[BaseModel] | None = None,
    ) -> dict:
        """
        Send a chat request to the model and return a structured response

        Parameters
        ----------
        system_prompt : str
            Instruction or context provided to guide the model's behavior
        user_prompt : str
            The user input or task description sent to the model
        temperature : float, optional
            Sampling temperature controlling randomness
        images : list[str] | None, optional
            List of image file paths to include in the request
        output_format : type[BaseModel] | None, optional
            Pydantic model defining the expected structured output

        Returns
        -------
        dict
            Parsed model response as a dictionary matching the provided output schema
        """
        user_message = {
            "role": "user",
            "content": user_prompt,
        }

        if images:
            user_message["images"] = [encode_image(path) for path in images]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "options": {"temperature": temperature},
        }

        if output_format is not None:
            payload["format"] = output_format.model_json_schema()

        response = chat(**payload)
        return output_format.model_validate_json(response.message.content).model_dump()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the agent

        Returns
        -------
        str
            The name
        """

    @abstractmethod
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
