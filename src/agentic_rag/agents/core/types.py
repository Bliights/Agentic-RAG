from typing import TypedDict

from pydantic import BaseModel

from agentic_rag.pipeline.types import HybridResult


class AgentState(TypedDict):
    query: str

    # Reformulation
    subqueries: list[str]

    # Retrieval
    retrieved_docs: list[HybridResult]

    # Verification of retrieved docs
    docs_are_sufficient: bool
    verification_reason: str
    new_queries: list[str]

    # Final answer
    draft_answer: str
    final_answer: str

    # Logic check
    logic_is_valid: bool
    logic_feedback: str

    # Loop control
    iteration: int
    max_iterations: int


class ReformulatingOutput(BaseModel):
    subqueries: list[str]


class RAGVerifOutput(BaseModel):
    docs_are_sufficient: bool
    verification_reason: str
    new_queries: list[str]


class AnswerGenerationOutput(BaseModel):
    draft_answer: str


class LogicVerifOutput(BaseModel):
    logic_is_valid: bool
    logic_feedback: str
    final_answer: str
