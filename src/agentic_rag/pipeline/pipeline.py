import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import END, START, StateGraph

from agentic_rag.agents.core.types import AgentState
from agentic_rag.agents.generation.agents import AnswerGenerationAgent
from agentic_rag.agents.reformulating.agents import ReformulationAgent
from agentic_rag.agents.verification.agents import LogicCheckAgent, RetrievalVerificationAgent
from agentic_rag.pipeline.types import HybridResult, PipelineAnswer, RetrieverMode
from agentic_rag.retriever.core.types import RetrievalResult, TextualResult, VisualResult
from agentic_rag.retriever.textual.retriever import TextualRetriever
from agentic_rag.retriever.visual.retriever import VisualRetriever
from agentic_rag.scorer.scorer import Scorer
from agentic_rag.vectordb.handler import QdrantHandler

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        textual_collection_name: str,
        visual_collection_name: str,
        textual_embedder_name: str,
        visual_embedder_name: str,
        scorer_model_path: str,
        llm_model_name: str,
        rrf_k: int = 60,
        textual_weight: float = 1.0,
        visual_weight: float = 1.0,
        max_iterations: int = 1,
        nb_docs: int = 5,
    ) -> None:
        """
        Initialize a pipeline using an hybrid RAG

        Parameters
        ----------
        qdrant_host : str
            Hostname or IP address of the Qdrant server
        qdrant_port : int
            Port used to connect to the Qdrant server
        textual_collection_name : str
            Name of the Qdrant collection storing textual embeddings
        visual_collection_name : str
            Name of the Qdrant collection storing visual embeddings
        textual_embedder_name : str
            Name of the embedding model used for textual queries
        visual_embedder_name : str
            Name of the embedding model used for visual queries
        scorer_model_path : str
            Path to the scorer model
        llm_model_name : str
            LLM used in agent name
        rrf_k : int, optional
            Constant used in RRF
        textual_weight : float, optional
            Weight assigned to textual retrieval scores for RRF
        visual_weight : float, optional
            Weight assigned to visual retrieval scores for RRF
        max_iteration : float, optional
            Max number of retry for the agent loop
        nb_docs : int, optional
            Number of docs to give to the agents
        """
        self.db = QdrantHandler(host=qdrant_host, port=qdrant_port)

        self.textual_retriever = TextualRetriever(
            collection_name=textual_collection_name,
            db_handler=self.db,
            embedder_name=textual_embedder_name,
        )

        self.visual_retriever = VisualRetriever(
            collection_name=visual_collection_name,
            db_handler=self.db,
            embedder_name=visual_embedder_name,
        )

        self.scorer = Scorer(scorer_model_path)
        self.rrf_k = rrf_k
        self.textual_weight = textual_weight
        self.visual_weight = visual_weight
        self.max_iterations = max_iterations
        self.nb_docs = nb_docs

        self.reformulation_agent = ReformulationAgent(llm_model_name, self.nb_docs)
        self.rag_verif_agent = RetrievalVerificationAgent(llm_model_name, self.nb_docs)
        self.generation_agent = AnswerGenerationAgent(llm_model_name, self.nb_docs)
        self.logic_check_agent = LogicCheckAgent(llm_model_name, self.nb_docs)

        self.workflow = self._build_agent_workflow()
        self.answer("Warmup query !")
        logger.info("Pipeline ready !")

    def _result_key(self, result: RetrievalResult) -> tuple[int, str, int]:
        """
        Build a unique key identifying a retrieved page within the corpus

        Parameters
        ----------
        result : RetrievalResult
            A retrieval result containing corpus, document, and page identifiers

        Returns
        -------
        tuple[int, str, int]
            (corpus_id, doc_id, page_id)
        """
        return (result.corpus_id, result.doc_id, result.page_id)

    def _rrf_term(self, rank: int, weight: float) -> float:
        """
        Compute the Reciprocal Rank Fusion contribution for one ranked result

        Parameters
        ----------
        rank : int
            Rank position of the result in the retriever output
        weight : float
            Weight applied to the retriever contribution

        Returns
        -------
        float
            The weighted RRF contribution for the given rank
        """
        return weight / (self.rrf_k + rank)

    def _fuse_with_rrf(
        self,
        textual_results: list[TextualResult],
        visual_results: list[VisualResult],
    ) -> list[HybridResult]:
        """
        Fuse textual and visual retrieval results using RRF

        Parameters
        ----------
        textual_results : list[TextualResult]
            Ranked list of results returned by the textual retriever
        visual_results : list[VisualResult]
            Ranked list of results returned by the visual retriever

        Returns
        -------
        list[HybridResult]
            A list of fused hybrid results sorted by descending RRF score
        """
        fused = {}
        for rank, result in enumerate(textual_results, start=1):
            key = self._result_key(result)
            if key not in fused:
                fused[key] = HybridResult(
                    corpus_id=result.corpus_id,
                    doc_id=result.doc_id,
                    page_id=result.page_id,
                    score=self._rrf_term(rank, self.textual_weight),
                    chunk_id=result.chunk_id,
                    content=result.content,
                    image_path=None,
                    textual_rank=rank,
                    visual_rank=None,
                )

        for rank, result in enumerate(visual_results, start=1):
            key = self._result_key(result)
            if key not in fused:
                fused[key] = HybridResult(
                    corpus_id=result.corpus_id,
                    doc_id=result.doc_id,
                    page_id=result.page_id,
                    score=0.0,
                    chunk_id=None,
                    content=None,
                    image_path=None,
                    textual_rank=None,
                    visual_rank=None,
                )

            fused[key].score += self._rrf_term(rank, self.visual_weight)
            fused[key].image_path = result.image_path
            fused[key].visual_rank = rank

        return sorted(fused.values(), key=lambda x: x.score, reverse=True)

    def _fuse_with_alpha(
        self,
        query: str,
        textual_results: list[TextualResult],
        visual_results: list[VisualResult],
    ) -> list[HybridResult]:
        """
        Fuse textual and visual retrieval results using an alpha-based strategy

        Parameters
        ----------
        query : str
            User query used to compute the alpha
        textual_results : list[TextualResult]
            Ranked list of results returned by the textual retriever
        visual_results : list[VisualResult]
            Ranked list of results returned by the visual retriever

        Returns
        -------
        list[HybridResult]
            A list of fused hybrid results sorted by descending score
        """
        alpha = self.scorer.compute_alpha(query)

        textual_map = {}
        textual_rank_map = {}
        for rank, r in enumerate(textual_results, start=1):
            key = self._result_key(r)
            if key not in textual_map:
                textual_map[key] = r
                textual_rank_map[key] = rank

        visual_map = {}
        visual_rank_map = {}
        for rank, r in enumerate(visual_results, start=1):
            key = self._result_key(r)
            if key not in visual_map:
                visual_map[key] = r
                visual_rank_map[key] = rank

        all_keys = set(textual_map.keys()) | set(visual_map.keys())
        fused = []
        for key in all_keys:
            t = textual_map.get(key)
            v = visual_map.get(key)

            t_score = t.score if t else 0.0
            v_score = v.score if v else 0.0

            fused_score = self.scorer.fuse_scores(alpha, t_score, v_score)

            fused.append(
                HybridResult(
                    corpus_id=key[0],
                    doc_id=key[1],
                    page_id=key[2],
                    score=fused_score,
                    chunk_id=t.chunk_id if t else None,
                    content=t.content if t else None,
                    image_path=v.image_path if v else None,
                    textual_rank=textual_rank_map.get(key),
                    visual_rank=visual_rank_map.get(key),
                ),
            )

        return sorted(fused, key=lambda x: x.score, reverse=True)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        per_retriever_k: int | None = None,
        mode: RetrieverMode = RetrieverMode.ALPHA,
    ) -> list[HybridResult]:
        """
        Retrieve and fuse textual and visual results for a given query

        Parameters
        ----------
        query : str
            Query string used for retrieval
        k : int, optional
            Number of final fused results to return
        per_retriever_k : int | None, optional
            Number of candidates to fetch from each retriever before fusion
        mode : RetrieverMode, optional
            Fusion strategy used to combine textual and visual results

        Returns
        -------
        list[HybridResult]
            The top-k fused hybrid retrieval results
        """
        candidate_k = per_retriever_k if per_retriever_k is not None else k * 3

        textual_results = self.textual_retriever.search(query=query, k=candidate_k)
        visual_results = self.visual_retriever.search(query=query, k=candidate_k)

        if mode == RetrieverMode.RRF:
            fused_results = self._fuse_with_rrf(
                textual_results=textual_results,
                visual_results=visual_results,
            )
        if mode == RetrieverMode.ALPHA:
            fused_results = self._fuse_with_alpha(
                query=query,
                textual_results=textual_results,
                visual_results=visual_results,
            )

        return fused_results[:k]

    def _build_agent_workflow(self) -> StateGraph:
        """
        Build and compile the agent workflow as a state graph

        Returns
        -------
        StateGraph
            A compiled state graph representing the full agent workflow
        """
        graph = StateGraph(AgentState)

        graph.add_node(self.reformulation_agent.name, self.reformulation_agent.invoke)
        graph.add_node("retrieval", self.retrieval_node)
        graph.add_node(
            self.rag_verif_agent.name,
            self.rag_verif_agent.invoke,
        )
        graph.add_node("expand_queries", self.expand_queries_node)
        graph.add_node(self.generation_agent.name, self.generation_agent.invoke)
        graph.add_node(self.logic_check_agent.name, self.logic_check_agent.invoke)

        graph.add_edge(START, self.reformulation_agent.name)
        graph.add_edge(self.reformulation_agent.name, "retrieval")
        graph.add_edge("retrieval", self.rag_verif_agent.name)

        graph.add_conditional_edges(
            self.rag_verif_agent.name,
            self.route_after_verification,
            {
                "expand_queries": "expand_queries",
                "answer_generation": self.generation_agent.name,
            },
        )

        graph.add_edge("expand_queries", "retrieval")
        graph.add_edge(self.generation_agent.name, self.logic_check_agent.name)
        graph.add_edge(self.logic_check_agent.name, END)

        return graph.compile()

    def display_graph(self) -> str:
        """
        Generate a Mermaid diagram representation of the workflow graph

        Returns
        -------
        str
            A Mermaid-formatted string describing the workflow structure
        """
        return self.workflow.get_graph().draw_mermaid()

    def retrieval_node(self, state: AgentState) -> dict:
        """
        Retrieve documents for the given query or subqueries

        Parameters
        ----------
        state : AgentState
            The current agent state containing the query, subqueries, and previously retrieved documents

        Returns
        -------
        dict
            Updated state fragment containing a list of unique retrieved documents sorted by score
        """
        subqueries = state.get("subqueries", [])
        if not subqueries:
            subqueries = [state["query"]]

        collected_docs = state.get("retrieved_docs", [])
        seen_keys = {(doc.corpus_id, doc.doc_id, doc.page_id) for doc in collected_docs}

        with ThreadPoolExecutor(max_workers=max(1, len(subqueries))) as executor:
            futures = {
                executor.submit(
                    self.retrieve,
                    subquery,
                ): subquery
                for subquery in subqueries
            }

            for future in as_completed(futures):
                results = future.result()
                for result in results:
                    key = (result.corpus_id, result.doc_id, result.page_id)
                    if key not in seen_keys:
                        collected_docs.append(result)
                        seen_keys.add(key)

        collected_docs.sort(key=lambda x: x.score, reverse=True)

        return {
            "retrieved_docs": collected_docs,
        }

    def expand_queries_node(self, state: AgentState) -> dict:
        """
        Update the state with newly generated subqueries for iterative retrieval

        Parameters
        ----------
        state : AgentState
            The current agent state

        Returns
        -------
        dict
            Updated state
        """
        new_queries = state.get("new_queries", [])

        return {
            "subqueries": new_queries,
            "iteration": state.get("iteration", 0) + 1,
        }

    def route_after_verification(self, state: AgentState) -> str:
        """
        Determine the next step after retrieval verification

        Parameters
        ----------
        state : AgentState
            The current agent state containing verification results and iteration info

        Returns
        -------
        str
            The name of the next node in the workflow
        """
        if state.get("docs_are_sufficient", False):
            return "answer_generation"

        if state.get("iteration", 0) >= state.get("max_iterations", self.max_iterations):
            return "answer_generation"

        if state.get("new_queries"):
            return "expand_queries"

        return "answer_generation"

    def answer(
        self,
        query: str,
    ) -> PipelineAnswer:
        """
        Execute the full agent pipeline to answer a query

        Parameters
        ----------
        query : str
            The input user query

        Returns
        -------
        PipelineAnswer
            An object containing the final generated answer and the top retrieved documents used as context
        """
        initial_state = {
            "query": query,
            "subqueries": [],
            "retrieved_docs": [],
            "retrieved_context": "",
            "docs_are_sufficient": False,
            "verification_reason": "",
            "new_queries": [],
            "draft_answer": "",
            "final_answer": "",
            "logic_is_valid": False,
            "logic_feedback": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }

        start_time = time.perf_counter()
        prev_time = start_time
        final_state = initial_state

        logger.info(f"Starting execution for query: {query}")

        for event in self.workflow.stream(initial_state):
            now = time.perf_counter()
            elapsed = now - prev_time
            prev_time = now

            for node_name, node_output in event.items():
                logger.info("-" * 30 + f"[+{elapsed:.3f}s | {node_name}]" + "-" * 30)

                if node_name == "retrieval":
                    docs = node_output.get("retrieved_docs", [])
                    logger.info("Docs retrieved: ")
                    for doc in docs[: self.nb_docs]:
                        logger.info(
                            f"\tDoc {doc.doc_id} at page {doc.page_id} : {doc.score:.4f}",
                        )

                elif "subqueries" in node_output:
                    logger.info("Generated subqueries:")
                    for query in node_output.get("subqueries"):
                        logger.info(f"\t{query}")

                elif "docs_are_sufficient" in node_output:
                    logger.info(f"Are the docs sufficient ? {node_output['docs_are_sufficient']}")
                    logger.info(f"Reason: {node_output.get('verification_reason')}")

                    if not node_output["docs_are_sufficient"]:
                        logger.info("New queries: ")
                        for query in node_output.get("new_queries"):
                            logger.info(f"\t{query}")

                elif "draft_answer" in node_output:
                    logger.info(f"Draft answer generated : {node_output.get('draft_answer')}")

                elif "logic_is_valid" in node_output:
                    logger.info(f"Logic valid: {node_output.get('logic_is_valid')}")
                    logger.info(f"Feedback: {node_output.get('logic_feedback')}")

                else:
                    logger.debug(f"Raw output: {node_output}")

                if isinstance(node_output, dict):
                    final_state = {**final_state, **node_output}

        total_elapsed = time.perf_counter() - start_time
        logger.info(f"Execution finished in {total_elapsed:.3f}s")

        return PipelineAnswer(
            final_state.get("final_answer", ""),
            final_state.get("retrieved_docs", [])[: self.nb_docs],
        )
