from agentic_rag.pipeline.types import HybridResult, RetrieverMode
from agentic_rag.retriever.core.types import RetrievalResult, TextualResult, VisualResult
from agentic_rag.retriever.textual.retriever import TextualRetriever
from agentic_rag.retriever.visual.retriever import VisualRetriever
from agentic_rag.scorer.scorer import Scorer
from agentic_rag.vectordb.handler import QdrantHandler


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
        rrf_k: int = 60,
        textual_weight: float = 1.0,
        visual_weight: float = 1.0,
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
        rrf_k : int, optional
            Constant used in RRF
        textual_weight : float, optional
            Weight assigned to textual retrieval scores for RRF
        visual_weight : float, optional
            _Weight assigned to visual retrieval scores for RRF
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

        return sorted(fused.values(), key=lambda x: x.rrf_score, reverse=True)

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
