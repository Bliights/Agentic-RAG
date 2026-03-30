import logging
import time

import streamlit as st

from agentic_rag.pipeline.pipeline import HybridRAGPipeline
from scripts.app.config import (
    DB_HOST,
    DB_PORT,
    LLM_MODEL_NAME,
    SCORER_PATH,
)
from scripts.retriever.textual.config import DATABASE_NAME as TEXTUAL_DATABASE_NAME
from scripts.retriever.textual.config import EMBEDDER_MODEL_NAME as TEXTUAL_EMBEDDER
from scripts.retriever.visual.config import DATABASE_NAME as VISUAL_DATABASE_NAME
from scripts.retriever.visual.config import EMBEDDER_MODEL_NAME as VISUAL_EMBEDDER
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


st.set_page_config(page_title="Agentic-RAG", page_icon="💬")
st.title("Agentic-RAG")


@st.cache_resource
def get_pipeline() -> HybridRAGPipeline:
    return HybridRAGPipeline(
        DB_HOST,
        DB_PORT,
        TEXTUAL_DATABASE_NAME,
        VISUAL_DATABASE_NAME,
        TEXTUAL_EMBEDDER,
        VISUAL_EMBEDDER,
        scorer_model_path=SCORER_PATH,
        llm_model_name=LLM_MODEL_NAME,
    )


pipeline = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Écris ton message ici...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            start = time.perf_counter()
            logger.info("Answer generation...")
            result = pipeline.answer(user_input)
            elapsed = time.perf_counter() - start
            logger.info(f"answer : {result.answer}")
            logger.info("docs:")
            for doc in result.docs:
                logger.info(
                    f"\tDoc {doc.doc_id} at page {doc.page_id} : {doc.score:.4f}",
                )

        st.write(result.answer)
        st.caption(f"⏱️ Response time: {elapsed:.3f}s")

        if result.docs:
            with st.expander("Retrieved documents"):
                for i, doc in enumerate(result.docs, start=1):
                    st.markdown(
                        f"**Doc {i}** — doc_id: `{doc.doc_id}` | "
                        f"page_id: `{doc.page_id}` | "
                        f"score: `{doc.score:.4f}`",
                    )

                    if hasattr(doc, "content") and doc.content:
                        st.write(doc.content)

                    if hasattr(doc, "image_path") and doc.image_path:
                        st.caption(f"Image path: {doc.image_path}")

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer},
        )
        st.session_state.last_docs = result.docs
