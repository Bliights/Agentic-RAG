import base64

from agentic_rag.agents.core.types import AgentState


def encode_image(image_path: str) -> str:
    """
    Encode an image file into a base64 string

    Parameters
    ----------
    image_path : str
        Path to the image file to be encoded

    Returns
    -------
    str
        Base64-encoded string representation of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def collect_context(state: AgentState, nb_doc: int = 5) -> tuple[list[str], list[str]]:
    """
    Base64-encoded string representation of the image

    Parameters
    ----------
    state : AgentState
        The agent state containing retrieved documents
    nb_doc : int, optional
        Number of top documents to consider

    Returns
    -------
    tuple[list[str], list[str]]
        (list of image paths, list of text contents)
    """
    visual_context = []
    textual_context = []
    top_k = state.get("retrieved_docs", [])[:nb_doc]
    for doc in top_k:
        if doc.content:
            textual_context.append(doc.content)
        if doc.image_path:
            visual_context.append(doc.image_path)
    return (visual_context, textual_context)
