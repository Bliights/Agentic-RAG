import tiktoken


class Chunker:
    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 50,
        tokeniser_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokeniser_name = tokeniser_name
        self.tokeniser = tiktoken.get_encoding(tokeniser_name)

    def chunk_text(self, text: str) -> list[str]:
        tokens = self.tokeniser.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokeniser.decode(chunk_tokens)

            chunks.append(chunk_text)
            start += self.chunk_size - self.overlap

        return chunks
