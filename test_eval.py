def _try_build_embedding_client() -> OpenAICompatibleEmbeddingClient | None:
    if not ENABLE_EMBEDDINGS:
        logger.info("Embeddings disabled by ENABLE_EMBEDDINGS=false")
        return None
    try:
        return OpenAICompatibleEmbeddingClient(
            model=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
        )
    except Exception as exc:
        logger.warning("Embedding client unavailable (%s) - continuing without embeddings", exc)
        return None



class _EmbeddingData:
    __slots__ = ("index", "embedding")

    def __init__(self, index: int, embedding: list[float]):
        self.index = index
        self.embedding = embedding


class _EmbeddingResponse:
    __slots__ = ("data",)

    def __init__(self, data: list[_EmbeddingData]):
        self.data = data


class _EmbeddingsAPI:
    __slots__ = ("_client",)

    def __init__(self, client: EmbeddingClient):
        self._client = client

    def create(
        self,
        *,
        input: str | list[str],
        model: str | None = None,
        **kwargs,
    ) -> _EmbeddingResponse:
        texts = [input] if isinstance(input, str) else list(input)

        # Do not forward model. Underlying client already uses self.model.
        vectors = self._client.embed_many(texts)

        return _EmbeddingResponse(
            data=[_EmbeddingData(index=i, embedding=v) for i, v in enumerate(vectors)]
        )


class OpenAICompatibleEmbeddingClient:
    __slots__ = ("embeddings",)

    def __init__(self, model: str = EMBEDDING_MODEL, dimension: int = EMBEDDING_DIMENSION):
        client = EmbeddingClient(model=model, dimension=dimension)
        self.embeddings = _EmbeddingsAPI(client)
