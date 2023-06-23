from __future__ import annotations
from functools import lru_cache
from typing import List, Any, TYPE_CHECKING

from pydantic import BaseModel

from langchain.embeddings.base import Embeddings


if TYPE_CHECKING:
    import mlflow.gateway


@lru_cache()
def _get_client(gateway_uri: str) -> mlflow.gateway.MlflowGatewayClient:
    import mlflow.gateway

    return mlflow.gateway.MlflowGatewayClient(gateway_uri)


def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
      yield lst[i:i + n]


class MlflowGatewayEmbeddings(Embeddings, BaseModel):
    gateway_uri: str
    route: str

    def __init__(self, **kwargs: Any):
        import mlflow

        super().__init__(**kwargs)
        mlflow.gateway.set_gateway_uri(self.gateway_uri)

    def _query(self, texts: List[str]) -> List[List[float]]:
        resp = _get_client(self.gateway_uri).query(
            self.route,
            data={"text": texts},
        )
        return resp["embeddings"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
      text_chunks = chunks(texts, 1000)
      results = []
      for chunk in text_chunks:
        result = self._query(chunk)
        results.append(result)
      return results

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]
