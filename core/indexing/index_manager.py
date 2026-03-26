from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class IndexManager:
    def __init__(self, client: QdrantClient):
        self._client = client

    def list_collections(self) -> list[str]:
        response = self._client.get_collections()
        return [c.name for c in response.collections]

    def collection_exists(self, name: str) -> bool:
        return name in self.list_collections()

    def collection_info(self, name: str) -> dict:
        info = self._client.get_collection(name)
        return {
            "name": name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }

    def create_collection(
        self,
        name: str,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
    ) -> None:
        if self.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def delete_collection(self, name: str) -> bool:
        return self._client.delete_collection(name)
