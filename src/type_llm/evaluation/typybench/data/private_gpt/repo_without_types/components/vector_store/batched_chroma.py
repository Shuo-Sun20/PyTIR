from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.chroma import ChromaVectorStore  # type: ignore

if TYPE_CHECKING:
    from collections.abc import Mapping


def chunk_list(
    lst, max_chunk_size):
    """Yield successive max_chunk_size-sized chunks from lst.

    Args:
        lst (List[BaseNode]): list of nodes with embeddings
        max_chunk_size (int): max chunk size

    Yields:
        Generator[List[BaseNode], None, None]: list of nodes with embeddings
    """
    for i in range(0, len(lst), max_chunk_size):
        yield lst[i : i + max_chunk_size]


class BatchedChromaVectorStore(ChromaVectorStore):  # type: ignore
    """Chroma vector store, batching additions to avoid reaching the max batch limit.

    In this vector store, embeddings are stored within a ChromaDB collection.

    During query time, the index uses ChromaDB to query for the top
    k most similar nodes.

    Args:
        chroma_client (from chromadb.api.API):
            API instance
        chroma_collection (chromadb.api.models.Collection.Collection):
            ChromaDB collection instance

    """

    chroma_client = None

    def __init__(
        self, chroma_client, chroma_collection, host = None, port = None, ssl = False, headers = None, collection_kwargs = None):
        super().__init__(
            chroma_collection=chroma_collection,
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            collection_kwargs=collection_kwargs or {},
        )
        self.chroma_client = chroma_client

    def add(self, nodes, **add_kwargs: Any):
        """Add nodes to index, batching the insertion to avoid issues.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
            add_kwargs: _
        """
        if not self.chroma_client:
            raise ValueError("Client not initialized")

        if not self._collection:
            raise ValueError("Collection not initialized")

        max_chunk_size = self.chroma_client.max_batch_size
        node_chunks = chunk_list(nodes, max_chunk_size)

        all_ids = []
        for node_chunk in node_chunks:
            embeddings: list[Sequence[float]] = []
            metadatas: list[Mapping[str, Any]] = []
            ids = []
            documents = []
            for node in node_chunk:
                embeddings.append(node.get_embedding())
                metadatas.append(
                    node_to_metadata_dict(
                        node, remove_text=True, flat_metadata=self.flat_metadata
                    )
                )
                ids.append(node.node_id)
                documents.append(node.get_content(metadata_mode=MetadataMode.NONE))

            self._collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            all_ids.extend(ids)

        return all_ids
