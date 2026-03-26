"""MMR (Maximal Marginal Relevance) postprocessor for result diversity."""

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class MMRPostprocessor(BaseNodePostprocessor):
    """Remove overly similar results using MMR-style filtering.

    After reranking, this filters out nodes whose embedding is too similar
    to already-selected nodes, promoting diversity in the final results.
    """

    similarity_threshold: float = 0.95
    """Cosine similarity threshold — nodes more similar than this to any
    already-selected node are dropped."""

    @classmethod
    def class_name(cls) -> str:
        return "MMRPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return nodes

        selected: list[NodeWithScore] = [nodes[0]]
        selected_embeddings: list[list[float]] = []

        emb = nodes[0].node.embedding
        if emb is not None:
            selected_embeddings.append(emb)
        else:
            # No embeddings available — skip MMR filtering
            return nodes

        for candidate in nodes[1:]:
            cand_emb = candidate.node.embedding
            if cand_emb is None:
                selected.append(candidate)
                continue

            cand_vec = np.array(cand_emb)
            is_diverse = True
            for sel_emb in selected_embeddings:
                sel_vec = np.array(sel_emb)
                cos_sim = float(
                    np.dot(cand_vec, sel_vec)
                    / (np.linalg.norm(cand_vec) * np.linalg.norm(sel_vec) + 1e-10)
                )
                if cos_sim > self.similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)
                selected_embeddings.append(cand_emb)

        return selected
