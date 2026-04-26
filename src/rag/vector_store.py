import chromadb
from chromadb import Collection

from src.core.config import get_settings
from src.core.llm import embed_batch, embed_text
from src.core.models import Chunk

_settings = get_settings()


def _get_collection() -> Collection:
    client = chromadb.PersistentClient(path=_settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=_settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


class VectorStore:
    def __init__(self) -> None:
        self._col = _get_collection()

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = await embed_batch(texts)
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "meeting_id": c.meeting_id,
                    "speaker": c.speaker or "",
                    "start_time": c.start_time or 0.0,
                    "end_time": c.end_time or 0.0,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    async def search(
        self,
        query: str,
        meeting_id: str,
        k: int | None = None,
    ) -> list[Chunk]:
        top_k = k or _settings.top_k_retrieval
        q_emb = await embed_text(query, task_type="retrieval_query")
        results = self._col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"meeting_id": meeting_id},
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[Chunk] = []
        docs = results["documents"][0]  # type: ignore[index]
        metas = results["metadatas"][0]  # type: ignore[index]
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            chunks.append(
                Chunk(
                    id=results["ids"][0][i],
                    text=doc,
                    speaker=meta.get("speaker") or None,
                    start_time=meta.get("start_time"),
                    end_time=meta.get("end_time"),
                    meeting_id=meta["meeting_id"],
                    chunk_index=int(meta.get("chunk_index", i)),
                )
            )
        return chunks

    async def mmr_search(
        self,
        query: str,
        meeting_id: str,
        k: int | None = None,
        lambda_mult: float = 0.6,
    ) -> list[Chunk]:
        """Maximal marginal relevance — diverse + relevant chunks."""
        import numpy as np

        top_k = k or _settings.top_k_retrieval
        candidates = await self.search(query, meeting_id, k=top_k * 3)
        if not candidates:
            return []

        q_emb_raw = await embed_text(query, task_type="retrieval_query")
        q_emb = np.array(q_emb_raw)

        # embed candidates
        cand_embs_raw = await embed_batch([c.text for c in candidates])
        cand_embs = np.array(cand_embs_raw)

        selected_idx: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_idx) < top_k and remaining:
            scores = []
            for i in remaining:
                relevance = float(np.dot(q_emb, cand_embs[i]))
                if selected_idx:
                    redundancy = max(
                        float(np.dot(cand_embs[i], cand_embs[j]))
                        for j in selected_idx
                    )
                else:
                    redundancy = 0.0
                scores.append(lambda_mult * relevance - (1 - lambda_mult) * redundancy)

            best = remaining[int(np.argmax(scores))]
            selected_idx.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_idx]
