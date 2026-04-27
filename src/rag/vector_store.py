"""ChromaDB vector store with MMR search."""
import logging

import chromadb
from chromadb import Collection

from src.core.config import get_settings
from src.core.llm import embed_batch, embed_text
from src.core.models import Chunk

logger = logging.getLogger(__name__)
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

        # ChromaDB doesn't allow duplicate IDs — upsert handles re-ingestion
        self._col.upsert(
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
        logger.info(f"Stored {len(chunks)} chunks for meeting {chunks[0].meeting_id}")

    async def search(
        self,
        query: str,
        meeting_id: str,
        k: int | None = None,
    ) -> list[Chunk]:
        top_k = k or _settings.top_k_retrieval

        # Check collection has data
        count = self._col.count()
        if count == 0:
            logger.warning("Vector store is empty — no chunks to search")
            return []

        q_emb = await embed_text(query, task_type="retrieval_query")

        try:
            results = self._col.query(
                query_embeddings=[q_emb],
                n_results=min(top_k, count),
                where={"meeting_id": meeting_id},
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            # Retry without filter (maybe meeting_id doesn't match)
            try:
                results = self._col.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k, count),
                    include=["documents", "metadatas", "distances", "embeddings"],
                )
                logger.warning("Fallback search without meeting_id filter succeeded")
            except Exception as e2:
                logger.error(f"Fallback search also failed: {e2}")
                return []

        if not results["documents"] or not results["documents"][0]:
            logger.warning(f"No chunks found for meeting_id={meeting_id}")
            return []

        chunks: list[Chunk] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        embeddings_list = []
        raw_embeddings = results.get("embeddings")
        if raw_embeddings is not None and len(raw_embeddings) > 0:
            embeddings_list = raw_embeddings[0] if len(raw_embeddings[0]) > 0 else []

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            chunk = Chunk(
                id=results["ids"][0][i],
                text=doc,
                speaker=meta.get("speaker") or None,
                start_time=meta.get("start_time"),
                end_time=meta.get("end_time"),
                meeting_id=meta.get("meeting_id", meeting_id),
                chunk_index=int(meta.get("chunk_index", i)),
            )
            # Attach embedding for MMR (avoids re-embedding)
            if len(embeddings_list) > i:
                chunk.metadata["embedding"] = embeddings_list[i]
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for meeting_id={meeting_id}")
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

        # Fetch more candidates than needed for MMR to select from
        candidates = await self.search(query, meeting_id, k=top_k * 3)
        if not candidates:
            logger.warning("No candidates for MMR — returning empty")
            return []

        if len(candidates) <= top_k:
            # Not enough to do MMR — just return all
            return candidates

        q_emb = await embed_text(query, task_type="retrieval_query")
        q_vec = np.array(q_emb, dtype=np.float32)

        # Use embeddings from search results (already cached on chunks)
        cand_vecs = []
        for c in candidates:
            emb = c.metadata.get("embedding")
            if emb is not None:
                cand_vecs.append(np.array(emb, dtype=np.float32))
            else:
                # Fallback: re-embed this chunk (shouldn't happen normally)
                from src.core.llm import embed_text as et
                emb = await et(c.text, task_type="retrieval_query")
                cand_vecs.append(np.array(emb, dtype=np.float32))

        cand_embs = np.array(cand_vecs)

        # Normalize for cosine similarity
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        cand_norms = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)

        selected_idx: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_idx) < top_k and remaining:
            scores = []
            for i in remaining:
                relevance = float(np.dot(q_norm, cand_norms[i]))
                if selected_idx:
                    redundancy = max(
                        float(np.dot(cand_norms[i], cand_norms[j]))
                        for j in selected_idx
                    )
                else:
                    redundancy = 0.0
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                scores.append(mmr_score)

            best_idx = int(np.argmax(scores))
            best = remaining[best_idx]
            selected_idx.append(best)
            remaining.remove(best)

        result = [candidates[i] for i in selected_idx]
        logger.info(f"MMR selected {len(result)} diverse chunks from {len(candidates)} candidates")
        return result

    def get_meeting_ids(self) -> list[str]:
        """Return all unique meeting IDs in the store."""
        try:
            all_meta = self._col.get(include=["metadatas"])
            metas = all_meta.get("metadatas", [])
            return list(set(m.get("meeting_id", "") for m in metas if m))
        except Exception:
            return []
