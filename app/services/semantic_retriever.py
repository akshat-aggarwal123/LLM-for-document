import re
import uuid
from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class DocumentClause:
    id: str
    content: str
    clause_type: str
    relevance_score: float
    source_document: str
    page_number: int

class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name)
        self.client = chromadb.PersistentClient(path="./data/vector_store")
        self.collection_name = "doc_clauses"
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=self.ef
            )
        except Exception:
            self.collection = self.client.get_collection(name=self.collection_name)

    # ---------- chunk & index ----------
    def _chunk(self, text: str, size: int = 500, overlap: int = 50) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s*", text)
        chunks, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) > size:
                if buf:
                    chunks.append(buf.strip())
                buf = sent
            else:
                buf += " " + sent
        if buf:
            chunks.append(buf.strip())
        return [c for c in chunks if len(c) > 50]

    def _classify_clause(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["cover", "benefit", "eligible"]):
            return "coverage"
        if any(w in t for w in ["exclude", "not covered", "limitation"]):
            return "exclusion"
        if any(w in t for w in ["premium", "payment", "amount", "cost"]):
            return "payment"
        if any(w in t for w in ["claim", "submit", "procedure", "process"]):
            return "claim_process"
        return "general"

    def index_documents(self, docs: List[Tuple[str, str, int]]):
        texts, metas, ids = [], [], []
        for text, src, page in docs:
            for chk in self._chunk(text):
                ids.append(str(uuid.uuid4()))
                texts.append(chk)
                metas.append(
                    {"source": src, "page": page, "clause_type": self._classify_clause(chk)}
                )
        if texts:
            self.collection.add(documents=texts, metadatas=metas, ids=ids)

    # ---------- retrieval ----------
    def retrieve_relevant_clauses(
        self, query: str, entity, top_k: int = 10
    ) -> List[DocumentClause]:
        q = " ".join(
            [
                query,
                entity.procedure or "",
                f"age {entity.age}" if entity.age else "",
                entity.location or "",
                entity.policy_duration or "",
            ]
        )
        res = self.collection.query(query_texts=[q], n_results=top_k)
        clauses = []
        if res["documents"]:
            for doc, meta, rid, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["ids"][0],
                res["distances"][0],
            ):
                clauses.append(
                    DocumentClause(
                        id=rid,
                        content=doc,
                        clause_type=meta.get("clause_type", "general"),
                        relevance_score=1 - dist,
                        source_document=meta["source"],
                        page_number=meta["page"],
                    )
                )
        return clauses