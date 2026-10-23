"""Private RAG retrieval, end to end: a real retrieval model, real documents, and the plan checked on real queries.

The application of demo/extensions/private_rag.py, with real parts:
  * the encoder is e5-base-v2 (intfloat/e5-base-v2), a BERT-base retrieval model with mean pooling;
  * the server's documents are the 5,183 abstracts of SciFact (BEIR, mteb/scifact), which the server
    embeds in plaintext; the client's queries are SciFact's claims, labeled with the abstracts that
    support or refute them.

The workflow, one step each:
  1. the application in PyTorch: encoder -> MeanPool -> DocScores -> TopK, the new operators of
     private_rag.rag_extension with their kernels and costs, traced with the extension
     (cf.models.from_torch(..., extensions=[rag]));
  2. calibrate it on training queries, so that HE runs only where its kernels' approximations hold;
  3. compile it for the deployment;
  4. emulate the plan on the test queries (cf.runtime.Emulator(..., extensions=[rag])): every operator
     in its planned domain, fixed point in MPC, the HE kernels' approximations, every conversion and
     bootstrap;
  5. compare the private top-k with the plaintext model's, and score both against the relevance labels.

    python -m demo.extensions.private_rag_end_to_end                       # downloads the model and SciFact
    python -m demo.extensions.private_rag_end_to_end --queries 50 --network 1Gbps/20ms --out rag.json
    python -m demo.extensions.private_rag_end_to_end --uncalibrated     # also the plan made without step 2

Needs torch, "transformers<5" and datasets. The client computes the token embeddings in plaintext (the
model input is the hidden states, as everywhere in CipherFlow). Each query runs at its own length,
without padding, under the plan made for 128 tokens: CipherFlow has no attention masks, and the
operators are the same at every length. The retrieval operators' costs are the estimates of
private_rag.rag_extension, not measurements; the emulation checks the plan's numerics, it does not run
the cryptography.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch import nn

import cipherflow as cf
from cipherflow import runtime
from demo.extensions import private_rag

MODEL = "intfloat/e5-base-v2"
DATASET = "mteb/scifact"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cipherflow" / "rag"


class Retriever(nn.Module):
    """The application: the encoder, then the retrieval operators of private_rag. The query is not
    normalized: dividing all of a query's scores by its norm does not change their order."""

    def __init__(self, encoder: nn.Module, docs: torch.Tensor, k: int):
        super().__init__()
        self.encoder = encoder
        self.pool = private_rag.MeanPool()
        self.scores = private_rag.DocScores(docs)
        self.topk = private_rag.TopK(k)

    def forward(self, x):
        return self.topk(self.scores(self.pool(self.encoder(x)[0])))


def load_scifact():
    from datasets import load_dataset
    corpus = load_dataset(DATASET, "corpus", split="corpus")
    queries = {str(q["_id"]): q["text"] for q in load_dataset(DATASET, "queries", split="queries")}

    def qrels(split) -> Dict[str, Set[str]]:
        rel: Dict[str, Set[str]] = {}
        for r in load_dataset(DATASET, "default", split=split):
            if r["score"] > 0:
                rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
        return rel

    return corpus, queries, qrels("train"), qrels("test")


def embed_documents(tok, model, corpus, max_tokens: int, device: str) -> torch.Tensor:
    """The server's index: each abstract embedded in plaintext (mean over its tokens, normalized). Cached."""
    path = CACHE / f"{MODEL.replace('/', '--')}-{DATASET.replace('/', '--')}-{max_tokens}.pt"
    if path.exists():
        return torch.load(path)
    texts = [f"passage: {d['title']} {d['text']}".strip() for d in corpus]
    print(f"the server embeds its {len(texts)} documents in plaintext, on {device} (once; cached in {path.parent})",
          flush=True)
    model = model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu())
    model.to("cpu")
    docs = torch.cat(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(docs, path)
    return docs


def ndcg_at_10(ranked: List[str], relevant: Set[str]) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in relevant)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))


def recall(ranked: List[str], relevant: Set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=10, help="documents returned to the client")
    ap.add_argument("--queries", type=int, default=0, help="test queries to emulate (0: all 300)")
    ap.add_argument("--calibrate", type=int, default=256, help="training queries to calibrate on")
    ap.add_argument("--hardware", default="cpu")
    ap.add_argument("--network", default="100Mbps/20ms")
    ap.add_argument("--security", default="2pc", choices=["2pc", "2pc-dealer"])
    ap.add_argument("--doc-tokens", type=int, default=512, help="tokens of each abstract the server embeds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="where the server embeds its documents (plaintext)")
    ap.add_argument("--uncalibrated", action="store_true", help="also plan without the calibration, and emulate it")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    step = lambda n, what: print(f"\n[{n}/5] {what}  ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731

    # ---- the data: the server's documents, the client's queries
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, attn_implementation="eager").eval()   # explicit Q/K/V matmuls
    corpus, queries, train, test = load_scifact()
    doc_ids = [str(d["_id"]) for d in corpus]
    docs = embed_documents(tok, model, corpus, a.doc_tokens, a.device)
    test_ids = [q for q in test if q in queries][: a.queries or None]
    train_ids = [q for q in train if q in queries][: a.calibrate]

    def hidden(qid):                      # the client: its query's token embeddings, in plaintext
        ids = tok("query: " + queries[qid], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            return model.embeddings(input_ids=ids["input_ids"], token_type_ids=ids["token_type_ids"]).double()

    print(f"model {MODEL}; the server's index: SciFact's {len(doc_ids)} abstracts")
    print(f"the client's queries: {len(test_ids)} test claims, {len(train_ids)} training claims to calibrate on")

    # ---- 1-2: the application and its extension, traced
    step(1, "the application in PyTorch, traced with its extension")
    rag = private_rag.rag_extension()
    retriever = Retriever(model.encoder, docs, a.k).eval()
    graph = cf.models.from_torch(retriever, torch.zeros(1, 128, 768), extensions=[rag])
    print(f"  {len(graph.nodes)} operators; the new ones: " + " -> ".join(n.op_type for n in graph.nodes[-3:])
          + f" (top-{a.k} of {len(doc_ids)} documents)")

    step(2, f"calibrate on {len(train_ids)} training queries")
    cal = runtime.calibrate(retriever, (hidden(q) for q in train_ids), extensions=[rag], model=MODEL)

    step(3, f"compile for {a.hardware}, {a.network}, security {a.security}")

    def compile_with(calibration):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
            return cf.compile(graph, cf.Deployment.of(a.hardware, a.network), extensions=[rag],
                              calibration=calibration, security=a.security, verbose=False)

    plans = {"calibrated": compile_with(cal)}
    if a.uncalibrated:
        plans["uncalibrated"] = compile_with(None)
    plan = plans["calibrated"]
    lines = plan.explain().splitlines()
    kept = [ln.split("warning: ", 1)[1] for ln in lines if "kept out of HE" in ln]
    print("\n".join(ln for ln in lines if "kept out of HE" not in ln and "not used: security model" not in ln))
    if kept:
        print(f"  calibration: {len(kept)} operators kept out of HE, where the real queries leave the HE kernels' ranges")
        print(f"    e.g. {kept[0].split(' (')[0]}")
    rows = plan.explain_steps().splitlines()
    tail = next(i for i, ln in enumerate(rows) if "MeanPool" in ln)
    print("  the retrieval, after the encoder:")
    print("\n".join(rows[:1] + rows[max(1, tail - 2):]))
    if a.uncalibrated:
        u = plans["uncalibrated"]
        he = sorted({nd.op_type for nd in u.nodes if nd.domain == "HE"} - {nd.op_type for nd in plan.nodes if nd.domain == "HE"})
        print(f"  without the calibration: {u.he_backend} + {u.mpc_backend}, {u.cost['total_ms'] / 1e3:.0f} s; "
              f"also in HE: {', '.join(he) or 'nothing more'}")

    # ---- 4: each plan on the real queries
    step(4, f"emulate {'each plan' if len(plans) > 1 else 'the plan'} on {len(test_ids)} test queries")
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    results = {}
    for name, pl in plans.items():
        em = runtime.Emulator(pl, retriever, extensions=[rag])
        op_of = {nd.id: nd.op_type for nd in pl.nodes}
        same_list = same_set = 0
        overlap, plain_ndcg, priv_ndcg, plain_rec, priv_rec = [], [], [], [], []
        out_of_range: Set[str] = set()
        problems: List[str] = []
        for i, qid in enumerate(test_ids):
            r = em(hidden(qid))
            priv, plain = [doc_ids[j] for j in r.output[0].tolist()], [doc_ids[j] for j in r.reference[0].tolist()]
            same_list += priv == plain
            same_set += set(priv) == set(plain)
            overlap.append(len(set(priv) & set(plain)) / a.k)
            rel = test[qid]
            plain_ndcg.append(ndcg_at_10(plain, rel)), priv_ndcg.append(ndcg_at_10(priv, rel))
            plain_rec.append(recall(plain, rel, a.k)), priv_rec.append(recall(priv, rel, a.k))
            out_of_range |= set(r.out_of_range())
            problems = r.checks["problems"]
            if (i + 1) % 100 == 0:
                print(f"  {name}: {i + 1} queries ({time.time() - t0:.0f} s)", flush=True)
        kinds: Dict[str, int] = {}
        for nid in out_of_range:
            kinds[op_of.get(nid, nid)] = kinds.get(op_of.get(nid, nid), 0) + 1
        results[name] = {
            "backends": f"{pl.he_backend}+{pl.mpc_backend}", "latency_s": pl.cost["total_ms"] / 1e3,
            "comm_gib": pl.cost["comm_bytes"] / 2 ** 30,
            "retrieval_domains": {nd.op_type: nd.domain for nd in pl.nodes if nd.op_type in ("MeanPool", "DocScores", "TopK")},
            "same_list": same_list, "same_set": same_set, "overlap": mean(overlap),
            "ndcg10": {"plaintext": mean(plain_ndcg), "private": mean(priv_ndcg)},
            f"recall{a.k}": {"plaintext": mean(plain_rec), "private": mean(priv_rec)},
            "out_of_range": kinds, "plan_problems": problems}

    # ---- 5: the result
    n = len(test_ids)
    step(5, "the private retrieval against the plaintext model")
    res = list(results.values())
    plain = res[0]
    table = [
        ("", [name for name in results]),
        (f"same top-{a.k}, in the same order", [f"{x['same_list']}/{n} ({100 * x['same_list'] / n:.1f}%)" for x in res]),
        (f"same top-{a.k} documents", [f"{x['same_set']}/{n} ({100 * x['same_set'] / n:.1f}%)" for x in res]),
        ("documents in common, on average", [f"{100 * x['overlap']:.1f}%" for x in res]),
        (f"nDCG@10 (plaintext {plain['ndcg10']['plaintext']:.4f})", [f"{x['ndcg10']['private']:.4f}" for x in res]),
        (f"recall@{a.k} (plaintext {plain[f'recall{a.k}']['plaintext']:.4f})",
         [f"{x[f'recall{a.k}']['private']:.4f}" for x in res]),
        ("HE inputs outside their kernels' ranges",
         [", ".join(f"{op} x{c}" for op, c in sorted(x["out_of_range"].items())) or "none" for x in res]),
        ("plan check", ["passes" if not x["plan_problems"] else f"{len(x['plan_problems'])} problems" for x in res]),
        ("latency per query (estimate)", [f"{x['latency_s']:.0f} s" for x in res]),
        ("online communication (estimate)", [f"{x['comm_gib']:.1f} GiB" for x in res]),
    ]
    for label, cells in table:
        print(f"  {label:<42s}" + "".join(f"{c:<26s}" for c in cells).rstrip())
    print(f"  (the plans are made for queries of up to 128 tokens; {n} queries, top-{a.k} of {len(doc_ids)} documents)")

    if a.out:
        report = {"model": MODEL, "dataset": DATASET, "documents": len(doc_ids), "queries": n, "k": a.k,
                  "calibration_queries": len(train_ids),
                  "deployment": {"hardware": a.hardware, "network": a.network, "security": a.security},
                  "plans": results}
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"  results: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
