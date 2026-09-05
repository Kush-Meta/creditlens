"""Grid-search retrieval parameters against the labelled eval cases.

Run with `make sweep`. Prints a table used to set the defaults in
creditlens/config.py - the values there should always be justifiable by this
output rather than chosen by feel.
"""
from __future__ import annotations

import itertools
import os
import sys

os.environ.setdefault("CREDITLENS_LOG_LEVEL", "WARNING")

from creditlens.config import get_settings
from creditlens.db import session_scope
from creditlens.eval.dataset import load_suite
from creditlens.eval.metrics import (
    MetricAccumulator,
    ndcg_at_k,
    precision_at_k,
    recall_ceiling_normalized,
    reciprocal_rank,
)
from creditlens.eval.runner import relevant_chunk_ids
from creditlens.retrieval import corpus as corpus_mod
from creditlens.retrieval import hybrid

DENSE_WEIGHTS = (0.0, 0.2, 0.35, 0.5, 0.65, 1.0)
MMR_SETTINGS = ((False, 0.0), (True, 0.7), (True, 0.85), (True, 0.95))


def main() -> int:
    settings = get_settings()
    k = settings.retrieval_top_k
    with session_scope() as session:
        available = {r.ticker for r in corpus_mod.get_snapshot(session).records}
        cases = [
            c for c in load_suite("golden")
            if c.relevance and all(t.upper() in available for t in c.tickers)
        ]
        labels = {c.id: relevant_chunk_ids(session, c.relevance) for c in cases}
        cases = [c for c in cases if labels[c.id]]
        print(f"sweeping over {len(cases)} labelled cases, k={k}\n")
        print(f"{'dense_w':>8}{'mmr':>7}{'lambda':>8}"
              f"{'recall_n':>10}{'prec@k':>9}{'MRR':>8}{'nDCG':>8}")

        original = (settings.dense_weight, settings.mmr_lambda)
        rows = []
        try:
            for weight, (use_mmr, lam) in itertools.product(DENSE_WEIGHTS, MMR_SETTINGS):
                settings.dense_weight = weight
                settings.mmr_lambda = lam or settings.mmr_lambda
                acc = MetricAccumulator()
                for case in cases:
                    relevant = labels[case.id]
                    result = hybrid.search(
                        session, case.retrieval_query or case.question,
                        tickers=case.tickers or None, top_k=k, use_mmr=use_mmr,
                    )
                    ids = [h.record.chunk_id for h in result.chunks]
                    acc.add("recall_n", recall_ceiling_normalized(ids, relevant, k))
                    acc.add("prec", precision_at_k(ids, relevant, k))
                    acc.add("mrr", reciprocal_rank(ids, relevant))
                    acc.add("ndcg", ndcg_at_k(ids, relevant, k))
                summary = acc.summary()
                row = (weight, use_mmr, lam,
                       summary["recall_n"]["mean"], summary["prec"]["mean"],
                       summary["mrr"]["mean"], summary["ndcg"]["mean"])
                rows.append(row)
                print(f"{weight:>8.2f}{use_mmr!s:>7}{lam:>8.2f}"
                      f"{row[3]:>10.4f}{row[4]:>9.4f}{row[5]:>8.4f}{row[6]:>8.4f}")
        finally:
            settings.dense_weight, settings.mmr_lambda = original

        best = max(rows, key=lambda r: r[6])
        print(f"\nbest by nDCG@{k}: dense_weight={best[0]}, mmr={best[1]}, "
              f"lambda={best[2]} -> nDCG={best[6]:.4f}, recall_n={best[3]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
