from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from evals.retrieval_benchmark_v2 import aggregate,load_dataset,metrics,percentile,read_jsonl,validate

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/evals/retrieval/benchmark_v2"


def fixture():
    md={"ticker":"TEST","fiscal_year":2024,"form_type":"10-K","source_html":"source.html","source_sha256":"source","section_path":"Section"}
    docs=[{"id":i,"content":i,"content_sha256":hashlib.sha256(i.encode()).hexdigest(),"metadata":md} for i in ("a","b","p","n")]
    judgments=[{"evidence_id":d["id"],"grade":grade,"groups":[group],"spans":[{"start":0,"end":1,"quote":d["content"],"anchor":d["content"]}],
                "content_sha256":d["content_sha256"],**{k:md[k] for k in ("source_html","source_sha256","section_path")}}
               for d,grade,group in zip(docs,[2,2,1,0],["g1","g2","partial","negative"])]
    c={"id":"q","query":"Question","status":"answerable","stratum":"multi_evidence","required_groups":["g1","g2"],"judgments":judgments,
       **{k:md[k] for k in ("ticker","fiscal_year","form_type")}}
    return c,docs


def test_known_metrics_and_graded_gain():
    c,_=fixture()
    result=metrics(["p","n","a","b"],c)
    assert result["recall@5"]==result["recall@10"]==1
    assert result["mrr@10"]==pytest.approx(1/3)
    dcg=1+3/math.log2(4)+3/math.log2(5)
    ideal=3+3/math.log2(3)+1/math.log2(4)
    assert result["ndcg@5"]==pytest.approx(dcg/ideal)
    assert result["evidence_group_recall@10"]==1


def test_duplicate_predictions_consume_rank_without_gain():
    c,_=fixture()
    result=metrics(["a"]*5+["b"],c)
    assert result["recall@5"]==.5
    assert result["recall@10"]==1
    assert result["ndcg@10"]<1
    assert metrics(["unknown"]*9+["a"],c)["mrr@10"]==.1
    assert metrics(["unknown"]*10+["a"],c)["mrr@10"]==0


def test_empty_results_are_zero_not_omitted():
    c,_=fixture()
    assert all(v==0 for v in metrics([],c).values())


def test_partial_does_not_satisfy_binary_relevance():
    c,_=fixture()
    result=metrics(["p"],c)
    assert result["recall@10"]==result["mrr@10"]==0
    assert result["ndcg@10"]>0


@pytest.mark.parametrize("mutation,error",[
    ("duplicate_query","query IDs"),("duplicate_corpus","corpus IDs"),
    ("duplicate_label","judgment IDs"),("missing_label","Missing label"),
    ("missing_relevant","Missing relevant"),("wrong_year","filter incompatibility"),
    ("wrong_content","content hash"),("wrong_span","evidence span"),
    ("wrong_source","provenance"),("wrong_grade","relevance grade"),
    ("wrong_group","Missing relevant"),("wrong_anchor","evidence anchor"),
])
def test_invalid_contracts(mutation,error):
    def change(c,ds):
        cases=[c]
        if mutation=="duplicate_query":cases.append(deepcopy(c))
        if mutation=="duplicate_corpus":ds.append(deepcopy(ds[0]))
        if mutation=="duplicate_label":c["judgments"].append(deepcopy(c["judgments"][0]))
        if mutation=="missing_label":c["judgments"][0]["evidence_id"]="absent"
        if mutation=="missing_relevant":c["judgments"]=[]
        if mutation=="wrong_year":c["fiscal_year"]=2025
        if mutation=="wrong_content":ds[0]["content"]="changed"
        if mutation=="wrong_span":c["judgments"][0]["spans"][0]["end"]=2
        if mutation=="wrong_source":c["judgments"][0]["source_sha256"]="changed"
        if mutation=="wrong_grade":c["judgments"][0]["grade"]=True
        if mutation=="wrong_group":c["required_groups"].append("missing")
        if mutation=="wrong_anchor":c["judgments"][0]["spans"][0]["anchor"]="missing"
        return cases,ds
    c,ds=fixture()
    with pytest.raises(ValueError,match=error):validate(*change(c,ds))


def test_exclusion_is_explicit():
    c,ds=fixture()
    c.update(status="excluded",stratum="unanswerable",judgments=[],required_groups=[],exclusion_reason="Future result")
    assert validate([c],ds)["excluded"]==1
    with pytest.raises(ValueError,match="Excluded"):metrics([],c)
    c.pop("exclusion_reason")
    with pytest.raises(ValueError,match="exclusion"):validate([c],ds)


def test_percentiles_and_missing_latency():
    assert percentile(list(range(1,21)),.5)==10
    assert percentile(list(range(1,21)),.95)==19
    assert percentile([],.5) is None
    assert aggregate([])["latency"]["reranker_ms"]["p50"] is None
    with pytest.raises(ValueError):percentile([float("nan")],.5)


def test_errors_stay_in_denominator():
    c,_=fixture()
    rows=[{"metrics":metrics(["a","b","p"],c),"error":None,"retrieval_ms":10},
          {"metrics":metrics([],c),"error":{"type":"Timeout"},"retrieval_ms":100}]
    result=aggregate(rows)
    assert result["queries"]==2 and result["errors"]==1
    assert result["metrics"]["recall@10"]==.5
    assert result["latency"]["retrieval_ms"]["p95"]==100


def test_frozen_dataset_loading_is_deterministic_and_compatible():
    a=load_dataset(DATA,ROOT)
    b=load_dataset(DATA,ROOT)
    assert a==b
    assert a[2]["answerable"]==120 and a[2]["documents"]==948 and a[2]["excluded"]==6
    assert len({c["ticker"] for c in a[0]})==3


def test_jsonl_load_rejects_blank_nonobjects(tmp_path):
    p=tmp_path/"bad.jsonl"
    for text in ("\n", "[]\n"):
        p.write_text(text)
        with pytest.raises(ValueError):read_jsonl(p)


def test_frozen_manifest_tamper_rejected(tmp_path):
    (tmp_path/"dataset_manifest.json").write_text(json.dumps({"files":{"queries.jsonl":"wrong"}}))
    (tmp_path/"queries.jsonl").write_text("{}\n")
    with pytest.raises(ValueError,match="hash mismatch"):load_dataset(tmp_path)


def runner():
    spec=importlib.util.spec_from_file_location("benchmark_runner",ROOT/"scripts/evals/retrieval/run_benchmark_v2.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_balanced_schedule():
    cases,_,_=load_dataset(DATA,ROOT)
    config=json.loads((DATA/"comparison_config.json").read_text())
    mod=runner()
    s=mod.schedule(cases,config)
    assert s==mod.schedule(list(reversed(cases)),config)
    assert len(s)==480
    assert len({(c["id"],m) for c,m in s})==480
    from collections import Counter
    assert set(Counter(m for _,m in s[::4]).values())=={30}


def test_index_compatibility_requires_full_identity():
    _,docs=fixture()
    records=[{"payload":{"doc_id":d["id"],"content":d["content"],**d["metadata"]}} for d in docs]
    mod=runner()
    mod.verify_index(records,docs)
    records[0]["payload"]["fiscal_year"]=2025
    with pytest.raises(ValueError,match="metadata mismatch"):mod.verify_index(records,docs)


def test_untracked_runtime_overrides_are_rejected(monkeypatch):
    mod=runner()
    monkeypatch.setattr(mod,"git",lambda *args:"src/override.py" if args[0]=="ls-files" else "")
    with pytest.raises(ValueError,match="Untracked"):mod.clean_checkout()
    monkeypatch.setattr(mod,"git",lambda *args:"artifacts/measurement.json" if args[0]=="ls-files" else "")
    mod.clean_checkout()


def test_staged_or_unstaged_change_is_rejected(monkeypatch):
    mod=runner()
    monkeypatch.setattr(mod,"git",lambda *args:" M src/runtime.py")
    with pytest.raises(ValueError,match="dirty"):mod.clean_checkout()


def test_index_rejects_missing_or_duplicate_ids():
    _,docs=fixture()
    records=[{"payload":{"doc_id":d["id"],"content":d["content"],**d["metadata"]}} for d in docs]
    mod=runner()
    with pytest.raises(ValueError,match="ID-set mismatch"):mod.verify_index(records[:-1],docs)
    with pytest.raises(ValueError,match="Duplicate"):mod.verify_index(records+[records[0]],docs)


def test_offline_verifier_recomputes_metrics(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("verifier",ROOT/"scripts/evals/retrieval/verify_benchmark_v2.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    c,docs=fixture()
    monkeypatch.setattr(mod,"load_dataset",lambda *a:([c],docs,{"missing_labels":0}))
    dataset=tmp_path/"dataset";dataset.mkdir()
    (dataset/"dataset_manifest.json").write_text("{}")
    out=tmp_path/"baseline";out.mkdir()
    row={"id":"q","mode":"bm25_only","query":c["query"],"stratum":c["stratum"],"ticker":c["ticker"],"fiscal_year":c["fiscal_year"],
         "ranked_ids":["a","b"],"metrics":metrics(["a","b"],c),"error":None,"retrieval_ms":10}
    (out/"per_query.jsonl").write_text(json.dumps(row)+"\n")
    summary={"complete":True,"expected_pairs":1,"completed_pairs":1,"modes":{"bm25_only":{
        "overall":aggregate([row]),"by_stratum":{"multi_evidence":aggregate([row])},"by_company":{"TEST":aggregate([row])}}}}
    (out/"summary.json").write_text(json.dumps(summary))
    manifest={"config":{"modes":["bm25_only"]},"dataset_manifest_sha256":mod.sha256(dataset/"dataset_manifest.json"),
              "raw_sha256":{name:mod.sha256(out/name) for name in ("summary.json","per_query.jsonl")}}
    (out/"manifest.json").write_text(json.dumps(manifest))
    assert mod.verify(dataset,out)["hashes_and_metrics_verified"]
    row["metrics"]["recall@5"]=0.0
    (out/"per_query.jsonl").write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError,match="hash mismatch"):mod.verify(dataset,out)
    manifest["raw_sha256"]["per_query.jsonl"]=mod.sha256(out/"per_query.jsonl")
    (out/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="metric mismatch"):mod.verify(dataset,out)
