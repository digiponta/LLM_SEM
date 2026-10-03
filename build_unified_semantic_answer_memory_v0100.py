# build_unified_semantic_answer_memory_v0100.py
#
# LLM_SEM v0.10.0
# Integrates useful LLM_TRY v10.8.6-v10.9.1 knowledge-memory ideas:
# - concept auto-discovery from local corpus
# - proposition merge
# - relation facts
# - incoming/outgoing relation rendering
# - bare-concept and paraphrase query variants

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows=[]
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row=json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row,dict):
            rows.append(row)
    return rows


def sentence_units(text: str) -> list[str]:
    return [re.sub(r"\s+"," ",x).strip() for x in re.split(r"(?<=[。！？!?])|\n+",text) if x.strip()]


def plausible_concept(text: str) -> bool:
    t=text.strip(" 、。・()（）[]「」『』")
    if not (2 <= len(t) <= 24):
        return False
    if re.search(r"[\s、。！？!?：:；;,]",t):
        return False
    if t in {"これ","それ","あれ","ここ","そこ","ため","もの","こと","私","あなた","場合","よう"}:
        return False
    return not re.fullmatch(r"[0-9０-９]+",t)


def discover_concepts(units: list[str], min_occurrences: int=2) -> list[str]:
    counts=Counter()
    pats=(
        r"^([^、。！？!?]{2,24})は[、,]?",
        r"^([^、。！？!?]{2,24})も[、,]?",
        r"^([^、。！？!?]{2,24})とは[、,]?",
        r"^([^、。！？!?]{2,24})の定義は[、,]?",
    )
    for unit in units:
        s=unit.strip(" ・＊*#>-")
        for pat in pats:
            m=re.match(pat,s)
            if not m:
                continue
            concept=re.sub(r"(?:も|が|を|に|で)$","",m.group(1).strip()).strip()
            if plausible_concept(concept):
                counts[concept]+=1
            break
    return [c for c,n in counts.most_common() if n >= min_occurrences]


def extract_clause(concept: str, text: str) -> str | None:
    s=text.strip()
    if concept not in s or len(s)>180:
        return None
    m=re.match(rf"^{re.escape(concept)}(?:とは|の定義は|は|も)[、,]?(.*?)(?:。)?$",s)
    if m:
        body=m.group(1).strip(" 。")
        if body:
            return f"{concept}は、{body}。"
    return s.rstrip("。")+"。"


def dedupe_clauses(clauses: list[str]) -> list[str]:
    out=[]
    seen=[]
    for c in clauses:
        n=re.sub(r"[\s、。,.「」『』()（）]","",c.lower())
        duplicate=False
        for old in seen:
            shorter=min(len(n),len(old))
            if shorter>=8 and (n in old or old in n):
                duplicate=True
                break
        if not duplicate:
            out.append(c)
            seen.append(n)
    return out


def relation_fact(row: dict) -> dict | None:
    subject=str(row.get("subject","")).strip()
    relation=str(row.get("relation","")).strip()
    value=str(row.get("value",row.get("object",""))).strip()
    if not subject or not relation or not value:
        return None
    return {
        "subject":subject,
        "relation":relation,
        "value":value,
        "condition":str(row.get("condition","")).strip(),
        "relation_context":str(row.get("relation_context",row.get("context",""))).strip(),
    }


def render_outgoing(f: dict) -> str:
    s,r,v=f["subject"],f["relation"],f["value"]
    if r in {"definition","is","is_a","isa"}:
        clause=f"{s}は、{v}である"
    else:
        templates={
            "includes":f"{s}は、{v}を含む",
            "contains":f"{s}は、{v}を含む",
            "part_of":f"{s}は、{v}の一部である",
            "has":f"{s}は、{v}を持つ",
            "uses":f"{s}は、{v}を使う",
            "related_to":f"{s}は、{v}と関係する",
            "causes":f"{s}は、{v}を引き起こす",
            "depends_on":f"{s}は、{v}に依存する",
        }
        clause=templates.get(r,f"{s}は、{v}と{r}の関係にある")
    if f["relation_context"]:
        clause=f'{f["relation_context"]}、{clause}'
    if f["condition"]:
        clause=f'{f["condition"]}の場合、{clause}'
    return clause.rstrip("。")+"。"


def render_incoming(f: dict) -> str | None:
    s,r,v=f["subject"],f["relation"],f["value"]
    if r=="definition":
        return None
    templates={
        "includes":f"{v}は、{s}に含まれる対象として関係する。",
        "contains":f"{v}は、{s}に含まれる対象として関係する。",
        "is_a":f"{v}は、{s}の上位概念として関係する。",
        "isa":f"{v}は、{s}の上位概念として関係する。",
        "part_of":f"{v}は、{s}を一部として含む関係にある。",
        "has":f"{v}は、{s}が持つ対象として関係する。",
        "uses":f"{v}は、{s}が使う対象として関係する。",
        "related_to":f"{v}は、{s}と関係する。",
        "causes":f"{v}は、{s}によって引き起こされる対象として関係する。",
        "depends_on":f"{v}は、{s}が依存する対象として関係する。",
    }
    return templates.get(r,f"{v}は、{s}との{r}関係の対象である。")


def variants(concept: str) -> list[str]:
    return [concept,f"{concept}とは",f"{concept}について教えて",f"{concept}を説明して",f"{concept}って何"]


def parse_args():
    p=argparse.ArgumentParser(description="LLM_SEM v0.10.0 unified semantic answer-memory builder")
    p.add_argument("--corpus",default="data/data-nagato.txt")
    p.add_argument("--data-dir",default="data")
    p.add_argument("--base-answer-memory",default="data/semantic_guided_qa_v097.json")
    p.add_argument("--output",default="data/unified_semantic_answer_memory_v0100.jsonl")
    p.add_argument("--min-occurrences",type=int,default=2)
    p.add_argument("--max-concepts",type=int,default=200)
    return p.parse_args()


def main():
    args=parse_args()
    answers: dict[str,list[str]]=defaultdict(list)
    labels: dict[str,str]={}

    base=Path(args.base_answer_memory)
    if base.exists():
        obj=json.loads(base.read_text(encoding="utf-8"))
        for row in obj.get("samples",[]):
            concept=(row.get("concepts") or [""])[0]
            if concept and row.get("answer"):
                answers[str(concept)].append(str(row["answer"]))
                labels[str(concept)]=str(row.get("label","unknown"))

    corpus=Path(args.corpus)
    if corpus.exists():
        units=sentence_units(corpus.read_text(encoding="utf-8"))
        for concept in discover_concepts(units,args.min_occurrences)[:args.max_concepts]:
            clauses=[extract_clause(concept,u) for u in units if concept in u]
            clauses=dedupe_clauses([c for c in clauses if c])[:6]
            if clauses:
                answers[concept].extend(clauses)

    facts=[]
    data_dir=Path(args.data_dir)
    if data_dir.exists():
        for p in data_dir.rglob("*.jsonl"):
            for row in load_jsonl(p):
                f=relation_fact(row)
                if f:
                    facts.append(f)

    outgoing=defaultdict(list)
    incoming=defaultdict(list)
    seen=set()
    for f in facts:
        key=(f["subject"],f["relation"],f["value"],f["condition"],f["relation_context"])
        if key in seen:
            continue
        seen.add(key)
        outgoing[f["subject"]].append(f)
        incoming[f["value"]].append(f)

    concepts=set(answers)|set(outgoing)|set(incoming)
    rows=[]
    for concept in sorted(concepts):
        parts=dedupe_clauses(answers.get(concept,[]))
        for f in outgoing.get(concept,[]):
            parts=dedupe_clauses(parts+[render_outgoing(f)])
        for f in incoming.get(concept,[]):
            c=render_incoming(f)
            if c:
                parts=dedupe_clauses(parts+[c])
        if not parts:
            continue
        answer="また、".join(x.rstrip("。") for x in parts[:8])+"。"
        label=labels.get(concept,"unknown")
        for q in variants(concept):
            rows.append({
                "query":q,
                "answer":answer,
                "label":label,
                "intent":"definition",
                "concepts":[concept],
                "truth_status":"UNVERIFIED",
                "source":"llm-try-unified-semantic-integration-v0100",
                "relation_count":len(outgoing.get(concept,[]))+len(incoming.get(concept,[])),
            })

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row,ensure_ascii=False)+"\n")

    print("="*92)
    print(" LLM_SEM v0.10.0 Unified Semantic Answer Memory Builder")
    print("="*92)
    print("Concepts :",len({r["concepts"][0] for r in rows}))
    print("Records  :",len(rows))
    print("Relations:",len(facts))
    print("Saved    :",out)


if __name__=="__main__":
    main()
