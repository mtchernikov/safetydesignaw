import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI


# ============================================================
# Helpers
# ============================================================

def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def normalize_id(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def try_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        cleaned = text.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {"raw_text": text}


def load_yaml(upload, default_path: str) -> Dict[str, Any]:
    if upload is not None:
        return yaml.safe_load(upload.read().decode("utf-8"))
    with open(default_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_fact(
    graph: List[Dict[str, Any]],
    subject: str,
    relation: str,
    object_: str,
    relation_group: str = "unknown",
    source: str = "derived",
    confidence: float = 0.7,
    evidence: str = "",
    line_id: Optional[int] = None,
    rule: Optional[str] = None,
    needs_validation: bool = False,
) -> None:
    fact = {
        "subject": normalize_id(subject),
        "relation": relation,
        "object": normalize_id(object_),
        "relation_group": relation_group,
        "source": source,
        "confidence": float(confidence or 0.7),
        "evidence": evidence,
        "line_id": line_id,
        "needs_validation": bool(needs_validation),
    }
    if rule:
        fact["rule"] = rule

    key = (fact["subject"], fact["relation"], fact["object"], fact["source"], fact.get("rule"))
    existing = {
        (f.get("subject"), f.get("relation"), f.get("object"), f.get("source"), f.get("rule"))
        for f in graph
    }
    if key not in existing:
        graph.append(fact)


def graph_nodes(graph: List[Dict[str, Any]]) -> set:
    return {f.get("subject") for f in graph if f.get("subject")} | {f.get("object") for f in graph if f.get("object")}


def graph_has_concept(graph: List[Dict[str, Any]], concept: str) -> bool:
    concept = normalize_id(concept)
    return concept in graph_nodes(graph)


def graph_has_any_concept(graph: List[Dict[str, Any]], concepts: List[str]) -> bool:
    return any(graph_has_concept(graph, c) for c in concepts)


def has_fact(graph: List[Dict[str, Any]], subject: str = "*", relation: str = "*", object_: str = "*") -> bool:
    s = normalize_id(subject) if subject != "*" else "*"
    o = normalize_id(object_) if object_ != "*" else "*"
    for f in graph:
        if s != "*" and normalize_id(f.get("subject")) != s:
            continue
        if relation != "*" and f.get("relation") != relation:
            continue
        if o != "*" and normalize_id(f.get("object")) != o:
            continue
        return True
    return False


def find_relation_matches(graph: List[Dict[str, Any]], pattern: List[str]) -> List[str]:
    if len(pattern) != 3:
        return []
    s, r, o = pattern
    out = []
    for f in graph:
        if s != "*" and normalize_id(f.get("subject")) != normalize_id(s):
            continue
        if r != "*" and f.get("relation") != r:
            continue
        if o != "*" and normalize_id(f.get("object")) != normalize_id(o):
            continue
        out.append(f'{f.get("subject")} — {f.get("relation")} → {f.get("object")}')
    return out


# ============================================================
# LLM normalization
# ============================================================

def ontology_summary(ontology: Dict[str, Any]) -> str:
    lines = []
    for cid, c in ontology.get("concepts", {}).items():
        syns = ", ".join(c.get("synonyms", [])[:10])
        props = ", ".join(c.get("properties", [])[:10])
        lines.append(f"- {cid}: type={c.get('type')}; synonyms=[{syns}]; properties=[{props}]")
    return "\n".join(lines)


def relation_summary(relations: Dict[str, Any]) -> str:
    lines = []
    for rid, r in relations.get("relations", {}).items():
        lines.append(f"- {rid}: {r.get('description', '')}")
    return "\n".join(lines)


def llm_normalize(api_key: str, model: str, description: str, ontology: Dict[str, Any], relations: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a safety engineering extraction and normalization module for a toaster / household appliance demo.

Task:
Normalize the system description into controlled concepts and graph triples.

Rules:
- Do NOT decide final hazards or applicable standards.
- Use ontology concepts where possible.
- Use only allowed relation IDs.
- Mark direct facts as source="explicit".
- Mark cautious inferences as source="inferred".
- Return valid JSON only.

Ontology:
{ontology_summary(ontology)}

Allowed relations:
{relation_summary(relations)}

System description:
{description}

Return JSON:
{{
  "entities": [
    {{
      "id": "E1",
      "original_text": "",
      "canonical": "",
      "type": "",
      "line_id": 1,
      "confidence": "low|medium|high"
    }}
  ],
  "triples": [
    {{
      "subject": "",
      "relation": "",
      "object": "",
      "relation_group": "",
      "source": "explicit|inferred",
      "line_id": 1,
      "confidence": 0.0,
      "evidence": ""
    }}
  ],
  "unknowns": []
}}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only. Be conservative and traceable."},
            {"role": "user", "content": prompt},
        ],
    )
    return try_parse_json(response.choices[0].message.content or "{}")


def fallback_normalize(description: str) -> Dict[str, Any]:
    text = description.lower()
    entities = []
    triples = []

    def ent(original: str, canonical: str, typ: str, conf: str = "medium"):
        entities.append({
            "id": f"E{len(entities)+1}",
            "original_text": original,
            "canonical": canonical,
            "type": typ,
            "line_id": 1,
            "confidence": conf,
        })

    def tr(s, r, o, group, source="explicit", conf=0.75, ev=""):
        triples.append({
            "subject": s,
            "relation": r,
            "object": o,
            "relation_group": group,
            "source": source,
            "line_id": 1,
            "confidence": conf,
            "evidence": ev,
        })

    if "toaster" in text:
        ent("toaster", "toaster", "device", "high")
    if "220" in text or "230" in text or "mains" in text or "ac" in text:
        ent("mains supply", "mains_supply", "component", "high")
        tr("toaster", "has_component", "mains_supply", "structural", ev="mains supply")
        tr("mains_supply", "is_a", "power_supply_220v", "classification", ev="220/230 V mains")
    if "heating" in text or "heater" in text or "element" in text:
        ent("heating element", "heating_element", "component", "high")
        tr("toaster", "has_component", "heating_element", "structural", ev="heating element")
        tr("heating_element", "supplied_by", "mains_supply", "energy", "inferred", 0.65, "heating element in mains toaster")
    if "metal" in text or "stainless" in text or "steel" in text:
        ent("metal housing", "metal_housing", "component", "high")
        tr("toaster", "has_component", "metal_housing", "structural", ev="metal housing")
        tr("metal_housing", "material", "conductive_material", "material", ev="metal housing")
        if "user" in text or "touch" in text or "accessible" in text or "housing" in text:
            tr("metal_housing", "user_accessible", "user", "human_interaction", "inferred", 0.7, "housing is normally touchable")
    if "slot" in text or "bread" in text:
        ent("bread slot", "bread_slot", "access_opening", "high")
        tr("toaster", "has_component", "bread_slot", "structural", ev="bread slot")
        tr("bread_slot", "user_accessible", "user", "human_interaction", "inferred", 0.75, "user inserts bread")
        tr("bread_slot", "near", "heating_element", "structural", "inferred", 0.7, "bread slot near heating element")
    if "plastic" in text:
        ent("plastic part", "plastic_part", "component", "medium")
        tr("plastic_part", "material", "plastic", "material", ev="plastic")
        if "heating" in text or "element" in text:
            tr("plastic_part", "near", "heating_element", "structural", "inferred", 0.55, "plastic near heat unclear")
    if "crumb" in text or "food residue" in text or "residue" in text:
        ent("crumbs", "crumbs", "residue", "medium")
        tr("crumbs", "near", "heating_element", "structural", "inferred", 0.65, "crumbs can accumulate near heating zone")
    if "water" in text or "moist" in text or "clean" in text or "liquid" in text:
        ent("moisture", "moisture", "environmental_condition", "medium")
        tr("toaster", "exposed_to", "moisture", "exposure", "explicit", 0.7, "cleaning/liquid/moisture mentioned")
    if "metal object" in text or "knife" in text or "fork" in text:
        ent("metal object", "metal_object", "foreign_object", "high")
        tr("metal_object", "inserted_into", "bread_slot", "human_interaction", "explicit", 0.8, "metal object inserted")

    return {
        "entities": entities,
        "triples": triples,
        "unknowns": [
            "Is the product protective-earthed or double-insulated?",
            "What insulation separates live parts from accessible parts?",
            "Can the user touch live parts through the bread slot?",
            "Maximum normal and single-fault surface temperatures",
            "Material grade and flammability rating of plastic parts",
            "Overtemperature and overcurrent protection concept",
        ],
    }


# ============================================================
# Graph building and propagation
# ============================================================

def build_graph(normalized: Dict[str, Any], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph: List[Dict[str, Any]] = []

    for e in normalized.get("entities", []):
        canonical = normalize_id(e.get("canonical") or e.get("original_text"))
        typ = normalize_id(e.get("type", "unknown"))
        conf = {"high": 0.9, "medium": 0.7, "low": 0.4}.get(e.get("confidence"), 0.7)
        add_fact(graph, canonical, "is_a", typ, "classification", "llm_extraction", conf, e.get("original_text", ""), e.get("line_id"))

    for t in normalized.get("triples", []):
        add_fact(
            graph,
            t.get("subject", ""),
            t.get("relation", "related_to"),
            t.get("object", ""),
            t.get("relation_group", "unknown"),
            t.get("source", "explicit"),
            float(t.get("confidence", 0.7) or 0.7),
            t.get("evidence", ""),
            t.get("line_id"),
        )

    # Ontology expansion. Repeat because implied facts can introduce new ontology concepts.
    concepts = ontology.get("concepts", {})
    for _ in range(4):
        before = len(graph)
        nodes = graph_nodes(graph)
        for cid, c in concepts.items():
            cidn = normalize_id(cid)
            if cidn not in nodes:
                continue
            for prop in c.get("properties", []):
                add_fact(graph, cidn, "has_property", prop, "ontology_property", "ontology", 0.75, f"Ontology property of {cid}")
            for imp in c.get("implied_facts", []):
                add_fact(
                    graph,
                    imp.get("subject", "$self").replace("$self", cidn),
                    imp.get("relation", "related_to"),
                    imp.get("object", ""),
                    imp.get("relation_group", "ontology_implied"),
                    "ontology",
                    float(imp.get("confidence", 0.7)),
                    f"Ontology implied fact of {cid}",
                    needs_validation=bool(imp.get("needs_validation", False)),
                )
        if len(graph) == before:
            break
    return graph


def apply_propagation(graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before = list(graph)

    # Hazardous voltage + accessible conductive part + unknown protection -> electric shock pathway
    hv = graph_has_any_concept(graph, ["power_supply_220v", "hazardous_voltage", "mains_voltage", "mains_supply"])
    accessible_conductive = (
        graph_has_any_concept(graph, ["metal_housing", "conductive_material", "user_accessible_metal_housing"])
        and (has_fact(graph, "metal_housing", "user_accessible", "user") or has_fact(graph, "*", "user_accessible", "user"))
    )
    protection_known = graph_has_any_concept(graph, ["protective_earth", "double_insulation", "reinforced_insulation"])

    if hv and accessible_conductive and not protection_known:
        add_fact(
            graph, "system", "has_potential_hazard_pattern", "electric_shock_accessible_conductive_path",
            "causal_hazard", "propagation_rule", 0.70,
            "Hazardous voltage and accessible conductive housing are present; protection concept not found.",
            rule="PR_HAZARDOUS_VOLTAGE_ACCESSIBLE_CONDUCTIVE_PATH", needs_validation=True,
        )
        add_fact(graph, "system", "has_property", "unknown_touch_protection", "missing_info", "propagation_rule", 0.65, rule="PR_UNKNOWN_PROTECTION", needs_validation=True)

    # Fire triangle
    oxidizer = graph_has_any_concept(graph, ["air", "oxygen", "oxidizer", "normal_air"])
    fuel = graph_has_any_concept(graph, ["plastic", "plastic_part", "crumbs", "food_residue", "possible_fuel", "combustible_material"])
    ignition = graph_has_any_concept(graph, ["heating_element", "hot_surface", "spark", "arc", "relay", "ignition_source"])
    local = has_fact(graph, "*", "near", "heating_element") or has_fact(graph, "heating_element", "located_inside", "bread_slot") or has_fact(graph, "bread_slot", "near", "heating_element")
    if oxidizer and fuel and ignition and local:
        add_fact(
            graph, "system", "has_potential_hazard_pattern", "fire_triangle",
            "causal_hazard", "propagation_rule", 0.75,
            "Oxidizer, possible fuel and ignition source are present in local context.",
            rule="PR_FIRE_TRIANGLE", needs_validation=True,
        )

    # Burn injury
    hot = graph_has_any_concept(graph, ["heating_element", "hot_surface", "high_temperature"])
    user_access = has_fact(graph, "*", "user_accessible", "user")
    if hot and user_access:
        add_fact(
            graph, "system", "has_potential_hazard_pattern", "burn_from_accessible_hot_part",
            "causal_hazard", "propagation_rule", 0.62,
            "Hot element/surface exists and user-accessible opening or housing exists.",
            rule="PR_HOT_SURFACE_USER_ACCESS", needs_validation=True,
        )

    # Moisture + mains
    if hv and graph_has_any_concept(graph, ["moisture", "cleaning_liquid", "water"]):
        add_fact(
            graph, "system", "has_potential_hazard_pattern", "mains_moisture_shock_fire",
            "causal_hazard", "propagation_rule", 0.72,
            "Mains voltage and moisture/cleaning liquid exposure are present.",
            rule="PR_MAINS_MOISTURE", needs_validation=True,
        )

    # Metal object insertion
    if graph_has_concept(graph, "metal_object") and graph_has_concept(graph, "bread_slot") and hv:
        add_fact(
            graph, "system", "has_potential_hazard_pattern", "metal_object_insertion_shock_short",
            "causal_hazard", "propagation_rule", 0.80,
            "Conductive object can be inserted into bread slot in a mains-powered toaster.",
            rule="PR_METAL_OBJECT_INSERTION", needs_validation=True,
        )

    return [f for f in graph if f not in before]


# ============================================================
# Fault-tree template matching
# ============================================================

def match_event(match: Dict[str, Any], graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = []
    if not match:
        return {"matched": False, "status": "no_evidence", "confidence_score": 0.0, "evidence": []}

    if "concepts_any" in match:
        for c in match.get("concepts_any", []):
            if graph_has_concept(graph, c):
                evidence.append(f"concept found: {c}")
                return {"matched": True, "status": "matched", "confidence_score": 0.85, "evidence": evidence}

    if "relations_any" in match:
        for pat in match.get("relations_any", []):
            ev = find_relation_matches(graph, pat)
            if ev:
                return {"matched": True, "status": "matched", "confidence_score": 0.85, "evidence": ev}

    if "hazard_pattern_any" in match:
        for p in match.get("hazard_pattern_any", []):
            if has_fact(graph, "system", "has_potential_hazard_pattern", p):
                evidence.append(f"hazard pattern found: {p}")
                return {"matched": True, "status": "matched", "confidence_score": 0.80, "evidence": evidence}

    return {"matched": False, "status": "missing", "confidence_score": 0.0, "evidence": []}


def evaluate_node(node: Dict[str, Any], graph: List[Dict[str, Any]]) -> Dict[str, Any]:
    ntype = node.get("type", "EVENT")
    label = node.get("label", node.get("node_id", ""))

    if ntype == "EVENT":
        res = match_event(node.get("match", {}), graph)
        res.update({"node_id": node.get("node_id"), "label": label, "type": ntype, "missing_if_absent": node.get("missing_if_absent", [])})
        if not res["matched"] and node.get("default_if_missing_information"):
            res["matched"] = True
            res["status"] = "assumed_unknown"
            res["confidence_score"] = 0.45
            res["evidence"] = ["No protection evidence found; treated as unknown for early warning."]
        return res

    child_results = [evaluate_node(c, graph) for c in node.get("children", [])]
    evidence = []
    missing = []
    for r in child_results:
        evidence.extend(r.get("evidence", []))
        if not r.get("matched"):
            missing.extend(r.get("missing_if_absent", []))
            missing.append(r.get("label", "missing branch"))

    if ntype == "AND":
        matched_count = sum(1 for r in child_results if r.get("matched"))
        total = len(child_results) or 1
        matched = matched_count == total
        partial = matched_count > 0
        score = sum(r.get("confidence_score", 0.0) for r in child_results) / total
        status = "confirmed_context" if matched and score >= 0.8 else "strong_potential" if matched else "potential_pathway" if partial else "no_evidence"
        return {"matched": matched or partial, "status": status, "confidence_score": score, "evidence": evidence, "missing": missing, "children": child_results, "label": label, "type": ntype}

    if ntype == "OR":
        matched_children = [r for r in child_results if r.get("matched")]
        if matched_children:
            best = max(matched_children, key=lambda x: x.get("confidence_score", 0.0))
            return {"matched": True, "status": best.get("status", "matched"), "confidence_score": best.get("confidence_score", 0.0), "evidence": best.get("evidence", []), "missing": [], "children": child_results, "label": label, "type": ntype}
        return {"matched": False, "status": "missing", "confidence_score": 0.0, "evidence": [], "missing": missing, "children": child_results, "label": label, "type": ntype}

    return {"matched": False, "status": "no_evidence", "confidence_score": 0.0, "evidence": [], "missing": []}


def confidence_label(score: float) -> str:
    if score >= 0.78:
        return "High"
    if score >= 0.50:
        return "Medium"
    if score > 0:
        return "Low"
    return "None"


def match_fault_trees(graph: List[Dict[str, Any]], templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for t in templates.get("templates", []):
        result = evaluate_node(t.get("fault_tree", {}), graph)
        if not result.get("matched"):
            continue
        score = round(result.get("confidence_score", 0.0), 2)
        status = result.get("status", "potential_pathway")
        if status == "no_evidence":
            continue
        missing = list(dict.fromkeys((result.get("missing", []) or []) + t.get("missing_information", [])))
        evidence = list(dict.fromkeys(result.get("evidence", []) or []))
        rows.append({
            "hazard_id": t.get("id"),
            "Hazard": t.get("title"),
            "LLM or Deterministic": "Deterministic fault-tree",
            "Rationale why this hazard is possible": t.get("reasoning", "") + " Evidence: " + "; ".join(evidence[:8]) + f" Status: {status}.",
            "Confidence level": confidence_label(score),
            "Confidence score": score,
            "Missing information": "; ".join(missing[:12]),
        })
    return rows


# ============================================================
# LLM backward investigation
# ============================================================

def llm_backward(api_key: str, model: str, description: str, graph: List[Dict[str, Any]], templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key)
    compact_graph = [{k: f.get(k) for k in ["subject", "relation", "object", "source", "confidence", "needs_validation", "evidence"]} for f in graph]
    compact_templates = [{"id": t.get("id"), "title": t.get("title"), "top_event": t.get("top_event"), "reasoning": t.get("reasoning"), "missing_information": t.get("missing_information", [])} for t in templates.get("templates", [])]
    prompt = f"""
You are doing backward safety investigation for an early toaster design.

Task:
Start from each hazard template and investigate whether the design graph contains ingredients making this hazard possible.

Rules:
- Use the graph as main evidence.
- Use the raw description only as supporting evidence.
- Do not invent facts.
- Do not claim a hazard is confirmed unless the graph explicitly supports it.
- Use status: confirmed_context, strong_potential, potential_pathway, partial_evidence, no_evidence.
- Return valid JSON only.

Raw design description:
{description}

System graph:
{as_json(compact_graph)}

Hazard templates:
{as_json(compact_templates)}

Return JSON:
{{
  "results": [
    {{
      "hazard_id": "",
      "hazard": "",
      "status": "confirmed_context|strong_potential|potential_pathway|partial_evidence|no_evidence",
      "rationale": "",
      "confidence": "low|medium|high",
      "missing_information": []
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return JSON only. Be conservative."},
            {"role": "user", "content": prompt},
        ],
    )
    parsed = try_parse_json(response.choices[0].message.content or "{}")
    rows = []
    for r in parsed.get("results", []):
        if r.get("status") == "no_evidence":
            continue
        conf = str(r.get("confidence", "medium")).capitalize()
        rows.append({
            "hazard_id": r.get("hazard_id", ""),
            "Hazard": r.get("hazard", r.get("hazard_id", "")),
            "LLM or Deterministic": "LLM backward",
            "Rationale why this hazard is possible": f"Status: {r.get('status')}. {r.get('rationale', '')}",
            "Confidence level": conf,
            "Confidence score": {"Low": 0.35, "Medium": 0.6, "High": 0.8}.get(conf, 0.6),
            "Missing information": "; ".join(r.get("missing_information", [])),
        })
    return rows


# ============================================================
# Norm resolution
# ============================================================

def resolve_norms(hazard_rows: List[Dict[str, Any]], standards: Dict[str, Any]) -> List[Dict[str, Any]]:
    hazard_ids = {r.get("hazard_id") for r in hazard_rows if r.get("hazard_id")}
    rows = []
    for sid, s in standards.get("standards", {}).items():
        matched = []
        level = None
        confidence = 0.0
        for link in s.get("links", []):
            if link.get("hazard_id") in hazard_ids:
                matched.append(link)
                # Keep strongest level roughly
                lvl = link.get("level", "conditional")
                if level is None or lvl == "primary" or (lvl == "conditional" and level == "supporting"):
                    level = lvl
                confidence = max(confidence, float(link.get("confidence", 0.5)))
        if not matched:
            continue
        rows.append({
            "Norm / regulation": s.get("display_name", sid),
            "Hazards addressed": "; ".join(sorted({m.get("hazard") or m.get("hazard_id") for m in matched})),
            "Level of relation": level or "conditional",
            "Confidence": confidence_label(confidence),
            "Confidence score": round(confidence, 2),
            "Rationale": s.get("rationale", "") + " " + " ".join(m.get("rationale", "") for m in matched),
            "Evidence source": "standards_catalogue.yaml",
        })
    return rows


# ============================================================
# Visualization
# ============================================================

def graph_to_dot(graph: List[Dict[str, Any]]) -> str:
    def nid(x):
        return "n_" + normalize_id(x)
    colors = {
        "classification": "gray",
        "structural": "blue",
        "human_interaction": "darkgreen",
        "material": "brown",
        "energy": "orange",
        "causal_hazard": "red",
        "ontology_property": "gray",
        "hazard_role": "red",
        "exposure": "purple",
        "missing_info": "gray",
    }
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="transparent"];',
        '  node [shape=box, style="rounded,filled", fillcolor="white", color="#444444", fontname="Arial"];',
        '  edge [fontname="Arial", color="#555555"];',
    ]
    for n in sorted(graph_nodes(graph)):
        lines.append(f'  {nid(n)} [label="{str(n).replace("_", " ")}"];')
    for f in graph:
        s, o, r = f.get("subject"), f.get("object"), f.get("relation")
        if not s or not o:
            continue
        color = colors.get(f.get("relation_group"), "black")
        style = "dashed" if f.get("needs_validation") else "solid"
        lines.append(f'  {nid(s)} -> {nid(o)} [label="{r}", color="{color}", style="{style}"];')
    lines.append("}")
    return "\n".join(lines)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Safety Co-Pilot Toaster PoC", layout="wide")

st.title("Safety Co-Pilot PoC — Toaster")
st.markdown(
    """
This demo performs **Early Design Hazard Awareness** for a toaster design.

```text
system description → LLM normalization → Safety Context Graph → fault-tree hazard matching → backward investigation → norms table
```
"""
)

api_key = get_api_key()

with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    use_fallback = st.checkbox("Use fallback extractor if no API key", value=True)
    use_backward = st.checkbox("Run LLM backward investigation", value=True)

    st.header("Upload YAML artifacts")
    ontology_upload = st.file_uploader("ontology.yaml", type=["yaml", "yml"])
    relations_upload = st.file_uploader("relations.yaml", type=["yaml", "yml"])
    templates_upload = st.file_uploader("fault_tree_hazard_templates.yaml", type=["yaml", "yml"])
    standards_upload = st.file_uploader("standards_catalogue.yaml", type=["yaml", "yml"])

ontology = load_yaml(ontology_upload, "data/ontology.yaml")
relations = load_yaml(relations_upload, "data/relations.yaml")
templates = load_yaml(templates_upload, "data/fault_tree_hazard_templates.yaml")
standards = load_yaml(standards_upload, "data/standards_catalogue.yaml")

default_description = """The product is a household toaster.
The toaster is powered by 220 V mains.
The heating element is located inside the bread slot.
The housing is made of stainless steel and can be touched by the user.
The toaster has a plastic lever near the heating area.
Crumbs can accumulate in the bread slot.
The user may clean the housing with a damp cloth."""

st.subheader("1. System description")
description = st.text_area("Input or edit the toaster design description. Press Proceed when ready.", value=default_description, height=190)

if st.button("Proceed", type="primary"):
    if not api_key and not use_fallback:
        st.error("No OPENAI_API_KEY found. Add it to Streamlit Secrets or enable fallback extractor.")
        st.stop()

    with st.spinner("Normalizing description, building graph and matching fault-tree templates..."):
        if api_key:
            normalized = llm_normalize(api_key, model, description, ontology, relations)
            source = "LLM"
        else:
            normalized = fallback_normalize(description)
            source = "Fallback extractor"

        graph = build_graph(normalized, ontology)
        derived = apply_propagation(graph)
        det_rows = match_fault_trees(graph, templates)
        llm_rows = []
        if api_key and use_backward:
            llm_rows = llm_backward(api_key, model, description, graph, templates)
        hazard_rows = det_rows + llm_rows
        norm_rows = resolve_norms(hazard_rows, standards)

    st.success(f"Analysis complete. Normalization source: {source}")

    st.subheader("2. Normalized system description")
    st.json(normalized)

    st.subheader("3. Safety Context Graph")
    tab_graph, tab_triples, tab_derived = st.tabs(["Graph view", "Triples", "Derived facts"])
    with tab_graph:
        try:
            st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
        except Exception as e:
            st.warning(f"Graph rendering failed: {e}")
            st.code(graph_to_dot(graph))
    with tab_triples:
        st.dataframe(pd.DataFrame(graph), use_container_width=True, hide_index=True)
    with tab_derived:
        if derived:
            st.dataframe(pd.DataFrame(derived), use_container_width=True, hide_index=True)
        else:
            st.info("No derived facts were added.")

    st.subheader("4. Hazard analysis table")
    if hazard_rows:
        hazard_df = pd.DataFrame(hazard_rows)
        display_df = hazard_df[["Hazard", "LLM or Deterministic", "Rationale why this hazard is possible", "Confidence level", "Missing information"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button("Download hazard table as CSV", display_df.to_csv(index=False).encode("utf-8"), "toaster_hazard_table.csv", "text/csv")
    else:
        st.info("No hazard patterns found.")

    st.subheader("5. Potentially applicable norms table")
    if norm_rows:
        norm_df = pd.DataFrame(norm_rows)
        st.dataframe(norm_df, use_container_width=True, hide_index=True)
        st.download_button("Download norms table as CSV", norm_df.to_csv(index=False).encode("utf-8"), "toaster_norms_table.csv", "text/csv")
    else:
        st.info("No norm candidates resolved from the standards catalogue.")

else:
    st.info("Edit the design description or upload YAML artifacts, then press Proceed.")
