# Safety Co-Pilot Streamlit PoC — Toaster

This is a Streamlit Cloud proof of concept for **Early Design Hazard Awareness** using a toaster example.

## Workflow

```text
system description
→ LLM normalization
→ Safety Context Graph
→ deterministic fault-tree hazard matching
→ optional LLM backward investigation
→ hazard table
→ potentially applicable norms table
```

## Files

```text
app.py
requirements.txt
README.md

data/
  ontology.yaml
  relations.yaml
  fault_tree_hazard_templates.yaml
  standards_catalogue.yaml

.streamlit/
  config.toml
  secrets.toml.example
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add `.streamlit/secrets.toml` locally:

```toml
OPENAI_API_KEY = "sk-..."
```

For Streamlit Cloud, put the same secret into the app settings.

## Suggested test input

```text
The product is a household toaster.
The toaster is powered by 220 V mains.
The heating element is located inside the bread slot.
The housing is made of stainless steel and can be touched by the user.
The toaster has a plastic lever near the heating area.
Crumbs can accumulate in the bread slot.
The user may clean the housing with a damp cloth.
```
