# GraphRAG-Driven Payment Safety Assistant

This project is a master's dissertation prototype for explainable payment fraud prevention.

The system combines:

- Neo4j knowledge graph
- GraphRAG-based contextual evidence retrieval
- Deterministic fraud-risk scoring
- Conversational LLM guidance
- Streamlit user interface

## Main Features

- Assess a proposed payment before completion
- Retrieve historical fraud evidence from Neo4j
- Calculate a deterministic risk score
- Display risk level and risk indicators
- Generate customer-friendly fraud prevention guidance
- Provide conversational explanations grounded in graph evidence

## Technology Stack

- Python
- Streamlit
- Neo4j Aura
- NetworkX
- Large Language Model
- GraphRAG

## Project Structure

```text
customer_app.py
neo4j_fraud_service.py
neo4j_risk_engine.py
customer_guidance.py
local_llm.py
requirements.txt
README.md
.gitignore
