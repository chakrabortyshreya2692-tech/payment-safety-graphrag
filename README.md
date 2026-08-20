# Payment Safety Assistant

## GraphRAG-Driven Conversational LLM for Explainable Payment Fraud Prevention

This repository contains the implementation developed for a Master's dissertation at the University of Limerick.

The project demonstrates a GraphRAG-driven conversational approach to explainable payment fraud prevention. It combines Neo4j graph evidence, a deterministic fraud-risk engine, a hosted Large Language Model (LLM), evidence validation, and a Streamlit user interface.

The prototype uses synthetic payment transaction data and is intended for academic research and evaluation purposes only.

---

## Live Demo

**https://payment-safety-graphrag.streamlit.app/**

The demo allows a user to:
- enter a proposed payment;
- retrieve historical graph evidence from Neo4j;
- calculate an explainable fraud-risk score;
- view the transaction graph;
- receive grounded fraud-prevention guidance;
- ask follow-up questions through the conversational GraphRAG assistant.

---

## Source Code

**Repository URL:**  
`https://github.com/chakrabortyshreya2692-tech/payment-safety-graphrag/`

## System Architecture

```text
Proposed Payment
      |
      v
Streamlit User Interface
      |
      v
Neo4j Knowledge Graph
      |
      v
GraphRAG Evidence Retrieval
      |
      v
Deterministic Risk Engine
      |
      v
Risk Score + Risk Level + Risk Factors
      |
      v
Hosted Conversational LLM
      |
      v
Evidence Validation
   /             \
 PASS             FAIL
  |                |
  v                v
LLM Guidance   Deterministic
               Safe Fallback
      |
      v
Conversational Payment Safety Assistant
```

---

## Main Components

### `customer_app.py`
Main Streamlit application for transaction input, risk assessment display, graph visualisation, fraud-prevention guidance, validation, fallback handling, and conversational interaction.

### `neo4j_fraud_service.py`
Connects to Neo4j Aura and retrieves graph evidence relevant to a proposed transaction.

### `neo4j_risk_engine.py`
Calculates the deterministic fraud-risk score and produces the risk level, risk factors, assessment status, evidence confidence, and recommended action.

The LLM does **not** calculate or modify the risk score.

### `customer_guidance.py`
Generates customer-facing payment-safety guidance using only approved evidence and validates generated responses.

### `hosted_llm.py`
Connects the application to OpenRouter.

For final dissertation evaluation, use the fixed model:

`openai/gpt-oss-20b:free`

Recommended evaluation setting:

`temperature = 0.0`

---

## Evidence Validation and Hallucination Control

The LLM is instructed to use only approved evidence supplied by the application.

The validation layer checks that:
- the deterministic risk score is preserved;
- the risk level is preserved;
- approved risk factors are retained;
- unsupported fraud observations are not introduced;
- the recommended action is preserved;
- unsupported external reputation information is not introduced;
- the LLM does not claim that a transaction is definitely fraudulent or definitely safe.

If generation fails or validation does not pass, deterministic grounded guidance is shown instead.

---

## Example Questions

```text
Why is this payment risky?
Why is the risk level Medium?
What should I check before paying?
What evidence is available about this merchant?
Is this payment definitely safe?
```

If the supplied graph evidence is insufficient, the assistant is instructed not to invent information.

---

## Installation

```bash
pip install -r requirements.txt
```

Use the repository's `requirements.txt` as the authoritative dependency list.

---

## Configuration

The application requires Neo4j Aura and an OpenRouter API key.

For Streamlit deployment:

```toml
NEO4J_URI = "neo4j+s://YOUR-INSTANCE.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "YOUR-NEO4J-PASSWORD"

OPENROUTER_API_KEY = "YOUR-OPENROUTER-API-KEY"
```

Do not commit real credentials to GitHub.

---

## Running Locally

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
pip install -r requirements.txt
streamlit run customer_app.py
```

---

## Evaluation

The final system can be evaluated using:
- risk classification performance;
- evidence coverage;
- risk-factor correctness;
- evidence-validation pass rate;
- grounded-response accuracy;
- unsupported-claim rate;
- fallback rate;
- conversational consistency;
- end-to-end response latency.

A repeated-question consistency experiment can be performed using the fixed model to verify that the conversational assistant preserves the same graph-supported evidence across repeated runs.

---

## Data

The prototype uses synthetic payment transaction data for research and evaluation. Data preprocessing, mapping, imputation, and dataset integration procedures are documented separately in the dissertation and auxiliary materials.

---

## Security

Do not commit:

```text
.env
.streamlit/secrets.toml
OPENROUTER_API_KEY
NEO4J_PASSWORD
```

Use `.gitignore` to prevent accidental upload of secret files.

---

## Research Scope and Disclaimer

This software is a research prototype developed for academic evaluation.

It is **not a production financial decision system** and should not be used as the sole basis for approving, rejecting, or blocking real financial transactions.

---

## Reproducibility

Preserve the final dissertation implementation as a tagged GitHub release:

`v1.0-dissertation-submission`

The tagged version should correspond to the implementation used to generate the results reported in the dissertation.

---

## Academic Context

Master's Dissertation  
University of Limerick

This repository accompanies the dissertation and is provided for academic assessment, reproducibility, and demonstration purposes.
