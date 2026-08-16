
import json


CUSTOMER_SYSTEM_PROMPT = """
You are a customer-facing payment-safety assistant.

Use ONLY the approved evidence supplied by the application.

Do not call the risk score a probability.
Do not guarantee that a payment or merchant is safe.
Do not invent fraud facts.
Do not mention external review websites or organisations that were not supplied.

For verified observations, use only the approved risk factors.
Give short, practical payment-safety guidance.
""".strip()


def generate_customer_safe_payment_guidance(
    transaction,
    graph_context,
    customer_risk_result,
    llm,
):
    approved_factors = [
        factor["description"]
        for factor in customer_risk_result["risk_factors"]
    ]

    evidence = {
        "transaction_id": transaction.get("transaction_id"),
        "merchant_id": transaction.get("merchant_id"),
        "payment_method": transaction.get("payment_method"),
        "country": transaction.get("country"),
        "rule_based_customer_risk_score":
            customer_risk_result["risk_score"],
        "risk_level":
            customer_risk_result["risk_level"],
        "assessment_status":
            customer_risk_result["assessment_status"],
        "evidence_confidence":
            customer_risk_result["decision_confidence"],
        "approved_risk_factors":
            approved_factors,
        "missing_evidence":
            customer_risk_result["missing_evidence"],
        "recommended_action":
            customer_risk_result["recommended_action"],
    }

    prompt = f"""
Create a customer-facing payment safety response using ONLY this evidence:

{json.dumps(evidence, indent=2, default=str)}

Use these headings:

Customer Payment Safety Check

Rule-based risk score: <score>/100
Risk level: <level>

Verified observations:
- Copy each approved risk-factor description exactly.

What you should check before paying:
- Give generic payment-safety checks only.

Recommended next action:
<copy the supplied recommended action exactly>
""".strip()

    return llm.generate(
        prompt=prompt,
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
        max_new_tokens=700,
        temperature=0.0,
    )


def _extract_bullets(text, start_heading, end_heading):
    lower = text.lower()

    start = lower.find(start_heading.lower())

    if start == -1:
        return []

    start += len(start_heading)

    end = lower.find(
        end_heading.lower(),
        start,
    )

    segment = (
        text[start:]
        if end == -1
        else text[start:end]
    )

    return [
        line.strip()[1:].strip()
        for line in segment.splitlines()
        if line.strip().startswith("-")
    ]


def validate_customer_audience_guidance(
    text,
    risk_result,
):
    violations = []

    expected_factors = {
        factor["description"]
        for factor in risk_result["risk_factors"]
    }

    if str(risk_result["risk_score"]) not in text:
        violations.append(
            "Customer risk score not preserved."
        )

    if (
        risk_result["risk_level"].lower()
        not in text.lower()
    ):
        violations.append(
            "Customer risk level not preserved."
        )

    if risk_result["recommended_action"] not in text:
        violations.append(
            "Recommended action not preserved."
        )

    bullets = set(
        _extract_bullets(
            text,
            "Verified observations:",
            "What you should check before paying:",
        )
    )

    if expected_factors:
        unsupported = bullets - expected_factors
        omitted = expected_factors - bullets

        if unsupported:
            violations.append(
                "Unsupported observation(s): "
                + " | ".join(sorted(unsupported))
            )

        if omitted:
            violations.append(
                "Approved factor(s) omitted: "
                + " | ".join(sorted(omitted))
            )

    forbidden_terms = [
        "trustpilot",
        "better business bureau",
        "google reviews",
        "secretary of state",
    ]

    for term in forbidden_terms:
        if term in text.lower():
            violations.append(
                f"Unsupported external reference: {term}"
            )

    return {
        "accepted": len(violations) == 0,
        "violations": violations,
    }


def deterministic_customer_audience_fallback(
    risk_result,
):
    factors = risk_result["risk_factors"]

    if factors:
        observations = "\n".join(
            f"- {factor['description']}"
            for factor in factors
        )
    else:
        observations = (
            "- No configured risk rules were triggered "
            "using the available evidence."
        )

    return f"""Customer Payment Safety Check

Rule-based risk score: {risk_result['risk_score']}/100
Risk level: {risk_result['risk_level']}

Verified observations:
{observations}

What you should check before paying:
- Independently verify the merchant or recipient.
- Confirm the amount and payment details.
- Avoid unexpected payment links.
- Contact your bank through an official channel if uncertain.

Recommended next action:
{risk_result['recommended_action']}"""

import json


CUSTOMER_CHAT_SYSTEM_PROMPT = """
You are an explainable payment-fraud prevention assistant.

A customer is considering a payment and may ask questions
about its risk assessment.

Use ONLY the approved evidence supplied to you.

The fraud-risk score has already been calculated by a
deterministic risk engine.

You must not:
- calculate or change the risk score,
- describe the score as a probability,
- claim the transaction is definitely fraudulent,
- claim the transaction is definitely safe,
- invent merchant history,
- invent device history,
- invent IP-address history,
- invent country history,
- invent fraud evidence,
- use external reputation information.

If the supplied evidence does not support the answer, say:

"The available graph evidence is insufficient to answer
this question."

Give a short, clear answer suitable for a customer.
""".strip()


def generate_customer_graphrag_answer(
    question,
    transaction,
    graph_context,
    risk_result,
    llm,
):

    approved_factors = [
        factor["description"]
        for factor
        in risk_result["risk_factors"]
    ]

    approved_evidence = {
        "transaction_id":
            transaction["transaction_id"],

        "transaction_amount":
            transaction["transaction_amount"],

        "country":
            transaction["country"],

        "payment_method":
            transaction["payment_method"],

        "channel":
            transaction["channel"],

        "risk_score":
            risk_result["risk_score"],

        "risk_level":
            risk_result["risk_level"],

        "assessment_status":
            risk_result[
                "assessment_status"
            ],

        "decision_confidence":
            risk_result[
                "decision_confidence"
            ],

        "approved_risk_factors":
            approved_factors,

        "missing_evidence":
            risk_result[
                "missing_evidence"
            ],

        "recommended_action":
            risk_result[
                "recommended_action"
            ],
    }

    prompt = f"""
Customer question:

{question}

Approved evidence:

{json.dumps(
    approved_evidence,
    indent=2,
    default=str
)}

Answer using only the approved evidence.
""".strip()

    return llm.generate(
        prompt=prompt,
        system_prompt=
            CUSTOMER_CHAT_SYSTEM_PROMPT,
        max_new_tokens=600,
        temperature=0.0,
    )
