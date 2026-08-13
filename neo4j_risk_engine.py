from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class RiskFactor:
    code: str
    description: str
    points: int
    evidence: dict[str, Any]


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse_hour(timestamp_value: Any) -> int | None:
    if timestamp_value is None:
        return None

    if hasattr(timestamp_value, "hour"):
        try:
            return int(timestamp_value.hour)
        except Exception:
            pass

    text = str(timestamp_value).strip()
    if not text:
        return None

    # ISO-8601 first
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).hour
    except ValueError:
        pass

    # Common fallbacks
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).hour
        except ValueError:
            continue

    return None


def calculate_risk(
    transaction: dict[str, Any],
    graph_context: dict[str, Any],
    *,
    amount_threshold: float = 1000.0,
    max_risk_points: int = 110,
    unusual_start_hour: int = 0,
    unusual_end_hour: int = 5,
    high_risk_countries: set[str] | None = None,
    risk_level_thresholds: dict[str, float] | None = None,
    min_evidence_coverage_for_assessment: float = 0.60,
) -> dict[str, Any]:
    """
    Use-case-aligned deterministic rule engine.

    Six rules:
      +20 high amount
      +20 first transaction with merchant
      +15 unfamiliar device
      +15 cross-border OR approved high-risk country
      +10 unusual time
      +30 merchant linked to previous fraud cases

    Fraud score (%) = triggered points / 110 * 100.
    """
    high_risk_countries = {
        str(x).strip().upper() for x in (high_risk_countries or set())
    }
    risk_level_thresholds = risk_level_thresholds or {
        "Critical": 75.0,
        "High": 50.0,
        "Medium": 25.0,
    }

    factors: list[RiskFactor] = []
    missing_evidence: list[str] = []
    available_checks = 0
    total_checks = 6

    # RULE 1 — amount threshold (+20)
    amount = _to_float(transaction.get("transaction_amount"))
    if amount is None:
        missing_evidence.append("Transaction amount unavailable.")
    else:
        available_checks += 1
        if amount > float(amount_threshold):
            factors.append(
                RiskFactor(
                    "HIGH_AMOUNT",
                    f"Transaction amount exceeds the configured threshold of {amount_threshold:g}.",
                    20,
                    {"transaction_amount": amount, "threshold": amount_threshold},
                )
            )

    # RULE 2 — first transaction with merchant (+20)
    known_merchant = graph_context.get("known_merchant")
    customer_found = bool(graph_context.get("customer_found"))
    if known_merchant is None or not customer_found:
        missing_evidence.append("Merchant familiarity could not be established.")
    else:
        available_checks += 1
        if known_merchant is False:
            factors.append(
                RiskFactor(
                    "FIRST_MERCHANT_TRANSACTION",
                    "This is the customer's first observed transaction with this merchant.",
                    20,
                    {"known_merchant": False},
                )
            )

    # RULE 3 — unfamiliar device (+15)
    familiar_device = graph_context.get("familiar_device")
    if familiar_device is None or not customer_found:
        missing_evidence.append("Device familiarity could not be established.")
    else:
        available_checks += 1
        if familiar_device is False:
            factors.append(
                RiskFactor(
                    "UNFAMILIAR_DEVICE",
                    "The device has not been observed in the customer's previous transactions.",
                    15,
                    {"familiar_device": False},
                )
            )

    # RULE 4 — cross-border OR configured high-risk country (+15)
    current_country = transaction.get("country")
    previous_countries = {
        str(x).strip().upper()
        for x in (graph_context.get("previous_countries") or [])
        if x is not None and str(x).strip()
    }

    if not current_country:
        missing_evidence.append("Transaction country unavailable.")
    elif not customer_found:
        missing_evidence.append("Customer history unavailable for cross-border comparison.")
    else:
        current_country_norm = str(current_country).strip().upper()
        is_high_risk_country = current_country_norm in high_risk_countries

        # With no historical countries, cross-border status is unknown.
        cross_border_known = len(previous_countries) > 0
        is_cross_border = (
            cross_border_known
            and current_country_norm not in previous_countries
        )

        if cross_border_known or is_high_risk_country:
            available_checks += 1
        else:
            missing_evidence.append(
                "No prior country history is available for cross-border comparison."
            )

        if is_cross_border or is_high_risk_country:
            reason = (
                "The transaction country has not appeared in the customer's prior observed country history."
                if is_cross_border
                else "The transaction country is in the configured high-risk-country list."
            )
            factors.append(
                RiskFactor(
                    "CROSS_BORDER_OR_HIGH_RISK_COUNTRY",
                    reason,
                    15,
                    {
                        "country": current_country_norm,
                        "previous_countries": sorted(previous_countries),
                        "configured_high_risk_country": is_high_risk_country,
                    },
                )
            )

    # RULE 5 — unusual time (+10)
    hour = _parse_hour(transaction.get("timestamp"))
    if hour is None:
        missing_evidence.append("Transaction time unavailable or unparseable.")
    else:
        available_checks += 1
        is_unusual = unusual_start_hour <= hour <= unusual_end_hour
        if is_unusual:
            factors.append(
                RiskFactor(
                    "UNUSUAL_TIME",
                    "The transaction occurred during the configured unusual-time window.",
                    10,
                    {
                        "hour": hour,
                        "unusual_start_hour": unusual_start_hour,
                        "unusual_end_hour": unusual_end_hour,
                    },
                )
            )

    # RULE 6 — merchant linked to previous fraud cases (+30)
    if not graph_context.get("merchant_found"):
        missing_evidence.append("Merchant history unavailable.")
    else:
        previous_fraud_count = int(
            graph_context.get("previous_fraud_transactions_at_merchant") or 0
        )
        historical_case_links = int(
            graph_context.get("historical_fraud_case_links") or 0
        )
        available_checks += 1

        if previous_fraud_count > 0 or historical_case_links > 0:
            factors.append(
                RiskFactor(
                    "MERCHANT_PREVIOUS_FRAUD",
                    "The merchant is linked to previous fraud evidence in the graph.",
                    30,
                    {
                        "previous_fraud_transactions_at_merchant": previous_fraud_count,
                        "historical_fraud_case_links": historical_case_links,
                    },
                )
            )

    raw_points = sum(f.points for f in factors)
    fraud_score = round((raw_points / max_risk_points) * 100, 1)

    if fraud_score >= risk_level_thresholds["Critical"]:
        risk_level = "Critical"
    elif fraud_score >= risk_level_thresholds["High"]:
        risk_level = "High"
    elif fraud_score >= risk_level_thresholds["Medium"]:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if risk_level == "Critical":
        recommended_action = (
            "Do not proceed until the payment and recipient have been independently verified."
        )
    elif risk_level == "High":
        recommended_action = (
            "Pause the payment and independently verify the merchant, device and payment details."
        )
    elif risk_level == "Medium":
        recommended_action = (
            "Review the payment details and verify anything unfamiliar before proceeding."
        )
    else:
        recommended_action = (
            "No configured high-risk rule was triggered; still verify the payment details before proceeding."
        )

    evidence_coverage = round(available_checks / total_checks, 3)
    if evidence_coverage >= 0.85:
        decision_confidence = "High"
    elif evidence_coverage >= min_evidence_coverage_for_assessment:
        decision_confidence = "Medium"
    else:
        decision_confidence = "Low"

    assessment_status = (
        "Insufficient evidence"
        if evidence_coverage < min_evidence_coverage_for_assessment
        else "Assessable"
    )

    return {
        "raw_risk_points": raw_points,
        "max_risk_points": max_risk_points,
        "risk_score": fraud_score,
        "risk_score_interpretation": (
            "Rule-based fraud risk score; not a calibrated probability of fraud."
        ),
        "risk_level": risk_level,
        "assessment_status": assessment_status,
        "risk_factors": [asdict(x) for x in factors],
        "missing_evidence": missing_evidence,
        "available_checks": available_checks,
        "total_checks": total_checks,
        "evidence_coverage": evidence_coverage,
        "decision_confidence": decision_confidence,
        "recommended_action": recommended_action,
    }
