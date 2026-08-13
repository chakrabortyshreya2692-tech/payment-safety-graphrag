from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


class Neo4jFraudService:
    """
    Read-only Neo4j Aura service for the implemented payment graph.

    Important:
    Transaction property names are detected from the live Neo4j database.
    This prevents warnings caused by referring to properties that do not exist,
    such as t.timestamp or t.amount when the imported CSV used different names.
    """

    TIMESTAMP_CANDIDATES = [
        "timestamp",
        "transaction_timestamp",
        "transaction_time",
        "transaction_datetime",
        "datetime",
        "date_time",
        "transaction_date",
        "date",
        "time",
    ]

    AMOUNT_CANDIDATES = [
        "transaction_amount",
        "amount",
        "transaction_value",
        "value",
        "payment_amount",
    ]

    FRAUD_CANDIDATES = [
        "is_fraud",
        "fraud_label",
        "fraud",
        "isFraud",
        "is_fraudulent",
        "label",
    ]

    TRANSACTION_ID_CANDIDATES = [
        "transaction_id",
        "transactionId",
        "tx_id",
        "id",
    ]

    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        missing = [
            key for key, value in {
                "NEO4J_URI": uri,
                "NEO4J_USERNAME": username,
                "NEO4J_PASSWORD": password,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing Neo4j environment variable(s): "
                + ", ".join(missing)
            )

        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=15,
            max_connection_lifetime=300,
        )

        # Filled after connectivity is verified.
        self.transaction_property_map: dict[str, str | None] = {}

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def verify_connection(self) -> None:
        self.driver.verify_connectivity()
        self.transaction_property_map = (
            self.detect_transaction_property_map()
        )

    def run_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Cypher query cannot be empty.")

        try:
            with self.driver.session(database=self.database) as session:
                return [
                    dict(record)
                    for record in session.run(
                        query,
                        parameters or {},
                    )
                ]

        except ServiceUnavailable as exc:
            raise RuntimeError(
                "Neo4j Aura is unavailable. Check Aura status, URI, "
                "credentials and internet connection."
            ) from exc

        except Neo4jError as exc:
            raise RuntimeError(
                "Neo4j query failed: "
                + getattr(exc, "message", str(exc))
            ) from exc

    def get_schema_summary(self) -> dict[str, list[str]]:
        labels = self.run_query(
            "CALL db.labels() YIELD label "
            "RETURN label ORDER BY label"
        )
        rels = self.run_query(
            "CALL db.relationshipTypes() "
            "YIELD relationshipType "
            "RETURN relationshipType "
            "ORDER BY relationshipType"
        )
        props = self.run_query(
            "CALL db.propertyKeys() YIELD propertyKey "
            "RETURN propertyKey ORDER BY propertyKey"
        )

        return {
            "labels": [r["label"] for r in labels],
            "relationship_types": [
                r["relationshipType"] for r in rels
            ],
            "property_keys": [
                r["propertyKey"] for r in props
            ],
        }

    def get_transaction_property_keys(self) -> list[str]:
        """
        Read only property names that actually exist on Transaction nodes.
        No candidate property is referenced directly here, so this query
        cannot produce an unknown-property warning.
        """
        rows = self.run_query(
            """
            MATCH (t:Transaction)
            WITH collect(keys(t)) AS all_key_lists
            UNWIND all_key_lists AS key_list
            UNWIND key_list AS property_key
            RETURN DISTINCT property_key
            ORDER BY property_key
            """
        )
        return [r["property_key"] for r in rows]

    @staticmethod
    def _pick_property(
        available: set[str],
        candidates: list[str],
    ) -> str | None:
        # Exact match first.
        for candidate in candidates:
            if candidate in available:
                return candidate

        # Case-insensitive fallback.
        lower_lookup = {
            name.lower(): name for name in available
        }
        for candidate in candidates:
            actual = lower_lookup.get(candidate.lower())
            if actual:
                return actual

        return None

    def detect_transaction_property_map(
        self,
    ) -> dict[str, str | None]:
        available = set(
            self.get_transaction_property_keys()
        )

        mapping = {
            "timestamp": self._pick_property(
                available,
                self.TIMESTAMP_CANDIDATES,
            ),
            "amount": self._pick_property(
                available,
                self.AMOUNT_CANDIDATES,
            ),
            "fraud": self._pick_property(
                available,
                self.FRAUD_CANDIDATES,
            ),
            "transaction_id": self._pick_property(
                available,
                self.TRANSACTION_ID_CANDIDATES,
            ),
        }

        return mapping

    def ensure_property_mapping(self) -> None:
        if not self.transaction_property_map:
            self.transaction_property_map = (
                self.detect_transaction_property_map()
            )

        missing_required = [
            key
            for key in [
                "timestamp",
                "amount",
                "transaction_id",
            ]
            if not self.transaction_property_map.get(key)
        ]

        if missing_required:
            actual_keys = self.get_transaction_property_keys()
            raise RuntimeError(
                "Could not automatically identify required "
                "Transaction property name(s): "
                + ", ".join(missing_required)
                + ".\nActual Transaction properties are:\n"
                + ", ".join(actual_keys)
                + "\n\nUpdate the candidate lists in "
                  "neo4j_fraud_service.py if your property uses "
                  "a different name."
            )

    def _property_expression(
        self,
        alias: str,
        logical_name: str,
        required: bool = True,
    ) -> str:
        self.ensure_property_mapping()
        actual = self.transaction_property_map.get(
            logical_name
        )

        if not actual:
            if required:
                raise RuntimeError(
                    f"No Transaction property was detected "
                    f"for '{logical_name}'."
                )
            return "NULL"

        # The value comes only from keys already returned by Neo4j.
        escaped = actual.replace("`", "``")
        return f"{alias}.`{escaped}`"

    @staticmethod
    def empty_context() -> dict[str, Any]:
        return {
            "customer_found": False,
            "merchant_found": False,
            "device_found": False,
            "ip_found": False,
            "country_found": False,

            "familiar_device": None,
            "known_merchant": None,
            "known_payment_method": None,
            "previous_countries": [],

            "customer_previous_tx_count_window": 0,
            "customer_previous_amounts_window": [],

            "device_previous_tx_count": 0,
            "device_previous_customer_count": 0,
            "device_previous_fraud_count": 0,

            "ip_previous_tx_count": 0,
            "ip_previous_customer_count": 0,
            "ip_previous_fraud_count": 0,

            "previous_fraud_transactions_at_merchant": 0,

            # Optional conceptual enrichments are neutral defaults
            # unless genuine graph nodes are added later.
            "historical_fraud_case_links": 0,
            "known_scam_links": [],
            "bank_warning_links": [],
            "chargeback_pattern_links": [],
        }

    def get_transaction_context(
        self,
        transaction: dict[str, Any],
        velocity_window_minutes: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve only evidence available before the current transaction.

        The query is built using the Transaction property names detected from
        the live database, while returned Python keys remain standardised as:
        transaction_id, timestamp, transaction_amount and is_fraud.
        """
        self.ensure_property_mapping()

        required_values = [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "device_id",
            "country",
            "transaction_amount",
            "timestamp",
        ]

        missing = [
            key for key in required_values
            if transaction.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Transaction is missing required value(s): "
                + ", ".join(missing)
            )

        parameters = {
            "transaction_id": str(
                transaction["transaction_id"]
            ),
            "customer_id": str(
                transaction["customer_id"]
            ),
            "merchant_id": str(
                transaction["merchant_id"]
            ),
            "device_id": str(
                transaction["device_id"]
            ),
            "ip_address": (
                None
                if transaction.get("ip_address") in (None, "")
                else str(transaction.get("ip_address"))
            ),
            "country": str(transaction["country"]),
            "payment_method": (
                None
                if transaction.get("payment_method") in (None, "")
                else str(
                    transaction.get("payment_method")
                )
            ),
            "timestamp": str(
                transaction["timestamp"]
            ),
            "velocity_window_minutes": int(
                velocity_window_minutes
            ),
        }

        # Dynamic expressions use only real property keys returned
        # by Neo4j, avoiding unknown-property warnings.
        h_ts = self._property_expression("h", "timestamp")
        vt_ts = self._property_expression("vt", "timestamp")
        dt_ts = self._property_expression("dt", "timestamp")
        it_ts = self._property_expression("it", "timestamp")
        pt_ts = self._property_expression("pt", "timestamp")
        prev_ts = self._property_expression("prev", "timestamp")

        h_id = self._property_expression(
            "h",
            "transaction_id",
        )
        vt_id = self._property_expression(
            "vt",
            "transaction_id",
        )
        dt_id = self._property_expression(
            "dt",
            "transaction_id",
        )
        it_id = self._property_expression(
            "it",
            "transaction_id",
        )
        pt_id = self._property_expression(
            "pt",
            "transaction_id",
        )
        prev_id = self._property_expression(
            "prev",
            "transaction_id",
        )

        vt_amount = self._property_expression(
            "vt",
            "amount",
        )

        dt_fraud = self._property_expression(
            "dt",
            "fraud",
            required=False,
        )
        it_fraud = self._property_expression(
            "it",
            "fraud",
            required=False,
        )
        prev_fraud = self._property_expression(
            "prev",
            "fraud",
            required=False,
        )

        # If the graph has no fraud property, historical-fraud checks
        # return zero instead of referencing a missing property.
        dt_fraud_test = (
            "false"
            if dt_fraud == "NULL"
            else (
                f"toLower(toString({dt_fraud})) IN "
                '["true","1","yes","fraud","fraudulent"]'
            )
        )
        it_fraud_test = (
            "false"
            if it_fraud == "NULL"
            else (
                f"toLower(toString({it_fraud})) IN "
                '["true","1","yes","fraud","fraudulent"]'
            )
        )
        prev_fraud_test = (
            "false"
            if prev_fraud == "NULL"
            else (
                f"toLower(toString({prev_fraud})) IN "
                '["true","1","yes","fraud","fraudulent"]'
            )
        )

        query = f"""
        OPTIONAL MATCH
            (c:Customer {{customer_id: $customer_id}})
        OPTIONAL MATCH
            (m:Merchant {{merchant_id: $merchant_id}})
        OPTIONAL MATCH
            (d:Device {{device_id: $device_id}})
        OPTIONAL MATCH
            (ip:IPAddress {{ip_address: $ip_address}})
        OPTIONAL MATCH
            (co:Country {{country: $country}})

        CALL (c) {{
            OPTIONAL MATCH
                (c)-[:MADE]->(h:Transaction)
                   -[:OCCURRED_IN]->(pc:Country)
            WHERE {h_id} <> $transaction_id
              AND {h_ts} IS NOT NULL
              AND datetime(toString({h_ts}))
                    < datetime($timestamp)
            RETURN
                collect(DISTINCT pc.country)
                    AS previous_countries
        }}

        CALL (c) {{
            OPTIONAL MATCH
                (c)-[:MADE]->(h:Transaction)
                   -[:USED_DEVICE]->(kd:Device)
            WHERE {h_id} <> $transaction_id
              AND {h_ts} IS NOT NULL
              AND datetime(toString({h_ts}))
                    < datetime($timestamp)
              AND kd.device_id = $device_id
            RETURN count(kd) > 0 AS familiar_device
        }}

        CALL (c) {{
            OPTIONAL MATCH
                (c)-[:MADE]->(h:Transaction)
                   -[:PAID_TO]->(km:Merchant)
            WHERE {h_id} <> $transaction_id
              AND {h_ts} IS NOT NULL
              AND datetime(toString({h_ts}))
                    < datetime($timestamp)
              AND km.merchant_id = $merchant_id
            RETURN count(km) > 0 AS known_merchant
        }}

        CALL (m) {{
            OPTIONAL MATCH
                (prev:Transaction)-[:PAID_TO]->(m)
            WHERE {prev_id} <> $transaction_id
              AND {prev_ts} IS NOT NULL
              AND datetime(toString({prev_ts}))
                    < datetime($timestamp)
              AND {prev_fraud_test}
            RETURN
                count(prev)
                    AS previous_fraud_transactions_at_merchant
        }}

        CALL (c) {{
            OPTIONAL MATCH
                (c)-[:MADE]->(vt:Transaction)
            WHERE {vt_id} <> $transaction_id
              AND {vt_ts} IS NOT NULL
              AND datetime(toString({vt_ts}))
                    < datetime($timestamp)
              AND datetime(toString({vt_ts}))
                    >= datetime($timestamp)
                    - duration({{
                        minutes:
                            $velocity_window_minutes
                    }})
            RETURN
                count(vt)
                    AS customer_previous_tx_count_window,
                collect({vt_amount})
                    AS customer_previous_amounts_window
        }}

        CALL (d) {{
            OPTIONAL MATCH
                (dc:Customer)-[:MADE]->(dt:Transaction)
                   -[:USED_DEVICE]->(d)
            WHERE {dt_id} <> $transaction_id
              AND {dt_ts} IS NOT NULL
              AND datetime(toString({dt_ts}))
                    < datetime($timestamp)
            RETURN
                count(dt)
                    AS device_previous_tx_count,
                count(DISTINCT dc)
                    AS device_previous_customer_count,
                sum(
                    CASE
                        WHEN {dt_fraud_test}
                        THEN 1 ELSE 0
                    END
                ) AS device_previous_fraud_count
        }}

        CALL (ip) {{
            OPTIONAL MATCH
                (ic:Customer)-[:MADE]->(it:Transaction)
                   -[:ORIGINATED_FROM]->(ip)
            WHERE {it_id} <> $transaction_id
              AND {it_ts} IS NOT NULL
              AND datetime(toString({it_ts}))
                    < datetime($timestamp)
            RETURN
                count(it)
                    AS ip_previous_tx_count,
                count(DISTINCT ic)
                    AS ip_previous_customer_count,
                sum(
                    CASE
                        WHEN {it_fraud_test}
                        THEN 1 ELSE 0
                    END
                ) AS ip_previous_fraud_count
        }}

        CALL (c) {{
            OPTIONAL MATCH
                (c)-[:MADE]->(pt:Transaction)
                   -[:USED_PAYMENT_METHOD]
                    ->(ppm:PaymentMethod)
            WHERE {pt_id} <> $transaction_id
              AND {pt_ts} IS NOT NULL
              AND datetime(toString({pt_ts}))
                    < datetime($timestamp)
              AND $payment_method IS NOT NULL
              AND ppm.payment_method = $payment_method
            RETURN
                count(ppm) > 0
                    AS known_payment_method
        }}

        RETURN
            c IS NOT NULL AS customer_found,
            m IS NOT NULL AS merchant_found,
            d IS NOT NULL AS device_found,
            ip IS NOT NULL AS ip_found,
            co IS NOT NULL AS country_found,

            familiar_device,
            known_merchant,
            known_payment_method,
            previous_countries,

            customer_previous_tx_count_window,
            customer_previous_amounts_window,

            device_previous_tx_count,
            device_previous_customer_count,
            device_previous_fraud_count,

            ip_previous_tx_count,
            ip_previous_customer_count,
            ip_previous_fraud_count,

            previous_fraud_transactions_at_merchant
        """

        rows = self.run_query(query, parameters)

        if not rows:
            return self.empty_context()

        context = self.empty_context()
        context.update(rows[0])

        context["previous_countries"] = [
            value
            for value in (
                context.get("previous_countries")
                or []
            )
            if value is not None
        ]

        context[
            "customer_previous_amounts_window"
        ] = [
            value
            for value in (
                context.get(
                    "customer_previous_amounts_window"
                )
                or []
            )
            if value is not None
        ]

        return context

    def get_temporal_test_transactions(
        self,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Return a chronological test slice while standardising live Neo4j
        property names into the Python keys expected by the notebook.
        """
        self.ensure_property_mapping()

        t_ts = self._property_expression(
            "t",
            "timestamp",
        )
        t_amount = self._property_expression(
            "t",
            "amount",
        )
        t_id = self._property_expression(
            "t",
            "transaction_id",
        )
        t_fraud = self._property_expression(
            "t",
            "fraud",
            required=False,
        )

        if t_fraud == "NULL":
            raise RuntimeError(
                "No fraud-label property could be detected "
                "on Transaction nodes. Classification metrics "
                "require a fraud label."
            )

        query = f"""
        MATCH
            (c:Customer)-[:MADE]->(t:Transaction)

        OPTIONAL MATCH
            (t)-[:PAID_TO]->(m:Merchant)
        OPTIONAL MATCH
            (t)-[:USED_DEVICE]->(d:Device)
        OPTIONAL MATCH
            (t)-[:ORIGINATED_FROM]->(ip:IPAddress)
        OPTIONAL MATCH
            (t)-[:OCCURRED_IN]->(co:Country)
        OPTIONAL MATCH
            (t)-[:USED_PAYMENT_METHOD]->(pm:PaymentMethod)
        OPTIONAL MATCH
            (t)-[:USED_CHANNEL]->(ch:Channel)

        WHERE {t_fraud} IS NOT NULL
          AND {t_ts} IS NOT NULL

        WITH
            c, t, m, d, ip, co, pm, ch,
            datetime(toString({t_ts})) AS parsed_transaction_dt
        ORDER BY parsed_transaction_dt DESC
        LIMIT $limit

        RETURN
            {t_id} AS transaction_id,
            c.customer_id AS customer_id,
            m.merchant_id AS merchant_id,
            d.device_id AS device_id,
            ip.ip_address AS ip_address,
            co.country AS country,
            {t_amount} AS transaction_amount,
            pm.payment_method AS payment_method,
            ch.channel AS channel,
            toString({t_ts}) AS timestamp,
            {t_fraud} AS is_fraud

        ORDER BY parsed_transaction_dt ASC
        """

        return self.run_query(
            query,
            {"limit": int(limit)},
        )
