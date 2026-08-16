import os
import uuid

from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Payment Safety Assistant",
    page_icon="🛡️",
    layout="centered",
)


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from neo4j_fraud_service import (
    Neo4jFraudService,
)

from neo4j_risk_engine import (
    calculate_risk,
)

from customer_guidance import (
    generate_customer_safe_payment_guidance,
    validate_customer_audience_guidance,
    deterministic_customer_audience_fallback,
    generate_customer_graphrag_answer,
)

from hosted_llm import (
    HostedLLM,
)


# =========================================================
# STREAMLIT SECRETS
# =========================================================

# Neo4j credentials
os.environ["NEO4J_URI"] = (
    st.secrets["NEO4J_URI"]
)

os.environ["NEO4J_USERNAME"] = (
    st.secrets["NEO4J_USERNAME"]
)

os.environ["NEO4J_PASSWORD"] = (
    st.secrets["NEO4J_PASSWORD"]
)


# =========================================================
# RISK ENGINE SETTINGS
# =========================================================

RISK_ENGINE_KWARGS = {

    "amount_threshold": 1000,

    "max_risk_points": 110,

    "high_risk_countries": set(),

    "unusual_start_hour": 0,

    "unusual_end_hour": 5,

    "risk_level_thresholds": {
        "Critical": 75,
        "High": 50,
        "Medium": 25,
    },

    "min_evidence_coverage_for_assessment": 0.60,
}


# =========================================================
# DRAW TRANSACTION GRAPH
# =========================================================

def draw_transaction_graph(
    transaction,
    graph_context,
):

    G = nx.DiGraph()

    # -----------------------------------------------------
    # Create readable node labels
    # -----------------------------------------------------

    tx_node = (
        "Transaction\n"
        + transaction["transaction_id"][:12]
    )

    customer_node = (
        "Customer\n"
        + transaction["customer_id"][:12]
    )

    merchant_node = (
        "Merchant\n"
        + transaction["merchant_id"][:12]
    )

    device_node = (
        "Device\n"
        + transaction["device_id"][:12]
    )

    country_node = (
        "Country\n"
        + str(transaction["country"])
    )


    # -----------------------------------------------------
    # Core transaction relationships
    # -----------------------------------------------------

    G.add_edge(
        customer_node,
        tx_node,
        relation="MADE",
    )

    G.add_edge(
        tx_node,
        merchant_node,
        relation="PAID_TO",
    )

    G.add_edge(
        tx_node,
        device_node,
        relation="USED_DEVICE",
    )


    # -----------------------------------------------------
    # IP address
    # -----------------------------------------------------

    if transaction.get(
        "ip_address"
    ):

        ip_node = (
            "IP\n"
            + transaction["ip_address"]
        )

        G.add_edge(
            tx_node,
            ip_node,
            relation="ORIGINATED_FROM",
        )


    # -----------------------------------------------------
    # Country
    # -----------------------------------------------------

    G.add_edge(
        tx_node,
        country_node,
        relation="OCCURRED_IN",
    )


    # -----------------------------------------------------
    # Payment method
    # -----------------------------------------------------

    if transaction.get(
        "payment_method"
    ):

        payment_node = (
            "Payment Method\n"
            + transaction["payment_method"]
        )

        G.add_edge(
            tx_node,
            payment_node,
            relation="USED_PAYMENT_METHOD",
        )


    # -----------------------------------------------------
    # Channel
    # -----------------------------------------------------

    if transaction.get(
        "channel"
    ):

        channel_node = (
            "Channel\n"
            + transaction["channel"]
        )

        G.add_edge(
            tx_node,
            channel_node,
            relation="USED_CHANNEL",
        )


    # -----------------------------------------------------
    # Draw graph
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.4,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2500,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=18,
        width=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        ax=ax,
    )

    edge_labels = (
        nx.get_edge_attributes(
            G,
            "relation",
        )
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax,
    )

    ax.axis(
        "off"
    )

    return fig


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title(
    "🛡️ Payment Safety Assistant"
)

st.write(
    "Check a proposed payment before proceeding. "
    "The assistant uses historical graph evidence "
    "and a deterministic fraud-risk engine."
)


# =========================================================
# DISSERTATION INFORMATION
# =========================================================

st.info(
    """
    **Dissertation Research Prototype**

    This application demonstrates a GraphRAG-driven
    conversational approach to explainable payment
    fraud prevention.

    It combines Neo4j graph evidence, deterministic
    risk scoring and grounded LLM-generated guidance.

    The prototype uses synthetic payment transaction
    data for research and evaluation purposes.
    """
)


# =========================================================
# LOAD NEO4J SERVICE
# =========================================================

if (
    "neo4j_service"
    not in st.session_state
):

    with st.spinner(
        "Connecting to fraud knowledge graph..."
    ):

        try:

            service = (
                Neo4jFraudService()
            )

            service.verify_connection()

            st.session_state.neo4j_service = (
                service
            )

        except Exception as e:

            st.error(
                "Unable to connect to the "
                "Neo4j fraud knowledge graph."
            )

            st.exception(
                e
            )

            st.stop()


service = (
    st.session_state.neo4j_service
)


# =========================================================
# LOAD HOSTED LLM
# =========================================================

if (
    "hosted_llm"
    not in st.session_state
):

    with st.spinner(
        "Connecting to conversational fraud assistant..."
    ):

        try:

            st.session_state.hosted_llm = (
                HostedLLM(
                    api_token=(
                        st.secrets[
                            "OPENROUTER_API_KEY"
                        ]
                    ),
                    model="openrouter/free",
                    max_new_tokens=300,
                    temperature=0.0,
                )
            )

        except Exception as e:

            st.error(
                "Unable to connect to the hosted LLM."
            )

            st.exception(
                e
            )

            st.stop()


llm = (
    st.session_state.hosted_llm
)


# =========================================================
# CUSTOMER TRANSACTION FORM
# =========================================================

with st.form(
    "payment_form"
):

    customer_id = st.text_input(
        "Customer ID",
        key="customer_id_input",
    )

    merchant_id = st.text_input(
        "Merchant ID",
        key="merchant_id_input",
    )

    device_id = st.text_input(
        "Device ID",
        key="device_id_input",
    )

    ip_address = st.text_input(
        "IP Address",
        key="ip_address_input",
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=1.0,
        key="amount_input",
    )

    country = st.text_input(
        "Country",
        value="IE",
        key="country_input",
    )

    payment_method = st.text_input(
        "Payment Method",
        value="Credit Card",
        key="payment_method_input",
    )

    channel = st.text_input(
        "Channel",
        value="E-commerce",
        key="channel_input",
    )

    transaction_date = (
        st.date_input(
            "Transaction Date",
            key="transaction_date_input",
        )
    )

    transaction_time = (
        st.time_input(
            "Transaction Time",
            key="transaction_time_input",
        )
    )

    submitted = (
        st.form_submit_button(
            "Check Payment Safety"
        )
    )


# =========================================================
# PROCESS PAYMENT
# =========================================================

if submitted:

    required_values = {

        "Customer ID":
            customer_id,

        "Merchant ID":
            merchant_id,

        "Device ID":
            device_id,

        "Country":
            country,
    }


    missing = [

        name

        for name, value
        in required_values.items()

        if not value.strip()
    ]


    if missing:

        st.error(
            "Please complete: "
            + ", ".join(
                missing
            )
        )

        st.stop()


    # -----------------------------------------------------
    # Create proposed transaction
    # -----------------------------------------------------

    transaction_id = (
        "UI_"
        + str(
            uuid.uuid4()
        )
    )


    timestamp = (
        datetime.combine(
            transaction_date,
            transaction_time,
        ).isoformat()
    )


    transaction = {

        "transaction_id":
            transaction_id,

        "customer_id":
            customer_id.strip(),

        "merchant_id":
            merchant_id.strip(),

        "device_id":
            device_id.strip(),

        "ip_address":
            ip_address.strip(),

        "country":
            country.strip(),

        "transaction_amount":
            float(
                amount
            ),

        "payment_method":
            payment_method.strip(),

        "channel":
            channel.strip(),

        "timestamp":
            timestamp,
    }


    # -----------------------------------------------------
    # Retrieve GraphRAG evidence
    # -----------------------------------------------------

    with st.spinner(
        "Checking graph history and "
        "fraud indicators..."
    ):

        try:

            graph_context = (
                service.get_transaction_context(
                    transaction
                )
            )

        except Exception as e:

            st.error(
                "Unable to retrieve graph evidence."
            )

            st.exception(
                e
            )

            st.stop()


        # -------------------------------------------------
        # Calculate deterministic risk
        # -------------------------------------------------

        try:

            risk_result = (
                calculate_risk(
                    transaction=transaction,
                    graph_context=graph_context,
                    **RISK_ENGINE_KWARGS,
                )
            )

        except Exception as e:

            st.error(
                "Unable to calculate payment risk."
            )

            st.exception(
                e
            )

            st.stop()


        # -------------------------------------------------
        # Store assessed transaction
        # -------------------------------------------------

        st.session_state.current_transaction = (
            transaction
        )

        st.session_state.current_graph_context = (
            graph_context
        )

        st.session_state.current_risk_result = (
            risk_result
        )

        st.session_state.messages = []


        # -------------------------------------------------
        # Generate customer guidance
        # -------------------------------------------------

        try:

            raw_guidance = (
                generate_customer_safe_payment_guidance(
                    transaction,
                    graph_context,
                    risk_result,
                    llm,
                )
            )


            # ---------------------------------------------
            # Validate generated guidance
            # ---------------------------------------------

            validation = (
                validate_customer_audience_guidance(
                    raw_guidance,
                    risk_result,
                )
            )


            # ---------------------------------------------
            # Use LLM result or deterministic fallback
            # ---------------------------------------------

            if validation.get(
                "accepted",
                False,
            ):

                final_guidance = (
                    raw_guidance
                )

                fallback_used = False

            else:

                final_guidance = (
                    deterministic_customer_audience_fallback(
                        risk_result
                    )
                )

                fallback_used = True


        except Exception as e:

            # If hosted LLM fails,
            # preserve fraud assessment
            # and show deterministic guidance.

            final_guidance = (
                deterministic_customer_audience_fallback(
                    risk_result
                )
            )

            raw_guidance = None

            validation = {
                "accepted": False,
                "reason":
                    "LLM generation unavailable",
            }

            fallback_used = True

            st.warning(
                "The conversational model was unavailable. "
                "Deterministic fraud-prevention guidance "
                "has been shown instead."
            )


        # -------------------------------------------------
        # Save guidance state
        # -------------------------------------------------

        st.session_state.raw_guidance = (
            raw_guidance
        )

        st.session_state.validation = (
            validation
        )

        st.session_state.final_guidance = (
            final_guidance
        )

        st.session_state.fallback_used = (
            fallback_used
        )


# =========================================================
# DISPLAY PAYMENT SAFETY RESULT
# =========================================================

if (
    "current_risk_result"
    in st.session_state
):

    risk_result = (
        st.session_state.current_risk_result
    )

    transaction = (
        st.session_state.current_transaction
    )

    graph_context = (
        st.session_state.current_graph_context
    )


    st.divider()

    st.header(
        "Payment Safety Result"
    )


    # =====================================================
    # TWO-COLUMN RESULT LAYOUT
    # =====================================================

    left_col, right_col = (
        st.columns(
            [1, 2]
        )
    )


    # -----------------------------------------------------
    # LEFT: RISK ASSESSMENT
    # -----------------------------------------------------

    with left_col:

        st.subheader(
            "Payment Risk"
        )

        st.metric(
            "Risk Score",
            (
                f"{risk_result['risk_score']}"
                "/100"
            ),
        )

        st.metric(
            "Risk Level",
            risk_result[
                "risk_level"
            ],
        )

        st.write(
            "**Assessment status:**",
            risk_result[
                "assessment_status"
            ],
        )

        st.write(
            "**Evidence confidence:**",
            risk_result[
                "decision_confidence"
            ],
        )


        st.subheader(
            "Risk Indicators"
        )


        factors = (
            risk_result.get(
                "risk_factors",
                [],
            )
        )


        if factors:

            for factor in factors:

                st.write(
                    f"• {factor['description']} "
                    f"(+{factor['points']} points)"
                )

        else:

            st.write(
                "No configured customer risk rules "
                "were triggered."
            )


        risk_level = (
            risk_result[
                "risk_level"
            ]
        )


        if (
            risk_result[
                "assessment_status"
            ]
            == "Insufficient evidence"
        ):

            st.warning(
                "There is not enough historical "
                "evidence for a confident assessment."
            )


        elif risk_level in {
            "High",
            "Critical",
        }:

            st.error(
                "Important risk indicators were detected. "
                "Pause and verify the payment "
                "before proceeding."
            )


        elif (
            risk_level
            == "Medium"
        ):

            st.warning(
                "Some risk indicators were detected. "
                "Review the payment details carefully."
            )


        else:

            st.success(
                "No high-risk threshold was reached "
                "using the available evidence."
            )


    # -----------------------------------------------------
    # RIGHT: TRANSACTION GRAPH
    # -----------------------------------------------------

    with right_col:

        st.subheader(
            "Transaction Graph"
        )

        fig = (
            draw_transaction_graph(
                transaction,
                graph_context,
            )
        )

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )


# =========================================================
# FRAUD PREVENTION GUIDANCE
# =========================================================

if (
    "final_guidance"
    in st.session_state
):

    st.divider()

    st.subheader(
        "Fraud Prevention Guidance"
    )

    st.markdown(
        st.session_state.final_guidance
    )


    if st.session_state.get(
        "fallback_used",
        False,
    ):

        st.caption(
            "A deterministic grounded response "
            "was shown because the generated "
            "LLM explanation was unavailable "
            "or did not pass the evidence "
            "validation check."
        )


# =========================================================
# TECHNICAL GRAPHRAG EVIDENCE
# =========================================================

if (
    "current_graph_context"
    in st.session_state
):

    with st.expander(
        "Show GraphRAG Evidence"
    ):

        st.json(
            st.session_state.current_graph_context
        )


# =========================================================
# START NEW PAYMENT
# =========================================================

if (
    "current_transaction"
    in st.session_state
):

    if st.button(
        "Start New Payment"
    ):

        keys_to_clear = [

            # Assessment state
            "current_transaction",
            "current_graph_context",
            "current_risk_result",

            "raw_guidance",
            "validation",
            "final_guidance",
            "fallback_used",

            "messages",

            # Form fields
            "customer_id_input",
            "merchant_id_input",
            "device_id_input",
            "ip_address_input",
            "amount_input",
            "country_input",
            "payment_method_input",
            "channel_input",
            "transaction_date_input",
            "transaction_time_input",
        ]


        for key in keys_to_clear:

            st.session_state.pop(
                key,
                None,
            )


        st.rerun()


# =========================================================
# CONVERSATIONAL GRAPHRAG ASSISTANT
# =========================================================

st.divider()

st.subheader(
    "Ask the Payment Safety Assistant"
)


if (
    "current_transaction"
    not in st.session_state
):

    st.info(
        "Check a payment first before asking "
        "questions about its fraud-risk assessment."
    )


else:

    st.caption(
        "Ask questions about the graph evidence "
        "used for this payment assessment."
    )


    # -----------------------------------------------------
    # Initialise chat history
    # -----------------------------------------------------

    if (
        "messages"
        not in st.session_state
    ):

        st.session_state.messages = []


    # -----------------------------------------------------
    # Display previous chat messages
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # Customer question
    # -----------------------------------------------------

    prompt = st.chat_input(
        "Ask why this payment is risky..."
    )


    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )


        transaction = (
            st.session_state.current_transaction
        )

        graph_context = (
            st.session_state.current_graph_context
        )

        risk_result = (
            st.session_state.current_risk_result
        )


        # -------------------------------------------------
        # Generate GraphRAG conversational response
        # -------------------------------------------------

        with st.spinner(
            "Reviewing graph evidence..."
        ):

            try:

                response = (
                    generate_customer_graphrag_answer(
                        question=prompt,
                        transaction=transaction,
                        graph_context=graph_context,
                        risk_result=risk_result,
                        llm=llm,
                    )
                )


            except Exception:

                response = (
                    "I could not generate a conversational "
                    "response at this time. Please review "
                    "the deterministic risk assessment and "
                    "GraphRAG evidence shown above."
                )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )


        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                response
            )


# =========================================================
# RESEARCH DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "Research prototype developed for academic evaluation. "
    "The system uses synthetic transaction data and should "
    "not be treated as a production financial decision system."
)
