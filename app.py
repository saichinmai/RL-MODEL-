
import streamlit as st
import pandas as pd
import traceback

from src.filtering import RelevanceFilter
from src.reasoner import ReasoningModel


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Relevance-Aware Mathematical Reasoning",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# PAGE STYLE
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #17365d;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #52647a;
    font-size: 16px;
}

.result-box {
    padding: 16px;
    border-radius: 10px;
    background-color: #eef4ff;
    border: 1px solid #c9dcff;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_filter():
    return RelevanceFilter()


@st.cache_resource
def load_reasoner():
    return ReasoningModel()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 Relevance-Aware Mathematical Reasoning</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Reducing irrelevant-context distraction using sentence relevance classification'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# PROJECT DESCRIPTION
# --------------------------------------------------

with st.expander("About this project", expanded=False):

    st.write("""
    This project investigates how irrelevant information can distract
    language models while solving mathematical word problems.

    The proposed framework uses:

    1. TF-IDF feature extraction.
    2. Logistic Regression relevance classification.
    3. Sentence-level irrelevant-context filtering.
    4. Llama 3.2 for mathematical reasoning.

    The filtered question is passed to the reasoning model to compare
    its response with the original question.
    """)


# --------------------------------------------------
# INPUT
# --------------------------------------------------

st.subheader("Enter your mathematical question")

default_question = (
    "Steve is 5 feet 6 inches tall. "
    "He grows 6 inches. "
    "The height of Emma is 8 feet. "
    "How tall is Steve in inches?"
)

question = st.text_area(
    "Enter a question containing relevant and irrelevant information:",
    value=default_question,
    height=150,
    placeholder="Type your mathematical word problem here..."
)


# --------------------------------------------------
# BUTTON
# --------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    analyze_button = st.button(
        "Analyze Question",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "Clear",
        use_container_width=True
    )


if clear_button:
    st.rerun()


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

if analyze_button:

    if not question.strip():

        st.warning("Please enter a mathematical question.")

    else:

        try:

            with st.spinner("Loading relevance classifier..."):

                relevance_filter = load_filter()

            with st.spinner("Analyzing sentence relevance..."):

                details, filtered_question, removed_sentences = (
                    relevance_filter.analyze_question(question)
                )

            st.success("Sentence relevance analysis completed.")

            # ------------------------------------------
            # CLASSIFICATION TABLE
            # ------------------------------------------

            st.subheader("1. Sentence Relevance Classification")

            df = pd.DataFrame(details)

            if not df.empty:

                display_df = df.copy()

                display_df["relevance_score"] = (
                    display_df["relevance_score"].map(
                        lambda x: f"{x:.4f}"
                    )
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------
            # ORIGINAL QUESTION
            # ------------------------------------------

            st.subheader("2. Original Question")

            st.info(question)

            # ------------------------------------------
            # FILTERED QUESTION
            # ------------------------------------------

            st.subheader("3. Filtered Question")

            st.success(filtered_question)

            # ------------------------------------------
            # REMOVED SENTENCES
            # ------------------------------------------

            st.subheader("4. Removed Sentences")

            if removed_sentences:

                for sentence in removed_sentences:
                    st.write("•", sentence)

            else:

                st.write("No sentences were removed.")

            # ------------------------------------------
            # REASONING
            # ------------------------------------------

            st.divider()

            st.subheader("5. Mathematical Reasoning Comparison")

            with st.spinner(
                "Connecting to Ollama and generating answers..."
            ):

                reasoner = load_reasoner()

                original_answer = reasoner.answer(question)

                filtered_answer = reasoner.answer(filtered_question)

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Original Question Answer**")

                st.write(original_answer)

            with col2:

                st.markdown("**Filtered Question Answer**")

                st.write(filtered_answer)

            # ------------------------------------------
            # SUMMARY
            # ------------------------------------------

            st.divider()

            st.subheader("6. Analysis Summary")

            total_sentences = len(details)

            relevant_count = sum(
                1 for item in details
                if item["classification"] in [
                    "Relevant",
                    "Retained (safety fallback)"
                ]
            )

            removed_count = len(removed_sentences)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Sentences",
                total_sentences
            )

            col2.metric(
                "Retained Sentences",
                relevant_count
            )

            col3.metric(
                "Removed Sentences",
                removed_count
            )

            st.caption(
                "The relevance score is the classifier's estimated "
                "probability for the relevant class."
            )

        except FileNotFoundError as error:

            st.error(str(error))

        except Exception as error:

            st.error(
                "An error occurred while processing the question."
            )

            st.code(str(error))

            st.info(
                "Check that Ollama is running and that the "
                "llama3.2:3b model is installed."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research implementation | TF-IDF + Logistic Regression + Llama 3.2"
)
