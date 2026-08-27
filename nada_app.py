import streamlit as st
import openai
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import json

# ─────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🤖 NADA Assistant")

try:
    # This tells the code: "Go look for the key in Streamlit's settings, not here."
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    # This tells the code: "Look for the connection link in Streamlit's settings."
    engine = create_engine(st.secrets["SUPABASE_URI"])
except Exception as e:
    st.error("Could not initialize connections. Check secrets configuration.")
    st.stop()

DISTRICTS = [
    "Bentong", "Bera", "Cameron Highlands", "Jerantut", "Kuantan",
    "Lipis", "Maran", "Pekan", "Raub", "Rompin", "Temerloh"
]

# ─────────────────────────────────────────────
# 2. Context-aware UI
# ─────────────────────────────────────────────
context = st.radio(
    "Dashboard Page Context:",
    ["Executive Overview & Labour Mismatch", "Vulnerability Index", "Economic Nowcasting"],
    horizontal=True
)

# ─────────────────────────────────────────────
# 3. Retrieval helpers (with error handling)
# ─────────────────────────────────────────────
def safe_query(sql, params=None):
    """Run a query and return a DataFrame, or None on failure."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params or {})
        return df
    except SQLAlchemyError as e:
        st.warning("Data lookup failed — answering from general context only.")
        return None

def detect_district(user_text):
    """Very simple keyword match against known district names."""
    for d in DISTRICTS:
        if d.lower() in user_text.lower():
            return d
    return None

def get_district_context(district):
    df = safe_query(
        """
        SELECT district, year, poverty_rate_pct, unemployment_rate_pct, lfpr_pct,
               dvi_score, vulnerability_level, dominant_pillar, dominant_indicator,
               dominant_contribution_pct
        FROM district_summary
        WHERE district = :district
        ORDER BY year DESC
        LIMIT 3
        """,
        {"district": district}
    )
    if df is None or df.empty:
        return None
    return df.to_dict(orient="records")

def get_nowcast_context(district):
    df = safe_query(
        """
        SELECT district, year, poverty_nowcast, poverty_actual_pct,
               income_nowcast_rm, income_actual_rm, is_estimated
        FROM nowcast_estimates
        WHERE district = :district
        ORDER BY year DESC
        LIMIT 3
        """,
        {"district": district}
    )
    if df is None or df.empty:
        return None
    return df.to_dict(orient="records")

def get_model_validation():
    df = safe_query("SELECT * FROM model_validation")
    if df is None or df.empty:
        return None
    return df.to_dict(orient="records")

def get_glossary_matches(user_text):
    """Pull glossary rows whose term appears in the question."""
    df = safe_query("SELECT term, plain_language_definition, example_or_context FROM indicator_glossary")
    if df is None or df.empty:
        return None
    matches = df[df["term"].apply(lambda t: str(t).lower() in user_text.lower())]
    return matches.to_dict(orient="records") if not matches.empty else None

def build_retrieved_context(user_text):
    """Assemble whatever real data is relevant to this question."""
    pieces = []

    district = detect_district(user_text)
    if district:
        d_ctx = get_district_context(district)
        if d_ctx:
            pieces.append(f"DISTRICT DATA for {district} (most recent years):\n{json.dumps(d_ctx, default=str)}")

        n_ctx = get_nowcast_context(district)
        if n_ctx:
            pieces.append(f"NOWCAST DATA for {district}:\n{json.dumps(n_ctx, default=str)}")

    if any(k in user_text.lower() for k in ["trust", "reliab", "accurate", "confidence", "r2", "r²", "rmse"]):
        v_ctx = get_model_validation()
        if v_ctx:
            pieces.append(f"MODEL VALIDATION METRICS:\n{json.dumps(v_ctx, default=str)}")

    g_ctx = get_glossary_matches(user_text)
    if g_ctx:
        pieces.append(f"GLOSSARY DEFINITIONS:\n{json.dumps(g_ctx, default=str)}")

    if not pieces:
        return None
    return "\n\n".join(pieces)

# ─────────────────────────────────────────────
# 4. System prompt (page-aware, jargon-safe)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are NADA, the AI assistant embedded in the Pahang Economic Vulnerability
Dashboard (DAX Challenge — DOSM Pahang sub-theme). You help non-technical users
(policymakers, district officers, the public) understand the dashboard's data
and models. All data comes from DOSM (Department of Statistics Malaysia).

Current dashboard page: {context}

PAGE-SPECIFIC FOCUS:
- If 'Executive Overview & Labour Mismatch': Focus on explaining the gap between
  labour force participation/unemployment and household poverty. Be direct that
  the tested correlation between unemployment and poverty was NOT statistically
  significant (p = 0.836) — this is the dashboard's key finding, not a flaw to
  hide. Explain that low unemployment can still hide real poverty.
- If 'Vulnerability Index': Explain DVI using its actual 3 components — Exposure
  (mainly inflation), Sensitivity (unemployment, employment-population ratio,
  dependency ratio), and Lack of Adaptive Capacity (school enrolment, LFPR).
  Do not invent additional dimensions beyond these three.
- If 'Economic Nowcasting': Always distinguish between 'nowcasted' (modeled
  estimate) and 'actual' (measured/survey) values. State model reliability
  honestly using the validation metrics (R², RMSE, MAE, 95% CI coverage) —
  do not present nowcasted figures as certain.

RULES:
1. Never use unexplained statistical jargon (p-value, R², LOOCV, DVI, IPCC
   pillar, etc.). Define any such term in one plain-language clause the first
   time you use it.
2. Always translate a number into what it means for the user, not just what
   it is.
3. If a result is weak or not statistically significant, say so plainly rather
   than glossing over it.
4. When citing a figure, mention its source and whether it is measured or
   estimated (e.g., "based on 2023 DOSM data" vs. "this is a modeled estimate
   for a non-survey year").
5. Never state a policy recommendation as certain fact when the underlying
   confidence is weak (e.g., low R², below-target CI coverage) — flag the
   uncertainty instead.
6. If a question is outside the dashboard's data (e.g., asks about a district
   not covered, or a topic unrelated to this dataset), say so honestly rather
   than guessing or fabricating an answer.
7. When DATA CONTEXT is provided below a user question, base your answer on
   those real figures rather than estimating from memory. If no data context
   is provided for a question that needs specific numbers, say you don't have
   that data rather than inventing a figure.
8. Keep answers to 2-4 sentences by default; offer to go deeper only if asked.
"""

# ─────────────────────────────────────────────
# 5. Chat state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
else:
    # Keep the system prompt current if the user switches page context
    st.session_state.messages[0] = {"role": "system", "content": SYSTEM_PROMPT}

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# ─────────────────────────────────────────────
# 6. Handle input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask about a district, a term, or how much you can trust an estimate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    retrieved = build_retrieved_context(prompt)
    if retrieved:
        augmented_prompt = f"{prompt}\n\n[DATA CONTEXT — use this to answer accurately]\n{retrieved}"
    else:
        augmented_prompt = prompt

    api_messages = st.session_state.messages[:-1] + [{"role": "user", "content": augmented_prompt}]

    try:
        with st.spinner("Thinking..."):
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=api_messages,
                timeout=20,
            )
        msg = response.choices[0].message.content
    except Exception as e:
        msg = ("Sorry, I couldn't process that just now — there may be a "
               "connection issue. Please try again in a moment.")

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)