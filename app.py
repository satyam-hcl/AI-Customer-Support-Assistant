import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, AuthenticationError, NotFoundError, RateLimitError
import streamlit as st


load_dotenv(Path(__file__).with_name(".env"))

SYSTEM_PROMPT = """You are a helpful e-commerce customer support assistant.
Be empathetic, concise, and practical. Ask focused questions when details are
missing. Never claim to access an order, issue a refund, or take an action you
cannot actually perform. Do not invent store policies, shipping updates, or
order details; explain when the customer should contact the store directly."""

APP_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root {
    --ink: #1b3026;
    --muted: #748078;
    --forest: #19392e;
    --leaf: #2d7658;
    --coral: #d66d50;
    --paper: #f3f6f2;
    --line: #dfe7e1;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', 'Avenir Next', sans-serif;
    color: var(--ink);
}
.stApp {
    background-color: var(--paper);
    background-image: linear-gradient(rgba(25, 57, 46, .025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(25, 57, 46, .025) 1px, transparent 1px);
    background-size: 36px 36px;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1000px; padding: 2.4rem 2rem 8rem; }
h1, h2, h3 { color: var(--ink); }
.brand-lockup { display: flex; align-items: center; gap: 12px; margin: 4px 0 34px; }
.brand-mark {
    display: grid; place-items: center; width: 38px; height: 38px;
    border: 1px solid rgba(255,255,255,.22); border-radius: 10px;
    color: #f4c46d; font-size: 13px; font-weight: 700;
}
.brand-name { color: #f5f7f3; font-size: 14px; font-weight: 700; }
.brand-sub { display: block; margin-top: 3px; color: #a9bdb1; font-size: 9px; font-weight: 600; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(18, 38, 29, .18); }
[data-testid="stSidebar"] > div:first-child {
    background: var(--forest); padding: 1.7rem 1.2rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: #e8eee9; }
.side-label, .eyebrow, .rail-label, .section-label {
    color: var(--muted); font-size: 10px; font-weight: 700;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .side-label { color: #a9bdb1; margin: 0 0 10px; }
[data-testid="stSidebar"] [data-testid="stCode"] {
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.12);
    border-radius: 8px;
}
[data-testid="stSidebar"] code { color: #f4c46d; }
.page-head {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 20px; margin: 0 0 2.2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--line);
}
.eyebrow { margin-bottom: 10px; color: var(--leaf); }
.page-title {
    margin: 0; color: var(--ink); font-family: 'Newsreader', Georgia, serif;
    font-size: 46px; font-weight: 500; line-height: 1.04;
}
.status-pill {
    display: inline-flex; align-items: center; gap: 8px; margin-bottom: 5px;
    padding: 8px 11px; border: 1px solid var(--line); border-radius: 999px;
    background: rgba(255,255,255,.72); color: #53645a;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #5a9d73; }
.section-label { margin: 0 0 15px; color: #79877e; }
.empty-title { margin: 28px 0 14px; color: var(--ink); font-size: 16px; font-weight: 600; }
.rail-label { margin: 4px 0 20px; color: var(--leaf); }
.help-item { display: flex; gap: 13px; padding: 15px 0; border-top: 1px solid var(--line); }
.help-index { color: var(--coral); font-family: 'Newsreader', Georgia, serif; font-size: 18px; }
.help-title { color: var(--ink); font-size: 13px; font-weight: 600; }
.help-detail { margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.5; }
.service-note {
    margin-top: 26px; padding: 15px 0 0; border-top: 1px solid var(--line);
    color: var(--muted); font-size: 12px; line-height: 1.65;
}
.service-note strong { display: block; margin-bottom: 6px; color: var(--ink); font-size: 11px; }
.stButton > button {
    min-height: 54px; border: 1px solid var(--line); border-radius: 9px;
    background: rgba(255,255,255,.88); color: var(--ink); text-align: left;
    font-size: 13px; font-weight: 600; transition: border-color .16s ease, transform .16s ease;
}
.stButton > button:hover { border-color: #8bb49b; color: var(--forest); transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    border-color: var(--coral); background: var(--coral); color: white; text-align: center;
}
.stButton > button[kind="primary"]:hover { border-color: #c45d42; background: #c45d42; color: white; }
[data-testid="stChatMessage"] {
    margin: 10px 0; padding: 14px 16px; border: 1px solid var(--line);
    border-radius: 12px; background: rgba(255,255,255,.9);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: var(--ink); font-size: 14px; line-height: 1.7;
}
[data-testid="stChatInput"] {
    margin-top: 1.5rem; padding: 22px 0 12px;
    background: linear-gradient(180deg, rgba(243,246,242,0), var(--paper) 38%);
}
[data-testid="stChatInput"] > div {
    display: flex; align-items: flex-end; gap: 8px;
    padding: 8px 9px 8px 12px; border: 1px solid #cfdbd1;
    border-radius: 14px; background: #fff;
    box-shadow: 0 9px 24px rgba(27,48,38,.08);
    transition: border-color .16s ease, box-shadow .16s ease;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #6b9b7c; box-shadow: 0 0 0 3px rgba(45,118,88,.1), 0 9px 24px rgba(27,48,38,.08);
}
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    border: 0 !important; border-radius: 9px; background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    min-height: 44px; max-height: 160px; padding: 11px 8px;
    border: 0 !important; border-radius: 9px; background: transparent !important;
    box-shadow: none !important; color: var(--ink);
    font-family: 'DM Sans', 'Avenir Next', sans-serif; font-size: 14px; line-height: 1.5;
}
[data-testid="stChatInput"] textarea:focus { outline: none; }
[data-testid="stChatInput"] textarea::placeholder { color: #87948b; opacity: 1; }
[data-testid="stChatInput"] button {
    width: 40px; min-width: 40px; height: 40px; margin-bottom: 2px;
    border: 0 !important; border-radius: 10px !important;
    background: var(--forest) !important; color: #fff !important;
    transition: background .16s ease, transform .16s ease;
}
[data-testid="stChatInput"] button:hover:not(:disabled) {
    background: var(--leaf) !important; transform: translateY(-1px);
}
[data-testid="stChatInput"] button:disabled {
    background: #e3ebe5 !important; color: #87948b !important;
}
@media (max-width: 900px) {
    .block-container { padding: 1.5rem 1rem 7rem; }
    .page-head { align-items: flex-start; flex-direction: column; margin-bottom: 1.4rem; }
    .page-title { font-size: 38px; }
}
</style>
"""


def build_chain():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to .env."
        )

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm


def main():
    st.set_page_config(page_title="Customer Support", layout="centered")
    st.markdown(APP_STYLES, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup"><div class="brand-mark">CS</div>'
            '<div class="brand-name">CARE DESK'
            '<span class="brand-sub">ECOMMERCE SUPPORT</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">WORKSPACE</div>', unsafe_allow_html=True)
        if st.button("New conversation", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.pop("pending_question", None)
            st.rerun()
        # st.markdown('<div class="side-label" style="margin-top:34px">MODEL</div>', unsafe_allow_html=True)
        # st.code(model, language=None)
        # st.caption("Connected through OpenRouter")

    try:
        chain = build_chain()
    except ValueError as error:
        chain = None
        setup_error = str(error)
    else:
        setup_error = None

    configured = chain is not None
    status_text = "READY" if configured else "CONFIGURATION NEEDED"
    status_color = "#5a9d73" if configured else "#d66d50"
    st.markdown(
        '<div class="page-head"><div><div class="eyebrow">CUSTOMER CARE / ONLINE STORE</div>'
        '<h1 class="page-title">Let\'s get this sorted.</h1></div>'
        f'<div class="status-pill"><span class="status-dot" style="background:{status_color}"></span>{status_text}</div></div>',
        unsafe_allow_html=True,
    )

    if setup_error:
        st.warning(f"{setup_error} Add your key to .env to start a conversation.")

    st.markdown('<div class="section-label">CONVERSATION</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])

    if not st.session_state.messages:
        st.markdown('<div class="empty-title">Start with a topic</div>', unsafe_allow_html=True)
        examples = [
            "My order is delayed",
            "I received a damaged item",
            "Can I return a purchase?",
            "I'm waiting on a refund",
        ]
        for row in range(2):
            columns = st.columns(2)
            for column, example in zip(columns, examples[row * 2 : row * 2 + 2]):
                if column.button(example, use_container_width=True):
                    st.session_state.pending_question = example
                    st.rerun()

    question = st.chat_input("Write a message...")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
    if not question:
        return

    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not configured:
            answer = setup_error
            is_error = True
        else:
            try:
                with st.spinner("Preparing a response..."):
                    response = chain.invoke(
                        {"history": st.session_state.history, "question": question}
                    )
                answer = str(response.content)
                is_error = False
            except AuthenticationError:
                answer = "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env."
                is_error = True
            except NotFoundError:
                answer = "The configured model was not found. Check OPENROUTER_MODEL."
                is_error = True
            except RateLimitError:
                answer = "OpenRouter rate limit or quota reached. Try again later."
                is_error = True
            except APIConnectionError:
                answer = "Could not connect to OpenRouter. Check your network and try again."
                is_error = True
            except Exception as error:
                answer = f"The request failed: {error}"
                is_error = True
            else:
                st.session_state.history.extend(
                    [HumanMessage(content=question), AIMessage(content=answer)]
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "error": is_error}
        )
        if is_error:
            st.error(answer)
        else:
            st.markdown(answer)


if __name__ == "__main__":
    main()