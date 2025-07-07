import streamlit as st
import time
import random
from rag_engine import load_chunks, build_index, search, normalize_input, ask_gpt_with_memory, is_small_talk, handle_small_talk
from memory import ContextMemory
from symspell_setup import correct_query
from config import BOT_PERSONALITY, DEFAULT_VOCAB

st.set_page_config(page_title="🎓 CrescentBot AI Assistant", layout="wide")
st.title("🤖 Crescent University Assistant Bot")

with st.spinner("🔍 Loading university knowledge base..."):
    chunks, raw_data = load_chunks("qa_dataset.json")
    index, model, embeddings = build_index(chunks)

if "context" not in st.session_state:
    st.session_state.context = ContextMemory()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

st.markdown("---")
st.markdown(BOT_PERSONALITY["greeting"])
user_query = st.text_input("Ask me anything about Crescent University:")

if user_query:
    with st.spinner("CrescentBot is typing..."):
        time.sleep(random.uniform(0.8, 1.5))

        corrected = correct_query(user_query)

        if is_small_talk(corrected):
            response = handle_small_talk(corrected)
        else:
            norm_query = normalize_input(corrected, DEFAULT_VOCAB, model)
            top_match, score = search(norm_query, index, model, chunks, top_k=1)

            if score > 0.6:
                response = top_match[0]
            else:
                history = st.session_state.context.get_context()
                messages = [{"role": "system", "content": (
                    "You are CrescentBot, a helpful, friendly, and empathetic assistant for Crescent University students. "
                    "You always respond in a natural, conversational tone. Keep replies short, clear, and human-like. "
                    "Use emojis sparingly to keep things warm but professional. Guide students politely when you don't know something."
                )}]
                for turn in history:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["bot"]})
                messages.append({"role": "user", "content": user_query})
                response, _ = ask_gpt_with_memory(messages)
                if "?" in response or len(response.split()) < 10:
                    response += f" {BOT_PERSONALITY['clarify']}"

        st.markdown(f"**You:** {user_query}")
        st.markdown(f"**{BOT_PERSONALITY['name']}:** {response}")

        follow_up = random.choice([
            "Would you like more details on that?",
            "Is that helpful for now?",
            "Let me know if you want to go deeper on this!",
            "Would you like me to explain another topic?"
        ])
        st.markdown(f"🗨️ _{follow_up}_")

        st.session_state.context.add_turn(user_query, response)
        st.session_state.chat_log.append({"user": user_query, "bot": response})

if st.button("🔧 Download Chat History"):
    import json
    st.download_button("Download Chat JSON", json.dumps(st.session_state.chat_log, indent=2), "chat_log.json")
