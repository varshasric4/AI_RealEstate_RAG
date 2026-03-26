
import os
import json
import uuid
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
from config import *

# ----------------------------------------
# STEP 1: Load Embeddings
# ----------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("✅ Embeddings ready!")

# ----------------------------------------
# STEP 2: Load FAISS Database
# ----------------------------------------
if not os.path.exists(FAISS_PATH):
    print("❌ Database not found! Run create_database.py first")
    exit()

print("Loading database...")
vectorstore = FAISS.load_local(
    FAISS_PATH, embeddings, allow_dangerous_deserialization=True
)
print(f"✅ Database loaded! Vectors: {vectorstore.index.ntotal}")

# ----------------------------------------
# STEP 3: Setup LLMs
# ----------------------------------------
print("Setting up LLM...")
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_base=MODEL_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.3,
    max_tokens=1024,
    default_headers={"HTTP-Referer": "http://localhost:7860", "X-Title": "RE Advisor"}
)
correction_llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_base=MODEL_BASE_URL,
    openai_api_key=OPENROUTER_API_KEY,
    temperature=0.1,
    max_tokens=100,
    default_headers={"HTTP-Referer": "http://localhost:7860", "X-Title": "RE Advisor"}
)
print("✅ LLM ready!")

# ----------------------------------------
# STEP 4: Nearby Areas Map
# ----------------------------------------
NEARBY_MAP = {
    "kondapur":        ["madhapur", "gachibowli", "nanakramguda", "serilingampally", "hitec city"],
    "bachupally":      ["nizampet", "kompally", "miyapur", "kukatpally"],
    "badepally":       ["jadcherla", "mahbubnagar", "shadnagar"],
    "nizampet":        ["bachupally", "kukatpally", "miyapur", "bhel"],
    "kukatpally":      ["miyapur", "kphb", "bachupally", "nizampet", "kondapur"],
    "gachibowli":      ["kondapur", "nanakramguda", "financial district", "madhapur"],
    "madhapur":        ["kondapur", "hitec city", "jubilee hills", "gachibowli"],
    "kompally":        ["bachupally", "shamirpet", "medchal", "nizampet"],
    "secunderabad":    ["begumpet", "trimulgherry", "malkajgiri"],
    "lb nagar":        ["dilsukhnagar", "saroor nagar", "hayathnagar"],
    "jadcherla":       ["badepally", "mahbubnagar", "shadnagar"],
    "miyapur":         ["kukatpally", "bachupally", "bhel", "nizampet"],
    "hitec city":      ["madhapur", "kondapur", "gachibowli"],
    "nallagandla":     ["serilingampally", "tellapur", "kollur"],
    "manchirevula":    ["gandipet", "narsingi", "kokapet"],
    "gandipet":        ["manchirevula", "narsingi", "kokapet", "serilingampally"],
    "narsingi":        ["gandipet", "manchirevula", "kokapet"],
    "kokapet":         ["narsingi", "gandipet", "financial district"],
    "uppal":           ["lb nagar", "habsiguda", "ramanthapur"],
    "dilsukhnagar":    ["lb nagar", "saroor nagar", "moosapet"],
    "ameerpet":        ["begumpet", "punjagutta", "sr nagar"],
    "banjara hills":   ["jubilee hills", "road no 12", "panjagutta"],
    "jubilee hills":   ["banjara hills", "madhapur", "film nagar"],
    "manikonda":       ["narsingi", "puppalaguda", "lanco hills"],
    "serilingampally": ["kondapur", "nallagandla", "tellapur", "miyapur"],
}

def get_nearby_areas(question_lower):
    found_areas, found_nearby = [], []
    for area, nearby in NEARBY_MAP.items():
        if area in question_lower:
            found_areas.append(area)
            found_nearby.extend(nearby)
    return found_areas, list(set(found_nearby))

# ----------------------------------------
# STEP 5: Build RAG Chains
# ----------------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 50, "lambda_mult": 0.7}
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an AI Real Estate Advisor for Hyderabad property market.

RULES:
1. Use retrieved documents to answer with exact numbers where available
2. If exact area not found, use data from nearby/similar areas and clearly mention it
3. ALWAYS provide useful numbers - never give a vague answer
4. Use **bold** for key numbers and area names
5. For rental trends: always give rupee ranges per month for different BHK types
6. For property rates: give per sq.ft or per sq.yard rates

--- CONVERSATION HISTORY ---
{chat_history}

--- RETRIEVED DOCUMENTS ---
{context}

--- QUESTION ---
{question}

Answer with specific numbers:"""
)

table_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an AI Real Estate Advisor for Hyderabad property market.
The user wants a TABLE comparison. You MUST respond with a proper markdown table.

MANDATORY TABLE FORMAT:
- Use markdown table syntax with | separators
- Always include columns: Area | Rate/Value | Unit | Notes
- If comparing two areas: Area | Metric | Area1 Value | Area2 Value
- Fill ALL cells - never leave empty cells
- After the table, add a brief 2-line summary

--- CONVERSATION HISTORY ---
{chat_history}

--- RETRIEVED DOCUMENTS ---
{context}

--- QUESTION ---
{question}

RESPOND WITH MARKDOWN TABLE ONLY (then brief summary):"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt}, verbose=False
)
qa_chain_table = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": table_prompt}, verbose=False
)
print("✅ RAG Chain Ready!")

# ----------------------------------------
# STEP 6: Sessions
# ----------------------------------------
sessions = {}

def create_new_session():
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = {
        "title": "New Chat",
        "clean_messages": [],
        "display_messages": [],
        "created_at": datetime.now().strftime("%b %d, %H:%M")
    }
    return sid

# ----------------------------------------
# STEP 7: Helpers
# ----------------------------------------
def correct_question(raw):
    try:
        r = correction_llm.invoke(
            f"Fix spelling and grammar only. Keep meaning same.\n"
            f"Return only the corrected question, nothing else.\n"
            f"Original: {raw}\nCorrected:"
        )
        return r.content.strip()
    except:
        return raw

def is_table_request(q):
    return any(kw in q.lower() for kw in [
        "table","tabular","in table","as table","compare","comparison",
        "vs","versus","side by side","list all","show all rates","tabulate"
    ])

def build_history_tuples(msgs):
    tuples = []
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            h = msgs[i]["content"]   if msgs[i]["role"]   == "user"      else ""
            a = msgs[i+1]["content"] if msgs[i+1]["role"] == "assistant" else ""
            if h and a:
                tuples.append((h, a))
    return tuples

def get_answer(question, history_tuples, use_table=False):
    weak = ["don't have enough data","not explicitly mentioned","cannot find",
            "no information","not available","do not have","i don't have","no specific","not found"]
    chain = qa_chain_table if use_table else qa_chain

    try:
        r = chain.invoke({"question": question, "chat_history": history_tuples})
        ans, src = r["answer"], r.get("source_documents", [])
        if not any(p in ans.lower() for p in weak):
            return ans, src
    except Exception as e:
        print(f"Try1 failed: {e}")

    found_areas, nearby = get_nearby_areas(question.lower())
    if nearby:
        enriched = (f"{question} Also search for data from nearby areas: "
                    f"{', '.join(nearby[:6])}. Use any available data.")
        try:
            r2 = chain.invoke({"question": enriched, "chat_history": history_tuples})
            ans2, src2 = r2["answer"], r2.get("source_documents", [])
            if not any(p in ans2.lower() for p in weak):
                note = ", ".join([a.title() for a in found_areas]) if found_areas else "requested area"
                return f"📍 *Nearby area data for **{note}**:*\n\n{ans2}", src2
        except Exception as e:
            print(f"Try2 failed: {e}")

    try:
        general = (f"Property rates, rental trends and market data in Hyderabad? "
                   f"User asked: '{question}'. Provide relevant data.")
        r3 = chain.invoke({"question": general, "chat_history": history_tuples})
        return f"📍 *General Hyderabad data:*\n\n{r3['answer']}", r3.get("source_documents", [])
    except Exception as e:
        print(f"Try3 failed: {e}")

    return "Unable to retrieve data. Please rephrase or ask about a specific area.", []

# ----------------------------------------
# STEP 8: Chat Function
# ----------------------------------------
def chat(user_message, current_sid, all_sessions_json):
    if not user_message.strip():
        return gr.update(), current_sid, all_sessions_json, "", ""

    global sessions
    if all_sessions_json and all_sessions_json != "{}":
        sessions = json.loads(all_sessions_json)
    if not current_sid or current_sid not in sessions:
        current_sid = create_new_session()

    corrected = correct_question(user_message)
    correction_note = f"✏️ Corrected: \"{corrected}\"" if corrected.lower() != user_message.lower() else ""

    history = build_history_tuples(sessions[current_sid]["clean_messages"])
    use_table = is_table_request(corrected)

    try:
        answer, _ = get_answer(corrected, history, use_table)
        display = (f"*{correction_note}*\n\n" + answer) if correction_note else answer
    except Exception as e:
        answer = f"Error: {e}"
        display = f"❌ **Error:** {e}"

    sessions[current_sid]["clean_messages"]  += [{"role":"user","content":corrected}, {"role":"assistant","content":answer}]
    sessions[current_sid]["display_messages"] += [{"role":"user","content":user_message}, {"role":"assistant","content":display}]

    if len(sessions[current_sid]["clean_messages"]) == 2:
        sessions[current_sid]["title"] = corrected[:38].strip()

    display_msgs = [{"role":m["role"],"content":m["content"]} for m in sessions[current_sid]["display_messages"]]
    return display_msgs, current_sid, json.dumps(sessions), correction_note, ""

# ----------------------------------------
# STEP 9: Session Handlers
# ----------------------------------------
def build_choices():
    return [(f"{d['title']}  ·  {d['created_at']}", sid) for sid,d in reversed(list(sessions.items()))]

def new_chat(sj):
    global sessions
    if sj and sj != "{}": sessions = json.loads(sj)
    sid = create_new_session()
    return [], sid, json.dumps(sessions), "", gr.update(choices=build_choices(), value=sid)

def load_chat(sid, sj):
    global sessions
    if sj and sj != "{}": sessions = json.loads(sj)
    if not sid or sid not in sessions: return [], sid, sj, ""
    msgs = [{"role":m["role"],"content":m["content"]} for m in sessions[sid]["display_messages"]]
    return msgs, sid, sj, ""

def delete_chat(sid, sj):
    global sessions
    if sj and sj != "{}": sessions = json.loads(sj)
    if sid in sessions: del sessions[sid]
    new_sid = list(sessions.keys())[-1] if sessions else create_new_session()
    msgs = [{"role":m["role"],"content":m["content"]} for m in sessions[new_sid]["display_messages"]]
    return msgs, new_sid, json.dumps(sessions), "", gr.update(choices=build_choices(), value=new_sid)

def handle_chat(msg, sid, sj):
    msgs, sid, sj, corr, _ = chat(msg, sid, sj)
    return msgs, sid, sj, corr, "", gr.update(choices=build_choices(), value=sid)

def handle_new(sj):
    msgs, sid, sj, _, dd = new_chat(sj)
    return msgs, sid, sj, "", dd

def handle_load(sid, sj):
    if not sid: return gr.update(), gr.update(), sj, ""
    msgs, sid, sj, _ = load_chat(sid, sj)
    return msgs, sid, sj, ""

def handle_delete(sid, sj):
    msgs, sid, sj, _, dd = delete_chat(sid, sj)
    return msgs, sid, sj, "", dd

# ----------------------------------------
# STEP 10: CSS
# ----------------------------------------
CSS = """
/* ── HARD RESET ── */
*, *::before, *::after { box-sizing: border-box !important; margin: 0 !important; padding: 0 !important; }
html, body { height: 100vh !important; overflow: hidden !important; background: #f9f9f8 !important; font-family: ui-sans-serif, -apple-system, 'Segoe UI', sans-serif !important; font-size: 14px !important; }
footer, .footer { display: none !important; }

/* ── GRADIO CONTAINER ── */
.gradio-container {
    max-width: 100vw !important; width: 100vw !important;
    height: 100vh !important; overflow: hidden !important;
    padding: 0 !important; margin: 0 !important;
    background: #f9f9f8 !important;
}

/* Kill ALL Gradio internal padding/gaps */
.contain, .gap, .wrap, .padded,
.gradio-container > div,
.gradio-container > div > div { 
    padding: 0 !important; margin: 0 !important;
    gap: 0 !important; border: none !important;
    background: transparent !important;
}

/* ── TWO-COLUMN SHELL ── */
#app-shell {
    display: flex !important;
    width: 100vw !important;
    height: 100vh !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* ── SIDEBAR ── */
#sidebar {
    width: 240px !important; min-width: 240px !important; max-width: 240px !important;
    height: 100vh !important; background: #f0efe9 !important;
    border-right: 1px solid #ddddd8 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    padding: 10px 8px !important; flex-shrink: 0 !important;
    display: flex !important; flex-direction: column !important;
}

#brand-box { padding: 0 2px 10px 2px !important; border-bottom: 1px solid #ddddd8 !important; margin-bottom: 8px !important; }
#brand-box p { font-size: 14px !important; font-weight: 700 !important; color: #1a1a1a !important; }

#new_chat_btn button {
    background: #fff !important; color: #1a1a1a !important;
    border: 1px solid #c8c8c2 !important; border-radius: 7px !important;
    font-size: 12px !important; font-weight: 500 !important;
    width: 100% !important; padding: 7px 10px !important;
    margin-bottom: 10px !important; cursor: pointer !important; text-align: left !important;
}
#new_chat_btn button:hover { background: #e8e8e4 !important; }

.sec-lbl {
    font-size: 10px !important; font-weight: 700 !important; color: #8a8a85 !important;
    text-transform: uppercase !important; letter-spacing: 0.6px !important;
    padding: 6px 2px 3px 2px !important; display: block !important;
}

#session_list label, #session_list .label-wrap { display: none !important; }
#session_list select {
    background: #fff !important; color: #1a1a1a !important;
    border: 1px solid #c8c8c2 !important; border-radius: 7px !important;
    font-size: 11px !important; padding: 6px 8px !important; width: 100% !important;
}

#delete_btn button {
    background: transparent !important; color: #999 !important;
    border: 1px solid #ddddd8 !important; border-radius: 7px !important;
    font-size: 11px !important; width: 100% !important;
    padding: 5px 8px !important; margin: 3px 0 10px 0 !important; cursor: pointer !important;
}
#delete_btn button:hover { background: #fde8e8 !important; color: #c0392b !important; border-color: #f5c6c6 !important; }

.eq-btn button {
    background: transparent !important; color: #4a4a45 !important;
    border: none !important; border-radius: 5px !important; font-size: 11px !important;
    text-align: left !important; padding: 5px 6px !important; width: 100% !important;
    cursor: pointer !important; white-space: normal !important; line-height: 1.3 !important;
}
.eq-btn button:hover { background: #e4e4e0 !important; color: #1a1a1a !important; }

/* ── MAIN PANEL: flex column fills remaining width ── */
#main-col {
    flex: 1 1 0 !important; min-width: 0 !important;
    height: 100vh !important;
    display: flex !important; flex-direction: column !important;
    overflow: hidden !important; background: #f9f9f8 !important;
}

/* ── TOP BAR: compact, fixed ── */
#topbar {
    flex: 0 0 auto !important;
    background: #f9f9f8 !important; border-bottom: 1px solid #e8e8e4 !important;
    padding: 7px 16px !important;
}
#topbar-title { font-size: 14px !important; font-weight: 600 !important; color: #1a1a1a !important; line-height: 1.3 !important; }
#topbar-sub   { font-size: 11px !important; color: #8a8a85 !important; line-height: 1.2 !important; }

/* ── CHATBOT: flex-grow fills all middle space ── */
#chatbot {
    flex: 1 1 0 !important; min-height: 0 !important;
    overflow-y: auto !important; background: transparent !important;
    border: none !important; padding: 12px 16px !important;
}

/* Flatten Gradio's internal chatbot wrappers */
#chatbot > div,
#chatbot > div > div {
    height: 100% !important; border: none !important;
    background: transparent !important; overflow-y: visible !important;
    padding: 0 !important; margin: 0 !important;
}

#chatbot .message-wrap { gap: 10px !important; padding: 0 !important; }
#chatbot .message      { border: none !important; box-shadow: none !important; background: transparent !important; }

/* User bubble — WIDE, right side */
#chatbot .user {
    display: flex !important;
    justify-content: flex-end !important;
    background: transparent !important;
    padding: 0 !important;
}
#chatbot .user > div,
#chatbot .user .prose {
    background: #e9e9e4 !important;
    color: #1a1a1a !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 9px 14px !important;
    max-width: 75% !important;          /* wider user bubble */
    width: fit-content !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    word-break: break-word !important;
}

/* Bot bubble — left */
#chatbot .bot {
    display: flex !important;
    justify-content: flex-start !important;
    background: transparent !important;
    padding: 0 !important;
}
#chatbot .bot > div,
#chatbot .bot .prose {
    background: #ffffff !important; color: #1a1a1a !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 10px 16px !important; max-width: 86% !important;
    font-size: 13px !important; line-height: 1.6 !important;
    border: 1px solid #e4e4e0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    word-break: break-word !important;
}

/* All text inside bot: dark */
#chatbot .bot p, #chatbot .bot span, #chatbot .bot li,
#chatbot .bot strong, #chatbot .bot em,
#chatbot .bot td, #chatbot .bot th,
#chatbot .bot h1, #chatbot .bot h2, #chatbot .bot h3 { color: #1a1a1a !important; }

/* ── TABLES ── */
#chatbot table {
    border-collapse: collapse !important; width: 100% !important;
    font-size: 12px !important; border: 1px solid #ddddd8 !important; margin: 8px 0 !important;
}
#chatbot th { background: #f0efe9 !important; color: #1a1a1a !important; font-weight: 700 !important; padding: 8px 12px !important; border: 1px solid #ddddd8 !important; text-align: left !important; }
#chatbot td { padding: 7px 12px !important; border: 1px solid #e8e8e4 !important; color: #1a1a1a !important; vertical-align: top !important; }
#chatbot tr:nth-child(even) td { background: #fafaf8 !important; }
#chatbot tr:hover td { background: #f4f4f0 !important; }
#chatbot strong { font-weight: 700 !important; color: #1a1a1a !important; }
#chatbot code { background: #f0efe9 !important; padding: 1px 5px !important; border-radius: 3px !important; font-size: 11px !important; color: #c7254e !important; }

/* ── INPUT WRAP: pinned to bottom, auto height ── */
#input-wrap {
    flex: 0 0 auto !important;
    background: #f9f9f8 !important; border-top: 1px solid #e8e8e4 !important;
    padding: 6px 16px 10px 16px !important;
}

/* Auto-corrected: hidden when empty, small when shown */
#corrected_box { margin-bottom: 4px !important; }
#corrected_box textarea {
    background: #fffef0 !important; color: #7c5c00 !important;
    border: 1px solid #e8d88a !important; border-radius: 6px !important;
    font-size: 11px !important; padding: 3px 10px !important;
    min-height: 0 !important; line-height: 1.3 !important;
}
#corrected_box label span { color: #9a7a00 !important; font-size: 10px !important; }

/* Input box — full width, at very bottom */
#input-box {
    background: #fff !important;
    border: 1.5px solid #c8c8c2 !important; border-radius: 12px !important;
    display: flex !important; align-items: flex-end !important;
    padding: 5px 5px 5px 14px !important; gap: 6px !important;
    box-shadow: 0 1px 5px rgba(0,0,0,0.08) !important;
    width: 100% !important;
}
#input-box:focus-within { border-color: #8a8a85 !important; }

/* Text input inside the box */
#msg_input { flex: 1 1 0 !important; min-width: 0 !important; border: none !important; background: transparent !important; }
#msg_input textarea {
    background: transparent !important; color: #1a1a1a !important;
    border: none !important; outline: none !important;
    font-size: 13px !important; padding: 6px 0 !important;
    resize: none !important; line-height: 1.4 !important;
    box-shadow: none !important; width: 100% !important;
}
#msg_input textarea::placeholder { color: #bbb !important; }
#msg_input textarea:focus { border: none !important; outline: none !important; box-shadow: none !important; }

/* Send button */
#send_btn { flex: 0 0 auto !important; }
#send_btn button {
    background: #1a1a1a !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    width: 36px !important; height: 36px !important; min-width: 36px !important;
    font-size: 18px !important; cursor: pointer !important;
    padding: 0 !important; line-height: 1 !important;
}
#send_btn button:hover { background: #333 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c8c8c2; border-radius: 4px; }
"""

# JS to fix layout after Gradio renders
JS = """
function() {
    function fix() {
        // Force the Gradio row containing sidebar+main to be a flex row
        var shell = document.getElementById('app-shell');
        if (shell) {
            shell.style.cssText = 'display:flex!important;width:100vw!important;height:100vh!important;overflow:hidden!important;gap:0!important;';
            var p = shell.parentElement;
            while (p && p !== document.body) {
                p.style.cssText += 'padding:0!important;margin:0!important;height:100vh!important;overflow:hidden!important;background:transparent!important;';
                p = p.parentElement;
            }
        }

        var sidebar   = document.getElementById('sidebar');
        var mainCol   = document.getElementById('main-col');
        var topbar    = document.getElementById('topbar');
        var chatbot   = document.getElementById('chatbot');
        var inputWrap = document.getElementById('input-wrap');

        if (sidebar) sidebar.style.cssText = 'width:240px!important;min-width:240px!important;max-width:240px!important;height:100vh!important;background:#f0efe9!important;border-right:1px solid #ddddd8!important;overflow-y:auto!important;overflow-x:hidden!important;padding:10px 8px!important;flex-shrink:0!important;display:flex!important;flex-direction:column!important;';

        if (mainCol) {
            mainCol.style.cssText = 'flex:1 1 0!important;min-width:0!important;height:100vh!important;display:flex!important;flex-direction:column!important;overflow:hidden!important;background:#f9f9f8!important;';
            // Fix all intermediate Gradio wrapper divs inside main-col
            var kids = mainCol.children;
            for (var i = 0; i < kids.length; i++) {
                var kid = kids[i];
                if (kid.id === 'topbar' || kid.id === 'chatbot' || kid.id === 'input-wrap') continue;
                kid.style.cssText += 'display:flex!important;flex-direction:column!important;flex:1 1 0!important;min-height:0!important;padding:0!important;margin:0!important;overflow:hidden!important;border:none!important;background:transparent!important;';
            }
        }

        if (topbar)    topbar.style.cssText    = 'flex:0 0 auto!important;background:#f9f9f8!important;border-bottom:1px solid #e8e8e4!important;padding:7px 16px!important;';
        if (chatbot)   chatbot.style.cssText   = 'flex:1 1 0!important;min-height:0!important;overflow-y:auto!important;background:transparent!important;border:none!important;padding:12px 16px!important;';
        if (inputWrap) inputWrap.style.cssText = 'flex:0 0 auto!important;background:#f9f9f8!important;border-top:1px solid #e8e8e4!important;padding:6px 16px 10px 16px!important;';
    }

    fix();
    setTimeout(fix, 100);
    setTimeout(fix, 400);
    setTimeout(fix, 900);
}
"""

# ----------------------------------------
# STEP 11: Build UI
# ----------------------------------------
with gr.Blocks(title="🏠 AI Real Estate Advisor", css=CSS, js=JS) as app:

    current_sid_state   = gr.State("")
    sessions_json_state = gr.State("{}")

    with gr.Row(elem_id="app-shell"):

        # ── SIDEBAR ──────────────────────────────────
        with gr.Column(scale=0, min_width=240, elem_id="sidebar"):
            gr.HTML('<div id="brand-box"><p>🏠 RE Advisor &nbsp;<span style="font-size:11px;font-weight:400;color:#8a8a85;">Hyderabad AI</span></p></div>')
            new_chat_btn = gr.Button("✏️  New Chat", elem_id="new_chat_btn")
            gr.HTML("<span class='sec-lbl'>Previous Chats</span>")
            session_dropdown = gr.Dropdown(choices=[], value=None, label="",
                                           interactive=True, elem_id="session_list", show_label=False)
            delete_btn = gr.Button("🗑️  Delete this chat", elem_id="delete_btn", size="sm")
            gr.HTML("<span class='sec-lbl'>Quick Questions</span>")
            examples = [
                "Property rates in Kondapur?",
                "Compare Kukatpally vs Kondapur rates in table",
                "Rental trends in Hyderabad in table",
                "Stamp duty in Telangana?",
                "Guideline rate in Gandipet?",
                "Documents for registration?",
                "Flat valuation in Badepally?",
                "Building permit fee Serilingampally?",
                "Compare Jadcherla vs Badepally in table",
            ]
            example_btns = [gr.Button(ex, size="sm", elem_classes=["eq-btn"]) for ex in examples]

        # ── MAIN CHAT ─────────────────────────────────
        with gr.Column(scale=1, elem_id="main-col"):

            gr.HTML("""
            <div id="topbar">
                <div id="topbar-title">🏠 AI Real Estate Insight &amp; Trend Advisor</div>
                <div id="topbar-sub">RAG · Hyderabad · 11,000+ PDFs · Say "compare in table" for tables</div>
            </div>
            """)

            chatbot = gr.Chatbot(
                value=[], label="", height=520,
                elem_id="chatbot", show_label=False, render_markdown=True,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=re&backgroundColor=d1e8d1")
            )

            with gr.Column(elem_id="input-wrap"):
                corrected_display = gr.Textbox(
                    label="✏️ Auto-corrected", interactive=False,
                    lines=1, elem_id="corrected_box",
                    show_label=True, container=True
                )
                with gr.Row(elem_id="input-box"):
                    msg_input = gr.Textbox(
                        placeholder="Ask about Hyderabad real estate... (say 'compare in table' for tables!)",
                        lines=1, max_lines=4, scale=10,
                        elem_id="msg_input", show_label=False, container=False
                    )
                    send_btn = gr.Button("↑", variant="primary", scale=0,
                                        elem_id="send_btn", min_width=36)

    # ── EVENTS ───────────────────────────────────────
    send_btn.click(fn=handle_chat,
        inputs=[msg_input, current_sid_state, sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display, msg_input, session_dropdown])

    msg_input.submit(fn=handle_chat,
        inputs=[msg_input, current_sid_state, sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display, msg_input, session_dropdown])

    new_chat_btn.click(fn=handle_new,
        inputs=[sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display, session_dropdown])

    session_dropdown.change(fn=handle_load,
        inputs=[session_dropdown, sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display])

    delete_btn.click(fn=handle_delete,
        inputs=[current_sid_state, sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display, session_dropdown])

    for btn in example_btns:
        btn.click(fn=lambda t: t, inputs=[btn], outputs=[msg_input])

    app.load(fn=handle_new,
        inputs=[sessions_json_state],
        outputs=[chatbot, current_sid_state, sessions_json_state, corrected_display, session_dropdown])

# ── LAUNCH ───────────────────────────────────────────
print("\n🚀 Starting AI Real Estate Advisor...")
print("Open browser at: http://localhost:7860")
app.launch(server_name="0.0.0.0", server_port=7860, share=False)