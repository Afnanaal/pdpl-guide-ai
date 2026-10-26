import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import (
    DOCUMENT_NAME,
    DOCUMENT_SUBTITLE,
    ISSUING_AUTHORITY,
    DOCUMENT_VERSION,
    LEGAL_DISCLAIMER_AR,
    DEFAULT_TOP_K
)
from src.extractor import load_and_process_pdf, locate_pdf_path
from src.rag import RAGIndex, answer_question

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PDPL Guide AI | مساعد قواعد تعيين مسؤول حماية البيانات",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic RTL Layout & Polished Design Aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&family=Cairo:wght@400;600;700;900&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Tajawal', sans-serif;
    direction: rtl;
    text-align: right;
}

/* Titles and Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
    text-align: right;
}

/* Header Banner */
.hero-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #064E3B 100%);
    color: white;
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #F8FAFC, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #94A3B8;
    margin-top: 0.6rem;
    line-height: 1.6;
}

.hero-badge-container {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 500;
}

/* Mode Indicators */
.mode-badge-llm {
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

.mode-badge-retrieval {
    background: rgba(59, 130, 246, 0.12);
    color: #3B82F6;
    border: 1px solid rgba(59, 130, 246, 0.3);
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

/* Source Box Styling */
.source-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-right: 4px solid #10B981;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-weight: 700;
    color: #0F172A;
}

.source-page-badge {
    background: #064E3B;
    color: #ECFDF5;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
}

.source-excerpt {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #334155;
    background: white;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    border: 1px dashed #CBD5E1;
    margin-top: 0.5rem;
    white-space: pre-wrap;
}

/* Disclaimer Box */
.disclaimer-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-right: 4px solid #F59E0B;
    color: #92400E;
    padding: 0.85rem 1.2rem;
    border-radius: 8px;
    font-size: 0.88rem;
    line-height: 1.6;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Chat bubble enhancements */
.stChatMessage {
    direction: rtl;
    text-align: right;
    border-radius: 12px;
}

/* RTL Sidebar */
[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}

/* RTL Form Inputs */
.stTextInput > div > div > input {
    direction: rtl;
    text-align: right;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Cache PDF Extraction and Vector Index
@st.cache_resource(show_spinner="جاري فحص الوثيقة الرسمية وبناء الفهرس الدلالي...")
def get_rag_engine():
    pdf_path, chunks = load_and_process_pdf()
    if not pdf_path or not chunks:
        return None, None, None
    index = RAGIndex(chunks)
    return pdf_path, chunks, index

pdf_path, chunks, rag_index = get_rag_engine()

# Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("### 🏛️ سدايا — SDAIA")
    st.markdown("#### نظام حماية البيانات الشخصية")
    st.markdown(f"**الوثيقة الرسمية:**  \n«{DOCUMENT_NAME}»")
    st.caption(f"{DOCUMENT_VERSION} | الصادرة عن سدايا")
    
    st.divider()
    
    st.markdown("### ⚙️ إعدادات النموذج والتطبيق")
    
    # Model Provider Selector
    provider_options = {
        "gemini": "Google Gemini (موصى به للذكاء التوليدي)",
        "openai": "OpenAI (GPT-4o-mini / GPT-4o)",
        "groq": "Groq (Llama-3)",
        "retrieval_only": "وضع الاسترجاع المباشر فقط (بدون LLM)"
    }
    
    selected_provider_key = st.selectbox(
        "مزود الخدمة:",
        options=list(provider_options.keys()),
        format_func=lambda x: provider_options[x],
        index=0
    )
    
    # API Key Configuration
    env_key = ""
    if selected_provider_key == "gemini":
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        default_model = "gemini-1.5-flash"
    elif selected_provider_key == "openai":
        env_key = os.getenv("OPENAI_API_KEY") or ""
        default_model = "gpt-4o-mini"
    elif selected_provider_key == "groq":
        env_key = os.getenv("GROQ_API_KEY") or ""
        default_model = "llama-3.3-70b-versatile"
    else:
        default_model = "none"

    user_api_key = ""
    if selected_provider_key != "retrieval_only":
        user_api_key = st.text_input(
            "مفتاح الـ API (اختياري إذا تم تعيينه في .env):",
            value=env_key,
            type="password",
            help="إذا تركت هذا الحقل فارغاً ولم يتوفر مفتاح في ملف .env، سينتقل التطبيق تلقائياً إلى وضع الاسترجاع المباشر الأصيل المعتمد."
        )
        selected_model = st.text_input("اسم النموذج:", value=default_model)
    else:
        selected_model = "retrieval-only"

    # Top-K Passages Slider
    top_k = st.slider("عدد المقاطع المسترجعة (Top-K):", min_value=1, max_value=5, value=DEFAULT_TOP_K)
    
    # Current Mode Badge
    has_active_llm = bool(user_api_key or env_key) and (selected_provider_key != "retrieval_only")
    st.markdown("#### الوضع الحالي للتطبيق:")
    if has_active_llm:
        st.markdown(
            '<div class="mode-badge-llm">🟢 وضع الذكاء الاصطناعي التوليدي (LLM Mode)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="mode-badge-retrieval">🔵 وضع الاسترجاع المباشر (Retrieval-Only Mode)</div>',
            unsafe_allow_html=True
        )
        st.caption("يعرض النصوص الرسمية المسترجعة من الوثيقة بدقة دون استدعاء نموذج توليدي خارجي.")

    st.divider()

    # Document Status & Metadata
    if pdf_path:
        st.markdown("### 📊 بيانات الفهرسة والوثيقة")
        st.success("✅ تم فحص الوثيقة وفهرستها بنجاح")
        st.write(f"- **اسم الملف:** `{os.path.basename(pdf_path)}`")
        st.write(f"- **عدد المقاطع المفهرسة:** {len(chunks)} مقطع قانوني")
        st.write("- **إجمالي الصفحات:** 11 صفحة")
    else:
        st.error("❌ لم يتم العثور على ملف PDF داخل مجلد `data/`")
        st.info("يرجى وضع ملف الوثيقة الرسمية في مجلد `data/` ثم إعادة تحميل الصفحة.")

    # Reset Chat Button
    if st.button("🗑️ مسح المحادثة وإعادة الضبط", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header / Hero
st.markdown(
    f"""
    <div class="hero-card">
        <h1 class="hero-title">PDPL Guide AI</h1>
        <div class="hero-subtitle">{DOCUMENT_SUBTITLE}</div>
        <div class="hero-badge-container">
            <span class="hero-badge">⚖️ نظام حماية البيانات الشخصية السعودي</span>
            <span class="hero-badge">📜 {DOCUMENT_VERSION}</span>
            <span class="hero-badge">🏛️ سدايا SDAIA</span>
            <span class="hero-badge">🔍 Retrieval-Augmented Generation (RAG)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Mandatory Legal Disclaimer Banner
st.markdown(
    f"""
    <div class="disclaimer-card">
        <strong>⚖️ تنبيه قانوني:</strong> {LEGAL_DISCLAIMER_AR}
    </div>
    """,
    unsafe_allow_html=True
)

# Missing Document Warning (if applicable)
if not pdf_path or not rag_index:
    st.error(
        """
        ### ⚠️ ملف الوثيقة الرسمية غير موجود!
        لم يتم العثور على وثيقة PDF داخل مجلد `data/`.
        يرجى وضع وثيقة:
        `قواعد تعيين مسؤول حماية البيانات الشخصية.pdf`
        داخل مسار `data/` ثم تحديث الصفحة لبناء الفهرس الدلالي.
        """
    )
    st.stop()

# Quick Sample Question Prompts for Testing
st.markdown("##### 💡 أسئلة نموذجية سريعة للتجربة والاختبار:")
col1, col2, col3, col4 = st.columns(4)

sample_q = None
with col1:
    if st.button("📌 متطلبات التعيين (المادة 4)", use_container_width=True):
        sample_q = "ما المتطلبات التي يجب توفرها عند تعيين مسؤول حماية البيانات الشخصية؟"
with col2:
    if st.button("📌 حالات التعيين (المادة 5)", use_container_width=True):
        sample_q = "ما الحالات التي يجب فيها تعيين مسؤول حماية البيانات الشخصية؟"
with col3:
    if st.button("📌 مهام المسؤول (المادة 8)", use_container_width=True):
        sample_q = "ما المهام التي يقوم بها مسؤول حماية البيانات الشخصية؟"
with col4:
    if st.button("❓ سؤال خارج النطاق (فرنسا)", use_container_width=True):
        sample_q = "ما عاصمة فرنسا؟"

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render Sources if available
        if msg.get("sources"):
            st.markdown("#### 📚 المواد المستشهد بها من الوثيقة الرسمية:")
            for s in msg["sources"]:
                st.markdown(
                    f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span>📖 {s['article_number']}: {s['article_title']}</span>
                            <span class="source-page-badge">صفحة {s['page_number']}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748B;">الوثيقة: {s['source_doc']}</div>
                        <div class="source-excerpt">{s['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # Render Expandable Retrieved Passages
        if msg.get("retrieved_chunks"):
            with st.expander("🔍 المقاطع المسترجعة (RAG Inspection Details)"):
                for idx, r in enumerate(msg["retrieved_chunks"], start=1):
                    c = r["chunk"]
                    score = r["score"]
                    st.markdown(f"**المقطع {idx}:** {c['article_number']} - {c['article_title']} (صفحة {c['page_number']}) — *درجة التطابق: `{score:.4f}`*")
                    st.code(c["content"], language="text")

# Chat Input Handler
user_input = st.chat_input("اكتب سؤالك هنا حول قواعد تعيين مسؤول حماية البيانات الشخصية...")
active_query = sample_q or user_input

if active_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)
        
    # Process with RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("جاري استرجاع النصوص النظامية وتحليل الإجابة..."):
            resolved_key = user_api_key or env_key
            result = answer_question(
                query=active_query,
                rag_index=rag_index,
                top_k=top_k,
                api_key=resolved_key,
                provider=selected_provider_key,
                model_name=selected_model
            )
            
            # Display Answer
            st.markdown(result["answer"])
            
            # Display Source Citations
            if result.get("sources"):
                st.markdown("#### 📚 المواد المستشهد بها من الوثيقة الرسمية:")
                for s in result["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-header">
                                <span>📖 {s['article_number']}: {s['article_title']}</span>
                                <span class="source-page-badge">صفحة {s['page_number']}</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #64748B;">الوثيقة: {s['source_doc']}</div>
                            <div class="source-excerpt">{s['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display Expandable Retrieved Chunks
            if result.get("retrieved_chunks"):
                with st.expander("🔍 المقاطع المسترجعة (RAG Inspection Details)"):
                    for idx, r in enumerate(result["retrieved_chunks"], start=1):
                        c = r["chunk"]
                        score = r["score"]
                        st.markdown(f"**المقطع {idx}:** {c['article_number']} - {c['article_title']} (صفحة {c['page_number']}) — *درجة التطابق: `{score:.4f}`*")
                        st.code(c["content"], language="text")
                        
            # Save Assistant Response to Session State
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "retrieved_chunks": result.get("retrieved_chunks", [])
            })
