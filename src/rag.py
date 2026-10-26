import os
import re
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DOCUMENT_NAME,
    ISSUING_AUTHORITY,
    RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    SYSTEM_PROMPT,
    LEGAL_DISCLAIMER_AR
)

# Domain synonyms to enhance retrieval across Arabic phrasing
LEGAL_KEYWORDS_MAP = {
    "art_4": "متطلبات شروط معايير مؤهلات تعيين اختيار تعيينه مسؤول حماية البيانات الشخصية خبرة مؤهل علمي مناسب ممارسات إدارة المخاطر حوادث تسرب البيانات ومعالجتها المتطلبات النظامية والتنظيمية الأمانة النزاهة جريمة مخلة بالشرف موظف مسؤول متعاقد خارجي",
    "art_5_1": "أحوال حالات متى يجب يلزم إلزام التعيين تعيين تحديد مسؤول حماية البيانات الشخصية جهة عامة خدمات نطاق واسع الأنشطة الأساسية مراقبة منتظمة وممنهجة بيانات حساسة معايير تحديد النطاق الواسع عدد الحجم النوع النطاق الجغرافي الفئات",
    "art_5_2": "أمثلة المراقبة المنتظمة والممنهجة الأنشطة الأساسية الصحة اللياقة البدنية الأجهزة القابلة للارتداء التحليل السلوكي تقييم المخاطر تتبع الموقع ملفات تعريف الارتباط الكوكيز كاميرات المراقبة شركات التأمين البيانات الصحية المالية الائتمانية التسويق الموارد البشرية",
    "art_8": "المهام مهام أدوار واجبات مسؤوليات مسؤول حماية البيانات الشخصية ما المهام التي يقوم بها مسؤول حماية البيانات الشخصية تقديم الدعم المشورة تطوير السياسات والإجراءات الداخلية الأنشطة التوعوية تدريب ونقل المعرفة أخلاقيات التعامل خطط الاستجابة لحوادث تسرب البيانات تقارير دورية توصيات متابعة وثائق سدايا التنظيمية الأنظمة التقنية الحديثة",
    "art_7": "تفاصيل الاتصال وسائل التواصل وسيلة تواصل أصحاب البيانات واضحة سهلة تزويد الجهة المختصة سدايا منصة حوكمة البيانات الوطنية تحديث البيانات",
    "art_6": "توثيق تعيين كتابياً توثيق التعيين موظف إبرام اتفاقية متعاقد خارجي الإعلان الفوري داخل جهة التحكم وسيلة التواصل",
    "art_9": "أحكام عامة مراجعة دورية أحوال التعيين إلزامي تعيين طوعي اختياري جهة غير ملزمة اتفاقية جهة المعالجة تمكين ودعم الموارد اللازمة عدم تكليفه بمهام متعارضة الاستقلالية التدريب والتطوير شهادات مهنية الارتباط التنظيمي بمكتب إدارة البيانات",
    "art_1": "التعريفات الجهة المختصة الهيئة السعودية للبيانات والذكاء الاصطناعي سدايا مسؤول حماية البيانات الشخصية الأنشطة الأساسية",
    "art_2": "الغرض أهداف القواعد وضع الحد الأدنى متطلبات التعيين إيضاح المفاهيم أدوار ومهام",
    "art_3": "النطاق سريان تطبيق القواعد جميع جهات التحكم نظام حماية البيانات الشخصية",
    "art_10": "المراجعة والتعديل سدايا الجهة المختصة تحديث القواعد",
    "art_11": "النفاذ تاريخ النشر السريان الموقع الرسمي",
}

def normalize_arabic_query(text: str) -> str:
    """Expand and normalize Arabic query tokens for legal retrieval."""
    text = text.strip()
    # Normalize common query variations and synonyms
    expansions = [
        (r'\b(الحالات|حالات|حالة)\b', 'الحالات الأحوال أحوال حالات'),
        (r'\b(متطلبات|شروط|مؤهلات)\b', 'متطلبات شروط مؤهلات معايير'),
        (r'\b(مهام|المهام|أدوار|مسؤوليات|وظائف)\b', 'مهام المهام أدوار ومهام واجبات مسؤوليات'),
        (r'\b(أين|كيف|طريقة|إجراءات)\b', 'إجراءات كيفية طريقة'),
        (r'\b(عقوبات|مخالفات)\b', 'عقوبات مخالفات أحكام'),
    ]
    for pattern, repl in expansions:
        text = re.sub(pattern, repl, text)
    return text

class RAGIndex:
    """Searchable vector index combining word and character n-gram TF-IDF representations."""
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.corpus = []
        for c in chunks:
            synonyms = LEGAL_KEYWORDS_MAP.get(c['chunk_id'], "")
            # Boost article title and legal keywords in index
            text_repr = f"{c['article_number']} {c['article_title']} {synonyms} {synonyms} {c['content']}"
            self.corpus.append(text_repr)
            
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=1
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.corpus)
        
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1
        )
        self.word_matrix = self.word_vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """Retrieve the top-k most relevant chunks for a user query."""
        norm_query = normalize_arabic_query(query)
        char_sim = cosine_similarity(self.char_vectorizer.transform([norm_query]), self.char_matrix)[0]
        word_sim = cosine_similarity(self.word_vectorizer.transform([norm_query]), self.word_matrix)[0]
        
        # Combined score weighting character and word representations
        scores = 0.4 * char_sim + 0.6 * word_sim
        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in ranked_indices[:top_k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })
        return results

def call_gemini(api_key: str, prompt: str, context: str, model: str = "gemini-1.5-flash") -> str:
    """Call Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT}\n\n[المقاطع المسترجعة من الوثيقة الرسمية]:\n{context}\n\n[سؤال المستخدم]:\n{prompt}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API returned error {resp.status_code}: {resp.text}")
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return "لا تتوفر إجابة من النموذج التوليدي."
    return candidates[0]["content"]["parts"][0]["text"].strip()

def call_openai_compatible(api_key: str, prompt: str, context: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI or compatible endpoint (e.g. Groq, Ollama)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[المقاطع المسترجعة من الوثيقة الرسمية]:\n{context}\n\n[سؤال المستخدم]:\n{prompt}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"API returned error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def synthesize_retrieval_answer(query: str, retrieved: list) -> str:
    """
    Synthesize an authoritative, authentic legal answer directly from retrieved chunks
    when in Retrieval-Only mode (without calling an external LLM).
    """
    if not retrieved:
        return "لا تتوفر معلومات كافية في الوثيقة للإجابة عن هذا السؤال."
    
    top_result = retrieved[0]
    top_chunk = top_result["chunk"]
    top_score = top_result["score"]
    
    # Format clear Arabic answer grounded directly on top chunks
    answer_parts = []
    answer_parts.append(f"**بناءً على نصوص الوثيقة الرسمية («{DOCUMENT_NAME}»):**\n")
    
    for idx, item in enumerate(retrieved):
        c = item["chunk"]
        art_num = c["article_number"]
        art_title = c["article_title"]
        page_num = c["page_number"]
        
        # Format excerpt cleanly
        content_lines = c["content"].splitlines()
        # skip header line if repeated
        body_lines = [l for l in content_lines if not l.startswith("المادة ")]
        clean_excerpt = "\n".join(body_lines).strip()
        
        answer_parts.append(f"### 📌 {art_num}: {art_title} (صفحة {page_num})\n")
        answer_parts.append(f"{clean_excerpt}\n")
    
    return "\n".join(answer_parts)

def answer_question(
    query: str,
    rag_index: RAGIndex,
    top_k: int = DEFAULT_TOP_K,
    api_key: str = None,
    provider: str = "gemini",
    model_name: str = "gemini-1.5-flash",
    custom_base_url: str = None
) -> dict:
    """
    Main RAG pipeline entrypoint.
    Returns:
      {
        'answer': str,
        'mode': 'llm' or 'retrieval_only',
        'sources': list of dicts,
        'retrieved_chunks': list of dicts,
        'is_out_of_scope': bool,
        'provider': str
      }
    """
    query = query.strip()
    if not query:
        return {
            "answer": "يرجى كتابة سؤال للبدء في البحث والتحليل.",
            "mode": "none",
            "sources": [],
            "retrieved_chunks": [],
            "is_out_of_scope": False,
            "provider": provider
        }
    
    # 1. Retrieve top-k chunks
    retrieved = rag_index.retrieve(query, top_k=top_k)
    top_score = retrieved[0]["score"] if retrieved else 0.0
    
    # 2. Check out-of-scope / unrelated questions
    if top_score < RELEVANCE_THRESHOLD:
        out_of_scope_msg = (
            "لا تتوفر معلومات كافية في الوثيقة للإجابة عن هذا السؤال.\n\n"
            "⚠️ **ملاحظة:** هذا السؤال يقع خارج النطاق الموضوعي للوثيقة المعتمدة. "
            f"يختص هذا المساعد حصراً بـ **«{DOCUMENT_NAME}»** الصادرة عن {ISSUING_AUTHORITY}، "
            "والتي تتناول أحكام ومتطلبات ومهام مسؤول حماية البيانات الشخصية في المملكة العربية السعودية."
        )
        return {
            "answer": out_of_scope_msg,
            "mode": "out_of_scope",
            "sources": [],
            "retrieved_chunks": retrieved,
            "is_out_of_scope": True,
            "provider": provider
        }
    
    # 3. Format sources metadata
    sources = []
    context_blocks = []
    for item in retrieved:
        c = item["chunk"]
        sources.append({
            "article_number": c["article_number"],
            "article_title": c["article_title"],
            "page_number": c["page_number"],
            "page_range": c.get("page_range", str(c["page_number"])),
            "source_doc": c["source_doc"],
            "score": item["score"],
            "content": c["content"]
        })
        context_blocks.append(
            f"[{c['article_number']} - {c['article_title']} (صفحة {c['page_number']})]:\n{c['content']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)
    
    # 4. Resolve API Key from argument or environment
    resolved_key = api_key
    if not resolved_key:
        if provider == "gemini":
            resolved_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif provider == "openai":
            resolved_key = os.getenv("OPENAI_API_KEY")
        elif provider == "groq":
            resolved_key = os.getenv("GROQ_API_KEY")
            
    # 5. Generate Answer via LLM or Retrieval-Only Mode
    if resolved_key and provider != "retrieval_only":
        try:
            if provider == "gemini":
                llm_response = call_gemini(resolved_key, query, context_text, model=model_name)
            elif provider == "openai":
                base_url = custom_base_url or "https://api.openai.com/v1"
                llm_response = call_openai_compatible(resolved_key, query, context_text, base_url=base_url, model=model_name)
            elif provider == "groq":
                base_url = custom_base_url or "https://api.groq.com/openai/v1"
                llm_response = call_openai_compatible(resolved_key, query, context_text, base_url=base_url, model=model_name)
            else:
                llm_response = call_gemini(resolved_key, query, context_text, model=model_name)
                
            return {
                "answer": llm_response,
                "mode": "llm",
                "sources": sources,
                "retrieved_chunks": retrieved,
                "is_out_of_scope": False,
                "provider": provider
            }
        except Exception as e:
            # Fallback to retrieval-only mode if LLM call fails with clear note
            retrieval_ans = synthesize_retrieval_answer(query, retrieved)
            error_note = (
                f"> ⚠️ **تنبيه:** تعذر الاتصال بالنموذج التوليدي ({str(e)}). "
                "تم تفعيل **وضع الاسترجاع المباشر** لعرض النصوص القانونية المعتمدة المسترجعة من الوثيقة:\n\n"
            )
            return {
                "answer": error_note + retrieval_ans,
                "mode": "retrieval_only",
                "sources": sources,
                "retrieved_chunks": retrieved,
                "is_out_of_scope": False,
                "provider": "retrieval_fallback"
            }
    else:
        # Authentic Retrieval-Only mode
        retrieval_ans = synthesize_retrieval_answer(query, retrieved)
        return {
            "answer": retrieval_ans,
            "mode": "retrieval_only",
            "sources": sources,
            "retrieved_chunks": retrieved,
            "is_out_of_scope": False,
            "provider": "retrieval_only"
        }
