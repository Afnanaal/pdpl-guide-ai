import os
import glob
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from bidi.algorithm import get_display

from src.config import DATA_DIR, DOCUMENT_NAME

def fix_word(w: str) -> str:
    """Fix individual word issues resulting from PDF glyph encoding."""
    clean_punct_start = ""
    clean_punct_end = ""
    m_lead = re.match(r'^([\s\.,:;\-\(\)\[\]!؟]+)(.*)', w)
    if m_lead:
        clean_punct_start = m_lead.group(1)
        w = m_lead.group(2)
    m_tail = re.match(r'(.*?)([\s\.,:;\-\(\)\[\]!؟]+)$', w)
    if m_tail:
        clean_punct_end = m_tail.group(2)
        w = m_tail.group(1)

    # In visual order PDFs, tanween words get reversed by bidi; invert them back
    if w.startswith('ً') or w.startswith('اً') or (len(w) > 1 and w[1] == 'ً'):
        w = w[::-1]
    
    # Clean decorative kashida/tatweel artifacts
    no_tatweel = w.replace('ـ', '')
    replacements = {
        "اتلبطمت": "متطلبات",
        "اتلبطتم": "متطلبات",
        "ؤولمس": "مسؤول",
        "ؤولسم": "مسؤول",
        "لوؤسم": "مسؤول",
        "ايةحم": "حماية",
        "ةياحم": "حماية",
        "ةيامح": "حماية",
        "والأح": "أحوال",
        "لاحوأ": "أحوال",
        "لاوحأ": "أحوال",
        "نييعتع": "تعيين",
        "نييعت": "تعيين",
        "تييعن": "تعيين",
        "لاصتلاا": "الاتصال",
        "لاصتلا": "الاتصال",
        "اتوذ": "ذات",
        "شخاصالأ": "الأشخاص",
        "شخاصال": "الأشخاص",
        "بياناتال": "البيانات",
        "شخصيةال": "الشخصية",
        "لمعي": "يعمل",
        "لمعُي": "يُعمل",
        "دصقي": "يقصد",
        "دصقُي": "يُقصد",
    }
    if no_tatweel in replacements:
        w = replacements[no_tatweel]
        
    return clean_punct_start + w + clean_punct_end

def fix_arabic_line(line: str) -> str:
    """Reconstruct an extracted PDF Arabic line into natural logical order."""
    line = line.strip()
    if not line:
        return ""
    # Treat the visual character sequence with bidi algorithm
    bidi_line = get_display(line)
    words = bidi_line.split()
    words.reverse()
    fixed_words = [fix_word(w) for w in words]
    fixed = " ".join(fixed_words)
    fixed = unicodedata.normalize('NFKC', fixed)
    fixed = re.sub(r'ـ+', '', fixed)
    fixed = re.sub(r'([0-9]+)\s*-\s*', r'\1- ', fixed)
    return fixed

def locate_pdf_path() -> str:
    """Find the official PDF in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        return ""
    
    pdf_files = glob.glob(str(DATA_DIR / "*.pdf"))
    if not pdf_files:
        return ""
    # Return first PDF found
    return pdf_files[0]

def extract_raw_pages(pdf_path: str) -> dict:
    """Extract and clean text page by page from the official PDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    pages = {}
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        lines = []
        for l in raw_text.split('\n'):
            cleaned_line = fix_arabic_line(l)
            if cleaned_line:
                lines.append(cleaned_line)
        pages[i + 1] = "\n".join(lines)
    return pages

def build_structured_chunks(pages: dict) -> list:
    """
    Parse the cleaned document pages into legal article chunks
    with complete metadata (Article Number, Title, Page, Source).
    """
    chunks = []
    
    # 1. المقدمة (Page 4)
    p4 = pages.get(4, "")
    intro_match = re.search(r'المقدمة\s*(.*?)(?=المادة الأولى|\Z)', p4, re.DOTALL)
    intro_text = intro_match.group(1).strip() if intro_match else ""
    chunks.append({
        "chunk_id": "intro",
        "article_number": "المقدمة",
        "article_title": "المقدمة والسند النظامي لإصدار القواعد",
        "page_number": 4,
        "page_range": "4",
        "source_doc": DOCUMENT_NAME,
        "content": "المقدمة:\n" + intro_text,
        "summary": "بيان السند النظامي لإصدار سدايا لهذه القواعد استناداً للمادة 30 من نظام حماية البيانات الشخصية والمادة 32 من لائحته التنفيذية."
    })

    # 2. المادة الأولى: التعريفات (Page 4)
    art1_match = re.search(r'المادة الأولى\s*:\s*التعريفات\s*(.*?)(?=المادة الثانية|\Z)', p4, re.DOTALL)
    art1_text = art1_match.group(1).strip() if art1_match else ""
    chunks.append({
        "chunk_id": "art_1",
        "article_number": "المادة الأولى",
        "article_title": "التعريفات (الجهة المختصة، مسؤول حماية البيانات، الأنشطة الأساسية)",
        "page_number": 4,
        "page_range": "4",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الأولى : التعريفات\n" + art1_text,
        "summary": "تحديد المعاني والمصطلحات: الجهة المختصة (سدايا)، مسؤول حماية البيانات الشخصية، والأنشطة الأساسية لجهة التحكم."
    })

    # 3. المادة الثانية: الغرض (Pages 4 and 5)
    p5 = pages.get(5, "")
    art2_p4 = re.search(r'المادة الثانية\s*:\s*الغرض\s*(.*?)$', p4, re.DOTALL)
    art2_p5 = re.search(r'^(.*?)(?=المادة الثالثة|\Z)', p5, re.DOTALL)
    t2_4 = art2_p4.group(1).strip() if art2_p4 else ""
    t2_5 = art2_p5.group(1).strip() if art2_p5 else ""
    t2_5 = re.sub(r'^\d+\s*تصنيف الوثيقة\s*:\s*عام\s*', '', t2_5).strip()
    art2_text = (t2_4 + "\n" + t2_5).strip()
    chunks.append({
        "chunk_id": "art_2",
        "article_number": "المادة الثانية",
        "article_title": "الغرض من القواعد",
        "page_number": 4,
        "page_range": "4-5",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الثانية : الغرض\n" + art2_text,
        "summary": "الأهداف: وضع الحد الأدنى لمتطلبات التعيين، إيضاح أحوال التعيين الإلزامي، وتحديد أدوار ومهام مسؤول الحماية."
    })

    # 4. المادة الثالثة: النطاق (Page 5)
    art3_match = re.search(r'المادة الثالثة\s*:\s*النطاق\s*(.*?)(?=المادة الرابعة|\Z)', p5, re.DOTALL)
    art3_text = art3_match.group(1).strip() if art3_match else ""
    chunks.append({
        "chunk_id": "art_3",
        "article_number": "المادة الثالثة",
        "article_title": "نطاق تطبيق القواعد",
        "page_number": 5,
        "page_range": "5",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الثالثة : النطاق\n" + art3_text,
        "summary": "سريان القواعد وتطبيقها على جميع جهات التحكم المشمولة بأحكام نظام حماية البيانات الشخصية ولوائحه التنفيذية."
    })

    # 5. المادة الرابعة: متطلبات تعيين مسؤول حماية البيانات الشخصية (Page 5)
    art4_match = re.search(r'المادة الرابعة\s*:\s*.*?\s*البيانات الشخصية\s*(.*?)$', p5, re.DOTALL)
    art4_text = art4_match.group(1).strip() if art4_match else ""
    chunks.append({
        "chunk_id": "art_4",
        "article_number": "المادة الرابعة",
        "article_title": "متطلبات تعيين مسؤول حماية البيانات الشخصية",
        "page_number": 5,
        "page_range": "5",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الرابعة : متطلبات تعيين مسؤول حماية البيانات الشخصية\n" + art4_text,
        "summary": "الشروط الإلزامية: 1- المؤهل العلمي والخبرة، 2- معرفة ممارسات إدارة المخاطر والتسرب، 3- معرفة المتطلبات النظامية، 4- الأمانة والنزاهة، مع جواز كونه موظفاً أو متعاقداً خارجياً."
    })

    # 6. المادة الخامسة: أحوال تعيين مسؤول حماية البيانات الشخصية (Pages 6 and 7)
    p6 = pages.get(6, "")
    p7 = pages.get(7, "")
    art5_p6 = re.search(r'المادة الخامسة\s*:\s*.*?\s*البيانات الشخصية\s*(.*?)$', p6, re.DOTALL)
    art5_p7 = re.search(r'^(.*?)(?=المادة السادسة|\Z)', p7, re.DOTALL)
    t5_6 = art5_p6.group(1).strip() if art5_p6 else ""
    t5_7 = art5_p7.group(1).strip() if art5_p7 else ""
    t5_7 = re.sub(r'^\d+\s*تصنيف الوثيقة\s*:\s*عام\s*', '', t5_7).strip()
    
    chunks.append({
        "chunk_id": "art_5_1",
        "article_number": "المادة الخامسة",
        "article_title": "أحوال تعيين مسؤول حماية البيانات الشخصية (الحالات الإلزامية ومعايير النطاق الواسع)",
        "page_number": 6,
        "page_range": "6",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الخامسة : أحوال تعيين مسؤول حماية البيانات الشخصية\n" + t5_6,
        "summary": "الحالات الإلزامية: 1- جهة عامة تقدم خدمات بمعالجة واسعة، 2- الأنشطة الأساسية تتطلب مراقبة منتظمة وممنهجة، 3- الأنشطة الأساسية تعالج بيانات حساسة. مع معايير النطاق الواسع (العدد، الحجم، النوع، النطاق الجغرافي، الفئات)."
    })

    chunks.append({
        "chunk_id": "art_5_2",
        "article_number": "المادة الخامسة",
        "article_title": "المراقبة المنتظمة والممنهجة ومفهوم الأنشطة الأساسية وأمثلتها",
        "page_number": 7,
        "page_range": "7",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الخامسة (تابع - المراقبة المنتظمة والممنهجة والأنشطة الأساسية):\n" + t5_7,
        "summary": "تعريف المراقبة المنتظمة والممنهجة وأمثلتها (الأجهزة القابلة للارتداء، التحليل السلوكي، ملفات الارتباط وكاميرات المراقبة) وحكم الأنشطة الأساسية مثل التأمين والتمويل والتسويق."
    })

    # 7. المادة السادسة: توثيق تعيين مسؤول حماية البيانات الشخصية (Pages 7 and 8)
    p8 = pages.get(8, "")
    art6_p7 = re.search(r'المادة السادسة\s*:\s*توثيق تعيين مسؤول حماية البيانات\s*الشخصية\s*(.*?)$', p7, re.DOTALL)
    art6_p8 = re.search(r'^(.*?)(?=المادة السابعة|\Z)', p8, re.DOTALL)
    t6_7 = art6_p7.group(1).strip() if art6_p7 else ""
    t6_8 = art6_p8.group(1).strip() if art6_p8 else ""
    t6_8 = re.sub(r'^\d+\s*تصنيف الوثيقة\s*:\s*عام\s*', '', t6_8).strip()
    art6_text = (t6_7 + "\n" + t6_8).strip()
    chunks.append({
        "chunk_id": "art_6",
        "article_number": "المادة السادسة",
        "article_title": "توثيق تعيين مسؤول حماية البيانات الشخصية",
        "page_number": 7,
        "page_range": "7-8",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة السادسة : توثيق تعيين مسؤول حماية البيانات الشخصية\n" + art6_text,
        "summary": "إلزامية التعيين كتابياً، وتوثيقه للموظف أو إبرام اتفاقية للمتعاقد الخارجي، والإعلان الفوري داخلياً عن التعيين ووسيلة التواصل."
    })

    # 8. المادة السابعة: تفاصيل الاتصال بمسؤول حماية البيانات الشخصية (Page 8)
    art7_match = re.search(r'المادة السابعة\s*:\s*.*?\s*البيانات الشخصية\s*(.*?)(?=المادة الثامنة|\Z)', p8, re.DOTALL)
    art7_text = art7_match.group(1).strip() if art7_match else ""
    chunks.append({
        "chunk_id": "art_7",
        "article_number": "المادة السابعة",
        "article_title": "تفاصيل الاتصال بمسؤول حماية البيانات الشخصية وإشعار سدايا",
        "page_number": 8,
        "page_range": "8",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة السابعة : تفاصيل الاتصال بمسؤول حماية البيانات الشخصية\n" + art7_text,
        "summary": "توفير وسيلة تواصل واضحة وسهلة لأصحاب البيانات، وتزويد سدايا فوراً بالبيانات عبر منصة حوكمة البيانات الوطنية وتحديثها دورياً."
    })

    # 9. المادة الثامنة: أدوار ومهام مسؤول حماية البيانات الشخصية (Pages 8 and 9)
    p9 = pages.get(9, "")
    art8_p8 = re.search(r'المادة الثامنة\s*:\s*أدوار ومهام مسؤول حماية البيانات\s*الشخصية\s*(.*?)$', p8, re.DOTALL)
    art8_p9 = re.search(r'^(.*?)(?=المادة التاسعة|\Z)', p9, re.DOTALL)
    t8_8 = art8_p8.group(1).strip() if art8_p8 else ""
    t8_9 = art8_p9.group(1).strip() if art8_p9 else ""
    t8_9 = re.sub(r'^\d+\s*تصنيف الوثيقة\s*:\s*عام\s*', '', t8_9).strip()
    art8_text = (t8_8 + "\n" + t8_9).strip()
    chunks.append({
        "chunk_id": "art_8",
        "article_number": "المادة الثامنة",
        "article_title": "أدوار ومهام مسؤول حماية البيانات الشخصية",
        "page_number": 8,
        "page_range": "8-9",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الثامنة : أدوار ومهام مسؤول حماية البيانات الشخصية\n" + art8_text,
        "summary": "المهام: 1- الدعم والمشورة وتطوير السياسات الداخلية، 2- الأنشطة التوعوية وتدريب المنسوبين، 3- مراجعة خطط حوادث التسرب، 4- إعداد تقارير دورية وتوصيات، 5- متابعة وثائق سدايا التنظيمية، 6- مشورة الأنظمة التقنية."
    })

    # 10. المادة التاسعة: أحكام عامة (Page 9)
    art9_match = re.search(r'المادة التاسعة\s*:\s*أحكام عامة\s*(.*?)$', p9, re.DOTALL)
    art9_text = art9_match.group(1).strip() if art9_match else ""
    chunks.append({
        "chunk_id": "art_9",
        "article_number": "المادة التاسعة",
        "article_title": "أحكام عامة (المراجعة الدورية، التعيين الطوعي، الاستقلالية، الارتباط التنظيمي)",
        "page_number": 9,
        "page_range": "9",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة التاسعة : أحكام عامة\n" + art9_text,
        "summary": "المراجعة الدورية لإلزامية التعيين، جواز التعيين الطوعي، متطلبات جهة المعالجة، توفير الموارد، عدم تعارض المهام والاستقلالية، التطوير المهني، والارتباط بمكتب إدارة البيانات."
    })

    # 11. المادة العاشرة: المراجعة والتعديل (Page 10)
    p10 = pages.get(10, "")
    art10_match = re.search(r'المادة العاشرة\s*:\s*المراجعة والتعديل\s*(.*?)(?=المادة الحادية عشرة|\Z)', p10, re.DOTALL)
    art10_text = art10_match.group(1).strip() if art10_match else ""
    chunks.append({
        "chunk_id": "art_10",
        "article_number": "المادة العاشرة",
        "article_title": "المراجعة والتعديل",
        "page_number": 10,
        "page_range": "10",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة العاشرة : المراجعة والتعديل\n" + art10_text,
        "summary": "اختصاص الجهة المختصة (سدايا) بمراجعة القواعد عند الاقتضاء وإجراء أي تعديل أو تحديث عليها."
    })

    # 12. المادة الحادية عشرة: النفاذ (Page 10)
    art11_match = re.search(r'المادة الحادية عشرة\s*:\s*النفاذ\s*(.*?)$', p10, re.DOTALL)
    art11_text = art11_match.group(1).strip() if art11_match else ""
    chunks.append({
        "chunk_id": "art_11",
        "article_number": "المادة الحادية عشرة",
        "article_title": "النفاذ وتاريخ العمل بالقواعد",
        "page_number": 10,
        "page_range": "10",
        "source_doc": DOCUMENT_NAME,
        "content": "المادة الحادية عشرة : النفاذ\n" + art11_text,
        "summary": "نفاذ وسريان القواعد والعمل بها اعتباراً من تاريخ نشرها في الموقع الرسمي للجهة المختصة (سدايا)."
    })

    return chunks

def load_and_process_pdf():
    """High-level function to locate, extract, and chunk the document."""
    pdf_path = locate_pdf_path()
    if not pdf_path:
        return None, []
    pages = extract_raw_pages(pdf_path)
    chunks = build_structured_chunks(pages)
    return pdf_path, chunks
