# =====================================================================
# rag.py - חלק ב של המערכת: שאלה ותשובה. רץ בכל שאלה מחדש.
# שאלה -> ווקטור -> חיפוש Hybrid (ווקטורי + מילים) -> סף דמיון ->
# Prompt -> מודל התשובה -> תשובה + המקורות
# =====================================================================

import math                                            # בשביל log (לחישוב דמיון הקוסינוס, ראי למטה)
import re                                              # ביטויים רגולריים: פירוק השאלה למילים
from datetime import date                              # התאריך של היום, כדי שהמודל יבין "הבחינה הבאה"
from llama_index.core import PromptTemplate, VectorStoreIndex  # תבנית Prompt, והאינדקס הווקטורי
from llama_index.vector_stores.chroma import ChromaVectorStore  # החיבור ל־Chroma
from llama_index.retrievers.bm25 import BM25Retriever          # החיפוש לפי מילים
from ingest import get_chroma_collection               # פתיחת Chroma (אותה פונקציה שבהכנה)
from models import get_embed_model, get_llm            # אותו מודל Embedding בדיוק כמו בהכנה, ומודל התשובה
import config                                          # ההגדרות שלנו

# ---------- ה־Prompt (החלטה 12) ----------
# {today}, {context} ו־{question} הם "חורים" שימולאו בכל שאלה מחדש.
PROMPT = PromptTemplate(
    "אתה העוזר הדיגיטלי של מזכירות מכללת אופק. ענה לסטודנט בעברית, בקצרה ובבהירות.\n"
    "תאריך היום: {today}\n\n"
    "כללים:\n"
    "1. ענה רק על סמך הקטעים שלמטה. אל תשתמש בידע אחר ואל תמציא.\n"
    "2. אם יש בקטעים עדכון שסותר מידע אחר (למשל שינוי כיתה), העדף את העדכון וציין זאת.\n"
    "3. אם הקטעים לא עונים על השאלה, כתוב שלא מצאת מידע מתאים, "
    "ובקש מהסטודנט לפרט או לפנות למזכירות הסטודנטים.\n"
    "4. בסוף התשובה ציין מאיזה מסמך נלקח המידע.\n\n"
    "קטעים:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n\n"
    "שאלה: {question}\n"
    "תשובה:"
)

# התשובה שמוחזרת כשלא נמצא אף קטע מעל הסף (החלטה 10): בלי לפנות למודל התשובה בכלל
CLARIFY_MESSAGE = (
    "לא מצאתי מידע מתאים לשאלה. אפשר לפרט יותר? "
    "למשל לציין את שם הקורס, את סוג הבקשה או את הנושא. "
    "אם המידע לא נמצא אצלי, מזכירות הסטודנטים (חדר 101, טלפון 03-5551200) תשמח לעזור."
)


def keyword_query(question: str) -> str:
    """מכינה את השאלה לחיפוש לפי מילים: מוחקת מילים נפוצות שלא עוזרות לחיפוש.

    למה? המילה "מה" מופיעה כמעט בכל קטע. אם נחפש לפיה, נקבל קטעים לא קשורים.
    """
    words = re.findall(r"[\w\-]+", question)           # מפרקים למילים. \w = אות או ספרה, ו־"-" כדי ש־CS-2104 יישאר שלם
    keep = [w for w in words if w not in config.HEBREW_STOPWORDS]  # משאירים רק מילים שאינן ברשימת המילים הנפוצות
    extra = []                                         # גרסאות נוספות של מילים, בלי אות התחילית
    for w in keep:                                     # עוברים על המילים שנשארו
        # "בטופס" -> גם "טופס". רק במילים של 4 אותיות ומעלה, כדי לא לפגוע במילים קצרות.
        # אם האות הראשונה היא חלק מהמילה עצמה (כמו ב"בחינה"), הגרסה בלי האות פשוט לא תימצא, ולא יקרה נזק.
        if len(w) >= 4 and w[0] in config.HEBREW_PREFIXES:
            extra.append(w[1:])                        # מוסיפים את המילה בלי האות הראשונה
    return " ".join(keep + extra)                      # מחברים הכל בחזרה למחרוזת אחת


class RagBot:
    """הבוט: נטען פעם אחת, ואז עונה על כמה שאלות שרוצים."""

    def __init__(self):
        """טוענת את המאגר מהדיסק (בלי לעבד מחדש) ומכינה את שני החיפושים ואת מודל התשובה."""
        vector_store = ChromaVectorStore(chroma_collection=get_chroma_collection())  # מתחברים ל־Chroma שבדיסק
        index = VectorStoreIndex.from_vector_store(    # טוענים אינדקס קיים, בלי ליצור ווקטורים מחדש
            vector_store,
            embed_model=get_embed_model(),             # חובה: אותו מודל כמו בהכנה, כי בו ניצור ווקטור לשאלה
        )
        self.vector_retriever = index.as_retriever(similarity_top_k=config.VECTOR_TOP_K)  # החיפוש הווקטורי
        self.bm25_retriever = BM25Retriever.from_persist_dir(str(config.BM25_DIR))        # החיפוש לפי מילים, נטען מהדיסק
        self.bm25_retriever.token_pattern = config.BM25_TOKEN_PATTERN  # הספרייה לא שומרת את כלל הפירוק בדיסק, אז מגדירים שוב: חייב להיות זהה לבנייה
        self.bm25_retriever.skip_stemming = True       # גם זה לא נשמר בדיסק. כמו בבנייה: בלי "שורשים"
        self.llm = get_llm()                           # מודל התשובה

    def retrieve(self, question: str) -> list[dict]:
        """שלבים 6+7 במסלול: חיפוש Hybrid. מחזירה רשימת קטעים עם הציונים שלהם."""
        # --- חיפוש 1: ווקטורי (לפי משמעות) ---
        vector_hits = []                               # כאן יישמרו התוצאות הווקטוריות שעברו את הסף
        for hit in self.vector_retriever.retrieve(question):  # ה־retriever יוצר ווקטור לשאלה ומחפש ב־Chroma
            # Chroma מחזיר ציון בנוסחה exp(-(1 - cosine)). ההמרה הבאה מחזירה את דמיון הקוסינוס עצמו (0 עד 1):
            cosine = 1 + math.log(hit.score)
            if cosine >= config.VECTOR_MIN_COSINE:     # סף הדמיון: קטע רחוק מדי במשמעות לא נכנס
                vector_hits.append((hit, cosine))      # שומרים את הקטע יחד עם הדמיון שלו

        # --- חיפוש 2: לפי מילים (BM25) ---
        bm25_hits = []                                 # כאן יישמרו התוצאות של חיפוש המילים
        query = keyword_query(question)                # השאלה בלי המילים הנפוצות
        if query:                                      # אם נשארו מילים לחפש (שאלה כמו "מה זה?" לא משאירה כלום)
            for hit in self.bm25_retriever.retrieve(query):  # מחפשים קטעים שמכילים את המילים
                if hit.score > 0:                      # ציון 0 = אף מילה מהשאלה לא מופיעה בקטע
                    bm25_hits.append(hit)              # שומרים רק קטעים עם התאמה אמיתית

        # --- איחוד שתי הרשימות: RRF (Reciprocal Rank Fusion) ---
        # שני החיפושים נותנים ציונים בסולמות שונים, ואי אפשר להשוות אותם ישירות.
        # לכן מסתכלים רק על המקום ברשימה: מקום 1 שווה הכי הרבה, מקום 2 פחות, וכן הלאה.
        # קטע שמופיע גבוה בשתי הרשימות מקבל ציון משני החיפושים, ולכן עולה לראש.
        found: dict[str, dict] = {}                    # מילון: מזהה הקטע -> הפרטים שלו
        for rank, (hit, cosine) in enumerate(vector_hits, start=1):  # עוברים על התוצאות הווקטוריות, לפי הסדר
            item = found.setdefault(hit.node.node_id, {"node": hit.node, "score": 0.0, "cosine": None, "bm25_rank": None})
            item["score"] += 1 / (config.RRF_K + rank) # הנוסחה: 1 חלקי (60 + המקום ברשימה)
            item["cosine"] = round(cosine, 3)          # שומרים את הדמיון להצגה במסך
        for rank, hit in enumerate(bm25_hits, start=1):  # אותו דבר לתוצאות של חיפוש המילים
            item = found.setdefault(hit.node.node_id, {"node": hit.node, "score": 0.0, "cosine": None, "bm25_rank": None})
            item["score"] += 1 / (config.RRF_K + rank) # מוסיפים לציון (אם הקטע כבר נמצא בחיפוש הווקטורי, הציונים מצטברים)
            item["bm25_rank"] = rank                   # שומרים את המקום בחיפוש המילים להצגה במסך

        ranked = sorted(found.values(), key=lambda item: item["score"], reverse=True)  # ממיינים מהציון הגבוה לנמוך
        return ranked[: config.TOP_K]                  # מחזירים לכל היותר TOP_K קטעים

    def answer(self, question: str) -> dict:
        """כל המסלול של שאלה אחת: חיפוש, Prompt, תשובה. מחזירה את התשובה ואת המקורות."""
        sources = self.retrieve(question)              # שלבים 6+7: מוצאים את הקטעים הרלוונטיים
        if not sources:                                # אף קטע לא עבר את הסף?
            return {"answer": CLARIFY_MESSAGE, "sources": []}  # מבקשים לפרט, בלי פנייה למודל התשובה (חוסך זמן וטוקנים)

        context = "\n\n".join(                         # שלב 8: מחברים את הקטעים לטקסט אחד
            f"[קטע {i}]\n{item['node'].text}"          # כל קטע עם מספר, כדי שהמודל יבחין ביניהם
            for i, item in enumerate(sources, start=1)
        )
        prompt = PROMPT.format(                        # ממלאים את ה"חורים" בתבנית
            today=date.today().strftime("%d.%m.%Y"),   # התאריך של היום, בפורמט ישראלי
            context=context,                           # הקטעים שנמצאו
            question=question,                         # שאלת הסטודנטית
        )
        response = self.llm.complete(prompt)           # שלב 9: שולחים ל־Gemini ומחכים לתשובה
        return {"answer": response.text, "sources": sources}  # מחזירים את התשובה ואת הקטעים שעליהם התבססה
