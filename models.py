# =====================================================================
# models.py - יצירת שני המודלים של Gemini, במקום אחד
# ingest.py ו־rag.py שניהם משתמשים כאן באותו מודל Embedding בדיוק,
# וכך מובטח שהווקטורים של המסמכים ושל השאלות יהיו באותה "שפה".
# =====================================================================

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding  # החיבור של LlamaIndex למודלי Embedding של Google
from llama_index.llms.google_genai import GoogleGenAI                # החיבור של LlamaIndex למודלי השפה של Google
import config                                                        # ההגדרות שלנו


def get_embed_model() -> GoogleGenAIEmbedding:
    """מחזירה את מודל ה־Embedding (שלבים 3 ו־6 במסלול).

    הערה מעניינת: הספרייה שולחת ל־Gemini סוג משימה שונה למסמך (RETRIEVAL_DOCUMENT)
    ולשאלה (RETRIEVAL_QUERY). זה עדיין אותו מודל ואותה "שפה", רק עם רמז שעוזר לו
    לייצר ווקטור מתאים יותר לכל צד. לא צריך לעשות שום דבר, זה קורה לבד.
    """
    return GoogleGenAIEmbedding(
        model_name=config.EMBED_MODEL,       # gemini-embedding-001
        api_key=config.GOOGLE_API_KEY,       # המפתח מקובץ ‎.env‎
        embed_batch_size=20,                 # כמה קטעים לשלוח בכל פנייה. מעט פניות = פחות סיכוי להגיע למגבלת המסלול החינמי
        retries=5,                           # אם Gemini עונה "יותר מדי בקשות", לנסות שוב עד 5 פעמים, עם המתנה הולכת וגדלה
    )


def get_llm() -> GoogleGenAI:
    """מחזירה את מודל התשובה (שלב 9 במסלול)."""
    return GoogleGenAI(
        model=config.LLM_MODEL,              # gemini-2.5-flash-lite
        api_key=config.GOOGLE_API_KEY,       # המפתח מקובץ ‎.env‎
        temperature=config.LLM_TEMPERATURE,  # נמוך = עקבי ומדויק
        max_tokens=config.LLM_MAX_TOKENS,    # תקרה לאורך התשובה
    )
