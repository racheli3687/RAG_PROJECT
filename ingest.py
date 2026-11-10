# =====================================================================
# ingest.py - חלק א של המערכת: הכנה (Indexing). רץ פעם אחת בלבד.
# טעינה -> חיתוך -> Embedding -> שמירה ב־Chroma (ובנוסף: אינדקס מילים BM25)
#
# הרצה ידנית:  python ingest.py            (בונה רק אם עוד לא נבנה)
#              python ingest.py --rebuild  (מוחק ובונה מחדש. למשל אחרי הוספת מסמך)
# =====================================================================

import shutil                                         # מחיקת תיקיות (בבנייה מחדש)
import sys                                            # קריאת הפרמטרים שנכתבו בשורת הפקודה (--rebuild)
import time                                           # המתנה בין קבוצות (מגבלת המסלול החינמי)
import chromadb                                       # ה־Vector DB שבחרנו (החלטה 7)
from llama_index.core import StorageContext, VectorStoreIndex  # האובייקטים של LlamaIndex שמחזיקים את המאגר
from llama_index.vector_stores.chroma import ChromaVectorStore  # החיבור בין LlamaIndex ל־Chroma
from llama_index.retrievers.bm25 import BM25Retriever          # החיפוש לפי מילים (החלטה 8)
from chunking import load_all_nodes                   # שלבים 1+2: טעינה וחיתוך (הקוד שלנו)
from models import get_embed_model                    # המודל ליצירת ווקטורים
import config                                         # ההגדרות שלנו


def get_chroma_collection():
    """פותחת (או יוצרת) את "הטבלה" ב־Chroma שבה נשמרים הקטעים והווקטורים."""
    client = chromadb.PersistentClient(                # Persistent = נשמר בדיסק, ולא נעלם כשהתוכנה נסגרת
        path=str(config.CHROMA_DIR),                   # איפה לשמור
        settings=chromadb.Settings(anonymized_telemetry=False),  # לא לשלוח נתוני שימוש ל־Chroma (חוסך הודעות שגיאה מיותרות)
    )
    return client.get_or_create_collection(            # פותחים את הטבלה אם קיימת, ואם לא, יוצרים אותה
        name=config.COLLECTION_NAME,                   # שם הטבלה
        metadata={"hnsw:space": "cosine"},             # איך מודדים מרחק בין ווקטורים: דמיון קוסינוס
    )


def is_index_ready() -> bool:
    """האם המאגר כבר נבנה? כך יודעים אם לטעון מהדיסק או לבנות מאפס."""
    for name in ("params.index.json", "retriever.json"):  # שני קבצים שנכתבים בשמירה של BM25
        if not (config.BM25_DIR / name).exists():      # אם אחד מהם חסר, השמירה לא הושלמה
            return False                               # המאגר לא מוכן, וצריך לבנות
    return get_chroma_collection().count() > 0         # מוכן רק אם יש קטעים ב־Chroma


def build_index(rebuild: bool = False) -> int:
    """בונה את המאגר: חיתוך, Embedding ושמירה. מחזירה את מספר הקטעים."""
    # תמיד מתחילים מאגר נקי: אם בנייה קודמת נקטעה באמצע, נשארו בו רק חלק מהקטעים,
    # ובנייה נוספת מעליו הייתה יוצרת כפילויות. (הפרמטר rebuild נשאר רק בשביל ההודעות בטרמינל.)
    if config.CHROMA_DIR.parent.exists():              # אם יש מאגר ישן
        shutil.rmtree(config.CHROMA_DIR.parent)        # מוחקים את תיקיית storage, כולל Chroma ו־BM25

    nodes = load_all_nodes(config.DATA_DIR)            # שלבים 1+2: קוראים את המסמכים ומחלקים לקטעים
    print(f"נוצרו {len(nodes)} קטעים. יוצרת ווקטורים...")  # הודעה למסך, כדי לדעת מה קורה

    collection = get_chroma_collection()               # פותחים את הטבלה ב־Chroma
    vector_store = ChromaVectorStore(chroma_collection=collection)  # עוטפים אותה כך ש־LlamaIndex ידע לעבוד איתה
    storage_context = StorageContext.from_defaults(vector_store=vector_store)  # אומרים ל־LlamaIndex: "תשמור ב־Chroma"

    index = VectorStoreIndex(                          # יוצרים אינדקס ריק, שמחובר ל־Chroma ולמודל ה־Embedding
        [],                                            # עוד בלי קטעים. נוסיף אותם בקבוצות, למטה
        storage_context=storage_context,               # איפה לשמור
        embed_model=get_embed_model(),                 # עם איזה מודל ליצור את הווקטורים
    )
    # שלבים 3+4 בקבוצות: המסלול החינמי מרשה עד 100 קטעים לדקה, אז שולחים קבוצה ומחכים.
    for start in range(0, len(nodes), config.EMBED_BATCH):  # start = 0, 50, 100...
        batch = nodes[start : start + config.EMBED_BATCH]   # הקבוצה הנוכחית (עד 50 קטעים)
        index.insert_nodes(batch)                       # יוצר ווקטור לכל קטע בקבוצה ושומר ב־Chroma
        done = start + len(batch)                       # כמה קטעים סיימנו עד עכשיו
        print(f"נשמרו {done} מתוך {len(nodes)} קטעים")  # התקדמות, כדי לדעת שהתוכנה לא תקועה
        if done < len(nodes):                           # אם נשארו עוד קטעים
            print(f"ממתינה {config.EMBED_PAUSE_SECONDS} שניות (מגבלת המסלול החינמי)...")
            time.sleep(config.EMBED_PAUSE_SECONDS)      # מחכים, כדי לא לעבור את המגבלה

    bm25 = BM25Retriever.from_defaults(                # בונים גם את אינדקס המילים, מאותם קטעים בדיוק
        nodes=nodes,                                   # אותם קטעים
        similarity_top_k=config.BM25_TOP_K,            # כמה תוצאות להחזיר בכל חיפוש
        skip_stemming=True,                            # "שורשים" (stemming) קיימים רק לאנגלית, אז מוותרים
        token_pattern=config.BM25_TOKEN_PATTERN,       # איך לפרק למילים (כולל מילה של תו אחד, כמו המספר 7)
    )
    bm25.persist(str(config.BM25_DIR))                 # שומרים את אינדקס המילים לדיסק
    print("המאגר נבנה ונשמר.")                          # סיימנו
    return len(nodes)                                  # מחזירים כמה קטעים נשמרו


# הקוד הזה רץ רק כשמריצים את הקובץ ישירות: python ingest.py
if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv                  # האם ביקשו בנייה מחדש?
    if is_index_ready() and not rebuild:               # אם כבר בנוי ולא ביקשו מחדש
        print("המאגר כבר קיים. לבנייה מחדש: python ingest.py --rebuild")
    else:                                              # אחרת: בונים
        build_index(rebuild=rebuild)
