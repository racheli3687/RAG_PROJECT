# =====================================================================
# app.py - הממשק (Streamlit). מכאן המשתמשת שולחת שאלה ורואה תשובה.
# הרצה: streamlit run app.py   (נפתח בדפדפן בכתובת http://localhost:8501)
# =====================================================================

import streamlit as st                                # ספריית הממשק: כל פקודת st מציירת משהו במסך
import config                                         # ההגדרות (בשביל לבדוק שיש מפתח API)
from ingest import build_index, is_index_ready        # בדיקה אם המאגר בנוי, ובנייה שלו
from rag import RagBot                                # הבוט עצמו

st.set_page_config(page_title="מזכירות דיגיטלית", page_icon="🎓", layout="centered")  # כותרת הלשונית בדפדפן ורוחב העמוד

# Streamlit מציג משמאל לימין כברירת מחדל. ה־CSS הזה הופך את העמוד לימין לשמאל (עברית):
st.markdown(
    "<style>body, .stApp, .stChatMessage, .stMarkdown, .stTextInput, textarea {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,                           # מאשרים ל־Streamlit להכניס קוד HTML/CSS שכתבנו בעצמנו
)

st.title("🎓 המזכירות הדיגיטלית של מכללת אופק")      # הכותרת הגדולה בראש העמוד
st.caption("שאלי על התקנון, מערכת השעות, הבחינות, שכר הלימוד, אישורים ועוד.")  # שורת הסבר קטנה מתחתיה

if not config.GOOGLE_API_KEY:                         # אם אין מפתח בקובץ ‎.env‎
    st.error("חסר מפתח API. צרי קובץ ‎.env‎ לפי הקובץ ‎.env.example‎, והפעילי מחדש.")  # הודעת שגיאה אדומה
    st.stop()                                         # עוצרים כאן, אין טעם להמשיך


@st.cache_resource(show_spinner=False)                # "זיכרון": הפונקציה רצה פעם אחת בלבד, ולא בכל לחיצה במסך
def load_bot() -> RagBot:
    """בפעם הראשונה בונה את המאגר. בכל פעם אחרת רק טוענת אותו מהדיסק."""
    if not is_index_ready():                          # המאגר עוד לא קיים?
        with st.spinner("בונה את מאגר הידע בפעם הראשונה... כ־3 דקות, בגלל מגבלת המסלול החינמי (זה קורה רק פעם אחת)"):  # עיגול טעינה עם הודעה
            build_index()                             # חלק א: חיתוך, Embedding ושמירה
    return RagBot()                                   # טוענים את הבוט מהמאגר השמור


# ---------- סרגל צד ----------
with st.sidebar:                                      # כל מה שבתוך ה־with מוצג בסרגל הצד
    st.header("ניהול")                                # כותרת בסרגל
    st.write("הוספת או שינוי מסמכים בתיקייה data? צריך לבנות את המאגר מחדש.")  # הסבר
    if st.button("בניית המאגר מחדש"):                  # כפתור. הקוד שבתוך ה־if רץ רק כשלוחצים עליו
        with st.spinner("בונה מחדש..."):              # עיגול טעינה
            count = build_index(rebuild=True)         # מוחקים ובונים מחדש
        load_bot.clear()                              # מוחקים את הבוט מה"זיכרון", כדי שייטען מחדש מהמאגר החדש
        st.success(f"המאגר נבנה מחדש: {count} קטעים.")  # הודעת הצלחה ירוקה
    show_sources = st.checkbox("הצגת הקטעים שנמצאו", value=True)  # תיבת סימון: להציג את המקורות מתחת לתשובה?

bot = load_bot()                                      # טוענים את הבוט (או בונים בפעם הראשונה)

# ---------- היסטוריית השיחה ----------
# Streamlit מריץ את כל הקובץ מחדש בכל פעולה במסך. כדי שהשיחה לא תימחק,
# שומרים אותה ב־session_state, "זיכרון" שנשאר כל עוד הדף פתוח.
if "messages" not in st.session_state:                # בפעם הראשונה עוד אין רשימה
    st.session_state.messages = []                    # יוצרים רשימה ריקה


def show_sources_box(sources: list[dict]):
    """מציגה את הקטעים שעליהם התבססה התשובה, בתיבה שנפתחת בלחיצה."""
    if not sources:                                   # אם אין מקורות (למשל כשביקשנו לפרט)
        return                                        # אין מה להציג
    with st.expander(f"מאיפה התשובה? ({len(sources)} קטעים)"):  # תיבה מקופלת
        for i, item in enumerate(sources, start=1):   # עוברים על הקטעים
            meta = item["node"].metadata              # ה־Metadata של הקטע: קובץ, סעיף, סוג
            # מציגים איך כל קטע נמצא: בחיפוש הווקטורי (עם הדמיון), בחיפוש המילים (עם המקום), או בשניהם.
            # כך רואים בעיניים אם החיפוש עבד, ואפשר לכוונן את VECTOR_MIN_COSINE ואת TOP_K.
            found_by = []                             # רשימה של "איך נמצא"
            if item["cosine"] is not None:            # נמצא בחיפוש הווקטורי?
                found_by.append(f"משמעות: {item['cosine']}")  # מוסיפים את ציון הדמיון
            if item["bm25_rank"] is not None:         # נמצא בחיפוש המילים?
                found_by.append(f"מילים: מקום {item['bm25_rank']}")  # מוסיפים את המקום
            st.markdown(f"**{i}. {meta['file_name']}** · {meta['section']} · {' | '.join(found_by)}")  # שורת כותרת לקטע
            st.text(item["node"].text)                # הטקסט המלא של הקטע


for message in st.session_state.messages:             # מציירים מחדש את כל ההודעות הקודמות
    with st.chat_message(message["role"]):            # בועה של "user" או של "assistant"
        st.markdown(message["content"])               # הטקסט של ההודעה
        if show_sources and message.get("sources"):   # אם זו תשובה עם מקורות, ותיבת הסימון מסומנת
            show_sources_box(message["sources"])      # מציגים את המקורות

question = st.chat_input("כתבי כאן את השאלה...")       # שורת הקלט בתחתית. מחזירה טקסט רק כשנשלחה שאלה

if question:                                          # נשלחה שאלה חדשה?
    st.session_state.messages.append({"role": "user", "content": question})  # שומרים אותה בהיסטוריה
    with st.chat_message("user"):                     # מציגים אותה בבועה של המשתמשת
        st.markdown(question)

    with st.chat_message("assistant"):                # בועה של הבוט
        with st.spinner("מחפשת במסמכים..."):          # עיגול טעינה בזמן החיפוש והתשובה
            try:                                      # מנסים לענות. אם יש תקלה, נתפוס אותה למטה
                result = bot.answer(question)         # כל המסלול: חיפוש, Prompt, מודל תשובה
            except Exception as error:                # תקלה: אין אינטרנט, מפתח שגוי, חריגה ממגבלת המסלול החינמי...
                result = {"answer": f"אירעה תקלה בפנייה ל־Gemini: {error}", "sources": []}
        st.markdown(result["answer"])                 # מציגים את התשובה
        if show_sources:                              # ואם צריך, גם את המקורות
            show_sources_box(result["sources"])

    st.session_state.messages.append(                 # שומרים את התשובה בהיסטוריה
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
