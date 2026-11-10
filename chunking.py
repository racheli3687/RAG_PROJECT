# =====================================================================
# chunking.py - שלב 2 במסלול: חיתוך המסמכים לקטעים (Nodes)
# כאן מממשים את החלטות 3, 4 ו־5 מהתכנון:
#   - חיתוך לפי מבנה המסמך (כותרות), כך שכל קטע הוא יחידת משמעות שלמה
#   - טבלה: כל שורה היא קטע, יחד עם כותרות העמודות ושם הקטע
#   - סעיף ארוך מדי: מחולק לחלקים עם חפיפה, ולכל חלק מוצמדת כותרת הסעיף
# =====================================================================

from pathlib import Path                            # עבודה עם נתיבי קבצים
from llama_index.core.node_parser import SentenceSplitter  # הכלי של LlamaIndex: חיתוך לפי טוקנים בגבולות משפט
from llama_index.core.schema import TextNode        # "Node" = קטע אחד: טקסט + Metadata
import config                                       # ההגדרות שלנו (גודל קטע, חפיפה וכו')


def approx_tokenizer(text: str) -> list[str]:
    """מחשבת בערך כמה טוקנים יש בטקסט.

    למה הערכה ולא ספירה מדויקת? לכל מודל יש "מחלק טוקנים" משלו.
    ל־Gemini אין כלי ספירה מקומי, וספירה מדויקת דרכו היא פנייה לאינטרנט על כל קטע.
    לחיתוך מספיקה הערכה: בעברית, טוקן אחד הוא בערך 3 תווים.
    """
    step = config.CHARS_PER_TOKEN                   # כמה תווים נחשבים כטוקן אחד
    return [text[i:i + step] for i in range(0, len(text), step)]  # מחלקים את הטקסט לחתיכות של 3 תווים. מספר החתיכות = הערכת הטוקנים


# הכלי שמחלק סעיף ארוך. נשתמש בו רק כשסעיף גדול מ־MAX_SECTION_TOKENS
splitter = SentenceSplitter(
    chunk_size=config.CHUNK_SIZE,                   # גודל כל חלק (בטוקנים)
    chunk_overlap=config.CHUNK_OVERLAP,             # חפיפה: כל חלק מתחיל בסוף של הקודם
    tokenizer=approx_tokenizer,                     # איך לספור טוקנים: בעזרת ההערכה שלמעלה
)


def count_tokens(text: str) -> int:
    """מחזירה את מספר הטוקנים המשוער בטקסט."""
    return len(approx_tokenizer(text))              # מספר החתיכות = מספר הטוקנים המשוער


def is_table_line(line: str) -> bool:
    """האם השורה היא שורה של טבלה? ב־Markdown, שורת טבלה מתחילה בתו |"""
    return line.strip().startswith("|")             # strip מוריד רווחים מההתחלה ומהסוף


def split_table_row(line: str) -> list[str]:
    """הופכת שורת טבלה כמו '| א | ב |' לרשימה ['א', 'ב']."""
    cells = line.strip().strip("|").split("|")      # מורידים את ה־| שבקצוות ומפרקים לפי | שבאמצע
    return [cell.strip() for cell in cells]         # מורידים רווחים מכל תא


def is_separator_row(cells: list[str]) -> bool:
    """שורת ההפרדה שמתחת לכותרות בטבלה נראית כך: |---|---|. אותה מדלגים."""
    return all(set(cell) <= set("-: ") for cell in cells)  # כל התאים מורכבים רק מ־"-", ":" או רווח


def make_header(title: str, section: str) -> str:
    """בונה את "תעודת הזהות" שמוצמדת לתחילת כל קטע: שם המסמך ושם הסעיף.

    זה מה שמבדיל בין קטעים דומים (זוכרת? "החזר נסיעות" מול "רכב חברה").
    """
    header = f"מסמך: {title}\n"                     # השורה הראשונה: מאיזה מסמך הקטע
    if section:                                     # אם הקטע נמצא בתוך סעיף מסוים
        header += f"סעיף: {section}\n"              # מוסיפים גם את שם הסעיף
    return header                                   # מחזירים את הכותרת המוכנה


def make_node(text: str, file_name: str, section: str, kind: str) -> TextNode:
    """יוצרת Node אחד: הטקסט עצמו + Metadata (מידע נוסף שנשמר לצד הטקסט)."""
    metadata = {                                    # ה־Metadata נשמר ב־Chroma לצד הווקטור
        "file_name": file_name,                     # שם הקובץ, כדי להציג למשתמשת מאיפה התשובה
        "section": section or "-",                  # שם הסעיף ("-" אם אין)
        "kind": kind,                               # סוג הקטע: "טקסט" או "שורת טבלה"
    }
    node = TextNode(text=text, metadata=metadata)   # יוצרים את ה־Node
    # כותרת המסמך והסעיף כבר כתובות בתוך הטקסט עצמו (make_header).
    # לכן אומרים ל־LlamaIndex לא להוסיף את ה־Metadata שוב לטקסט, לא לווקטור ולא למודל התשובה:
    node.excluded_embed_metadata_keys = list(metadata.keys())  # לא להכניס את ה־Metadata ל־Embedding
    node.excluded_llm_metadata_keys = list(metadata.keys())    # ולא ל־Prompt של מודל התשובה
    return node                                     # מחזירים את הקטע המוכן


def text_to_nodes(body: str, title: str, section: str, file_name: str) -> list[TextNode]:
    """הופכת טקסט של סעיף אחד לקטע אחד, או לכמה קטעים אם הוא ארוך מדי."""
    body = body.strip()                             # מורידים רווחים ושורות ריקות מהקצוות
    if not body:                                    # אם אין בכלל טקסט (למשל כותרת בלי תוכן)
        return []                                   # אין מה ליצור
    header = make_header(title, section)            # כותרת המסמך והסעיף
    if count_tokens(body) <= config.MAX_SECTION_TOKENS:  # סעיף קצר מספיק?
        return [make_node(header + body, file_name, section, "טקסט")]  # קטע אחד שלם, בלי חפיפה
    # סעיף ארוך: מחלקים לחלקים עם חפיפה, ולכל חלק מצמידים את הכותרת
    parts = splitter.split_text(body)               # SentenceSplitter מחלק בגבולות משפט, עם חפיפה
    return [make_node(header + part, file_name, section, "טקסט") for part in parts]  # Node לכל חלק


def table_to_nodes(rows: list[str], title: str, section: str, file_name: str) -> list[TextNode]:
    """הופכת טבלה לקטעים: כל שורה היא קטע, יחד עם שמות העמודות."""
    header_cells = split_table_row(rows[0])         # השורה הראשונה בטבלה: שמות העמודות
    nodes = []                                      # כאן נאסוף את הקטעים
    for row in rows[1:]:                            # עוברים על שאר השורות
        cells = split_table_row(row)                # מפרקים את השורה לתאים
        if is_separator_row(cells):                 # אם זו שורת ההפרדה |---|
            continue                                # מדלגים עליה
        # מחברים כל תא לשם העמודה שלו, למשל "מועד ב': יום רביעי, 10.2.2027".
        # בלי זה, מודל התשובה היה רואה רק "CS-2104 | 27.1.2027 | 10.2.2027" ולא היה יודע מה כל מספר.
        pairs = [f"{name}: {value}" for name, value in zip(header_cells, cells)]  # zip מצמיד עמודה לתא
        text = make_header(title, section) + "\n".join(pairs)  # כותרת המסמך והסעיף + כל הזוגות, שורה לכל זוג
        nodes.append(make_node(text, file_name, section, "שורת טבלה"))  # מוסיפים את הקטע לרשימה
    return nodes                                    # מחזירים את כל הקטעים של הטבלה


def markdown_to_nodes(path: Path) -> list[TextNode]:
    """קוראת קובץ Markdown אחד ומחלקת אותו לקטעים לפי המבנה שלו.

    הרעיון: הולכים שורה אחר שורה, וזוכרים באיזה סעיף אנחנו (לפי הכותרות).
    טקסט נאסף עד שמגיעים לכותרת חדשה או לטבלה. טבלה נאספת עד שהיא נגמרת.
    """
    lines = path.read_text(encoding="utf-8").splitlines()  # קוראים את הקובץ (בקידוד עברית תקין) ומפרקים לשורות
    file_name = path.name                           # שם הקובץ, למשל 03_luach_mivchanim.md
    title = file_name                               # שם המסמך. יוחלף בכותרת הראשית (#) כשנגיע אליה
    h2 = ""                                         # הכותרת הנוכחית ברמה 2 (##)
    h3 = ""                                         # הכותרת הנוכחית ברמה 3 (###)
    text_buffer: list[str] = []                     # שורות טקסט שנאספו ועוד לא הפכו לקטע
    table_buffer: list[str] = []                    # שורות טבלה שנאספו ועוד לא הפכו לקטעים
    nodes: list[TextNode] = []                      # כל הקטעים של הקובץ

    def section_name() -> str:
        """שם הסעיף הנוכחי, למשל 'שנה ב'' או 'שנה א' > תרגולים - שנה א''."""
        return " > ".join(part for part in (h2, h3) if part)  # מחברים רק את הכותרות שקיימות

    def flush_text():
        """הופכת את הטקסט שנאסף לקטעים, ומרוקנת את המאגר."""
        nodes.extend(text_to_nodes("\n".join(text_buffer), title, section_name(), file_name))
        text_buffer.clear()                         # מתחילים לאסוף מחדש

    def flush_table():
        """הופכת את הטבלה שנאספה לקטעים, ומרוקנת את המאגר."""
        if table_buffer:                            # רק אם באמת נאספה טבלה
            nodes.extend(table_to_nodes(table_buffer, title, section_name(), file_name))
        table_buffer.clear()                        # מתחילים לאסוף מחדש

    for line in lines:                              # עוברים על כל שורות הקובץ
        if line.startswith("#"):                    # שורה שמתחילה ב־# היא כותרת
            flush_table()                           # כותרת חדשה = הסעיף הקודם נגמר. סוגרים טבלה פתוחה
            flush_text()                            # וגם טקסט פתוח
            level = len(line) - len(line.lstrip("#"))  # כמה סימני # יש = רמת הכותרת
            heading = line.lstrip("#").strip()      # הטקסט של הכותרת, בלי ה־#
            if level == 1:                          # # = הכותרת הראשית של המסמך
                title = heading                     # זה שם המסמך
            elif level == 2:                        # ## = סעיף חדש
                h2, h3 = heading, ""                # סעיף חדש מאפס גם את תת־הסעיף
            else:                                   # ### ומטה = תת־סעיף (בשו"ת: כל שאלה)
                h3 = heading                        # שומרים את שם תת־הסעיף
        elif is_table_line(line):                   # שורה של טבלה
            if text_buffer:                         # אם לפני הטבלה היה טקסט
                flush_text()                        # סוגרים אותו כקטע נפרד
            table_buffer.append(line)               # מוסיפים את השורה לטבלה הנוכחית
        else:                                       # שורת טקסט רגילה (או שורה ריקה)
            flush_table()                           # אם הייתה טבלה, היא נגמרה
            text_buffer.append(line)                # אוספים את השורה לטקסט הנוכחי

    flush_table()                                   # בסוף הקובץ: סוגרים מה שנשאר פתוח
    flush_text()
    return nodes                                    # כל הקטעים של הקובץ


def load_all_nodes(data_dir: Path) -> list[TextNode]:
    """שלבים 1+2 במסלול: טוענת את כל קובצי ה־Markdown בתיקייה ומחלקת אותם לקטעים."""
    all_nodes: list[TextNode] = []                  # כל הקטעים מכל הקבצים
    for path in sorted(data_dir.glob("*.md")):      # כל קובצי ה־md בתיקייה, לפי סדר השמות
        all_nodes.extend(markdown_to_nodes(path))   # מוסיפים את הקטעים של הקובץ
    return all_nodes                                # מחזירים את כולם


# הרצה ישירה של הקובץ (python chunking.py) מציגה את הקטעים, בלי לפנות לשום שירות.
# זו בדיקה חשובה: לראות בעיניים שהחיתוך יצא כמו שתכננו.
if __name__ == "__main__":
    nodes = load_all_nodes(config.DATA_DIR)         # מחלקים את כל המסמכים
    for i, node in enumerate(nodes, start=1):       # עוברים על הקטעים, עם מספור מ־1
        print(f"===== קטע {i} | {node.metadata['kind']} | ~{count_tokens(node.text)} טוקנים =====")
        print(node.text)                            # הטקסט המלא של הקטע
        print()                                     # שורה ריקה בין קטעים
    print(f"סה\"כ: {len(nodes)} קטעים")             # כמה קטעים יצאו בסך הכל
