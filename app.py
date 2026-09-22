import os
import re
import time
import random

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MedQuAD Bot", page_icon="🏥", layout="centered")

# KONFIGURASI
ENABLE_SBERT = False   # True -> pakai embedding semantik (lebih pintar, lebih lambat)
CSV_CANDIDATES = ["medquad.csv", "data/medquad.csv", "/content/sample_data/medquad.csv"]

WELCOME = ("Halo! Saya **MedQuAD Bot**, asisten informasi kesehatan berbasis basis data NIH. "
           "Silakan tanyakan tentang gejala, penyebab, pengobatan, atau pencegahan suatu penyakit.")

SUGGESTIONS = ["What is Glaucoma ?",
               "What are the symptoms of Parkinson's Disease ?",
               "how to prevent high blood pressure",
               "gejala kanker payudara",
               "pengobatan asma",
               "symptoms of diabetes"]

# SETUP NLTK (dengan fallback) — dijalankan sekali
_FALLBACK_STOPWORDS = set("""a an the is are was were be been being of for to in on at by
    with from and or as it its this that these those which who whom do does did done you
    your i my we our they their he she his her not no nor can could should would may might
    must have has had there here about into over under more most other some such only own
    same than too very""".split())


@st.cache_resource(show_spinner="Menyiapkan NLTK…")
def setup_nltk():
    """Return (nltk_ok, english_stopwords, lemmatizer, word_tokenize)."""
    try:
        import nltk
        for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
            nltk.download(pkg, quiet=True)
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        word_tokenize("test sentence")
        lem = WordNetLemmatizer()
        lem.lemmatize("running", pos='v')
        return True, set(stopwords.words('english')), lem, word_tokenize
    except Exception:
        return False, set(_FALLBACK_STOPWORDS), None, None


NLTK_OK, english_stopwords, lemmatizer, _word_tokenize = setup_nltk()


@st.cache_resource(show_spinner="Memuat model SBERT…")
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


SBERT_MODEL, USE_SBERT = None, False
if ENABLE_SBERT:
    try:
        SBERT_MODEL = load_sbert()
        USE_SBERT = True
    except Exception:
        USE_SBERT = False


# EKSTRAKSI INTENT (rule-based regex)
# Urutan PENTING: yang lebih spesifik diletakkan lebih dulu.
INTENT_RULES = [
    ('gejala',      r'symptom|signs of|sign of|what to do for'),
    ('penyebab',    r'what causes|causes of|cause of'),
    ('pengobatan',  r'treatment|how to treat|therapy|therapies|medicat'),
    ('pencegahan',  r'prevent'),
    ('diagnosis',   r'diagnos|how to diagnose|test for|screening'),
    ('prevalensi',  r'how many people|how common|frequency'),
    ('genetik',     r'genetic change|inherit|hereditary|mutation'),
    ('penelitian',  r'research|clinical trial'),
    ('prognosis',   r'outlook|prognosis|survival'),
    ('komplikasi',  r'complication'),
    ('risiko',      r'who is at risk|risk factor'),
    ('definisi',    r'what is|what are|do you have information'),
]


def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in INTENT_RULES:
        if re.search(pattern, q):
            return intent
    return 'lainnya'


# STOPWORDS & JEMBATAN ISTILAH ID → EN
indonesian_stopwords = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','pada','adalah','atau','juga',
    'dalam','tidak','akan','ada','saya','kamu','anda','ia','mereka','kami','kita','bisa',
    'sudah','bila','jika','maka','oleh','karena','apa','bagaimana','berapa','kapan','dimana',
    'siapa','apakah','cara','lebih','sangat','dapat','nya','pun','lagi','belum','telah',
    'namun','tapi','serta','meski','agar','supaya','hal','dong','sih','kah','mau','ingin',
    'what','how','why','when','where','whose','which','who',
}

# Jangan pernah memasukkan istilah medis ke stopwords
MEDICAL_KEEP = {'pain','blood','heart','skin','eye','bone','lung','liver','kidney','brain',
                'cancer','risk','test','cell','gene','drug','care','diet','type','stage'}
all_stopwords = (english_stopwords | indonesian_stopwords) - MEDICAL_KEEP

ID_EN_MAP = {
    'gejala':'symptoms','tanda':'signs','penyebab':'causes','sebab':'causes',
    'pengobatan':'treatment','obat':'treatment medication','mengobati':'treatment',
    'terapi':'therapy','pencegahan':'prevention','mencegah':'prevent','cegah':'prevent',
    'diagnosa':'diagnosis','diagnosis':'diagnosis','pemeriksaan':'diagnosis test',
    'penyakit':'disease','penularan':'transmission','menular':'contagious',
    'keturunan':'inherited','genetik':'genetic','risiko':'risk','komplikasi':'complication',
    'penelitian':'research','harapan':'outlook','sembuh':'cure','kambuh':'relapse',
    'jantung':'heart','paru':'lung','ginjal':'kidney','hati':'liver','otak':'brain',
    'darah':'blood','tulang':'bone','sendi':'joint','kulit':'skin','mata':'eye',
    'telinga':'ear','lambung':'stomach','usus':'intestine','saraf':'nerve','otot':'muscle',
    'payudara':'breast','paru-paru':'lung','kandungan':'uterus','prostat':'prostate',
    'kanker':'cancer','tumor':'tumor','diabetes':'diabetes','kencing':'urinary',
    'manis':'mellitus','darah tinggi':'high blood pressure','hipertensi':'hypertension',
    'kolesterol':'cholesterol','stroke':'stroke','asma':'asthma','alergi':'allergy',
    'demam':'fever','panas':'fever','batuk':'cough','pilek':'cold','flu':'influenza',
    'nyeri':'pain','sakit':'pain','pusing':'dizziness','mual':'nausea','muntah':'vomiting',
    'lelah':'fatigue','letih':'fatigue','sesak':'shortness of breath','napas':'breath',
    'gatal':'itching','bengkak':'swelling','ruam':'rash','luka':'wound','infeksi':'infection',
    'radang':'inflammation','anemia':'anemia','osteoporosis':'osteoporosis',
    'depresi':'depression','cemas':'anxiety','stres':'stress','tidur':'sleep',
    'anak':'children','bayi':'infant','lansia':'elderly','hamil':'pregnancy',
    'berat badan':'weight','makanan':'diet food','olahraga':'exercise','rokok':'smoking',
    'keracunan':'poisoning','racun':'poison','diracun':'poisoning','tertelan':'swallowed poisoning',
    'gigitan':'bite','sengatan':'sting','luka bakar':'burn','tersedak':'choking',
    'sakit kepala':'headache','migrain':'migraine','diare':'diarrhea','sembelit':'constipation',
    'maag':'gastritis','tbc':'tuberculosis','tuberkulosis':'tuberculosis','glaukoma':'glaucoma',
    'katarak':'cataract','epilepsi':'epilepsy','patah tulang':'fracture','gula darah':'blood sugar',
}


def _strip_nya(text: str) -> str:
    """'gejalanya' -> 'gejala' (hanya jika akar katanya ada di kamus)."""
    def fix(m):
        stem = m.group(1)
        return stem if stem.lower() in ID_EN_MAP else m.group(0)
    return re.sub(r'\b([A-Za-z\-]+?)nya\b', fix, text)


def bridge_id_to_en(text: str) -> str:
    """Menambahkan padanan Inggris untuk istilah medis Indonesia yang terdeteksi."""
    text = _strip_nya(str(text))
    low = ' ' + text.lower() + ' '
    extra = []
    for idw, enw in ID_EN_MAP.items():
        if re.search(r'\b' + re.escape(idw) + r'\b', low):
            extra.append(enw)
    return text + ' ' + ' '.join(extra) if extra else text

# PREPROCESSING
def tokenize(text: str):
    if NLTK_OK:
        return _word_tokenize(text)
    return re.findall(r'[a-z]+', text)


def simple_lemma(word: str) -> str:
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


def preprocess_text(text: str) -> str:
    """jembatan ID→EN → lowercase → hapus non-huruf → tokenisasi → stopwords → lemmatisasi."""
    text = bridge_id_to_en(str(text))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    if NLTK_OK:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [simple_lemma(t) for t in tokens]
    return ' '.join(tokens)

# PENERJEMAH, DETEKSI BAHASA, FOLLOW-UP, PEMBERSIH JAWABAN
@st.cache_resource(ttl=600, show_spinner=False)
def translator_available() -> bool:
    try:
        from deep_translator import GoogleTranslator
        return bool(GoogleTranslator(source='en', target='id').translate('fever'))
    except Exception:
        return False


def _split_chunks(text, limit=4500):
    """Potong teks panjang di batas kalimat (batas Google Translate ±5000 karakter)."""
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ''
    for sent in re.split(r'(?<=[.!?])\s+', text):
        if len(cur) + len(sent) + 1 > limit and cur:
            chunks.append(cur)
            cur = ''
        cur += (' ' if cur else '') + sent
    if cur:
        chunks.append(cur)
    return chunks


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2000)
def _translate_cached(text: str, src: str, tgt: str) -> str:
    """Exception tidak di-cache oleh Streamlit, jadi kegagalan sesaat tidak 'menempel'."""
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source=src, target=tgt)
    out = ' '.join(tr.translate(c) or '' for c in _split_chunks(text)).strip()
    if not out:
        raise ValueError("terjemahan kosong")
    return out


_EN_QWORDS = {'what', 'how', 'why', 'when', 'where', 'whose', 'which', 'who'}
_EN_FUNC = _EN_QWORDS | {'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'should',
    'would', 'the', 'of', 'for', 'to', 'in', 'on', 'and', 'or', 'with', 'my', 'your', 'me', 'we',
    'it', 'this', 'that', 'there', 'have', 'has', 'about', 'tell', 'information', 'symptoms',
    'causes', 'treatment', 'treatments', 'prevent', 'signs', 'inherited'}
_ID_MARKERS = (indonesian_stopwords - _EN_QWORDS) | {
    k for k, v in ID_EN_MAP.items() if ' ' not in k and k != v and k not in {'flu'}}
_ID_AFFIX = re.compile(r'^(?:ke|pe|pen|per|peng|mem|men|meng|ber|ter|di)[a-z]{3,}(?:an|kan|nya)$')


def detect_language(text, known_en=None, default='id'):
    """'id' atau 'en'. Kalau ambigu (mis. hanya 'diabetes') pakai `default`."""
    id_score = en_score = 0
    for t in re.findall(r'[a-z]+', text.lower()):
        base = t[:-3] if t.endswith('nya') and t[:-3] in ID_EN_MAP else t
        if base in _ID_MARKERS:
            id_score += 1
        elif t in _EN_FUNC:
            en_score += 1
        elif _ID_AFFIX.match(t) and not (known_en and t in known_en):
            id_score += 1
    if id_score > en_score:
        return 'id'
    if en_score > id_score:
        return 'en'
    return default


FOLLOWUP_ID = {'gejala', 'tanda', 'penyebab', 'sebab', 'pengobatan', 'obat', 'mengobati', 'terapi',
               'pencegahan', 'mencegah', 'cegah', 'diagnosa', 'diagnosis', 'pemeriksaan', 'penularan',
               'menular', 'keturunan', 'genetik', 'risiko', 'komplikasi', 'penelitian', 'harapan',
               'sembuh', 'kambuh', 'definisi'}
FOLLOWUP_EN = set(preprocess_text(
    "symptoms signs causes treatment treatments medication therapy prevention prevent diagnosis "
    "test screening risk complication complications research outlook prognosis genetic inherited "
    "contagious transmission cure relapse definition").split())
FOLLOWUP_TERMS = FOLLOWUP_ID | FOLLOWUP_EN


def clean_answer(text: str) -> str:
    """Buang kalimat petunjuk video artefak scraping MedQuAD."""
    text = re.sub(r'\(Watch the video.*?keyboard\.\)', '', text, flags=re.S)
    return re.sub(r'\s{2,}', ' ', text).strip()

# DATASET: LOAD + CLEANING + INTENT + PREPROCESSING + INDEX (di-cache)
@st.cache_resource(show_spinner="Memuat dataset & membangun index TF-IDF (hanya sekali)…")
def build_resources(csv_path: str, file_mtime: float, use_sbert: bool):
    df_raw = pd.read_csv(csv_path)
    needed = {'question', 'answer', 'focus_area', 'source'}
    missing_cols = needed - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ada di CSV: {sorted(missing_cols)}. "
                         f"Kolom yang ditemukan: {list(df_raw.columns)}")

    df = df_raw.copy()
    n_raw = len(df)

    # 3.2 — cleaning
    df = df.dropna(subset=['question', 'answer', 'focus_area'])
    for col in ['question', 'answer', 'focus_area', 'source']:
        df[col] = df[col].astype(str).str.strip()
    df = df[df['answer'].str.len() >= 50]
    df['answer_len'] = df['answer'].str.len()
    df = (df.sort_values('answer_len', ascending=False)
            .drop_duplicates(subset=['question'], keep='first'))
    df = df.sort_index().reset_index(drop=True)

    # 3.3 — intent
    df['intent'] = df['question'].apply(detect_intent)
    df['category'] = df['focus_area']
    df['search_text'] = df['focus_area'] + ' ' + df['question'] + ' ' + df['intent']

    # 4.3 — preprocessing
    df['processed_question'] = df['search_text'].apply(preprocess_text)

    # 5.1 — TF-IDF
    t0 = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=60000,
                                 sublinear_tf=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df['processed_question'])

    sbert_embeddings = None
    if use_sbert and SBERT_MODEL is not None:
        sbert_embeddings = SBERT_MODEL.encode(df['search_text'].tolist(),
                                              convert_to_tensor=True, batch_size=64)

    return {
        "df": df,
        "n_raw": n_raw,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "vocab_words": set(vectorizer.get_feature_names_out()),
        "sbert_embeddings": sbert_embeddings,
        "index_seconds": time.time() - t0,
    }


# ENGINE CHATBOT
class MedQuADChatbotEngine:
    """
    Index (df, TF-IDF) dipakai bersama semua sesi lewat cache;
    state percakapan (last_focus, last_lang, dst.) milik masing-masing sesi.
    """

    POISON_PATTERN = r'keracunan|diracun|meracuni|\bracun\b|tertelan|poison'
    POISON_NOTE = (
        "**PENTING** **Jika keracunan sedang terjadi sekarang** (baru menelan, menghirup, atau terpapar zat berbahaya): "
        "segera hubungi **119 / 112** atau langsung ke IGD terdekat. Jangan memaksa muntah atau memberi "
        "makan/minum kecuali diarahkan tenaga medis, dan bawa kemasan atau sisa zat untuk ditunjukkan ke petugas."
    )

    def __init__(self, resources, threshold=None, top_k=5, max_answer_chars=1200,
                 translate_enabled=True):
        self.df = resources["df"]
        self.vectorizer = resources["vectorizer"]
        self.tfidf_matrix = resources["tfidf_matrix"]
        self.vocab_words = resources["vocab_words"]
        self.sbert_embeddings = resources["sbert_embeddings"]

        self.threshold = threshold if threshold is not None else (0.35 if USE_SBERT else 0.18)
        self.top_k = top_k
        self.max_answer_chars = max_answer_chars
        self.translate_enabled = translate_enabled

        self.reset()
        self._define_rules()

    # ---------- state ----------
    def reset(self):
        self.conversation_history = []
        self.last_focus = None
        self.last_lang = 'id'
        self.last_related = []

    # ---------- rules ----------
    def _define_rules(self):
        self.rules = {
            'emergency': {
                'patterns': [
                    r'(sesak.*(berat|parah|hebat)|nyeri dada.*(berat|hebat|menjalar))',
                    r'(tidak (bisa|dapat) bernapas|berhenti bernapas|henti jantung)',
                    r'\b(pingsan|kejang|tidak sadarkan diri|overdosis)\b',
                    r'(muntah darah|perdarahan hebat|bab hitam)',
                    r'(chest pain.*(severe|radiat)|can\'t breathe|cannot breathe)',
                    r'\b(unconscious|seizure|stroke symptoms now|heart attack now)\b',
                    r'(bunuh diri|mengakhiri hidup|suicide)',
                ],
                'responses': [
                    "🚨 **INI KONDISI DARURAT.**\n\n"
                    "Segera hubungi **119** (Ambulans Nasional) atau **112**, "
                    "atau langsung ke IGD rumah sakit terdekat.\n\n"
                    "Jangan menunggu jawaban chatbot untuk kondisi seperti ini. "
                    "Bila memungkinkan, minta orang di sekitar Anda untuk menemani."
                ],
            },
            'greeting': {
                'patterns': [r'^\s*(?:halo|hai|hei|hi|hello|hey|selamat|assalam\w*|pagi|siang|sore|malam)\b'
                             r'(?:\s+(?:bot|dok|dokter|semua|kak|min|admin|there|selamat|pagi|siang|sore|malam|apa kabar))*'
                             r'\s*[!.?,]*\s*$'],
                'responses': [
                    "Halo! Saya **MedQuAD Bot**, asisten informasi kesehatan berbasis "
                    "basis data resmi NIH.\n\nSilakan tanyakan tentang suatu penyakit — "
                    "definisi, gejala, penyebab, pengobatan, atau pencegahannya.",
                    "Hai! Ada yang ingin Anda ketahui seputar kesehatan hari ini?\n\n"
                    "Contoh: *\"gejala diabetes\"*, *\"how to prevent stroke\"*, "
                    "*\"pengobatan kanker payudara\"*.",
                ],
            },
            'thanks': {
                'patterns': [r'\b(terima kasih|makasih|thanks|thank you|tengkyu)\b'],
                'responses': [
                    "Sama-sama! Semoga membantu. Ingat, informasi ini bersifat edukatif — "
                    "untuk keluhan yang Anda alami sendiri, tetap konsultasikan ke dokter."
                ],
            },
            'capability': {
                'patterns': [r'(topik apa|bisa apa|\bkemampuan\b|what can you do|^\s*(?:help|bantuan)\s*[!?.]*\s*$)'],
                'responses': [None],
            },
        }

    def _check_rules(self, text):
        low = text.lower().strip()
        for intent, rule in self.rules.items():
            for pattern in rule['patterns']:
                if re.search(pattern, low):
                    if intent == 'capability' and len(low.split()) > 8:
                        continue
                    if intent == 'capability':
                        top_topics = self.df['focus_area'].value_counts().head(12).index.tolist()
                        return (
                            f"Saya punya **{len(self.df):,} pasangan tanya-jawab** medis "
                            f"mencakup **{self.df['focus_area'].nunique():,} topik penyakit** "
                            f"dari sumber resmi (NIH, CDC, NIDDK, NHLBI, NINDS, GARD, GHR).\n\n"
                            f"**Jenis pertanyaan yang saya pahami:** "
                            f"{', '.join(sorted(self.df['intent'].unique()))}\n\n"
                            f"**Topik terpopuler:** {', '.join(top_topics)}"
                        )
                    return random.choice(rule['responses'])
        return None

    def _safety_note(self, text):
        return self.POISON_NOTE if re.search(self.POISON_PATTERN, text.lower()) else None

    def _is_followup(self, user_input):
        if not self.last_focus or not re.search(r'[a-zA-Z]{2,}', user_input):
            return False
        return all(t in FOLLOWUP_TERMS for t in preprocess_text(user_input).split())

    # ---------- penerjemah ----------
    def _translate(self, text, src, tgt):
        if not self.translate_enabled or not text or not text.strip():
            return None
        try:
            return _translate_cached(text, src, tgt)
        except Exception:
            return None

    # ---------- retrieval ----------
    def _search_sbert(self, query):
        from sentence_transformers import util
        q_emb = SBERT_MODEL.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.sbert_embeddings)[0].cpu().numpy()
        top = scores.argsort()[-self.top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top]

    def _search_tfidf(self, query):
        q_vec = self.vectorizer.transform([preprocess_text(query)])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top = scores.argsort()[-self.top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top]

    def _find_best_match(self, query):
        results = self._search_sbert(query) if USE_SBERT else self._search_tfidf(query)
        results = self._rerank(query, results)
        return results[0]

    def _rerank(self, query, results):
        """+0.03 per kata yang persis sama, +0.10 bila intent cocok. Skor yang dikembalikan = skor ASLI."""
        q_intent = detect_intent(bridge_id_to_en(query))
        q_words = set(preprocess_text(query).split())

        boosted = []
        for idx, score in results:
            row = self.df.iloc[idx]
            doc_words = set(str(row['processed_question']).split())
            bonus = 0.03 * len(q_words & doc_words)
            if row['intent'] == q_intent:
                bonus += 0.10
            boosted.append((idx, score + bonus, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return [(i, raw) for i, _, raw in boosted]

    def _build_context_query(self, user_input, translated=None):
        core = f"{user_input} {translated}" if translated else user_input
        if self._is_followup(user_input):
            return f"{self.last_focus} {core}"
        return core

    # ---------- format ----------
    def _format_answer(self, row, score, lang='en'):
        answer = clean_answer(str(row['answer']))
        truncated = False
        if len(answer) > self.max_answer_chars:
            cut = answer[:self.max_answer_chars]
            dot = cut.rfind('. ')
            answer = cut[:dot + 1] if dot > 200 else cut
            truncated = True

        lang_note = ""
        if lang == 'id':
            tr = self._translate(answer, 'en', 'id')
            if tr:
                answer = tr
                lang_note = "*Diterjemahkan otomatis dari teks asli berbahasa Inggris — bisa kurang sempurna.*  \n"
            else:
                lang_note = "*Terjemahan otomatis nonaktif/tidak tersedia — teks di atas berbahasa Inggris (sesuai sumber).*  \n"
        if truncated:
            answer += " …\n\n*(jawaban dipotong — teks aslinya lebih panjang)*"

        method = "SBERT" if USE_SBERT else "TF-IDF"
        return (
            f"🩺 **{row['focus_area']}** — *{row['intent']}*\n\n"
            f"{answer}\n\n"
            f"---\n"
            f"📚 Sumber: **{row['source']}** (MedQuAD / NIH)  \n"
            f"{lang_note}"
            f"🎯 Relevansi: {score:.2f} ({method})  \n"
            f"⚠️ Informasi edukatif, bukan pengganti konsultasi dokter."
        )

    # ---------- fungsi utama ----------
    def get_response(self, user_input):
        self.last_related = []

        if not user_input or not user_input.strip():
            return "Silakan ketik pertanyaan Anda."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        lang = detect_language(user_input, known_en=self.vocab_words, default=self.last_lang)
        self.last_lang = lang

        note = self._safety_note(user_input)
        prefix = f"{note}\n\n---\n\n" if note else ""

        translated = self._translate(user_input, 'id', 'en') if lang == 'id' else None
        query = self._build_context_query(user_input, translated)

        results = self._search_sbert(query) if USE_SBERT else self._search_tfidf(query)
        if not results:
            return prefix + "Saya tidak menemukan informasi yang relevan."
        results = self._rerank(query, results)
        best_idx, best_score = results[0]

        if best_score < self.threshold:
            hint = self.df['focus_area'].value_counts().head(6).index.tolist()
            return prefix + (
                "Maaf, saya tidak menemukan jawaban yang cukup relevan di basis data saya "
                f"(skor tertinggi hanya {best_score:.2f}).\n\n"
                "Coba sebutkan **nama penyakitnya** secara spesifik, misalnya: "
                f"*{hint[0]}*, *{hint[1]}*, atau *{hint[2]}*.\n\n"
                "Basis data ini berbahasa Inggris, jadi menulis dalam Bahasa Inggris "
                "biasanya memberi hasil lebih akurat."
            )

        row = self.df.iloc[best_idx]
        self.conversation_history.append(user_input)
        self.last_focus = row['focus_area']

        response = prefix + self._format_answer(row, best_score, lang)

        related = self.df[(self.df['focus_area'] == row['focus_area']) &
                          (self.df.index != best_idx)]['question'].head(3).tolist()
        if related:
            if lang == 'id':
                tr = self._translate('\n'.join(related), 'en', 'id')
                tr_lines = [l.strip() for l in tr.split('\n') if l.strip()] if tr else []
                if len(tr_lines) == len(related):
                    related = tr_lines
            self.last_related = related
            response += "\n\n**❓ Pertanyaan terkait:**\n" + "\n".join(f"- {q}" for q in related)
        return response


# HELPER UI
def md(text: str) -> str:
    """Escape '$' supaya teks medis tidak dirender sebagai rumus LaTeX."""
    return text.replace("$", r"\$")


def queue_prompt(text: str):
    st.session_state.pending = text


def new_bot(res, **kw):
    return MedQuADChatbotEngine(res, **kw)


# SIDEBAR + PEMUATAN DATA
with st.sidebar:
    st.title("🏥 MedQuAD Bot")
    page = st.radio("Halaman", ["💬 Chatbot", "📊 Analisis Dataset", "🧪 Evaluasi"],
                    label_visibility="collapsed")
    st.divider()

    st.subheader("⚙️ Pengaturan")
    default_csv = next((p for p in CSV_CANDIDATES if os.path.exists(p)), CSV_CANDIDATES[0])
    csv_path = st.text_input("Lokasi medquad.csv", value=default_csv)

    tr_ok = translator_available()
    translate_on = st.toggle("Terjemahkan jawaban ke Bahasa Indonesia", value=tr_ok, disabled=not tr_ok,
                             help="Memakai deep-translator (Google Translate), butuh koneksi internet.")
    if not tr_ok:
        st.caption("⚠️ Penerjemah tidak tersedia (offline / deep-translator belum terpasang) — jawaban tetap berbahasa Inggris.")

    threshold = st.slider("Threshold relevansi", 0.05, 0.60, 0.35 if USE_SBERT else 0.18, 0.01,
                          help="Skor cosine minimum agar jawaban ditampilkan. Di bawah ini bot menjawab 'tidak tahu'.")
    top_k = st.slider("Top-K kandidat", 1, 10, 5)
    max_chars = st.slider("Maks. panjang jawaban (karakter)", 300, 4000, 1200, 100)

# --- pastikan CSV ada ---
if not os.path.exists(csv_path):
    st.title("🏥 MedQuAD Bot")
    st.error(f"File dataset tidak ditemukan: `{csv_path}`")
    up = st.file_uploader("Unggah **medquad.csv** (kolom: question, answer, source, focus_area)", type="csv")
    if up is not None:
        target = csv_path or "medquad.csv"
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "wb") as f:
            f.write(up.getbuffer())
        st.rerun()
    st.stop()

try:
    res = build_resources(csv_path, os.path.getmtime(csv_path), USE_SBERT)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

df = res["df"]

# --- state sesi ---
sig = (csv_path, os.path.getmtime(csv_path))
if st.session_state.get("bot_sig") != sig:
    st.session_state.bot = new_bot(res)
    st.session_state.bot_sig = sig
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
    st.session_state.pending = None

bot: MedQuADChatbotEngine = st.session_state.bot
bot.threshold, bot.top_k, bot.max_answer_chars = threshold, top_k, max_chars
bot.translate_enabled = bool(translate_on and tr_ok)

with st.sidebar:
    st.divider()
    st.caption(f"📚 **{len(df):,}** Q&A · **{df['focus_area'].nunique():,}** topik · "
               f"**{df['source'].nunique()}** sumber")
    st.caption(f"Retrieval: **{'SBERT' if USE_SBERT else 'TF-IDF'}** + keyword/intent re-ranking")
    if st.button("🔄 Reset percakapan", use_container_width=True):
        bot.reset()
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.session_state.pending = None
        st.rerun()

# HALAMAN 1 — CHATBOT\
def page_chat():
    st.title("🏥 MedQuAD Bot")
    st.caption(f"Chatbot informasi kesehatan berbasis {len(df):,} Q&A resmi NIH · "
               "Informasi edukatif, bukan pengganti konsultasi dokter.")
    st.info("🚨 Kondisi darurat? Hubungi **119** / **112** atau langsung ke IGD terdekat.", icon="⚠️")

    msgs = st.session_state.messages
    for m in msgs:
        with st.chat_message(m["role"], avatar="🩺" if m["role"] == "assistant" else "🧑"):
            st.markdown(md(m["content"]))

    # contoh pertanyaan (hanya saat percakapan masih kosong)
    if len(msgs) <= 1:
        st.markdown("**Coba tanyakan:**")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            cols[i % 2].button(s, key=f"sug_{i}", on_click=queue_prompt, args=(s,),
                               use_container_width=True)

    # tombol pertanyaan terkait untuk jawaban terakhir
    if len(msgs) > 1 and msgs[-1]["role"] == "assistant" and bot.last_related:
        st.markdown("**Lanjutkan dengan:**")
        for i, q in enumerate(bot.last_related):
            st.button(q, key=f"rel_{len(msgs)}_{i}", on_click=queue_prompt, args=(q,),
                      use_container_width=True)

    prompt = st.chat_input("Ketik pertanyaan medis Anda…")
    if st.session_state.get("pending") and not prompt:
        prompt = st.session_state.pending
    st.session_state.pending = None

    if prompt and prompt.strip():
        with st.chat_message("user", avatar="🧑"):
            st.markdown(md(prompt))
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("MedQuAD Bot sedang mencari…"):
                response = bot.get_response(prompt)
            st.markdown(md(response))
        msgs.append({"role": "user", "content": prompt})
        msgs.append({"role": "assistant", "content": response})
        st.rerun()

# HALAMAN 2 — ANALISIS DATASET
def page_analysis():
    st.title("📊 Analisis Dataset & Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris mentah", f"{res['n_raw']:,}")
    c2.metric("Setelah cleaning", f"{len(df):,}", delta=f"-{res['n_raw'] - len(df):,}", delta_color="off")
    c3.metric("Topik unik", f"{df['focus_area'].nunique():,}")
    c4.metric("Rata² jawaban", f"{df['answer_len'].mean():.0f} char")
    st.caption(f"Matriks TF-IDF: {res['tfidf_matrix'].shape[0]:,} dokumen × {res['tfidf_matrix'].shape[1]:,} fitur "
               f"· waktu indexing {res['index_seconds']:.2f} detik")

    st.subheader("Distribusi intent pertanyaan")
    st.bar_chart(df['intent'].value_counts())

    st.subheader("Jumlah Q&A per institusi sumber")
    st.bar_chart(df['source'].value_counts())

    st.subheader("Distribusi panjang jawaban")
    counts, edges = np.histogram(df['answer_len'].clip(upper=6000), bins=30)
    st.bar_chart(pd.DataFrame({"jumlah jawaban": counts}, index=edges[:-1].astype(int)))
    st.caption(f"Sumbu-x: jumlah karakter (dipotong di 6000). Batas tampil di chatbot saat ini: {bot.max_answer_chars} karakter.")

    st.subheader("Similarity score untuk query uji")
    test_cases = [
        ("what are the symptoms of glaucoma", "Glaucoma"),
        ("how to prevent diabetes", "Diabetes"),
        ("treatments for breast cancer", "Breast Cancer"),
        ("what causes asthma", "Asthma"),
        ("is parkinson disease inherited", "Parkinson"),
        ("how many people have osteoporosis", "Osteoporosis"),
        ("gejala kanker payudara", "Kanker (ID)"),
        ("xyzabc qwerty nonsense", "Out-of-scope"),
    ]
    rows = []
    for q, label in test_cases:
        idx, score = bot._find_best_match(q)
        rows.append({"Query": q, "Label": label, "Skor": round(score, 3),
                     "Lolos threshold": "✅" if score >= bot.threshold else "❌",
                     "Top-1 topik": df.iloc[idx]['focus_area']})
    sim_df = pd.DataFrame(rows)
    st.bar_chart(sim_df.set_index("Label")["Skor"])
    st.caption(f"Threshold saat ini = {bot.threshold:.2f}")
    st.dataframe(sim_df, hide_index=True, use_container_width=True)

    st.subheader("20 istilah paling informatif (rata-rata bobot TF-IDF)")
    names = np.array(bot.vectorizer.get_feature_names_out())
    mean_tfidf = np.asarray(bot.tfidf_matrix.mean(axis=0)).ravel()
    top20 = mean_tfidf.argsort()[-20:][::-1]
    st.dataframe(pd.DataFrame({"Istilah": names[top20], "Bobot": mean_tfidf[top20].round(5)}),
                 hide_index=True, use_container_width=True)

    with st.expander("🔎 Jelajahi dataset (500 baris pertama)"):
        f_intent = st.multiselect("Filter intent", sorted(df['intent'].unique()))
        view = df[df['intent'].isin(f_intent)] if f_intent else df
        st.dataframe(view[['source', 'focus_area', 'intent', 'question']].head(500),
                     hide_index=True, use_container_width=True)

# HALAMAN 3 — EVALUASI
INTENT_PHRASE = {
    'gejala': 'symptoms of', 'penyebab': 'what causes',
    'pengobatan': 'treatments for', 'pencegahan': 'how to prevent',
    'diagnosis': 'how to diagnose', 'prevalensi': 'how many people have',
    'genetik': 'is inherited', 'penelitian': 'research on',
    'prognosis': 'outlook for', 'komplikasi': 'complications of',
    'risiko': 'who is at risk for', 'definisi': 'what is',
    'lainnya': 'information about',
}

TEST_SCENARIOS = [
    ("What are the symptoms of Glaucoma ?", "glaucoma", "Pertanyaan persis seperti dataset"),
    ("symptoms of glaucoma", "glaucoma", "Parafrase — tanpa kata tanya"),
    ("my eye pressure is high what disease", "glaucoma", "Deskriptif — tanpa nama penyakit"),
    ("how do I prevent diabetes", "diabetes", "Parafrase intent pencegahan"),
    ("treatments for breast cancer", "breast cancer", "Intent pengobatan"),
    ("what causes asthma", "asthma", "Intent penyebab"),
    ("is parkinson disease inherited", "parkinson", "Intent genetik"),
    ("gejala kanker payudara", "breast cancer", "Query Bahasa Indonesia"),
    ("cara mencegah stroke", "stroke", "Query Bahasa Indonesia"),
    ("halo selamat pagi", "medquad", "Rule-based: salam"),
    ("topik apa saja yang tersedia", "topik", "Rule-based: kapabilitas"),
    ("terima kasih banyak", "sama-sama", "Rule-based: terima kasih"),
    ("ayah saya pingsan tidak sadarkan diri", "darurat", "Rule-based: DARURAT"),
    ("nyeri dada berat menjalar ke lengan", "darurat", "Rule-based: DARURAT"),
    ("zxcvbn qwerty asdfgh 12345", "tidak menemukan", "Out-of-scope harus ditolak"),
]


def page_eval():
    st.title("🧪 Evaluasi Chatbot")
    st.caption("Evaluasi memakai instance bot terpisah, jadi tidak mengganggu percakapan di halaman Chatbot.")
    eval_translate = st.checkbox("Aktifkan penerjemah saat evaluasi skenario (lebih lambat, butuh internet)", value=False)

    def fresh():
        return new_bot(res, threshold=bot.threshold, top_k=bot.top_k,
                       max_answer_chars=bot.max_answer_chars,
                       translate_enabled=bool(eval_translate and tr_ok))

    tab_a, tab_b, tab_c = st.tabs(["A. Otomatis (parafrase)", "B. Skenario manual", "C. Memori & regresi"])

    # ---------- A ----------
    with tab_a:
        n_test = st.slider("Jumlah query uji", 50, min(1000, len(df)), 400, 50)
        if st.button("▶️ Jalankan evaluasi otomatis", type="primary"):
            eb = fresh()
            sample = df.sample(n_test, random_state=42)
            top1 = top3 = focus = 0
            prog = st.progress(0.0)
            t0 = time.time()
            for i, (_, row) in enumerate(sample.iterrows(), 1):
                query = f"{INTENT_PHRASE[row['intent']]} {row['focus_area']}"
                results = eb._rerank(query, eb._search_tfidf(query))

                def match(j):
                    r = eb.df.iloc[j]
                    return r['focus_area'].lower() == row['focus_area'].lower() and r['intent'] == row['intent']

                top1 += match(results[0][0])
                top3 += any(match(j) for j, _ in results[:3])
                focus += (eb.df.iloc[results[0][0]]['focus_area'].lower() == row['focus_area'].lower())
                if i % 10 == 0 or i == n_test:
                    prog.progress(i / n_test)
            elapsed = time.time() - t0
            prog.empty()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Top-1 (focus+intent)", f"{top1 / n_test:.2%}", f"{top1}/{n_test}", delta_color="off")
            m2.metric("Top-3 (focus+intent)", f"{top3 / n_test:.2%}", f"{top3}/{n_test}", delta_color="off")
            m3.metric("Topic (focus saja)", f"{focus / n_test:.2%}", f"{focus}/{n_test}", delta_color="off")
            m4.metric("Waktu / query", f"{elapsed / n_test * 1000:.1f} ms")
            st.caption(f"Metode: {'SBERT' if USE_SBERT else 'TF-IDF'} + keyword/intent re-ranking · "
                       "query dibangun dari pola intent + nama topik (bukan kalimat asli dataset).")

    # ---------- B ----------
    with tab_b:
        if st.button("▶️ Jalankan 15 skenario", type="primary"):
            eb = fresh()
            rows, correct = [], 0
            prog = st.progress(0.0)
            for i, (q, expected, desc) in enumerate(TEST_SCENARIOS, 1):
                eb.last_focus = None
                resp = eb.get_response(q).lower()
                ok = expected.lower() in resp
                correct += ok
                rows.append({"": "✅" if ok else "❌", "Skenario": desc, "Input": q,
                             "Harapan": f"mengandung '{expected}'",
                             "Cuplikan": ' '.join(resp.split())[:110] + "…"})
                prog.progress(i / len(TEST_SCENARIOS))
            prog.empty()
            st.metric("Skor skenario", f"{correct}/{len(TEST_SCENARIOS)}",
                      f"{correct / len(TEST_SCENARIOS):.0%}", delta_color="off")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---------- C ----------
    with tab_c:
        if st.button("▶️ Jalankan uji memori & regresi", type="primary"):
            eb = fresh()

            st.markdown("**Uji memori percakapan** — follow-up singkat harus tetap membahas topik yang sama")
            eb.reset()
            mem_rows = []
            for q in ["what is glaucoma", "symptoms", "treatments"]:
                r = eb.get_response(q)
                topic = r.split('**')[1] if '**' in r else '-'
                mem_rows.append({"Input": q, "Topik terdeteksi": topic})
            st.dataframe(pd.DataFrame(mem_rows), hide_index=True, use_container_width=True)

            st.markdown("**Uji regresi**")
            reg = []

            eb.reset(); eb.get_response("what is glaucoma")
            r = eb.get_response("keracunan")
            reg.append(("Pergantian topik: 'keracunan' setelah glaucoma tidak dijawab Glaucoma",
                        "glaucoma" not in r.lower()))

            eb.reset(); eb.get_response("what is glaucoma")
            for q in ["symptoms", "gejalanya", "apa pengobatannya"]:
                r = eb.get_response(q)
                topic = r.split('**')[1] if '**' in r else '-'
                reg.append((f"Follow-up murni {q!r} tetap di topik Glaucoma (terdeteksi: {topic})",
                            "glaucoma" in topic.lower()))

            eb.reset(); r_id = eb.get_response("gejala kanker payudara")
            eb.reset(); r_en = eb.get_response("symptoms of breast cancer")
            reg.append(("Balasan query Indonesia memuat catatan terjemahan/bahasa (🌐)", "🌐" in r_id))
            reg.append(("Balasan query Inggris TIDAK diterjemahkan", "🌐" not in r_en))

            for q, should_greet in [("hipertensi", False), ("HIV AIDS", False),
                                    ("sore throat", False), ("halo selamat pagi", True)]:
                got = eb._check_rules(q) is not None
                reg.append((f"{q!r} {'dikenali' if should_greet else 'TIDAK dianggap'} sebagai salam", got == should_greet))

            st.dataframe(pd.DataFrame([{"": "✅" if ok else "❌", "Uji": d} for d, ok in reg]),
                         hide_index=True, use_container_width=True)
            st.metric("Lulus", f"{sum(ok for _, ok in reg)}/{len(reg)}")

# ROUTER
if page.startswith("💬"):
    page_chat()
elif page.startswith("📊"):
    page_analysis()
else:
    page_eval()
