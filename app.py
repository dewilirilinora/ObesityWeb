import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Prediksi Tingkat Risiko Obesitas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CSS CUSTOM
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #f5f3ef;
}

/* ── Semua label input (kini berada di halaman utama, bukan sidebar) ── */
.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stRadio label {
    color: #6e6e9e !important;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Header hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px;
    padding: 48px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(229,57,53,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(100,181,246,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(229,57,53,0.15);
    border: 1px solid rgba(229,57,53,0.35);
    color: #ef9a9a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 12px 0;
}
.hero-title span {
    color: #ef5350;
}
.hero-sub {
    color: #9e9eb8;
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}
.hero-meta {
    display: flex;
    gap: 24px;
    margin-top: 28px;
    flex-wrap: wrap;
}
.hero-meta-item {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 18px;
}
.hero-meta-item .label {
    font-size: 0.68rem;
    color: #6e6e9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.hero-meta-item .value {
    font-size: 0.92rem;
    color: #e8e4dc;
    font-weight: 500;
    margin-top: 2px;
}

/* ── Section cards ── */
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid #e8e2d8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1a1a2e;
    margin: 0 0 6px 0;
}
.section-subtitle {
    font-size: 0.82rem;
    color: #9e9e9e;
    margin: 0 0 24px 0;
    font-weight: 400;
}
.input-group-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e53935;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 4px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #f0ece5;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #f8f6f2;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #ede8e0;
    text-align: center;
}
.metric-card .m-icon { font-size: 1.6rem; margin-bottom: 6px; }
.metric-card .m-val  { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.metric-card .m-lbl  { font-size: 0.72rem; color: #9e9e9e; text-transform: uppercase;
                        letter-spacing: 0.07em; font-weight: 600; margin-top: 2px; }

/* ── Result banners ── */
.result-banner {
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
    border-left: 5px solid;
}
.result-banner.green  { background:#e8f5e9; border-color:#43a047; }
.result-banner.yellow { background:#fffde7; border-color:#f9a825; }
.result-banner.orange { background:#fff3e0; border-color:#ef6c00; }
.result-banner.red    { background:#ffebee; border-color:#e53935; }
.result-banner.darkred{ background:#fce4ec; border-color:#b71c1c; }

.result-banner .rb-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px; opacity: 0.7;
}
.result-banner .rb-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: #1a1a2e; margin: 0 0 6px 0;
}
.result-banner .rb-desc {
    font-size: 0.9rem; color: #555; line-height: 1.6; margin: 0;
}
.confidence-bar-wrap { margin-top: 16px; }
.confidence-bar-wrap .cb-label {
    font-size: 0.78rem; color: #666; margin-bottom: 6px;
    display: flex; justify-content: space-between;
}
.confidence-bar-outer {
    height: 8px; background: rgba(0,0,0,0.08);
    border-radius: 99px; overflow: hidden;
}
.confidence-bar-inner {
    height: 100%; border-radius: 99px;
    transition: width 0.8s ease;
}

/* ── Rekomendasi item ── */
.rek-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 14px 0; border-bottom: 1px solid #f0ece5;
}
.rek-item:last-child { border-bottom: none; }
.rek-icon {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
    background: #f0ece5;
}
.rek-text .rek-head { font-weight: 600; color: #1a1a2e; font-size: 0.9rem; }
.rek-text .rek-body { font-size: 0.82rem; color: #777; margin-top: 2px; line-height: 1.5; }

/* ── BMI gauge ── */
.bmi-wrap {
    background: #f8f6f2; border-radius: 14px; padding: 20px 24px;
    border: 1px solid #ede8e0; margin-bottom: 12px;
}
.bmi-row { display: flex; justify-content: space-between; align-items: center; }
.bmi-num { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
.bmi-tag { font-size: 0.8rem; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

/* ── Divider label ── */
.divider-label {
    display: flex; align-items: center; gap: 12px; margin: 20px 0;
}
.divider-label span { font-size: 0.75rem; font-weight: 600; color: #bbb;
    text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }
.divider-label::before, .divider-label::after {
    content:''; flex: 1; height: 1px; background: #e8e2d8;
}

/* ── Streamlit overrides ── */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e53935, #b71c1c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stFormSubmitButton"] button:hover,
div[data-testid="stButton"] button:hover { opacity: 0.88 !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e0dbd2 !important;
    background: #faf9f6 !important;
    color: #000
}

/* Footer */
.footer {
    text-align: center; padding: 28px 0 10px;
    font-size: 0.78rem; color: #bbb;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("xgboost_obesity_pipeline.pkl")
    le       = joblib.load("label_encoder.pkl")
    return pipeline, le

try:
    pipeline, le = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
CLASS_INFO = {
    "Insufficient_Weight": {
        "label": "Berat Badan Kurang",
        "color": "yellow",
        "bar_color": "#f9a825",
        "desc": "Berat badan kamu berada di bawah kisaran normal. Kondisi ini dapat memengaruhi imunitas, energi, dan kesehatan tulang."
    },
    "Normal_Weight": {
        "label": "Berat Badan Normal",
        "color": "green",
        "bar_color": "#43a047",
        "desc": "Selamat! Berat badanmu berada dalam kisaran normal. Pertahankan pola hidup sehat yang sudah kamu jalani."
    },
    "Overweight_Level_I": {
        "label": "Kelebihan Berat Badan Tk. I",
        "color": "orange",
        "bar_color": "#ef6c00",
        "desc": "Berat badanmu sedikit di atas normal. Perubahan kecil pada pola makan dan aktivitas fisik sudah cukup efektif."
    },
    "Overweight_Level_II": {
        "label": "Kelebihan Berat Badan Tk. II",
        "color": "orange",
        "bar_color": "#e64a19",
        "desc": "Berat badanmu berada di tingkat kelebihan yang lebih signifikan. Diperlukan perubahan gaya hidup yang lebih terstruktur."
    },
    "Obesity_Type_I": {
        "label": "Obesitas Tipe I",
        "color": "red",
        "bar_color": "#e53935",
        "desc": "Kamu termasuk dalam kategori obesitas tingkat I. Kondisi ini meningkatkan risiko penyakit jantung, diabetes, dan hipertensi."
    },
    "Obesity_Type_II": {
        "label": "Obesitas Tipe II",
        "color": "red",
        "bar_color": "#c62828",
        "desc": "Kategori obesitas tingkat II dengan risiko komplikasi kesehatan yang lebih serius. Penanganan medis sangat dianjurkan."
    },
    "Obesity_Type_III": {
        "label": "Obesitas Tipe III (Morbid)",
        "color": "darkred",
        "bar_color": "#b71c1c",
        "desc": "Kategori obesitas tingkat III (morbid obesity). Kondisi ini memerlukan penanganan medis segera dan komprehensif."
    },
}

def bmi_category(bmi):
    if bmi < 18.5:
        return "Kurang", "#f9a825", "#fff8e1"
    elif bmi < 25.0:
        return "Normal", "#43a047", "#e8f5e9"
    elif bmi < 30.0:
        return "Berlebih", "#ef6c00", "#fff3e0"
    else:
        return "Obesitas", "#e53935", "#ffebee"

def lifestyle_score(favc, fcvc, caec, family, faf, ch2o, calc):
    score = 0
    if favc == "yes": score += 1
    if fcvc < 2: score += 1
    if caec in ["Sometimes", "Frequently", "Always"]:
        score += (["Sometimes", "Frequently", "Always"].index(caec) + 1)
    if family == "yes": score += 1
    score += max(0, 2 - faf)
    if ch2o < 2: score += 1
    if calc in ["Sometimes", "Frequently", "Always"]: score += 1
    max_score = 10
    return min(score, max_score), max_score


def generate_recommendations(pred_label, favc, fcvc, caec, family, faf, ch2o,
                              calc, smoke, tue, mtrans, scc, ncp):
    """
    Menyusun rekomendasi secara dinamis berdasarkan logika faktor risiko gaya hidup
    yang diinput pengguna, dikombinasikan dengan tingkat risiko hasil prediksi model.
    Aturan dasar:
      - Faktor yang MENINGKATKAN kalori/risiko (FAVC, CAEC, CALC, TUE, kendaraan
        bermotor) -> direkomendasikan untuk DIKURANGI.
      - Faktor yang MEMBAKAR kalori/menurunkan risiko (FAF, transportasi aktif,
        konsumsi air, konsumsi sayur) -> direkomendasikan untuk DITAMBAH.
      - Untuk kategori Insufficient_Weight, logika sebagian dibalik karena tujuan
        utamanya adalah menaikkan berat badan secara sehat, bukan menurunkannya.
    """
    is_underweight = (pred_label == "Insufficient_Weight")
    is_normal = (pred_label == "Normal_Weight")
    is_overweight_or_obese = pred_label in [
        "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
    ]
    is_severe_obese = pred_label in ["Obesity_Type_II", "Obesity_Type_III"]

    # Dua kelompok terpisah: "neg" = faktor risiko yang perlu diperbaiki (selalu
    # diprioritaskan tampil lebih dulu secara utuh), "pos" = afirmasi kebiasaan baik
    # yang sudah diterapkan (hanya mengisi slot sisa apabila faktor negatif sedikit).
    neg, pos = [], []

    # 1) Konsumsi makanan tinggi kalori/lemak (FAVC)
    if favc == "yes" and not is_underweight:
        neg.append((
            "🍔", "Kurangi Makanan Tinggi Kalori dan Lemak",
            "Kamu tercatat sering mengonsumsi makanan tinggi kalori. Kurangi porsi "
            "gorengan, fast food, dan makanan berlemak jenuh secara bertahap, lalu "
            "gantikan dengan protein rendah lemak, sayur, dan karbohidrat kompleks."
        ))
    elif favc == "yes" and is_underweight:
        pos.append((
            "🥑", "Pilih Sumber Kalori yang Sehat",
            "Karena berat badanmu di bawah normal, pertahankan asupan kalori namun "
            "utamakan sumber yang sehat seperti alpukat, kacang-kacangan, dan minyak zaitun."
        ))
    elif favc == "no":
        pos.append((
            "🥗", "Pola Makan Rendah Kalori Sudah Baik",
            "Kamu jarang mengonsumsi makanan tinggi kalori. Pertahankan kebiasaan memilih "
            "makanan yang lebih sehat ini."
        ))

    # 2) Frekuensi konsumsi sayur (FCVC)
    if fcvc < 2:
        neg.append((
            "🥦", "Tambah Konsumsi Sayur dan Buah",
            "Frekuensi konsumsi sayurmu masih tergolong rendah. Tambahkan porsi sayur "
            "pada setiap makan utama untuk memenuhi kebutuhan serat, vitamin, dan mineral harian."
        ))
    elif fcvc >= 2.5:
        pos.append((
            "🥬", "Konsumsi Sayur Sudah Sangat Baik",
            "Frekuensi konsumsi sayur dan buahmu sudah tinggi. Pertahankan pola makan "
            "tinggi serat ini untuk menjaga kesehatan pencernaan dan berat badan."
        ))

    # 3) Kebiasaan ngemil (CAEC)
    if caec in ["Frequently", "Always"] and not is_underweight:
        neg.append((
            "🍪", "Kurangi Frekuensi Ngemil di Luar Jam Makan",
            "Kebiasaan ngemil di luar waktu makan utama tergolong sering. Ganti camilan "
            "tinggi gula/garam dengan pilihan lebih sehat seperti buah potong atau kacang tanpa garam."
        ))
    elif caec in ["Frequently", "Always"] and is_underweight:
        pos.append((
            "🥜", "Jadikan Camilan Sebagai Tambahan Kalori Sehat",
            "Kebiasaan ngemil dapat dimanfaatkan untuk menambah asupan kalori harian secara "
            "sehat, misalnya dengan kacang-kacangan, granola, atau susu tinggi protein."
        ))
    elif caec in ["no", "Sometimes"]:
        pos.append((
            "🍎", "Kebiasaan Ngemil Sudah Terkendali",
            "Frekuensi ngemil di luar jam makan utama tergolong terkendali. Pertahankan "
            "kebiasaan ini agar asupan kalori harian tetap terjaga."
        ))

    # 4) Jumlah makan utama (NCP) -> hanya relevan untuk underweight
    if is_underweight and ncp < 3:
        neg.append((
            "🍽️", "Tambah Frekuensi Makan Utama",
            "Jumlah makan utamamu masih kurang dari 3 kali sehari. Tambahkan menjadi "
            "3 kali makan utama disertai 1-2 kali selingan bergizi untuk membantu menaikkan berat badan."
        ))

    # 5) Aktivitas fisik (FAF)
    if faf < 1.5 and not is_underweight:
        neg.append((
            "🏃", "Tingkatkan Frekuensi Aktivitas Fisik",
            f"Frekuensi aktivitas fisikmu saat ini sekitar {faf:.1f} kali per minggu, tergolong "
            "rendah. Usahakan berolahraga minimal 3-5 kali per minggu selama 30 menit, seperti "
            "jalan cepat, bersepeda, atau latihan kekuatan ringan."
        ))
    elif faf < 1.0 and is_underweight:
        neg.append((
            "💪", "Tambahkan Latihan Kekuatan (Bukan Kardio Berlebihan)",
            "Untuk menaikkan berat badan secara sehat, fokuskan aktivitas fisik pada latihan "
            "kekuatan (strength training) guna membentuk massa otot, bukan aktivitas kardio yang berlebihan."
        ))
    elif faf >= 2.5:
        pos.append((
            "✅", "Pertahankan Rutinitas Aktivitas Fisik",
            "Frekuensi aktivitas fisikmu sudah cukup baik. Pertahankan rutinitas ini dan "
            "variasikan jenis olahraga agar tetap konsisten dalam jangka panjang."
        ))

    # 6) Konsumsi air putih (CH2O)
    if ch2o < 2:
        neg.append((
            "💧", "Tingkatkan Konsumsi Air Putih",
            "Konsumsi air putihmu saat ini di bawah 2 liter per hari. Usahakan minum minimal "
            "2 liter (setara 8 gelas) air putih setiap hari untuk menjaga metabolisme tubuh."
        ))
    elif ch2o >= 2.5:
        pos.append((
            "🚰", "Konsumsi Air Putih Sudah Optimal",
            "Asupan air putih harianmu sudah baik. Pertahankan kebiasaan ini untuk menjaga "
            "metabolisme dan fungsi tubuh secara keseluruhan."
        ))

    # 7) Konsumsi alkohol (CALC)
    if calc in ["Frequently", "Always"]:
        neg.append((
            "🍷", "Batasi Konsumsi Alkohol",
            "Frekuensi konsumsi alkoholmu tergolong sering. Alkohol menyumbang kalori kosong "
            "yang tinggi, sehingga membatasi konsumsinya dapat membantu mengontrol berat badan "
            "dan menjaga kesehatan hati."
        ))
    elif calc == "no":
        pos.append((
            "🚫", "Tidak Mengonsumsi Alkohol",
            "Kamu tidak mengonsumsi alkohol. Pertahankan kebiasaan baik ini karena turut "
            "membantu menjaga berat badan dan kesehatan hati."
        ))

    # 8) Merokok (SMOKE)
    if smoke == "yes":
        neg.append((
            "🚭", "Hentikan Kebiasaan Merokok",
            "Merokok tidak berkontribusi langsung terhadap berat badan, namun meningkatkan "
            "risiko penyakit kardiovaskular yang dapat diperparah oleh kondisi berat badan berlebih."
        ))
    else:
        pos.append((
            "🌿", "Tidak Merokok, Pertahankan",
            "Kamu tidak memiliki kebiasaan merokok. Pertahankan gaya hidup bebas rokok ini "
            "untuk menjaga kesehatan jantung dan paru-paru."
        ))

    # 9) Waktu penggunaan perangkat teknologi (TUE)
    if tue > 1.0 and not is_underweight:
        neg.append((
            "📵", "Kurangi Waktu Duduk di Depan Layar",
            "Durasi penggunaan perangkat teknologimu tergolong tinggi. Selingi dengan bergerak "
            "aktif setiap 30-60 menit untuk mengurangi perilaku sedentari (banyak duduk)."
        ))
    elif tue <= 1.0:
        pos.append((
            "⏱️", "Durasi Waktu Layar Sudah Terkendali",
            "Waktu penggunaan perangkat teknologimu tergolong wajar. Pertahankan agar "
            "waktu duduk pasif tidak berlebihan."
        ))

    # 10) Moda transportasi (MTRANS)
    if mtrans in ["Automobile", "Motorbike"] and not is_underweight:
        neg.append((
            "🚶", "Gunakan Transportasi Aktif Sesekali",
            "Kamu sehari-hari menggunakan kendaraan bermotor. Cobalah berjalan kaki atau "
            "bersepeda untuk jarak dekat guna menambah aktivitas fisik harian secara alami."
        ))
    elif mtrans in ["Walking", "Bike"]:
        pos.append((
            "🚴", "Sudah Menerapkan Transportasi Aktif",
            "Kamu sudah menggunakan moda transportasi aktif sehari-hari. Kebiasaan ini "
            "membantu menambah aktivitas fisik secara alami, pertahankan."
        ))

    # 11) Monitoring kalori (SCC)
    if scc == "no" and is_overweight_or_obese:
        neg.append((
            "📋", "Mulai Pantau Asupan Kalori Harian",
            "Kamu belum memiliki kebiasaan memantau asupan kalori. Mulailah mencatat makanan "
            "harian menggunakan aplikasi food diary untuk membantu mengontrol jumlah kalori yang masuk."
        ))
    elif scc == "yes":
        pos.append((
            "📈", "Sudah Memantau Asupan Kalori",
            "Kamu sudah memiliki kebiasaan memantau asupan kalori harian. Pertahankan "
            "kebiasaan ini untuk membantu menjaga berat badan tetap terkontrol."
        ))

    # 12) Riwayat keluarga -> edukasi tambahan (dikelompokkan sebagai faktor risiko)
    if family == "yes":
        neg.append((
            "🧬", "Perhatikan Riwayat Keluarga",
            "Kamu memiliki riwayat keluarga dengan kondisi overweight/obesitas. Lakukan "
            "pemeriksaan kesehatan berkala (tekanan darah, gula darah, kolesterol) sebagai langkah pencegahan dini."
        ))

    # 13) Rekomendasi konsultasi medis untuk obesitas tingkat lanjut -> prioritas utama
    if is_severe_obese:
        neg.insert(0, (
            "🏥", "Konsultasikan dengan Tenaga Medis",
            "Dengan kategori obesitas tingkat lanjut, sangat dianjurkan untuk berkonsultasi "
            "dengan dokter atau ahli gizi guna menyusun program penurunan berat badan yang "
            "aman dan terpantau secara medis."
        ))

    # 14) Prioritas khusus underweight di posisi teratas
    if is_underweight:
        neg.insert(0, (
            "🍚", "Tingkatkan Asupan Kalori Secara Sehat",
            "Berat badanmu berada di bawah normal. Tambahkan porsi makan dengan sumber kalori "
            "padat gizi seperti kacang-kacangan, alpukat, whole grains, dan protein tanpa lemak "
            "untuk mencapai berat badan ideal secara bertahap dan sehat."
        ))

    # 15) Jika sama sekali tidak ada faktor risiko maupun afirmasi yang terdeteksi
    if len(neg) == 0 and len(pos) == 0:
        pos.append((
            "✅", "Pertahankan Pola Hidup Sehat",
            "Tidak ditemukan faktor risiko gaya hidup yang signifikan dari data yang kamu "
            "masukkan. Pertahankan pola makan seimbang, aktivitas fisik rutin, dan waktu "
            "istirahat yang cukup."
        ))
    if is_normal:
        pos.append((
            "🎯", "Jaga Konsistensi",
            "Berat badanmu sudah berada dalam kategori normal. Jaga konsistensi pola makan "
            "dan aktivitas fisik yang sudah baik agar tetap berada pada kategori ini."
        ))

    # Faktor risiko (negatif) selalu diprioritaskan tampil lebih dulu secara utuh.
    # Afirmasi (positif) hanya ditambahkan untuk mengisi slot sisa apabila jumlah
    # faktor risiko yang terdeteksi sedikit, sehingga daftar tidak didominasi hal
    # yang sudah baik ketika masih banyak hal yang justru perlu diperbaiki.
    MAX_ITEMS = 6
    combined = neg[:MAX_ITEMS]
    if len(combined) < MAX_ITEMS:
        combined += pos[:MAX_ITEMS - len(combined)]

    return combined


# ──────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge"> Machine Learning · XGBoost · Stratified 10-Fold CV</div>
  <h1 class="hero-title">Prediksi Tingkat<br><span>Risiko Obesitas</span></h1>
  <p class="hero-sub">
    Sistem prediksi berbasis <em>machine learning</em> untuk mendeteksi tingkat risiko obesitas
    secara <em>real-time</em> berdasarkan data gaya hidup dan karakteristik fisik.
  </p>
  <div class="hero-meta">
    <div class="hero-meta-item">
      <div class="label">Algoritma</div>
      <div class="value">XGBoost</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Validasi</div>
      <div class="value">Stratified 10-Fold CV</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Akurasi CV</div>
      <div class="value">97.51%</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Kelas Output</div>
      <div class="value">7 Kategori Obesitas</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ File model tidak ditemukan. Pastikan `xgboost_obesity_pipeline.pkl` dan `label_encoder.pkl` berada di folder yang sama dengan `app.py`.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# INPUT DATA — kini berada di halaman utama (bukan sidebar)
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-card">
  <p class="section-title">📝 Input Data Pengguna</p>
  <p class="section-subtitle">Lengkapi data fisik, pola makan, dan gaya hidupmu di bawah ini</p>
""", unsafe_allow_html=True)

with st.form("form_input"):
    # ── Baris 1: input selectbox / number ──
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="input-group-title">Data Fisik</div>', unsafe_allow_html=True)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age    = st.number_input("Usia (tahun)", min_value=10, max_value=60, value=25)
        height = st.number_input("Tinggi Badan (m)", min_value=1.40, max_value=2.20,
                                  value=1.68, step=0.01, format="%.2f")
        weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=250.0,
                                  value=70.0, step=0.5, format="%.1f")
        family = st.selectbox("Riwayat Keluarga dengan Obesitas",
                               ["yes", "no"], format_func=lambda x: "Ya" if x == "yes" else "Tidak")

    with col2:
        st.markdown('<div class="input-group-title">Pola Makan</div>', unsafe_allow_html=True)
        favc = st.selectbox("Konsumsi Makanan Tinggi Kalori (FAVC)",
                             ["yes", "no"], format_func=lambda x: "Ya" if x == "yes" else "Tidak")
        caec = st.selectbox("Konsumsi Makanan Ringan (CAEC)",
                             ["no", "Sometimes", "Frequently", "Always"],
                             index=1,
                             format_func=lambda x: {
                                 "no": "Tidak", "Sometimes": "Kadang-kadang",
                                 "Frequently": "Sering", "Always": "Selalu"
                             }[x])
        # Slider untuk Pola Makan
        fcvc = st.slider("Frekuensi Sayur (FCVC)", 1.0, 3.0, 2.0, 0.5)
        ncp = st.slider("Makan Utama/Hari (NCP)", 1.0, 4.0, 3.0, 0.5)

    with col3:
        st.markdown('<div class="input-group-title">Kebiasaan & Gaya Hidup</div>', unsafe_allow_html=True)
        smoke = st.selectbox("Merokok (SMOKE)",
                              ["no", "yes"], format_func=lambda x: "Tidak" if x == "no" else "Ya")
        scc  = st.selectbox("Monitoring Konsumsi Kalori (SCC)",
                             ["no", "yes"], format_func=lambda x: "Tidak" if x == "no" else "Ya")
        mtrans = st.selectbox("Transportasi Harian (MTRANS)",
                               ["Public_Transportation", "Automobile", "Walking",
                                "Motorbike", "Bike"],
                               format_func=lambda x: {
                                   "Public_Transportation": "Transportasi Umum",
                                   "Automobile": "Mobil", "Walking": "Jalan Kaki",
                                   "Motorbike": "Motor", "Bike": "Sepeda"
                               }[x])
        # Slider untuk Gaya Hidup
        calc = st.selectbox("Konsumsi Alkohol (CALC)",
                             ["no", "Sometimes", "Frequently", "Always"],
                             format_func=lambda x: {
                                 "no": "Tidak", "Sometimes": "Kadang-kadang",
                                 "Frequently": "Sering", "Always": "Selalu"
                             }[x])
        ch2o = st.slider("Air Harian, Liter (CH2O)", 1.0, 3.0, 2.0, 0.5)
        faf = st.slider("Aktivitas Fisik/Minggu (FAF)", 0.0, 3.0, 1.0, 0.5)
        tue = st.slider("Layar per Hari, Jam (TUE)", 0.0, 2.0, 1.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.form_submit_button(" Analisis Risiko Obesitas", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close section-card input


# ──────────────────────────────────────────────────────────────
# HASIL — hanya tampil setelah tombol ditekan, di bawah form input
# ──────────────────────────────────────────────────────────────
if predict_btn:
    input_df = pd.DataFrame([{
        "Gender": gender, "Age": float(age), "Height": height, "Weight": weight,
        "family_history_with_overweight": family,
        "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec,
        "SMOKE": smoke, "CH2O": ch2o, "SCC": scc,
        "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
    }])

    with st.spinner("Memproses prediksi..."):
        pred_encoded = pipeline.predict(input_df)[0]
        pred_proba   = pipeline.predict_proba(input_df)[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]
        confidence   = pred_proba[pred_encoded] * 100

    info = CLASS_INFO.get(pred_label, {})
    color      = info.get("color", "yellow")
    bar_color  = info.get("bar_color", "#999")
    label_text = info.get("label", pred_label)
    desc_text  = info.get("desc", "")

    # Rekomendasi dihasilkan secara dinamis berdasarkan logika faktor risiko input
    reks = generate_recommendations(
        pred_label, favc, fcvc, caec, family, faf, ch2o,
        calc, smoke, tue, mtrans, scc, ncp
    )

    col_eval, col_result = st.columns([1, 1], gap="large")

    # ── Kolom kiri: Laporan Evaluasi Kesehatan ──
    with col_eval:
        st.markdown("""
        <div class="section-card">
          <p class="section-title">📊 Laporan Evaluasi Kesehatan</p>
          <p class="section-subtitle">Ringkasan kondisi fisik dan indikator gaya hidup</p>
        """, unsafe_allow_html=True)

        # BMI
        bmi = weight / (height ** 2)
        bmi_cat, bmi_col, bmi_bg = bmi_category(bmi)

        st.markdown(f"""
        <div class="bmi-wrap">
          <div class="bmi-row">
            <div>
              <div style="font-size:0.75rem;color:#999;text-transform:uppercase;
                          letter-spacing:0.07em;font-weight:600;margin-bottom:4px;">
                Indeks Massa Tubuh (BMI)
              </div>
              <div class="bmi-num">{bmi:.1f} <span style="font-size:1rem;color:#aaa;">kg/m²</span></div>
            </div>
            <div class="bmi-tag" style="background:{bmi_bg};color:{bmi_col};">
              {bmi_cat}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Lifestyle score
        ls, ls_max = lifestyle_score(favc, fcvc, caec, family, faf, ch2o, calc)
        ls_pct = int(ls / ls_max * 100)
        ls_color = "#43a047" if ls <= 3 else ("#ef6c00" if ls <= 6 else "#e53935")
        ls_label = "Baik" if ls <= 3 else ("Perlu Perhatian" if ls <= 6 else "Berisiko Tinggi")

        st.markdown(f"""
        <div class="bmi-wrap" style="margin-top:0;">
          <div class="bmi-row">
            <div>
              <div style="font-size:0.75rem;color:#999;text-transform:uppercase;
                          letter-spacing:0.07em;font-weight:600;margin-bottom:4px;">
                Skor Risiko Gaya Hidup
              </div>
              <div class="bmi-num" style="color:{ls_color};">{ls}<span style="font-size:1rem;color:#aaa;">/{ls_max}</span></div>
            </div>
            <div class="bmi-tag" style="background:{ls_color}22;color:{ls_color};">
              {ls_label}
            </div>
          </div>
          <div style="margin-top:12px;">
            <div class="confidence-bar-outer">
              <div class="confidence-bar-inner"
                   style="width:{ls_pct}%;background:{ls_color};"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Distribusi Probabilitas Semua Kelas (dipindahkan ke kolom kiri) ──
        st.markdown('<div class="divider-label"><span>Distribusi Probabilitas Semua Kelas</span></div>',
                    unsafe_allow_html=True)

        proba_df = pd.DataFrame({
            "Kategori": le.classes_,
            "Probabilitas (%)": (pred_proba * 100).round(2)
        }).sort_values("Probabilitas (%)", ascending=False)

        for _, row in proba_df.iterrows():
            pct = row["Probabilitas (%)"]
            is_pred = row["Kategori"] == pred_label
            bar_w = pct
            bc = bar_color if is_pred else "#ddd"
            tc = "#1a1a2e" if is_pred else "#999"
            fw = "700" if is_pred else "400"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.78rem;margin-bottom:3px;">
                <span style="color:{tc};font-weight:{fw};">{row['Kategori'].replace('_',' ')}</span>
                <span style="color:{tc};font-weight:{fw};">{pct:.1f}%</span>
              </div>
              <div class="confidence-bar-outer">
                <div class="confidence-bar-inner"
                     style="width:{bar_w:.1f}%;background:{bc};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Ringkasan input
        st.markdown('<div class="divider-label"><span>Ringkasan Input</span></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
              <div class="m-icon">🧑</div>
              <div class="m-val">{age} thn</div>
              <div class="m-lbl">Usia</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
              <div class="m-icon">⚖️</div>
              <div class="m-val">{weight:.1f} kg</div>
              <div class="m-lbl">Berat Badan</div>
            </div>
            """, unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
              <div class="m-icon">📏</div>
              <div class="m-val">{height:.2f} m</div>
              <div class="m-lbl">Tinggi Badan</div>
            </div>
            """, unsafe_allow_html=True)
        with col_d:
            st.markdown(f"""
            <div class="metric-card">
              <div class="m-icon">🏃</div>
              <div class="m-val">{faf:.1f}x</div>
              <div class="m-lbl">Aktivitas/Minggu</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close section-card

    # ── Kolom kanan: Hasil Prediksi ──
    with col_result:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-banner {color}">
          <div class="rb-label">Hasil Prediksi Model XGBoost</div>
          <div class="rb-title">{label_text}</div>
          <div class="rb-desc">{desc_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Rekomendasi (dinamis sesuai logika faktor risiko)
        st.markdown('<div class="divider-label"><span>Kesimpulan Klinis & Rekomendasi</span></div>',
                    unsafe_allow_html=True)
        rek_html = ""
        for icon, head, body in reks:
            rek_html += f"""
            <div class="rek-item">
              <div class="rek-icon">{icon}</div>
              <div class="rek-text">
                <div class="rek-head">{head}</div>
                <div class="rek-body">{body}</div>
              </div>
            </div>"""
        st.markdown(rek_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card" style="text-align:center;padding:60px 20px;">
      <div style="font-size:3.5rem;margin-bottom:16px;">🩺</div>
      <div style="font-size:1rem;font-weight:500;color:#999;">
        Lengkapi data di atas, lalu klik<br>
        <strong style="color:#e53935;">Analisis Risiko Obesitas</strong>
        untuk melihat laporan evaluasi dan hasil prediksi.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Sistem ini hanya untuk keperluan penelitian dan <strong>bukan pengganti diagnosis medis</strong>.<br>
  <span style="color:#d0ccc4;">Dewi Liri Linora  · Teknik Informatika · UMRI · 2026</span>
</div>
""", unsafe_allow_html=True)
