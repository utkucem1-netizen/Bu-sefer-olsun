
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import re

try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

st.set_page_config(
    page_title="Research Statistics AI Platform v5 Educational",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.hero {
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    background: linear-gradient(135deg, #f7fbff 0%, #f4fff7 100%);
    border: 1px solid #e3e9f0;
    margin-bottom: 1rem;
}
.card {
    border: 1px solid #e6e6e6;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    background: #fbfbfb;
    margin: .6rem 0 1rem 0;
}
.good {border-left: 7px solid #2ca25f; background:#f0fff4; padding:.9rem; border-radius:12px; margin:.5rem 0;}
.warn {border-left: 7px solid #f0ad4e; background:#fff8eb; padding:.9rem; border-radius:12px; margin:.5rem 0;}
.bad {border-left: 7px solid #d9534f; background:#fff1f1; padding:.9rem; border-radius:12px; margin:.5rem 0;}
.blue {border-left: 7px solid #2b7de9; background:#eef5ff; padding:.9rem; border-radius:12px; margin:.5rem 0;}
.metricbox {background:#ffffff; border:1px solid #e5e5e5; border-radius:14px; padding:1rem;}
.small {font-size:0.92rem; color:#555;}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Knowledge base
# ------------------------

TEST_INFO = {
    "Independent t-test": ["İki bağımsız grubun ortalamasını karşılaştırır.", "Sürekli sonuç + iki bağımsız grup", "Cohen's d", "Boxplot", "Analyze → Compare Means → Independent-Samples T Test"],
    "Mann–Whitney U": ["İki bağımsız grubu nonparametrik karşılaştırır.", "Sürekli/ordinal sonuç + iki bağımsız grup + normal değil", "r / rank-biserial", "Boxplot", "Analyze → Nonparametric Tests → Legacy Dialogs → 2 Independent Samples"],
    "Paired t-test": ["Aynı bireyin iki ölçümünü karşılaştırır.", "İki bağımlı sürekli ölçüm", "Cohen's dz", "Paired line plot", "Analyze → Compare Means → Paired-Samples T Test"],
    "Wilcoxon Signed-Rank": ["Aynı bireyin iki ölçümünü nonparametrik karşılaştırır.", "İki bağımlı ölçüm + normal değil", "r", "Paired line plot", "Analyze → Nonparametric Tests → Legacy Dialogs → 2 Related Samples"],
    "One-Way ANOVA": ["3+ bağımsız grubun ortalamasını karşılaştırır.", "Sürekli sonuç + 3+ bağımsız grup", "Eta squared", "Boxplot", "Analyze → Compare Means → One-Way ANOVA"],
    "Kruskal–Wallis": ["3+ bağımsız grubu nonparametrik karşılaştırır.", "Sürekli/ordinal sonuç + 3+ bağımsız grup + normal değil", "Epsilon squared", "Boxplot", "Analyze → Nonparametric Tests → Legacy Dialogs → K Independent Samples"],
    "Chi-Square": ["İki kategorik değişken ilişkisini inceler.", "Kategorik + kategorik", "Cramer's V", "Stacked bar chart", "Analyze → Descriptive Statistics → Crosstabs"],
    "Fisher Exact": ["Küçük örneklemli kategorik ilişki testidir.", "Kategorik + düşük hücre sayısı", "Odds ratio / Phi", "Bar chart", "Analyze → Descriptive Statistics → Crosstabs → Exact"],
    "Pearson Correlation": ["İki sürekli değişkenin doğrusal ilişkisini inceler.", "İki sürekli + normal", "r", "Scatter plot", "Analyze → Correlate → Bivariate"],
    "Spearman Correlation": ["Ordinal veya normal olmayan sürekli değişkenlerde ilişkiyi inceler.", "Ordinal veya normal olmayan sürekli", "rho", "Scatter plot", "Analyze → Correlate → Bivariate"],
    "Linear Regression": ["Sürekli sonucu etkileyen faktörleri inceler.", "Sürekli bağımlı değişken", "R² / β", "Residual plot", "Analyze → Regression → Linear"],
    "Logistic Regression": ["Binary sonucu etkileyen faktörleri inceler.", "Var/yok, doğru/yanlış sonuç", "Odds ratio", "Predicted probability / ROC", "Analyze → Regression → Binary Logistic"],
    "ROC Analysis": ["Tanısal ayırt ediciliği değerlendirir.", "Sürekli skor + gerçek binary durum", "AUC", "ROC curve", "Analyze → Classify → ROC Curve"],
    "ICC": ["Sürekli ölçüm güvenilirliğini ölçer.", "Tekrarlı sürekli ölçüm", "ICC", "Scatter/Bland–Altman", "Analyze → Scale → Reliability Analysis"],
    "Cohen's Kappa": ["Kategorik değerlendirme uyumunu ölçer.", "Tekrarlı kategorik sınıflama", "Kappa", "Agreement table", "Analyze → Descriptive Statistics → Crosstabs"],
    "Cronbach Alpha": ["Anket iç tutarlılığını ölçer.", "Çoklu Likert madde", "Alpha", "Item-total table", "Analyze → Scale → Reliability Analysis"],
    "EFA": ["Anket alt boyutlarını keşfeder.", "Çoklu Likert madde", "Factor loadings", "Scree plot", "Analyze → Dimension Reduction → Factor"]
}

DENTAL_PLANS = {
    "NPC-IOF-FPM morfometrik çalışma": {
        "question": "NPC çap/morfoloji grupları ile IOF/FPM ölçümleri arasında fark/ilişki var mı?",
        "variables": "Bağımlı: IOF/FPM mm ölçümleri. Bağımsız: NPC çap grubu veya morfoloji.",
        "analysis": "Ölçümler için ANOVA/Kruskal; morfolojiler için Chi-square/Fisher; reliability için ICC/Kappa.",
        "tables": "Descriptive table, group comparison table, post-hoc table, reliability table."
    },
    "Pediatrik CBCT endikasyon çalışması": {
        "question": "CBCT endikasyonları yaş/cinsiyet ve merkezlere göre değişiyor mu?",
        "variables": "Yaş sürekli; cinsiyet/endikasyon/merkez kategorik.",
        "analysis": "Cinsiyet-endikasyon: Chi-square. Yaş-endikasyon: ANOVA/Kruskal. Merkez farkı: Chi-square.",
        "tables": "Endikasyon frekans tablosu, yaş dağılımı, cinsiyet ilişkisi."
    },
    "Incidental findings prevalans çalışması": {
        "question": "İnsidental bulguların prevalansı ve demografik değişkenlerle ilişkisi nedir?",
        "variables": "Bulgu var/yok, bulgu tipi, yaş, cinsiyet.",
        "analysis": "Prevalans %, Chi-square/Fisher, gerekirse logistic regression.",
        "tables": "Prevalence table, demographic comparison, OR table."
    },
    "LLM prompt sensitivity çalışması": {
        "question": "Model doğruluğu prompt tipine/model tipine göre değişiyor mu?",
        "variables": "Doğru/yanlış binary; model ve prompt kategorik.",
        "analysis": "Chi-square, McNemar/Cochran Q tasarıma göre, logistic regression, accuracy metrics.",
        "tables": "Accuracy table, prompt comparison, model comparison, OR table."
    },
    "Tanısal doğruluk çalışması": {
        "question": "Bir ölçüm veya model pozitif/negatif olguları ayırt edebiliyor mu?",
        "variables": "Gerçek durum binary, test skoru sürekli.",
        "analysis": "ROC/AUC, cutoff, sensitivity, specificity, PPV, NPV.",
        "tables": "ROC table, confusion matrix, cutoff table."
    },
    "Anket/ölçek geliştirme çalışması": {
        "question": "Ölçek güvenilir ve alt boyut yapısı uygun mu?",
        "variables": "Likert maddeleri.",
        "analysis": "Cronbach alpha, EFA, KMO/Bartlett, item-total correlation.",
        "tables": "Reliability table, factor loading table, descriptive item table."
    }
}

# ------------------------
# Helper functions
# ------------------------

def infer_type(s):
    s2 = s.dropna()
    if len(s2) == 0:
        return "Boş"
    if pd.api.types.is_numeric_dtype(s2):
        unique = s2.nunique()
        vals = set(pd.to_numeric(s2, errors="coerce").dropna().unique())
        if unique <= 2:
            return "Binary sayısal"
        if unique <= 7 and vals.issubset(set(range(0, 8))):
            return "Ordinal/Likert sayısal"
        return "Sürekli/ölçümsel"
    unique = s2.astype(str).nunique()
    if unique == 2:
        return "Binary kategorik"
    if unique <= 15:
        return "Kategorik/ordinal"
    return "Metin/yüksek kategorili"

def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_cols(df):
    return [c for c in df.columns if c not in numeric_cols(df)]

def shapiro_text(s):
    if not SCIPY_OK:
        return None, "SciPy kurulu değil."
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 3:
        return None, "En az 3 sayısal değer gerekir."
    sample = vals.sample(5000, random_state=42) if len(vals) > 5000 else vals
    stat, p = stats.shapiro(sample)
    txt = f"Shapiro-Wilk p={p:.4f}. "
    txt += "Normal dağılım varsayımı kabul edilebilir." if p > 0.05 else "Normal dağılım varsayımı zayıf olabilir."
    return p, txt

def interpret_p(p):
    if p < 0.001:
        return "p<0.001: güçlü istatistiksel anlamlılık."
    if p < 0.05:
        return "p<0.05: istatistiksel olarak anlamlı."
    return "p≥0.05: istatistiksel olarak anlamlı değil."

def cohen_d(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan

def eta_squared(groups):
    vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand = np.nanmean(vals)
    ssb = sum(len(g) * (np.nanmean(g) - grand)**2 for g in groups)
    sst = np.nansum((vals - grand)**2)
    return ssb / sst if sst != 0 else np.nan

def cramers_v(table):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = n * min(table.shape[0]-1, table.shape[1]-1)
    return np.sqrt(chi2/denom) if denom > 0 else np.nan

def make_boxplot(df, group_col, value_col):
    if group_col == value_col:
        raise ValueError("Grup değişkeni ile sayısal değişken aynı olamaz. Grup için NPC_Group/Gender/Center gibi kategorik bir değişken; sayısal değişken için Age/NPC_Diameter_mm/IOF ölçümü gibi bir ölçüm seç.")
    temp = df[[group_col, value_col]].copy()
    temp.columns = ["__group__", "__value__"]
    temp["__value__"] = pd.to_numeric(temp["__value__"], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        raise ValueError("Seçilen değişkenlerde grafik/analiz için yeterli veri yok.")
    if temp["__group__"].nunique() < 2:
        raise ValueError("Boxplot için grup değişkeninde en az iki farklı grup olmalı. Patient_ID gibi benzersiz kimlik sütunları grup değişkeni olarak uygun değildir.")
    fig, ax = plt.subplots()
    temp.boxplot(column="__value__", by="__group__", ax=ax)
    ax.set_title(f"{value_col} by {group_col}")
    fig.suptitle("")
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    return fig

def make_scatter(df, x, y):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    return fig

def make_bar(df, col):
    fig, ax = plt.subplots()
    df[col].value_counts(dropna=False).plot(kind="bar", ax=ax)
    ax.set_title(f"{col} dağılımı")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekans")
    return fig

def make_hist(df, col):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(f"{col} histogram")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekans")
    return fig

def make_bland_altman(df, a, b):
    temp = df[[a,b]].dropna().astype(float)
    avg = temp.mean(axis=1)
    diff = temp[a] - temp[b]
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    fig, ax = plt.subplots()
    ax.scatter(avg, diff)
    ax.axhline(mean_diff, linestyle="--")
    ax.axhline(mean_diff + 1.96*sd_diff, linestyle="--")
    ax.axhline(mean_diff - 1.96*sd_diff, linestyle="--")
    ax.set_xlabel("Ortalama ölçüm")
    ax.set_ylabel("Fark")
    ax.set_title("Bland–Altman Plot")
    return fig, mean_diff, sd_diff

def fig_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    return buf.getvalue()

def download_fig(fig, filename):
    st.download_button("Grafiği indir (.png)", fig_bytes(fig), filename, "image/png")

def markdown_report(title, sections):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = f"# {title}\n\nGenerated: {now}\n\n"
    for h, b in sections:
        out += f"## {h}\n\n{b}\n\n"
    out += "---\nGenerated by Research Statistics AI Platform v5 Educational.\n"
    return out

def spss_syntax(test, var1="", var2="", group=""):
    if test == "Independent t-test":
        return f"T-TEST GROUPS={group}(1 2)\n/VARIABLES={var1}\n/CRITERIA=CI(.95)."
    if test == "One-Way ANOVA":
        return f"ONEWAY {var1} BY {group}\n/STATISTICS DESCRIPTIVES HOMOGENEITY\n/POSTHOC=TUKEY ALPHA(.05)."
    if test == "Chi-Square":
        return f"CROSSTABS\n/TABLES={var1} BY {var2}\n/STATISTICS=CHISQ PHI\n/CELLS=COUNT ROW COLUMN."
    if test == "Correlation":
        return f"CORRELATIONS\n/VARIABLES={var1} {var2}\n/PRINT=TWOTAIL NOSIG."
    if test == "Linear Regression":
        return f"REGRESSION\n/DEPENDENT {var1}\n/METHOD=ENTER {var2}."
    return "SPSS syntax şablonu bu test için henüz tanımlı değil."

def kappa_manual(a, b):
    temp = pd.DataFrame({"a": a, "b": b}).dropna().astype(str)
    cats = sorted(set(temp["a"]) | set(temp["b"]))
    tab = pd.crosstab(temp["a"], temp["b"]).reindex(index=cats, columns=cats, fill_value=0)
    n = tab.to_numpy().sum()
    if n == 0:
        return np.nan, tab
    po = np.trace(tab.to_numpy()) / n
    pe = (tab.sum(axis=1).to_numpy() * tab.sum(axis=0).to_numpy()).sum() / (n*n)
    k = (po - pe) / (1 - pe) if (1-pe)!=0 else np.nan
    return k, tab

def icc_simple(df, a, b):
    temp = df[[a,b]].dropna().astype(float)
    data = temp.to_numpy()
    n, k = data.shape
    mean_target = data.mean(axis=1)
    grand = data.mean()
    ss_between = k * ((mean_target - grand)**2).sum()
    ss_within = ((data - mean_target[:,None])**2).sum()
    ms_between = ss_between / (n-1)
    ms_within = ss_within / (n*(k-1))
    return (ms_between - ms_within) / (ms_between + (k-1)*ms_within)

def run_group_test(df, group_col, value_col, method="Otomatik"):
    if group_col == value_col:
        return None, "Grup değişkeni ile sonuç değişkeni aynı olamaz. Grup için NPC_Group/Gender/Center; sonuç için Age/NPC_Diameter_mm/IOF gibi sayısal ölçüm seç."
    temp = df[[group_col, value_col]].dropna()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna()
    groups_dict = {str(k): v[value_col].values for k, v in temp.groupby(group_col)}
    groups = list(groups_dict.values())
    names = list(groups_dict.keys())
    k = len(groups)
    if k < 2:
        return None, "En az iki grup gerekli."
    normal_flags = []
    if SCIPY_OK:
        for arr in groups:
            if len(arr) >= 3:
                _, pn = stats.shapiro(arr[:5000] if len(arr) > 5000 else arr)
                normal_flags.append(pn > 0.05)
    normal_auto = all(normal_flags) if normal_flags else False
    chosen = "Parametrik" if (method == "Otomatik" and normal_auto) else ("Nonparametrik" if method == "Otomatik" else method)
    res = {"temp": temp, "groups": groups, "names": names, "chosen": chosen, "k": k}
    if k == 2:
        if chosen == "Parametrik":
            stat, p = stats.ttest_ind(groups[0], groups[1], nan_policy="omit")
            res.update({"test":"Independent t-test", "stat_label":"t", "stat":stat, "p":p, "effect_label":"Cohen's d", "effect":cohen_d(groups[0], groups[1])})
        else:
            stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            res.update({"test":"Mann–Whitney U", "stat_label":"U", "stat":stat, "p":p, "effect_label":"Effect", "effect":np.nan})
    else:
        if chosen == "Parametrik":
            stat, p = stats.f_oneway(*groups)
            res.update({"test":"One-Way ANOVA", "stat_label":"F", "stat":stat, "p":p, "effect_label":"Eta squared", "effect":eta_squared(groups)})
            if STATSMODELS_OK and p < 0.05:
                try:
                    res["posthoc"] = str(pairwise_tukeyhsd(temp[value_col], temp[group_col], alpha=0.05))
                except Exception as e:
                    res["posthoc"] = str(e)
        else:
            stat, p = stats.kruskal(*groups)
            res.update({"test":"Kruskal–Wallis", "stat_label":"H", "stat":stat, "p":p, "effect_label":"Effect", "effect":np.nan})
    return res, None

def result_sentence_group(res, group_col, value_col):
    sig = "istatistiksel olarak anlamlı fark saptandı" if res["p"] < 0.05 else "istatistiksel olarak anlamlı fark saptanmadı"
    eff = "" if pd.isna(res.get("effect", np.nan)) else f", {res['effect_label']}={res['effect']:.3f}"
    return f"{res['test']} sonucunda {group_col} grupları arasında {value_col} açısından {sig} ({res['stat_label']}={res['stat']:.3f}, p={res['p']:.4f}{eff})."

# ------------------------
# Sidebar
# ------------------------

with st.sidebar:
    st.title("📊 Stats AI v5")
    st.write("Excel → analiz → tablo → grafik → sonuç → rapor")
    st.divider()
    st.markdown("### Kullanım sırası")
    st.write("1. Veri yükle")
    st.write("2. Veri kalitesini kontrol et")
    st.write("3. Analiz motorunu çalıştır")
    st.write("4. Tablo/grafik üret")
    st.write("5. Raporu indir")

# ------------------------
# Header
# ------------------------

st.title("📊 Research Statistics AI Platform v5 Educational")
st.caption("İstatistik bilmeyen kullanıcı için öğretici, analiz yapan ve rapor üreten akademik araştırma platformu.")

st.info("v7 Auto Manuscript: v6’daki tüm modüller korunmuştur. Boxplot/Group comparison hatası düzeltilmiştir. Grup değişkeni olarak Patient_ID yerine NPC_Group, Gender, Center gibi kategorik değişkenler seçilmelidir.")

st.markdown("""
<div class="hero">
<b>Yeni v5 Educational:</b> v4'teki analiz gücü korundu; üzerine kullanıcıyı adım adım öğreten açıklama kutuları, örnek araştırma senaryoları, mini dersler, test mantığı, yanlış seçim uyarıları ve quiz tabanlı öğrenme bölümleri eklendi.
</div>
""", unsafe_allow_html=True)

education_mode = st.sidebar.toggle("🎓 Eğitici açıklamaları göster", value=True)
beginner_mode = st.sidebar.toggle("🧒 Sıfırdan anlat modu", value=True)

if education_mode:
    st.markdown("""
<div class="blue">
<b>Bu site nasıl kullanılmalı?</b><br>
1) Önce verini yükle. 2) Veri kalitesi sekmesinde değişkenlerini tanı. 3) Test seçiciyle mantığı öğren. 
4) Analiz motorunda testi çalıştır. 5) Tablo/grafik üret. 6) Sonuç cümlesini ve raporu al.
</div>
""", unsafe_allow_html=True)


tabs = st.tabs([
    "🏠 Ana Sayfa",
    "📁 Veri Yükle",
    "🧹 Veri Kalitesi",
    "🧠 Test Seçici",
    "⚙️ Analiz Motoru",
    "📊 Tablo Üretici",
    "📈 Grafik Stüdyosu",
    "🎯 ROC & Tanısal",
    "🔁 Reliability",
    "🤖 ML / LLM Metrics",
    "📏 Power",
    "🧾 Reviewer",
    "🦷 Dental/CBCT",
    "🧪 Research Design",
    "💻 SPSS Syntax",
    "📥 Rapor",
    "📚 Öğren",
    "🧩 Örnek Senaryolar",
    "❓ Quiz",
    "🧭 Karar Ağacı"
])

# ------------------------
# Tab 0
# ------------------------
with tabs[0]:
    st.header("🏠 Platform Özeti")
    st.table([
        ["Modül", "Ne yapar?"],
        ["Veri yükleme", "Excel/CSV dosyasını tamamen okur ve değişken tiplerini tahmin eder."],
        ["Veri kalitesi", "Eksik veri, outlier, normallik, duplicate kontrolü yapar."],
        ["Test seçici", "Başlangıç seviyesine uygun test önerir."],
        ["Analiz motoru", "Temel testleri çalıştırır ve sonuç cümlesi üretir."],
        ["Tablo üretici", "Descriptive, group comparison, correlation table üretir."],
        ["Grafik stüdyosu", "Boxplot, histogram, scatter, bar, Bland–Altman üretir."],
        ["ROC", "AUC, cutoff, sensitivity, specificity hesaplar."],
        ["Reliability", "Kappa, ICC, Cronbach alpha modülleri."],
        ["ML/LLM", "Accuracy, precision, recall, F1, confusion matrix."],
        ["Reviewer", "Makale istatistik bölümünü eksikler açısından kontrol eder."],
        ["Research Design", "Araştırma sorusundan analiz planı çıkarır."],
    ])
    st.info("Bu platform eğitim ve araştırma planlama amaçlıdır. Final analizlerde çalışma tasarımı ve istatistik uzmanı görüşü önemlidir.")

# ------------------------
# Tab 1 Data Upload
# ------------------------
with tabs[1]:
    st.header("📁 Veri Yükle")
    uploaded = st.file_uploader("CSV veya Excel dosyası yükle", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state["df"] = df
            st.success(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun.")
            show = st.radio("Görüntüleme", ["İlk 30 satır", "İlk 100 satır", "Tüm veri"], horizontal=True)
            if show == "İlk 30 satır":
                st.dataframe(df.head(30), use_container_width=True)
            elif show == "İlk 100 satır":
                st.dataframe(df.head(100), use_container_width=True)
            else:
                st.warning("Büyük dosyalarda tüm veriyi göstermek yavaş olabilir.")
                st.dataframe(df, use_container_width=True)
            summary = pd.DataFrame([[c, str(df[c].dtype), int(df[c].isna().sum()), int(df[c].nunique(dropna=True)), infer_type(df[c])] for c in df.columns],
                                   columns=["Değişken", "Ham tip", "Eksik", "Benzersiz", "Muhtemel veri tipi"])
            st.subheader("Otomatik veri tipi tahmini")
            st.dataframe(summary, use_container_width=True)
            st.download_button("Değişken özetini indir", summary.to_csv(index=False).encode("utf-8"), "variable_summary.csv", "text/csv")
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
    else:
        st.info("Başlamak için Excel veya CSV yükle.")

# ------------------------
# Tab 2 Data quality
# ------------------------
with tabs[2]:
    st.header("🧹 Veri Kalitesi")
    if education_mode:
        st.markdown("""
<div class="blue">
<b>Neden önemli?</b> Eksik veri, uç değer ve normal dağılım kontrolü yanlış test seçimini önler. 
Örneğin sürekli bir değişken normal dağılmıyorsa t-test/ANOVA yerine nonparametrik test gerekebilir.
</div>
""", unsafe_allow_html=True)
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    else:
        st.metric("Satır", df.shape[0])
        st.metric("Sütun", df.shape[1])
        dup = df.duplicated().sum()
        st.metric("Duplicate satır", int(dup))
        missing = df.isna().sum().sort_values(ascending=False)
        st.subheader("Eksik veri")
        miss_df = pd.DataFrame({"Eksik sayı": missing, "Eksik %": (missing/len(df)*100).round(2)})
        st.dataframe(miss_df, use_container_width=True)
        st.download_button("Eksik veri tablosunu indir", miss_df.to_csv().encode("utf-8"), "missing_data.csv", "text/csv")

        nums = numeric_cols(df)
        if nums:
            st.subheader("Outlier kontrolü")
            out_rows = []
            for c in nums:
                x = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(x) > 0:
                    q1, q3 = x.quantile(.25), x.quantile(.75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
                    n_out = ((x < low) | (x > high)).sum()
                    out_rows.append([c, len(x), float(low), float(high), int(n_out), round(n_out/len(x)*100,2)])
            out_df = pd.DataFrame(out_rows, columns=["Değişken", "n", "Alt sınır", "Üst sınır", "Outlier sayı", "Outlier %"])
            st.dataframe(out_df, use_container_width=True)

            st.subheader("Normallik paneli")
            norm_rows = []
            for c in nums:
                p, txt = shapiro_text(df[c])
                norm_rows.append([c, txt])
            st.dataframe(pd.DataFrame(norm_rows, columns=["Değişken", "Yorum"]), use_container_width=True)

# ------------------------
# Tab 3 Test selector
# ------------------------
with tabs[3]:
    st.header("🧠 Akıllı Test Seçici")
    c1,c2,c3 = st.columns(3)
    with c1:
        purpose = st.selectbox("Amacın?", ["Grupları karşılaştırmak", "İlişki incelemek", "Etki/tahmin modeli", "Tanısal performans", "Güvenilirlik", "Anket/ölçek"])
        dtype = st.selectbox("Sonuç değişken tipi?", ["Sürekli", "Kategorik", "Ordinal", "Binary", "Emin değilim"])
    with c2:
        groupn = st.selectbox("Grup/ölçüm sayısı?", ["2", "3+", "Grup yok"])
        dep = st.selectbox("Gruplar?", ["Bağımsız", "Bağımlı/tekrarlı", "Uygun değil"])
    with c3:
        norm = st.selectbox("Normal dağılım?", ["Normal", "Normal değil", "Bilmiyorum"])
        cells = st.selectbox("Kategorik hücre sayısı?", ["Yeterli", "Düşük", "Bilmiyorum"])

    if st.button("Test öner"):
        recs = []
        if purpose == "Grupları karşılaştırmak":
            if dtype in ["Sürekli", "Ordinal", "Emin değilim"]:
                if groupn == "2" and dep == "Bağımsız":
                    recs = ["Independent t-test"] if norm == "Normal" else ["Mann–Whitney U"]
                elif groupn == "2" and dep == "Bağımlı/tekrarlı":
                    recs = ["Paired t-test"] if norm == "Normal" else ["Wilcoxon Signed-Rank"]
                elif groupn == "3+" and dep == "Bağımsız":
                    recs = ["One-Way ANOVA"] if norm == "Normal" else ["Kruskal–Wallis"]
            else:
                recs = ["Chi-Square"] if cells == "Yeterli" else ["Fisher Exact"]
        elif purpose == "İlişki incelemek":
            recs = ["Pearson Correlation"] if dtype == "Sürekli" and norm == "Normal" else ["Spearman Correlation" if dtype in ["Sürekli","Ordinal"] else "Chi-Square"]
        elif purpose == "Etki/tahmin modeli":
            recs = ["Linear Regression"] if dtype == "Sürekli" else ["Logistic Regression"]
        elif purpose == "Tanısal performans":
            recs = ["ROC Analysis"]
        elif purpose == "Güvenilirlik":
            recs = ["ICC"] if dtype == "Sürekli" else ["Cohen's Kappa"]
        else:
            recs = ["Cronbach Alpha", "EFA"]

        for r in recs:
            info = TEST_INFO.get(r, ["","","","",""])
            with st.container(border=True):
                st.subheader(f"✅ {r}")
                st.write(f"**Ne işe yarar?** {info[0]}")
                st.write(f"**Veri yapısı:** {info[1]}")
                st.write(f"**Effect size:** {info[2]}")
                st.write(f"**Grafik:** {info[3]}")
                st.code(info[4], language="text")

# ------------------------
# Tab 4 Analysis Engine
# ------------------------
with tabs[4]:
    st.header("⚙️ Analiz Motoru")
    if education_mode:
        st.markdown("""
<div class="blue">
<b>Nasıl düşünmeliyim?</b> Analiz seçerken önce sonuç değişkenini seç: ölçüm mü, kategori mi? 
Sonra grup değişkenini seç: iki grup mu, üçten fazla mı? Uygulama sonucu verir ama yorumda effect size ve grafik de önemlidir.
</div>
""", unsafe_allow_html=True)
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    else:
        nums = numeric_cols(df)
        all_cols = df.columns.tolist()
        analysis = st.selectbox("Analiz", ["Grup karşılaştırması", "Korelasyon", "Kategorik ilişki", "Basit lineer regresyon", "Çoklu lineer regresyon", "Lojistik regresyon"])
        if analysis == "Grup karşılaştırması":
            g = st.selectbox("Grup değişkeni", all_cols, key="analysis_group")
            v = st.selectbox("Sürekli sonuç", nums, key="analysis_value")
            method = st.selectbox("Yöntem", ["Otomatik", "Parametrik", "Nonparametrik"])
            if st.button("Çalıştır"):
                res, err = run_group_test(df, g, v, method)
                if err: st.error(err)
                else:
                    st.success(f"{res['test']}: {res['stat_label']}={res['stat']:.4f}, p={res['p']:.4f}")
                    st.write(interpret_p(res["p"]))
                    if not pd.isna(res.get("effect", np.nan)):
                        st.info(f"{res['effect_label']}={res['effect']:.4f}")
                    sent = result_sentence_group(res, g, v)
                    st.code(sent, language="text")
                    if "posthoc" in res:
                        st.text("Post-hoc")
                        st.text(res["posthoc"])
                    fig = make_boxplot(res["temp"], g, v)
                    st.pyplot(fig); download_fig(fig, "group_comparison.png")

        elif analysis == "Korelasyon":
            x = st.selectbox("X", nums)
            y = st.selectbox("Y", nums, index=min(1,len(nums)-1))
            corr = st.selectbox("Tip", ["Pearson", "Spearman"])
            if st.button("Korelasyon çalıştır"):
                temp = df[[x,y]].dropna()
                if corr == "Pearson":
                    r,p = stats.pearsonr(temp[x], temp[y])
                else:
                    r,p = stats.spearmanr(temp[x], temp[y])
                st.success(f"{corr}: r/rho={r:.4f}, p={p:.4f}")
                st.write(interpret_p(p))
                st.code(f"{corr} korelasyon analizinde {x} ile {y} arasında {'anlamlı' if p<0.05 else 'anlamlı olmayan'} ilişki saptandı (r/rho={r:.3f}, p={p:.4f}).")
                fig = make_scatter(temp,x,y); st.pyplot(fig); download_fig(fig,"correlation.png")

        elif analysis == "Kategorik ilişki":
            c1 = st.selectbox("1. kategorik", all_cols)
            c2 = st.selectbox("2. kategorik", all_cols, index=min(1,len(all_cols)-1))
            if st.button("Chi-square çalıştır"):
                table = pd.crosstab(df[c1], df[c2])
                st.dataframe(table)
                chi2,p,dof,expected = stats.chi2_contingency(table)
                cv = cramers_v(table)
                st.success(f"χ²={chi2:.4f}, df={dof}, p={p:.4f}, Cramer's V={cv:.4f}")
                st.code(f"Ki-kare testi sonucunda {c1} ile {c2} arasında {'anlamlı ilişki saptandı' if p<0.05 else 'anlamlı ilişki saptanmadı'} (χ²={chi2:.3f}, df={dof}, p={p:.4f}, Cramer's V={cv:.3f}).")

        elif analysis == "Basit lineer regresyon":
            y = st.selectbox("Bağımlı değişken", nums)
            x = st.selectbox("Bağımsız değişken", nums, index=min(1,len(nums)-1))
            if st.button("Regresyon çalıştır"):
                temp = df[[x,y]].dropna()
                X = sm.add_constant(temp[x])
                model = sm.OLS(temp[y], X).fit()
                st.text(model.summary())
                st.code(f"Basit lineer regresyonda {x}, {y} değişkenini {'anlamlı şekilde yordadı' if model.pvalues[x]<0.05 else 'anlamlı şekilde yordamadı'} (β={model.params[x]:.3f}, p={model.pvalues[x]:.4f}, R²={model.rsquared:.3f}).")

        elif analysis == "Çoklu lineer regresyon":
            y = st.selectbox("Bağımlı değişken", nums)
            xs = st.multiselect("Bağımsız sayısal değişkenler", [c for c in nums if c != y])
            if st.button("Çoklu regresyon çalıştır") and xs:
                temp = df[[y]+xs].dropna()
                X = sm.add_constant(temp[xs])
                model = sm.OLS(temp[y], X).fit()
                st.text(model.summary())
                coef = pd.DataFrame({"Beta": model.params, "p": model.pvalues})
                st.dataframe(coef)

        else:
            if not SKLEARN_OK:
                st.error("scikit-learn kurulu değil.")
            else:
                y = st.selectbox("Binary bağımlı değişken", all_cols)
                xs = st.multiselect("Bağımsız sayısal değişkenler", nums)
                positive = st.text_input("Pozitif sınıf etiketi", "1")
                if st.button("Lojistik regresyon çalıştır") and xs:
                    temp = df[[y]+xs].dropna()
                    ybin = (temp[y].astype(str) == positive).astype(int)
                    if ybin.nunique() < 2:
                        st.error("Pozitif sınıf etiketi yanlış olabilir.")
                    else:
                        X = temp[xs]
                        model = LogisticRegression(max_iter=1000).fit(X, ybin)
                        coefs = pd.DataFrame({"Variable": xs, "Coefficient": model.coef_[0], "Odds ratio": np.exp(model.coef_[0])})
                        st.dataframe(coefs)
                        st.info(f"Model accuracy: {model.score(X,ybin):.3f}")

# ------------------------
# Table Builder
# ------------------------
with tabs[5]:
    st.header("📊 Tablo Üretici")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    else:
        nums = numeric_cols(df)
        all_cols = df.columns.tolist()
        ttype = st.selectbox("Tablo türü", ["Tanımlayıcı tablo", "Gruba göre tanımlayıcı tablo", "Frekans tablosu", "Korelasyon matrisi", "Eksik veri tablosu"])
        if ttype == "Tanımlayıcı tablo":
            sel = st.multiselect("Değişkenler", nums, default=nums[:5])
            if sel:
                tab = df[sel].describe().T
                st.dataframe(tab); st.download_button("İndir", tab.to_csv().encode("utf-8"), "descriptive.csv","text/csv")
        elif ttype == "Gruba göre tanımlayıcı tablo":
            group = st.selectbox("Grup", all_cols)
            sel = st.multiselect("Sayısal", nums, default=nums[:3])
            if sel:
                tab = df.groupby(group)[sel].agg(["count","mean","std","median","min","max"])
                st.dataframe(tab); st.download_button("İndir", tab.to_csv().encode("utf-8"), "group_descriptives.csv","text/csv")
        elif ttype == "Frekans tablosu":
            col = st.selectbox("Kategorik değişken", all_cols)
            tab = df[col].value_counts(dropna=False).reset_index()
            tab.columns = [col, "Frekans"]
            st.dataframe(tab); st.download_button("İndir", tab.to_csv(index=False).encode("utf-8"), "frequency.csv","text/csv")
        elif ttype == "Korelasyon matrisi":
            sel = st.multiselect("Sayısal değişkenler", nums, default=nums[:6])
            if sel:
                tab = df[sel].corr()
                st.dataframe(tab); st.download_button("İndir", tab.to_csv().encode("utf-8"), "correlation_matrix.csv","text/csv")
        else:
            tab = pd.DataFrame({"Eksik sayı": df.isna().sum(), "Eksik %": (df.isna().mean()*100).round(2)})
            st.dataframe(tab); st.download_button("İndir", tab.to_csv().encode("utf-8"), "missing.csv","text/csv")

# ------------------------
# Graphics
# ------------------------
with tabs[6]:
    st.header("📈 Grafik Stüdyosu")
    if education_mode:
        st.markdown("""
<div class="blue">
<b>Grafik neden gerekli?</b> p değeri tek başına verinin dağılımını göstermez. 
Boxplot grup farklarını, scatter ilişkiyi, histogram dağılımı, Bland–Altman iki ölçüm yönteminin uyumunu gösterir.
</div>
""", unsafe_allow_html=True)
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    else:
        nums = numeric_cols(df)
        all_cols = df.columns.tolist()
        gtype = st.selectbox("Grafik", ["Boxplot", "Bar chart", "Scatter plot", "Histogram", "Bland–Altman"])
        if gtype == "Boxplot":
            g=st.selectbox("Grup", all_cols, key="graph_group"); v=st.selectbox("Sayısal", nums, key="graph_numeric")
            try:
                fig=make_boxplot(df[[g,v]].dropna(),g,v)
            except Exception as e:
                st.error(str(e))
                fig=None
        elif gtype == "Bar chart":
            c=st.selectbox("Kategorik", all_cols, key="graph_categorical")
            fig=make_bar(df,c)
        elif gtype == "Scatter plot":
            x=st.selectbox("X", nums, key="graph_x"); y=st.selectbox("Y", nums,index=min(1,len(nums)-1), key="graph_y")
            fig=make_scatter(df[[x,y]].dropna(),x,y)
        elif gtype == "Histogram":
            c=st.selectbox("Sayısal", nums, key="hist_numeric")
            fig=make_hist(df,c)
        else:
            a=st.selectbox("1. ölçüm", nums, key="ba1"); b=st.selectbox("2. ölçüm", nums,index=min(1,len(nums)-1), key="ba2")
            fig, md, sd = make_bland_altman(df,a,b)
            st.info(f"Mean difference={md:.3f}, SD difference={sd:.3f}")
        if fig is not None:
            st.pyplot(fig); download_fig(fig, "figure.png")

# ------------------------
# ROC
# ------------------------
with tabs[7]:
    st.header("🎯 ROC & Tanısal Performans")
    if education_mode:
        st.markdown("""
<div class="blue">
<b>ROC mantığı:</b> Bir ölçümün hasta/sağlıklı veya pozitif/negatif ayrımı yapıp yapamadığını gösterir. 
AUC 0.5 ise rastgeleye yakın, 0.8 iyi, 0.9 çok iyi ayırt edicilik anlamına gelir.
</div>
""", unsafe_allow_html=True)
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    elif not SKLEARN_OK:
        st.error("scikit-learn kurulu değil.")
    else:
        nums = numeric_cols(df); all_cols = df.columns.tolist()
        truth = st.selectbox("Gerçek durum", all_cols)
        score = st.selectbox("Skor/ölçüm", nums)
        pos = st.text_input("Pozitif sınıf etiketi", "1")
        if st.button("ROC hesapla"):
            temp = df[[truth,score]].dropna()
            ytrue = (temp[truth].astype(str)==pos).astype(int)
            if ytrue.nunique()<2:
                st.error("Pozitif sınıf etiketi doğru değil.")
            else:
                fpr,tpr,thr=roc_curve(ytrue,temp[score])
                aucv=auc(fpr,tpr)
                youden=tpr-fpr; idx=int(np.argmax(youden))
                cutoff=thr[idx]; sens=tpr[idx]; spec=1-fpr[idx]
                st.success(f"AUC={aucv:.4f}; cutoff={cutoff:.4f}; sensitivity={sens:.4f}; specificity={spec:.4f}")
                fig,ax=plt.subplots()
                ax.plot(fpr,tpr,label=f"AUC={aucv:.3f}")
                ax.plot([0,1],[0,1],linestyle="--")
                ax.set_xlabel("1-Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(); ax.set_title("ROC Curve")
                st.pyplot(fig); download_fig(fig,"roc_curve.png")
                st.code(f"ROC analizinde AUC={aucv:.3f} bulundu. Youden indeksine göre optimum cutoff={cutoff:.3f}, sensitivite={sens:.3f}, spesifite={spec:.3f}.")

# ------------------------
# Reliability
# ------------------------
with tabs[8]:
    st.header("🔁 Reliability")
    if education_mode:
        st.markdown("""
<div class="blue">
<b>Reliability mantığı:</b> Aynı ölçümü tekrar yaptığında benzer sonuç alıyor musun? 
Sürekli ölçümde ICC, kategorik sınıflamada Kappa, anket maddelerinde Cronbach alpha kullanılır.
</div>
""", unsafe_allow_html=True)
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    else:
        mode=st.selectbox("Analiz", ["Cohen's Kappa", "ICC", "Cronbach Alpha"])
        all_cols=df.columns.tolist(); nums=numeric_cols(df)
        if mode=="Cohen's Kappa":
            a=st.selectbox("1. değerlendirme", all_cols); b=st.selectbox("2. değerlendirme", all_cols,index=min(1,len(all_cols)-1))
            if st.button("Kappa hesapla"):
                k,tab=kappa_manual(df[a],df[b])
                st.success(f"Kappa={k:.4f}"); st.dataframe(tab)
                st.code(f"Kategorik uyum Cohen kappa ile değerlendirildi ve κ={k:.3f} bulundu.")
        elif mode=="ICC":
            a=st.selectbox("1. ölçüm", nums, key="ba1"); b=st.selectbox("2. ölçüm", nums,index=min(1,len(nums)-1), key="ba2")
            if st.button("ICC hesapla"):
                icc=icc_simple(df,a,b)
                st.success(f"Basit ICC={icc:.4f}")
                fig=make_scatter(df[[a,b]].dropna(),a,b); st.pyplot(fig)
                st.code(f"Sürekli ölçümlerde güvenilirlik ICC ile değerlendirildi ve ICC={icc:.3f} bulundu.")
        else:
            items=st.multiselect("Likert maddeleri", nums, default=nums[:5])
            if st.button("Cronbach alpha hesapla") and len(items)>=2:
                data=df[items].dropna()
                k=len(items)
                item_vars=data.var(axis=0,ddof=1)
                total_var=data.sum(axis=1).var(ddof=1)
                alpha=(k/(k-1))*(1-item_vars.sum()/total_var) if total_var !=0 else np.nan
                st.success(f"Cronbach alpha={alpha:.4f}")
                st.code(f"Ölçeğin iç tutarlılığı Cronbach alfa ile değerlendirildi ve α={alpha:.3f} bulundu.")

# ------------------------
# ML / LLM Metrics
# ------------------------
with tabs[9]:
    st.header("🤖 ML / LLM Metrics")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Önce veri yükle.")
    elif not SKLEARN_OK:
        st.error("scikit-learn kurulu değil.")
    else:
        all_cols=df.columns.tolist()
        true_col=st.selectbox("Gerçek etiket", all_cols)
        pred_col=st.selectbox("Tahmin/model cevabı", all_cols,index=min(1,len(all_cols)-1))
        positive=st.text_input("Pozitif sınıf", "1", key="mlpos")
        if st.button("Metrikleri hesapla"):
            temp=df[[true_col,pred_col]].dropna().astype(str)
            yt=(temp[true_col]==positive).astype(int)
            yp=(temp[pred_col]==positive).astype(int)
            acc=accuracy_score(yt,yp); prec=precision_score(yt,yp,zero_division=0); rec=recall_score(yt,yp,zero_division=0); f1=f1_score(yt,yp,zero_division=0)
            st.success(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall/Sensitivity={rec:.3f}, F1={f1:.3f}")
            cm=confusion_matrix(yt,yp)
            st.dataframe(pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))
            st.code(f"Model performansı accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f} ve F1-score={f1:.3f} olarak hesaplandı.")

# ------------------------
# Power
# ------------------------
with tabs[10]:
    st.header("📏 Power / Sample Size")
    if not STATSMODELS_OK:
        st.error("statsmodels kurulu değil.")
    else:
        test=st.selectbox("Test", ["İki bağımsız grup t-test", "One-Way ANOVA"])
        alpha=st.number_input("Alpha",0.001,0.2,0.05,0.001)
        power=st.number_input("Power",0.5,0.99,0.80,0.01)
        effect=st.number_input("Effect size",0.01,2.0,0.5,0.01)
        groups=st.number_input("ANOVA grup sayısı",2,10,3,1)
        if st.button("Örneklem hesapla"):
            if test=="İki bağımsız grup t-test":
                n=TTestIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1)
                st.success(f"Grup başına n≈{np.ceil(n):.0f}, toplam n≈{np.ceil(n)*2:.0f}")
            else:
                n=FTestAnovaPower().solve_power(effect_size=effect, alpha=alpha, power=power, k_groups=int(groups))
                st.success(f"Toplam n≈{np.ceil(n):.0f}, grup başına n≈{np.ceil(n/groups):.0f}")

# ------------------------
# Reviewer
# ------------------------
with tabs[11]:
    st.header("🧾 Manuscript / Statistical Reviewer")
    text=st.text_area("Metod/sonuç bölümünü yapıştır", height=260)
    if st.button("Kontrol et"):
        t=text.lower()
        issues=[]
        if "p" not in t: issues.append(("p değeri görünmüyor","Sonuçlarda p değeri raporlanmalı."))
        if not any(x in t for x in ["cohen","eta","cramer","auc","or","odds","icc","kappa","effect"]): issues.append(("Effect size eksik olabilir","p değeri yanında effect size eklenmeli."))
        if "anova" in t and not any(x in t for x in ["tukey","games","post","bonferroni"]): issues.append(("ANOVA sonrası post-hoc eksik olabilir","Anlamlı 3+ grup analizinde post-hoc belirtilmeli."))
        if any(x in t for x in ["observer","gözlemci","intra","inter"]) and not any(x in t for x in ["icc","kappa"]): issues.append(("Reliability eksik olabilir","Gözlemci ölçümü varsa ICC/Kappa raporlanmalı."))
        if "chi" in t and "cramer" not in t: issues.append(("Chi-square effect size eksik","Cramer's V eklenebilir."))
        if "normal" not in t and any(x in t for x in ["anova","t-test","regression"]): issues.append(("Varsayım kontrolü belirtilmemiş olabilir","Normallik/varyans varsayımları raporlanabilir."))
        if not issues: st.success("Temel kontrolde belirgin eksik bulunmadı.")
        for title,desc in issues:
            st.markdown(f"<div class='warn'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)

# ------------------------
# Dental
# ------------------------
with tabs[12]:
    st.header("🦷 Dental / CBCT Özel Mod")
    plan=st.selectbox("Çalışma tipi", list(DENTAL_PLANS.keys()))
    p=DENTAL_PLANS[plan]
    st.markdown(f"""
<div class="blue">
<b>Araştırma sorusu:</b> {p['question']}<br>
<b>Değişkenler:</b> {p['variables']}<br>
<b>Analiz:</b> {p['analysis']}<br>
<b>Tablolar:</b> {p['tables']}
</div>
""", unsafe_allow_html=True)

# ------------------------
# Research Design
# ------------------------
with tabs[13]:
    st.header("🧪 Research Design Wizard")
    rq=st.text_area("Araştırma sorunu", placeholder="Örn: 4 NPC çap grubu arasında IOF ölçümleri farklı mı?")
    depvar=st.text_input("Bağımlı değişken", placeholder="Örn: IOF horizontal ölçümü")
    indep=st.text_input("Bağımsız değişken", placeholder="Örn: NPC çap grubu")
    design=st.selectbox("Çalışma tipi", ["Kesitsel", "Retrospektif", "Prospektif", "Anket", "Tanısal doğruluk", "Reliability", "LLM/AI evaluation"])
    if st.button("Analiz planı yaz"):
        plan_text=f"""
Araştırma sorusu: {rq}

Bu çalışmada bağımlı değişken {depvar}, bağımsız değişken ise {indep} olarak tanımlanabilir. Çalışma tasarımı {design} niteliğindedir. İlk aşamada veri tipleri tanımlanmalı, eksik veri ve uç değerler kontrol edilmelidir. Sürekli değişkenler için normal dağılım Shapiro-Wilk testi ve grafiksel yöntemlerle değerlendirilmelidir. Grup karşılaştırmaları için veri dağılımına ve grup sayısına göre parametrik veya nonparametrik testler seçilmelidir. Üç veya daha fazla grup karşılaştırmasında anlamlılık saptanırsa uygun post-hoc analiz uygulanmalı, sonuçlar p değeri yanında effect size ile raporlanmalıdır.
"""
        st.code(plan_text, language="text")

# ------------------------
# SPSS
# ------------------------
with tabs[14]:
    st.header("💻 SPSS Syntax Üretici")
    test=st.selectbox("Test", ["Independent t-test","One-Way ANOVA","Chi-Square","Correlation","Linear Regression"])
    var1=st.text_input("Değişken 1 / bağımlı değişken")
    var2=st.text_input("Değişken 2 / bağımsız değişken")
    group=st.text_input("Grup değişkeni")
    if st.button("Syntax üret"):
        st.code(spss_syntax(test,var1,var2,group), language="spss")

# ------------------------
# Report
# ------------------------
with tabs[15]:
    st.header("📥 Rapor Oluştur")
    title=st.text_input("Rapor başlığı", "Statistical Analysis Report")
    notes=st.text_area("Notlar / analiz planı / sonuçlar", height=220)
    if st.button("Markdown raporu oluştur"):
        df=st.session_state.get("df")
        sections=[]
        if df is not None:
            sections.append(("Dataset", f"Rows: {df.shape[0]}\n\nColumns: {df.shape[1]}"))
            if numeric_cols(df):
                sections.append(("Descriptives", df[numeric_cols(df)].describe().T.to_markdown()))
            miss=pd.DataFrame({"Missing": df.isna().sum(), "Missing %": (df.isna().mean()*100).round(2)})
            sections.append(("Missing Data", miss.to_markdown()))
        sections.append(("Notes", notes if notes else "No notes entered."))
        rep=markdown_report(title, sections)
        st.code(rep, language="markdown")
        st.download_button("Raporu indir (.md)", rep.encode("utf-8"), "statistics_report.md", "text/markdown")

# ------------------------
# Learning
# ------------------------
with tabs[16]:
    st.header("📚 Öğrenme Merkezi")
    topic=st.selectbox("Konu", ["Veri tipi", "Normal dağılım", "Parametrik/nonparametrik", "Post-hoc", "Effect size", "Regresyon", "ROC", "Reliability", "Anket"])
    lessons={
        "Veri tipi":"Yaş, mm ve skor sürekli; kadın/erkek ve var/yok kategorik; Likert ordinal; doğru/yanlış binary veridir.",
        "Normal dağılım":"Parametrik testlerden önce sürekli değişkenlerde normal dağılım değerlendirilir. Shapiro-Wilk tek başına değil histogram/Q-Q plot ile yorumlanmalıdır.",
        "Parametrik/nonparametrik":"Parametrik testler ortalama ve varsayımlara dayanır. Nonparametrik testler daha esnektir ve normal dağılım olmadığında tercih edilir.",
        "Post-hoc":"ANOVA veya Kruskal genel farkı gösterir; post-hoc hangi grupların farklı olduğunu gösterir.",
        "Effect size":"p değeri anlamlılığı, effect size etkinin büyüklüğünü gösterir. İkisi birlikte raporlanmalıdır.",
        "Regresyon":"Sürekli sonuç için lineer regresyon, binary sonuç için lojistik regresyon kullanılır.",
        "ROC":"ROC/AUC bir testin pozitif-negatif ayrım gücünü gösterir. AUC 0.8 iyi, 0.9 çok iyi kabul edilir.",
        "Reliability":"Kategorik uyum Kappa, sürekli ölçüm uyumu ICC ile değerlendirilir.",
        "Anket":"Likert maddelerinde iç tutarlılık için Cronbach alpha, yapı geçerliği için EFA/CFA kullanılır."
    }
    st.info(lessons[topic])
    q=st.radio("mm ölçümü hangi veri tipidir?", ["Kategorik", "Sürekli", "Binary"])
    if st.button("Cevabı kontrol et"):
        st.success("Doğru.") if q=="Sürekli" else st.error("Yanlış. mm ölçümü sürekli veridir.")


# ------------------------
# Educational Scenario Library
# ------------------------
with tabs[17]:
    st.header("🧩 Örnek Senaryolarla Öğren")
    st.write("Bu bölümde kendi verini örnek araştırma durumlarıyla eşleştirebilirsin.")

    scenarios = {
        "İki grup arasında ölçüm farkı": {
            "example": "Kadın ve erkek hastaların NPC çap ortalaması farklı mı?",
            "data": "Cinsiyet = kategorik iki grup; NPC çapı = sürekli ölçüm.",
            "logic": "İki bağımsız grup + sürekli sonuç değişkeni. Veri normal ise Independent t-test, normal değilse Mann–Whitney U.",
            "test": "Independent t-test / Mann–Whitney U",
            "mistake": "Cinsiyet kategorik olduğu için cinsiyete ANOVA uygulanmaz; ölçüm değişkeni NPC çapıdır."
        },
        "Üç veya daha fazla grup arasında ölçüm farkı": {
            "example": "Normal, genişlemiş, patoloji şüphesi ve kist gruplarında IOF ölçümü farklı mı?",
            "data": "NPC grubu = kategorik 4 grup; IOF ölçümü = sürekli.",
            "logic": "3+ bağımsız grup + sürekli sonuç. Normal ise ANOVA; normal değilse Kruskal–Wallis.",
            "test": "One-Way ANOVA / Kruskal–Wallis",
            "mistake": "ANOVA anlamlı çıkarsa hangi grupların farklı olduğunu görmek için post-hoc gerekir."
        },
        "İki kategorik değişken ilişkisi": {
            "example": "Cinsiyet ile CBCT endikasyonu arasında ilişki var mı?",
            "data": "Cinsiyet = kategorik; endikasyon = kategorik.",
            "logic": "İki kategorik değişken ilişkisi için Chi-square kullanılır. Hücre sayıları düşükse Fisher Exact düşünülür.",
            "test": "Chi-square / Fisher Exact",
            "mistake": "Bu durumda t-test veya ANOVA kullanılmaz çünkü sonuç değişkeni ölçüm değil kategoridir."
        },
        "İki ölçüm arasındaki ilişki": {
            "example": "NPC çapı ile IOF horizontal ölçümü ilişkili mi?",
            "data": "NPC çapı = sürekli; IOF ölçümü = sürekli.",
            "logic": "İki sürekli değişken ilişkisi. Normal/doğrusal ise Pearson; normal değilse Spearman.",
            "test": "Pearson / Spearman",
            "mistake": "Korelasyon neden-sonuç göstermez; sadece ilişki gösterir."
        },
        "Gözlemci güvenilirliği": {
            "example": "Aynı radyoloğun iki hafta arayla yaptığı ölçümler uyumlu mu?",
            "data": "mm ölçümü ise sürekli; şekil sınıflaması ise kategorik.",
            "logic": "Sürekli ölçüm için ICC, kategorik sınıflama için Cohen's Kappa.",
            "test": "ICC / Cohen's Kappa",
            "mistake": "Sürekli ölçümde Kappa değil ICC kullanılmalıdır."
        },
        "Tanısal doğruluk": {
            "example": "Bir ölçüm değeri patolojiyi ayırt edebiliyor mu?",
            "data": "Gerçek durum = pozitif/negatif; ölçüm = sürekli skor.",
            "logic": "ROC analizi AUC, cutoff, sensitivite ve spesifite verir.",
            "test": "ROC Analysis",
            "mistake": "Sadece p değeri tanısal performansı anlatmak için yeterli değildir."
        },
        "Anket/ölçek": {
            "example": "Uzmanlık tercihi anketi güvenilir mi ve alt boyutları var mı?",
            "data": "Birden fazla Likert maddesi.",
            "logic": "İç tutarlılık için Cronbach alpha; alt boyut keşfi için EFA.",
            "test": "Cronbach Alpha / EFA",
            "mistake": "Tek bir Likert maddesi için Cronbach alpha hesaplanmaz."
        }
    }

    choice = st.selectbox("Örnek senaryo seç", list(scenarios.keys()))
    s = scenarios[choice]
    st.markdown(f"""
<div class="card">
<h4>{choice}</h4>
<b>Örnek araştırma sorusu:</b> {s['example']}<br><br>
<b>Veri yapısı:</b> {s['data']}<br><br>
<b>Mantık:</b> {s['logic']}<br><br>
<b>Önerilen test:</b> {s['test']}<br><br>
<b>Sık hata:</b> {s['mistake']}
</div>
""", unsafe_allow_html=True)

    st.subheader("Bu örnek benim verime benziyor mu?")
    st.write("Kendi verini düşün: Sonuç değişkenin ölçüm mü, kategori mi? Kaç grup var? Aynı kişilerden tekrar ölçüm mü aldın?")
    st.info("Bu üç soruya cevap verdiğinde doğru teste çok yaklaşırsın.")

# ------------------------
# Quiz
# ------------------------
with tabs[18]:
    st.header("❓ Mini Quiz ile Öğren")
    st.write("Bu bölüm kullanıcıya test seçme mantığını öğretir.")

    questions = [
        {
            "q": "Kadın ve erkek hastaların yaş ortalamalarını karşılaştırmak istiyorsun. Yaş normal dağılıyor. Hangi test?",
            "opts": ["Chi-square", "Independent t-test", "Cohen's Kappa", "ROC"],
            "ans": "Independent t-test",
            "why": "Yaş sürekli veridir, kadın/erkek iki bağımsız gruptur ve normal dağılım varsa Independent t-test uygundur."
        },
        {
            "q": "NPC şekli round/oval/heart-shaped ve cinsiyet kadın/erkek. İlişki var mı diye bakıyorsun. Hangi test?",
            "opts": ["ANOVA", "Pearson", "Chi-square", "Paired t-test"],
            "ans": "Chi-square",
            "why": "İki değişken de kategoriktir. Kategorik-kategorik ilişki için Chi-square kullanılır."
        },
        {
            "q": "Aynı hastanın tedavi öncesi ve sonrası ağrı skoru var, normal dağılmıyor. Hangi test?",
            "opts": ["Paired t-test", "Wilcoxon Signed-Rank", "Mann–Whitney U", "ANOVA"],
            "ans": "Wilcoxon Signed-Rank",
            "why": "Aynı bireylerde iki bağımlı ölçüm var ve normal dağılım yok. Bu durumda Wilcoxon uygundur."
        },
        {
            "q": "Dört farklı endikasyon grubunda yaş dağılımı normal değil. Hangi test?",
            "opts": ["Kruskal–Wallis", "Independent t-test", "Fisher Exact", "ICC"],
            "ans": "Kruskal–Wallis",
            "why": "3+ bağımsız grup ve normal dağılmayan sürekli değişken için Kruskal–Wallis kullanılır."
        },
        {
            "q": "İki ölçüm yöntemiyle aynı mm ölçümü yapılmış. Ölçümler uyumlu mu?",
            "opts": ["Chi-square", "ICC / Bland–Altman", "ROC", "Spearman"],
            "ans": "ICC / Bland–Altman",
            "why": "Sürekli ölçüm uyumu için ICC ve yöntem karşılaştırması için Bland–Altman uygundur."
        },
    ]

    score = 0
    for i, item in enumerate(questions, 1):
        st.markdown(f"### Soru {i}")
        ans = st.radio(item["q"], item["opts"], key=f"quiz_{i}")
        if st.button(f"Soru {i} cevabını kontrol et", key=f"check_{i}"):
            if ans == item["ans"]:
                st.success("Doğru.")
            else:
                st.error(f"Yanlış. Doğru cevap: {item['ans']}")
            st.info(item["why"])

# ------------------------
# Decision Tree
# ------------------------
with tabs[19]:
    st.header("🧭 İstatistik Karar Ağacı")
    st.write("Bu bölüm, test seçimini görsel mantıkla öğretir.")

    st.markdown("""
### 1. Önce veri tipini belirle

**Sonuç değişkenin ölçüm mü?**  
Örn: yaş, mm, skor, HU → Sürekli veri

**Sonuç değişkenin sınıf mı?**  
Örn: kadın/erkek, var/yok, oval/round → Kategorik veri

---

### 2. Amacını belirle

| Amaç | Düşünmen gereken şey |
|---|---|
| Grupları karşılaştırmak | Kaç grup var? Gruplar bağımsız mı? |
| İlişki incelemek | Değişkenler sürekli mi kategorik mi? |
| Etki/tahmin modeli | Sonuç değişkeni sürekli mi binary mi? |
| Güvenilirlik | Veri sürekli mi kategorik mi? |
| Tanısal performans | Gerçek pozitif/negatif durum var mı? |

---

### 3. Hızlı karar

| Durum | Test |
|---|---|
| 2 bağımsız grup + sürekli + normal | Independent t-test |
| 2 bağımsız grup + sürekli + normal değil | Mann–Whitney U |
| 2 bağımlı ölçüm + normal | Paired t-test |
| 2 bağımlı ölçüm + normal değil | Wilcoxon |
| 3+ bağımsız grup + normal | ANOVA |
| 3+ bağımsız grup + normal değil | Kruskal–Wallis |
| Kategorik + kategorik | Chi-square / Fisher |
| Sürekli + sürekli ilişki | Pearson / Spearman |
| Sürekli sonuç için model | Linear regression |
| Binary sonuç için model | Logistic regression |
| Tanısal performans | ROC / AUC |
| Sürekli ölçüm uyumu | ICC |
| Kategorik uyum | Kappa |
| Anket iç tutarlılık | Cronbach alpha |
""")

    st.success("Altın kural: Önce veri tipini, sonra grup sayısını, sonra bağımlı/bağımsız durumunu belirle.")

st.divider()
st.caption("Research Statistics AI Platform v5 Educational | Eğitim ve araştırma planlama amacıyla hazırlanmıştır. Klinik/akademik nihai kararlar uzman doğrulaması gerektirir.")
