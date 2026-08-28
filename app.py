"""
RFM Customer Segmentation - Prediction Application

Unsupervised machine learning prototype comparing K-Means, DBSCAN and Gaussian Mixture Models (GMM) for e-commerce customer segmentation.
"""

# IMPORTS
import streamlit as st # streamlit = builds the web interface
import numpy as np # numpy = maths on arrays (log transform, distances)
import pandas as pd # pandas = tables
import joblib # joblib = loads our saved .pkl model files
import os # os = checks whether files exist on disk
from datetime import datetime # datetime = timestamps each prediction in the history log
from datetime import datetime # datetime = timestamps each prediction in the history log

# CONSTANTS
MODEL_DIR = "saved_models"

# Training-data ranges, taken from our 5,878 customers. Used for two things:
# 1. normalising values for the RFM triangle chart
# 2. warning the user when an input falls outside the range the models learned from
TRAIN_RANGE = {
    "Recency":   (1, 739),
    "Frequency": (1, 398),
    "Monetary":  (2.95, 580_987.04),
}

# Colour assigned to each segment. Ordered semantically:
# teal = best, blue = neutral-positive, amber = warning, grey = inactive
SEGMENT_COLOUR = {
    "Champions": "#3DBFC9",
    "Promising": "#7EAEDD",
    "At-Risk":   "#E8B44A",
    "Dormant":   "#9AA7C4",
}

# The marketing action each segment calls for. This is what turns a cluster number into a business decision, which is the whole point of segmentation.
SEGMENT_ACTION = {
    "Champions": ("Reward and retain",
                  "Highest-value customers. Prioritise loyalty rewards, early access and personal "
                  "account contact. Losing one costs more than acquiring several new customers."),
    "Promising": ("Nurture into loyalty",
                  "Recently active but not yet frequent. Encourage a second and third purchase "
                  "through onboarding offers and product recommendations."),
    "At-Risk":   ("Win back now",
                  "Previously valuable but purchasing has lapsed. Time-sensitive reactivation "
                  "offers are worthwhile while the relationship is still recoverable."),
    "Dormant":   ("Minimal spend",
                  "Long inactive, typically single-purchase customers. Include in low-cost bulk "
                  "campaigns only; individual outreach is unlikely to pay back."),
}

# PAGE SETUP
# set_page_config must be the FIRST streamlit command in the script
st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# STYLING
# Custom CSS gives the app its own visual identity rather than the Streamlit default
# Type: Space Grotesk (display) + IBM Plex Sans (body) + IBM Plex Mono (all figures)
# Mono numerals are a deliberate choice - RFM values are ledger figures, so they are set like ledger figures.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --base:    #0F1729;   /* page ground - deep navy, deliberately not black */
    --surface: #18213A;   /* cards, sidebar */
    --line:    #2A3654;   /* hairlines */
    --ink:     #E8ECF5;   /* primary text */
    --muted:   #97A6C7;   /* secondary text */
    --teal:    #3DBFC9;   /* accent */
    --amber:   #E8B44A;   /* warning accent */
}

.stApp { background: var(--base); }

html, body, [class*="css"], .stMarkdown, p, span, div, label {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem;
}
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem; font-weight: 700; color: var(--ink);
    letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 0.35rem 0;
}
.page-sub { color: var(--muted); font-size: 0.94rem; margin-bottom: 1.4rem; }

.verdict {
    border: 1px solid var(--line); border-left: 5px solid var(--seg, var(--teal));
    background: var(--surface); border-radius: 4px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.verdict-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
}
.verdict-name {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700;
    color: var(--seg, var(--teal)); line-height: 1.15; margin: 0.15rem 0 0.4rem 0;
}
.verdict-desc { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }

.chip {
    border: 1px solid var(--line); background: var(--surface); border-radius: 4px;
    padding: 0.85rem 1rem; height: 100%;
}
.chip-model {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
}
.chip-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.35rem;
    font-weight: 600; color: var(--ink); margin: 0.2rem 0 0.1rem 0;
}
.chip-note { font-size: 0.76rem; color: var(--muted); line-height: 1.35; }

.stat { border-top: 2px solid var(--teal); padding-top: 0.55rem; }
.stat-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem;
    letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted);
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem;
    font-weight: 600; color: var(--ink); line-height: 1.25;
}
.stat-sub { font-size: 0.74rem; color: var(--muted); }

.section-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--line); padding-bottom: 0.4rem;
    margin: 1.6rem 0 0.9rem 0;
}

.stButton > button {
    background: var(--teal); color: #0F1729; border: none; border-radius: 3px;
    font-family: 'IBM Plex Sans', sans-serif; font-weight: 600;
    letter-spacing: 0.02em; padding: 0.55rem 1rem;
}
.stButton > button:hover { background: #5AD3DC; color: #0F1729; }

section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }

.stDataFrame, input { font-family: 'IBM Plex Mono', monospace !important; }

#MainMenu, footer { visibility: hidden; }

/* Overrides for Streamlit's own chrome */

/* View selector: styled as a tab bar rather than radio buttons */
div[role="radiogroup"] {
    gap: 1.8rem; border-bottom: 1px solid var(--line);
    padding-bottom: 0; margin-bottom: 1.2rem;
}
div[role="radiogroup"] > label {
    padding: 0.35rem 0 0.5rem 0; margin: 0;
    border-bottom: 2px solid transparent;
}
div[role="radiogroup"] > label:has(input:checked) {
    border-bottom-color: var(--teal);
}
div[role="radiogroup"] > label > div:first-child { display: none; }  /* hide the circle */
div[role="radiogroup"] > label p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.92rem !important; font-weight: 500; color: var(--muted) !important;
}
div[role="radiogroup"] > label:has(input:checked) p { color: var(--ink) !important; }

/* Tabs: replace the default red indicator with our teal, and set the type */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500; color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ink) !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    background-color: var(--teal) !important;
}
div[data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 1.6rem; }

/* Alerts: restyle info / warning / error to sit inside the palette */
div[data-testid="stAlert"] {
    border-radius: 4px; border: 1px solid var(--line);
    border-left: 4px solid var(--teal); background: var(--surface);
    font-family: 'IBM Plex Sans', sans-serif; color: var(--ink);
}
div[data-testid="stAlert"] p { color: var(--ink); font-size: 0.88rem; }

/* Number inputs and file uploader */
div[data-testid="stNumberInput"] input {
    border-radius: 3px; border: 1px solid var(--line);
    background: var(--base) !important; color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace;
}
div[data-testid="stNumberInput"] label, div[data-testid="stFileUploader"] label {
    font-size: 0.8rem !important; color: var(--muted) !important; font-weight: 500;
}
section[data-testid="stFileUploadDropzone"] {
    border: 1px dashed var(--line); background: var(--surface); border-radius: 4px;
}

/* Captions */
div[data-testid="stCaptionContainer"] p { color: var(--muted); font-size: 0.78rem; }

/* Tables */
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; }

/* Keyboard focus stays visible for accessibility */
*:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }

/* Respect users who prefer reduced motion */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* Narrow screens: let the segment cards stack instead of squeezing */
@media (max-width: 900px) {
    .page-title { font-size: 1.6rem; }
    .verdict-name { font-size: 1.5rem; }
}

</style>
""", unsafe_allow_html=True)


# MODEL LOADING
# @st.cache_resource tells Streamlit: "run this ONCE, then reuse the result".
# Without it the models would reload on every interaction, making the app slow
# This matches the "cached on app startup" box in the system flowchart
@st.cache_resource
def load_models():
    """
    Load the trained models and scaler from disk.

    Returns a dict on success, or a string describing the problem on failure.
    Returning the reason (rather than just None) lets the interface tell the
    user exactly what to fix.
    """
    required = ["kmeans.pkl", "gmm.pkl", "scaler.pkl", "segment_names.pkl"]

    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return f"Missing model files: {', '.join(missing)}"

    # Wrap the loading in try/except. A .pkl can exist but still fail to load. For example, if it was saved by a different scikit-learn version.
    try:
        models = {
            "kmeans": joblib.load(f"{MODEL_DIR}/kmeans.pkl"),
            "gmm":    joblib.load(f"{MODEL_DIR}/gmm.pkl"),
            "scaler": joblib.load(f"{MODEL_DIR}/scaler.pkl"),
            "names":  joblib.load(f"{MODEL_DIR}/segment_names.pkl"),
        }
    except Exception as e:
        return f"Model files could not be read: {e}"

    # DBSCAN core points are optional - the app degrades gracefully without them
    db_path = f"{MODEL_DIR}/dbscan_core.pkl"
    try:
        models["dbscan"] = joblib.load(db_path) if os.path.exists(db_path) else None
    except Exception:
        models["dbscan"] = None

    # Comparison of all three algorithms, used on the landing page. Optional
    comp_path = f"{MODEL_DIR}/comparison.pkl"
    try:
        models["comparison"] = joblib.load(comp_path) if os.path.exists(comp_path) else None
    except Exception:
        models["comparison"] = None

    # Customer-level export, used by the management report. Optional
    cust_path = f"{MODEL_DIR}/customer_segments.csv"
    try:
        models["customers"] = pd.read_csv(cust_path) if os.path.exists(cust_path) else None
    except Exception:
        models["customers"] = None

    # The segment profile table is optional too
    prof_path = f"{MODEL_DIR}/segment_profile.csv"
    try:
        models["profile"] = pd.read_csv(prof_path, index_col=0) if os.path.exists(prof_path) else None
    except Exception:
        models["profile"] = None

    return models


# PREDICTION LOGIC
def predict_dbscan(x_scaled, dbscan_data):
    """
    Classify a new point using DBSCAN's own definition.

    scikit-learn's DBSCAN has no .predict() method, because a density-based
    algorithm has no centroid to compare a new point against. We therefore apply
    the algorithm's defining rule directly: a new point joins a cluster if it
    lies within `eps` of one of that cluster's CORE points. If it is not close
    enough to any core point, it is noise.
    """
    core_points = dbscan_data["core_points"]
    core_labels = dbscan_data["core_labels"]
    eps = dbscan_data["eps"]

    # Euclidean distance from the new point to every core point
    # axis=1 sums across columns, giving one distance per core point
    distances = np.sqrt(((core_points - x_scaled) ** 2).sum(axis=1))

    nearest = float(distances.min())
    if nearest <= eps:
        return int(core_labels[distances.argmin()]), nearest #Joins that cluster
    return -1, nearest # Too far -> noise


def segment_key(full_name):
    """Extract the short segment name: 'Champions   (recent...)' -> 'Champions'."""
    return full_name.split("(")[0].strip()


def classify(recency, frequency, monetary, models):
    """
    Run the full prediction pipeline for one customer.

    Applies exactly the same transformations used in training:
      1. log1p to compress the skew
      2. the SAVED scaler - never a newly fitted one, or the numbers would not match what the models learned from.
    """
    raw = np.array([[recency, frequency, monetary]], dtype=float)
    x_log = np.log1p(raw)
    x_scaled = models["scaler"].transform(x_log)

    km = int(models["kmeans"].predict(x_scaled)[0])
    gm = int(models["gmm"].predict(x_scaled)[0])
    # predict_proba gives a probability for EVERY component; .max() is the confidence in the winning one.
    gm_conf = float(models["gmm"].predict_proba(x_scaled).max())

    if models["dbscan"] is not None:
        db, db_dist = predict_dbscan(x_scaled[0], models["dbscan"])
    else:
        db, db_dist = None, None

    return {
        "kmeans": km, "gmm": gm, "gmm_conf": gm_conf,
        "dbscan": db, "dbscan_dist": db_dist,
        "segment": segment_key(models["names"].get(km, f"Cluster {km}")),
    }


# VALIDATION
def validate(recency, frequency, monetary):
    """
    Check the input against hard constraints and business rules.

    Returns (errors, warnings).
      errors   -> block the prediction; the values are impossible
      warnings -> allow the prediction but tell the user it may be unreliable

    This implements the "Values valid and positive?" decision in the system flowchart, plus the business rules a marketing analyst would apply when reviewing a customer record.
    """
    errors, warnings = [], []

    # HARD CONSTRAINTS: values that cannot exist
    if recency <= 0:
        errors.append("Recency must be at least 1 day. A purchase cannot be zero or negative days ago.")
    if frequency <= 0:
        errors.append("Frequency must be at least 1. A customer with no orders is not a customer.")
    if monetary <= 0:
        errors.append("Monetary value must be above zero. Cancelled and zero-value orders are excluded from the model.")

    if errors:
        return errors, warnings # No point checking business rules on impossible values

    # BUSINESS RULE 1: average order value must be plausible
    # Total spend divided by order count gives the average order value. Values far outside a plausible retail range usually mean the figures have been entered into the wrong boxes
    aov = monetary / frequency
    if aov < 1:
        warnings.append(
            f"Average order value is GBP {aov:,.2f}, below GBP 1 per order. That is unusual "
            "for this retailer. Check that Frequency and Monetary have not been swapped."
        )
    elif aov > 50_000:
        warnings.append(
            f"Average order value is GBP {aov:,.2f}. That is exceptionally high and may indicate "
            "a data-entry error, or a genuine bulk wholesale account."
        )

    # BUSINESS RULE 2: frequency must fit the observation window
    # The dataset covers a 739-day trading period. A customer cannot have placed more separate orders than there were days available to place them
    if frequency > 739:
        errors.append(
            f"Frequency of {frequency:,} exceeds the 739-day observation window of the training "
            "data. A customer cannot place more orders than there are days."
        )

    # BUSINESS RULE 3: warn when extrapolating beyond training data
    # The models learned the structure of customers within these ranges. Outside them the prediction is extrapolation and should be treated with caution.
    for name, value in [("Recency", recency), ("Frequency", frequency), ("Monetary", monetary)]:
        low, high = TRAIN_RANGE[name]
        if value > high:
            warnings.append(
                f"{name} of {value:,.2f} is above the training maximum of {high:,.2f}. The "
                "prediction is extrapolating beyond observed customer behaviour."
            )

    return errors, warnings


# RFM TRIANGLE
def rfm_scores(recency, frequency, monetary):
    """
    Convert raw RFM values into three 0-1 scores for the triangle chart.

    Uses a log scale to match how the models see the data. Recency is INVERTED, because a low recency (bought recently) is a GOOD outcome and should produce a large shape, consistent with Frequency and Monetary.
    """
    def scale(value, low, high, invert=False):
        s = (np.log1p(value) - np.log1p(low)) / (np.log1p(high) - np.log1p(low))
        s = 1 - s if invert else s
        return float(np.clip(s, 0.05, 1.0)) # Floor at 0.05 keeps the shape visible

    return [
        scale(recency,   *TRAIN_RANGE["Recency"],   invert=True),
        scale(frequency, *TRAIN_RANGE["Frequency"]),
        scale(monetary,  *TRAIN_RANGE["Monetary"]),
    ]


def triangle_svg(customer_scores, segment_scores, colour):
    """
    Draw the RFM triangle: the customer's shape over their segment's median shape.

    Three axes at 120 degrees - Recency at the top, Frequency lower-right,
    Monetary lower-left. The further a vertex sits from the centre, the stronger
    the customer is on that dimension.
    """
    cx, cy, rad = 130, 122, 82
    angles = [-90, 30, 150] # Degrees: top, lower-right, lower-left

    def to_points(scores):
        pts = []
        for score, angle in zip(scores, angles):
            a = np.radians(angle)
            pts.append(f"{cx + rad*score*np.cos(a):.1f},{cy + rad*score*np.sin(a):.1f}")
        return " ".join(pts)

    # Reference rings at 25/50/75/100% so the reader can judge magnitude
    rings = "".join(
        f'<polygon points="{to_points([f, f, f])}" fill="none" stroke="#2A3654" stroke-width="1"/>'
        for f in (0.25, 0.5, 0.75, 1.0)
    )

    # Axis spokes and their labels
    spokes, labels = "", ""
    for label, angle in zip(["R", "F", "M"], angles):
        a = np.radians(angle)
        spokes += (f'<line x1="{cx}" y1="{cy}" x2="{cx+rad*np.cos(a):.1f}" '
                   f'y2="{cy+rad*np.sin(a):.1f}" stroke="#2A3654" stroke-width="1"/>')
        lx, ly = cx + (rad + 16) * np.cos(a), cy + (rad + 16) * np.sin(a)
        labels += (f'<text x="{lx:.1f}" y="{ly+4:.1f}" text-anchor="middle" '
                   f'font-family="IBM Plex Mono, monospace" font-size="11" '
                   f'font-weight="600" fill="#97A6C7">{label}</text>')

    return f"""
    <svg viewBox="0 0 260 250" width="100%" style="max-width:280px">
        {rings}{spokes}{labels}
        <polygon points="{to_points(segment_scores)}" fill="#E8ECF5" fill-opacity="0.06"
                 stroke="#8494B8" stroke-width="1.5" stroke-dasharray="4 3"/>
        <polygon points="{to_points(customer_scores)}" fill="{colour}" fill-opacity="0.22"
                 stroke="{colour}" stroke-width="2.5"/>
        <line x1="30" y1="232" x2="48" y2="232" stroke="{colour}" stroke-width="2.5"/>
        <text x="54" y="235.5" font-family="IBM Plex Mono, monospace"
              font-size="9" fill="#97A6C7">this customer</text>
        <line x1="138" y1="232" x2="156" y2="232" stroke="#8494B8"
              stroke-width="1.5" stroke-dasharray="4 3"/>
        <text x="162" y="235.5" font-family="IBM Plex Mono, monospace"
              font-size="9" fill="#97A6C7">segment median</text>
    </svg>"""



def cluster_bar(rows, labels=None, highlight_noise=False):
    """
    Draw one algorithm's clusters as a single horizontal bar.

    Each block's WIDTH is that cluster's share of customers, and its OPACITY is
    its share of revenue. A narrow but solid block therefore means "few
    customers, lots of money" - exactly the pattern that makes an outlier group
    commercially interesting.

    Wide blocks are labelled with the cluster name and its customer share.
    Every block, however narrow, carries a hover tooltip with the full figures,
    so no cluster is left unexplained.
    """
    labels = labels or {}
    segments, x = "", 0.0

    for r in rows:
        cid = int(r["C"])
        w = float(r["pct_cust"])
        rev = float(r["pct_rev"])
        is_noise = cid == -1

        colour = "#E8B44A" if (is_noise and highlight_noise) else "#3DBFC9"
        # Opacity scales with revenue share, so value is visible as well as headcount
        opacity = 0.25 + 0.75 * min(1.0, rev / 60)
        # Text sits on the block, so it needs the opposite tone to stay readable
        text_col = "#0F1729" if opacity > 0.55 else "#C7D3EA"

        name = labels.get(cid, ("Noise" if is_noise else f"C{cid}"))

        # Tooltip: full detail for every block, including the ones too small to label
        tip = (f"{name} — {w:.1f}% of customers, {rev:.1f}% of revenue "
               f"(R {r['R']:,.0f} d, F {r['F']:,.0f}, M GBP {r['M']:,.0f})")

        block = (f'<rect x="{x:.2f}%" y="0" width="{max(w - 0.4, 0.25):.2f}%" height="46" '
                 f'fill="{colour}" fill-opacity="{opacity:.2f}" rx="2"/>')

        cx = x + w / 2
        if w >= 9:        # room for both the name and the figure
            block += (f'<text x="{cx:.2f}%" y="20" text-anchor="middle" '
                      f'font-family="IBM Plex Sans, sans-serif" font-size="10.5" '
                      f'font-weight="600" fill="{text_col}">{name}</text>'
                      f'<text x="{cx:.2f}%" y="35" text-anchor="middle" '
                      f'font-family="IBM Plex Mono, monospace" font-size="10" '
                      f'fill="{text_col}" fill-opacity="0.85">{w:.1f}%</text>')
        elif w >= 4.5:    # room for the figure only
            block += (f'<text x="{cx:.2f}%" y="29" text-anchor="middle" '
                      f'font-family="IBM Plex Mono, monospace" font-size="10" '
                      f'font-weight="600" fill="{text_col}">{w:.1f}%</text>')

        # Wrap in <g> with a <title> so the whole block is hoverable
        segments += f'<g><title>{tip}</title>{block}</g>'
        x += w

    return f'<svg width="100%" height="46" style="display:block">{segments}</svg>'


def cluster_legend(rows, labels=None, highlight_noise=False):
    """
    List every cluster under the bar, including the ones too narrow to label.

    The bar shows proportion at a glance; this legend guarantees that no cluster
    is invisible, which matters most for the small high-value groups.
    """
    labels = labels or {}
    parts = []

    for r in rows:
        cid = int(r["C"])
        is_noise = cid == -1
        name = labels.get(cid, ("Noise" if is_noise else f"C{cid}"))
        colour = "#E8B44A" if (is_noise and highlight_noise) else "#3DBFC9"

        parts.append(
            f'<span style="white-space:nowrap;margin-right:1.1rem">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:1px;'
            f'background:{colour};margin-right:0.35rem"></span>'
            f'<span style="color:var(--ink)">{name}</span> '
            f'<span style="color:var(--muted)">{float(r["pct_cust"]):.1f}% cust '
            f'&#183; {float(r["pct_rev"]):.1f}% rev</span></span>'
        )

    return (f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
            f'line-height:2;margin-top:0.5rem">{"".join(parts)}</div>')



def share_bars(profile, names):
    """
    Draw customer share against revenue share for every segment.

    The pale bar is the share of customers; the solid bar is the share of
    revenue. Where the solid bar is far longer than the pale one, that segment
    returns more than its headcount suggests - which is the whole argument for
    unequal marketing spend.
    """
    ordered = profile.sort_values("Pct_Revenue", ascending=False)
    rows = ""
    row_h, bar_h, gap = 46, 15, 4

    for i, (cid, r) in enumerate(ordered.iterrows()):
        seg = segment_key(names.get(cid, f"Cluster {cid}"))
        colour = SEGMENT_COLOUR.get(seg, "#9AA7C4")
        top = i * row_h
        cust, rev = float(r["Pct_Customers"]), float(r["Pct_Revenue"])

        rows += (
            f'<text x="0" y="{top+16}" font-family="IBM Plex Sans, sans-serif" '
            f'font-size="12" font-weight="600" fill="{colour}">{seg}</text>'
            # pale bar - share of customers
            f'<rect x="90" y="{top+3}" width="{cust*0.78:.2f}%" height="{bar_h}" '
            f'fill="{colour}" fill-opacity="0.32" rx="2"/>'
            f'<text x="{90 + cust*0.78:.2f}%" y="{top+15}" dx="6" '
            f'font-family="IBM Plex Mono, monospace" font-size="10" '
            f'fill="#97A6C7">{cust:.1f}% customers</text>'
            # solid bar - share of revenue
            f'<rect x="90" y="{top+3+bar_h+gap}" width="{rev*0.78:.2f}%" height="{bar_h}" '
            f'fill="{colour}" rx="2"/>'
            f'<text x="{90 + rev*0.78:.2f}%" y="{top+15+bar_h+gap}" dx="6" '
            f'font-family="IBM Plex Mono, monospace" font-size="10" font-weight="600" '
            f'fill="{colour}">{rev:.1f}% revenue</text>'
        )

    return f'<svg width="100%" height="{len(ordered)*row_h}" style="display:block">{rows}</svg>'


# INTERFACE
st.markdown(
    '<div class="eyebrow">Unsupervised machine learning &#183; RFM segmentation</div>'
    '<div class="page-title">Customer Segmentation</div>'
    '<div class="page-sub">Classify a customer into a behavioural segment using three '
    'clustering algorithms trained on 5,878 customers.</div>',
    unsafe_allow_html=True,
)

models = load_models()

# If loading failed, `models` is a string explaining why.
# This is the "No" branch of the "Saved models exist?" decision in the flowchart.
if isinstance(models, str):
    st.error(
        f"**Cannot start.** {models}\n\n"
        "Run `RFM_Customer_Segmentation.ipynb` in Anaconda Jupyter Notebok to train the models, then reload this page."
    )
    st.stop()   # Halts the script - nothing below runs

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="eyebrow">Customer record</div>', unsafe_allow_html=True)

    recency = st.number_input(
        "Recency — days since last order",
        min_value=1, max_value=5000, value=15, step=1,
        help="Training range 1-739 days. Lower means more recently active.",
    )
    frequency = st.number_input(
        "Frequency — number of orders",
        min_value=1, max_value=5000, value=15, step=1,
        help="Training range 1-398 orders. Counts distinct invoices, not line items.",
    )
    monetary = st.number_input(
        "Monetary — total spend (GBP)",
        min_value=0.01, max_value=5_000_000.0, value=5000.0, step=100.0,
        help="Training range GBP 2.95-580,987.",
    )

    classify_clicked = st.button("Classify customer", use_container_width=True)

    # Live average order value, so the analyst sees it before submitting
    st.markdown(
        f'<div class="stat" style="margin-top:1rem"><div class="stat-label">Average order value</div>'
        f'<div class="stat-value">GBP {monetary/frequency:,.2f}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">Reference profiles</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:0.76rem;line-height:1.9;color:#97A6C7">'
        'Champions &nbsp;&nbsp;15 / 15 / 5000<br>'
        'Promising &nbsp;&nbsp;20 / 2 / 600<br>'
        'At-Risk &nbsp;&nbsp;&nbsp;&nbsp;200 / 5 / 1500<br>'
        'Dormant &nbsp;&nbsp;&nbsp;&nbsp;450 / 1 / 250</div>',
        unsafe_allow_html=True,
    )

# MAIN PANEL
# A radio is used rather than st.tabs() because Streamlit tabs are client-side
# only and cannot be switched from code. Holding the view in session_state lets
# the sidebar button return the user to the classification view automatically.
VIEWS = ["Classify Customer", "Management Report", "History"]

if "view" not in st.session_state:
    st.session_state.view = VIEWS[0]

# The prediction log lives in session_state, which Streamlit preserves across
# reruns within a session. Every classification appends one row.
if "history" not in st.session_state:
    st.session_state.history = []

# The prediction log lives in session_state, which Streamlit preserves across
# reruns within a session. Every classification appends one row.
if "history" not in st.session_state:
    st.session_state.history = []

# Set the view BEFORE the widget is created, otherwise Streamlit raises an error
# for modifying a widget's state after it has been instantiated.
if classify_clicked:
    st.session_state.view = VIEWS[0]

view = st.radio("View", VIEWS, key="view", horizontal=True,
                label_visibility="collapsed")

# VIEW 1: single-customer classification
if view == VIEWS[0]:
    if classify_clicked:
        # VALIDATION (flowchart: "Values valid and positive?")
        errors, warnings = validate(recency, frequency, monetary)

        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        for w in warnings:
            st.warning(w)

        # PREDICT
        result = classify(recency, frequency, monetary, models)
        seg = result["segment"]
        colour = SEGMENT_COLOUR.get(seg, "#3DBFC9")
        action, advice = SEGMENT_ACTION.get(seg, ("Review", "No action defined for this segment."))

        # LOG THIS PREDICTION
        # Appended once per click. Streamlit reruns the script on every
        # interaction, but classify_clicked is only True on the run triggered
        # by the button, so no duplicate rows are created.
        db_label = ("n/a" if result["dbscan"] is None
                    else "Noise" if result["dbscan"] == -1
                    else f"Cluster {result['dbscan']}")
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": round(monetary, 2),
            "Segment": seg,
            "K-Means": f"Cluster {result['kmeans']}",
            "DBSCAN": db_label,
            "GMM": f"Comp {result['gmm']}",
            "GMM confidence": f"{result['gmm_conf']:.0%}",
            "Warnings": len(warnings),
        })

        # VERDICT CARD
        st.markdown(
            f'<div class="verdict" style="--seg:{colour}">'
            f'<div class="verdict-label">Assigned segment</div>'
            f'<div class="verdict-name">{seg}</div>'
            f'<div class="verdict-desc"><strong>{action}.</strong> {advice}</div></div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1.25])

        # LEFT: The RFM triangle
        with left:
            st.markdown('<div class="section-head">RFM profile</div>', unsafe_allow_html=True)

            cust_scores = rfm_scores(recency, frequency, monetary)

            # Compare against this segment's median customer, if the profile exists
            if models["profile"] is not None and result["kmeans"] in models["profile"].index:
                row = models["profile"].loc[result["kmeans"]]
                seg_scores = rfm_scores(row["Med_Recency"], row["Med_Frequency"], row["Med_Monetary"])
            else:
                seg_scores = cust_scores

            st.markdown(triangle_svg(cust_scores, seg_scores, colour), unsafe_allow_html=True)

        # RIGHT: The three model verdicts
        with right:
            st.markdown('<div class="section-head">Model verdicts</div>', unsafe_allow_html=True)

            db = result["dbscan"]
            if db is None:
                db_val, db_note = "n/a", "Core points not saved"
            elif db == -1:
                db_val, db_note = "Noise", f"Nearest core point {result['dbscan_dist']:.2f} away"
            else:
                db_val, db_note = f"Cluster {db}", "Density-based"

            m1, m2, m3 = st.columns(3)
            for col, model_name, value, note in [
                (m1, "K-Means", f"Cluster {result['kmeans']}", "Partition-based &#183; deployed"),
                (m2, "DBSCAN", db_val, db_note),
                (m3, "GMM", f"Comp {result['gmm']}", f"Confidence {result['gmm_conf']:.0%}"),
            ]:
                col.markdown(
                    f'<div class="chip"><div class="chip-model">{model_name}</div>'
                    f'<div class="chip-value">{value}</div>'
                    f'<div class="chip-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

            # The outlier finding, surfaced when it applies
            if db == -1:
                st.markdown(
                    '<div style="border:1px solid #4A3E1E;border-left:5px solid #E8B44A;'
                    'background:#231D0E;border-radius:4px;padding:1rem 1.2rem;margin-top:1rem">'
                    '<div class="verdict-label" style="color:#E8B44A">Outlier detected</div>'
                    '<div style="font-size:0.87rem;line-height:1.55;margin-top:0.35rem">'
                    'DBSCAN isolates this customer as noise while K-Means absorbs them into an '
                    'ordinary cluster. In our analysis the 182 customers labelled as noise were '
                    '3.1% of the customer base but generated <strong>36.7% of total revenue</strong> '
                    '&#8212; typically wholesale accounts.</div></div>',
                    unsafe_allow_html=True,
                )

            # Comparison with the segment
            if models["profile"] is not None and result["kmeans"] in models["profile"].index:
                row = models["profile"].loc[result["kmeans"]]
                st.markdown('<div class="section-head">Against segment median</div>',
                            unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                for col, label, mine, theirs, fmt in [
                    (c1, "Recency",   recency,   row["Med_Recency"],   "{:,.0f} d"),
                    (c2, "Frequency", frequency, row["Med_Frequency"], "{:,.0f}"),
                    (c3, "Monetary",  monetary,  row["Med_Monetary"],  "GBP {:,.0f}"),
                ]:
                    col.markdown(
                        f'<div class="stat"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value">{fmt.format(mine)}</div>'
                        f'<div class="stat-sub">segment median {fmt.format(theirs)}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.caption(
                    f"This segment holds {int(row['Customers']):,} customers "
                    f"({row['Pct_Customers']}% of the base) and generates "
                    f"{row['Pct_Revenue']}% of total revenue."
                )

    else:
        # IDLE STATE: an invitation to act, not a blank page.
        # Shows what each of the three algorithms actually found, so the choice of
        # K-Means as the deployed model is evidenced rather than simply asserted.

        # ---- The four actionable segments, from the deployed model -------------
        st.markdown('<div class="section-head">Segments discovered by K-Means &#183; the deployed model</div>',
                    unsafe_allow_html=True)

        if models["profile"] is not None:
            # Ordered by revenue contribution, most valuable first
            ordered = models["profile"].sort_values("Pct_Revenue", ascending=False)
            cols = st.columns(len(ordered))

            for col, (cid, row) in zip(cols, ordered.iterrows()):
                name = segment_key(models["names"].get(cid, f"Cluster {cid}"))
                seg_colour = SEGMENT_COLOUR.get(name, "#9AA7C4")
                col.markdown(
                    f'<div class="verdict" style="--seg:{seg_colour};padding:1.1rem 1.2rem">'
                    f'<div class="verdict-label">{row["Pct_Customers"]}% of customers</div>'
                    f'<div class="verdict-name" style="font-size:1.25rem">{name}</div>'
                    f'<div class="stat" style="border-top-color:{seg_colour};margin-top:0.6rem">'
                    f'<div class="stat-label">Revenue share</div>'
                    f'<div class="stat-value" style="color:{seg_colour}">{row["Pct_Revenue"]}%</div></div>'
                    f'<div class="chip-note" style="margin-top:0.6rem">'
                    f'R {row["Med_Recency"]:,.0f} d &#183; F {row["Med_Frequency"]:,.0f} &#183; '
                    f'M GBP {row["Med_Monetary"]:,.0f}</div></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "Median values per segment. Derived from 5,878 customers and 779,425 "
                "transactions in the Online Retail II dataset (UCI Machine Learning Repository)."
            )

        # How all three algorithms partitioned the same customers
        if models["comparison"] is not None:
            st.markdown('<div class="section-head">How each algorithm divided the same 5,878 customers</div>',
                        unsafe_allow_html=True)

            for algo in models["comparison"]:
                # Sort blocks by customer share so the bar reads largest-first
                rows = sorted(algo["rows"], key=lambda r: -float(r["pct_cust"]))
                noise_note = (f' &#183; {algo["noise"]} noise' if algo["noise"] else "")

                # Only K-Means has business names; the other two are labelled by
                # cluster number, which is honest - we never named their clusters.
                if algo["name"] == "K-Means":
                    bar_labels = {cid: segment_key(nm) for cid, nm in models["names"].items()}
                else:
                    bar_labels = {}

                bar_col, metric_col = st.columns([2.1, 1])

                with bar_col:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);'
                        f'margin:0.9rem 0 0.4rem 0">{algo["name"]} &#183; '
                        f'{algo["clusters"]} clusters{noise_note}</div>'
                        + cluster_bar(rows, labels=bar_labels, highlight_noise=True)
                        + cluster_legend(rows, labels=bar_labels, highlight_noise=True),
                        unsafe_allow_html=True,
                    )

                with metric_col:
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                        f'color:var(--muted);padding-top:2.1rem;text-align:right">'
                        f'Sil {algo["silhouette"]} &#183; DB {algo["davies_bouldin"]} '
                        f'&#183; CH {algo["calinski"]:,.0f}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
            st.caption(
                "Block width is the share of customers in that cluster; block opacity is its share "
                "of revenue. Amber marks the customers DBSCAN labels as noise \u2014 3.1% of the base "
                "but 36.7% of revenue. K-Means produces four balanced, interpretable segments and "
                "scores best on all three indices, so it is the deployed model. Those indices are "
                "distance-based and structurally favour compact spherical clusters, however, so they "
                "understate what DBSCAN uniquely finds."
            )


        st.info("Enter a customer record in the sidebar, then select **Classify customer**.")


# VIEW 2: management report for company leadership
elif view == VIEWS[1]:
    prof = models["profile"]

    if prof is None:
        st.info("Segment profile not found. Run the notebook to generate saved_models/segment_profile.csv.")
    else:
        # Rebuild the segment table with business names and actions
        rep = prof.copy()
        rep["Segment"] = [segment_key(models["names"].get(c, f"Cluster {c}")) for c in rep.index]

        # Total revenue is recovered from the shares, so the app does not need
        # the raw dataset to report absolute figures.
        est_total = float((rep["Customers"] * rep["Med_Monetary"]).sum())
        if models["customers"] is not None:
            est_total = float(models["customers"]["Monetary"].sum())
            rev = models["customers"].groupby("Segment")["Monetary"].sum()
            rep["Revenue"] = [float(rev.get(s, 0)) for s in rep["Segment"]]
        else:
            rep["Revenue"] = est_total * rep["Pct_Revenue"] / 100

        rep["Value_Each"] = rep["Revenue"] / rep["Customers"]
        rep = rep.sort_values("Pct_Revenue", ascending=False)

        top = rep.iloc[0]
        bottom = rep.iloc[-1]
        at_risk = rep[rep["Segment"] == "At-Risk"]
        risk_rev = float(at_risk["Revenue"].iloc[0]) if len(at_risk) else 0.0
        ratio = top["Value_Each"] / bottom["Value_Each"]
        total_cust = int(rep["Customers"].sum())

        # HEADLINE
        st.markdown(
            f'<div class="verdict" style="--seg:#3DBFC9">'
            f'<div class="verdict-label">Headline</div>'
            f'<div class="verdict-name" style="font-size:1.5rem">'
            f'{top["Pct_Customers"]:.0f}% of customers generate {top["Pct_Revenue"]:.0f}% of revenue</div>'
            f'<div class="verdict-desc">A {top["Segment"]} customer is worth GBP '
            f'{top["Value_Each"]:,.0f} on average, against GBP {bottom["Value_Each"]:,.0f} for a '
            f'{bottom["Segment"]} customer &#8212; a difference of {ratio:.0f} times. Marketing '
            f'spend split evenly across the base is therefore spent mostly on customers who '
            f'return little.</div></div>',
            unsafe_allow_html=True,
        )

        # KPI STRIP
        k1, k2, k3, k4 = st.columns(4)
        for col, value, label, colour in [
            (k1, f"{total_cust:,}",            "Active customers",   "var(--ink)"),
            (k2, f"GBP {est_total/1e6:.1f}M",  "Total revenue",      "var(--ink)"),
            (k3, f"{top['Pct_Revenue']:.0f}%", f"From {top['Segment']}", "#3DBFC9"),
            (k4, f"GBP {risk_rev/1e6:.1f}M",   "Revenue at risk",    "#E8B44A"),
        ]:
            col.markdown(
                f'<div class="stat"><div class="stat-label">{label}</div>'
                f'<div class="stat-value" style="color:{colour}">{value}</div></div>',
                unsafe_allow_html=True,
            )

        # SHARE COMPARISON
        st.markdown('<div class="section-head">Customer share against revenue share</div>',
                    unsafe_allow_html=True)
        st.markdown(share_bars(prof, models["names"]), unsafe_allow_html=True)

        # SEGMENT DETAIL AND ACTIONS
        st.markdown('<div class="section-head">Segments and recommended action</div>',
                    unsafe_allow_html=True)

        detail = pd.DataFrame({
            "Segment": rep["Segment"],
            "Customers": rep["Customers"].astype(int),
            "% of base": rep["Pct_Customers"],
            "Revenue (GBP)": rep["Revenue"].round(0),
            "% of revenue": rep["Pct_Revenue"],
            "Value each (GBP)": rep["Value_Each"].round(0),
            "Recency (d)": rep["Med_Recency"].astype(int),
            "Orders": rep["Med_Frequency"].astype(int),
            "Action": [SEGMENT_ACTION.get(s, ("Review", ""))[0] for s in rep["Segment"]],
        }).set_index("Segment")
        st.dataframe(detail, use_container_width=True)

        # PRIORITIES
        st.markdown('<div class="section-head">Recommended priorities</div>',
                    unsafe_allow_html=True)

        for n, (_, r) in enumerate(rep.iterrows(), start=1):
            seg = r["Segment"]
            colour = SEGMENT_COLOUR.get(seg, "#9AA7C4")
            action, tactic = SEGMENT_ACTION.get(seg, ("Review", "No action defined."))
            st.markdown(
                f'<div style="display:flex;gap:0.9rem;align-items:flex-start;'
                f'padding:0.7rem 0;border-bottom:1px solid var(--line)">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;'
                f'font-weight:600;color:{colour};min-width:1.2rem">{n}</div><div>'
                f'<div style="font-weight:600;color:{colour}">{action} &#183; {seg}</div>'
                f'<div style="font-size:0.85rem;color:var(--muted);line-height:1.5">'
                f'{int(r["Customers"]):,} customers carrying GBP {r["Revenue"]/1e6:.2f}M '
                f'({r["Pct_Revenue"]:.1f}% of revenue), median recency '
                f'{int(r["Med_Recency"])} days. {tactic}.</div></div></div>',
                unsafe_allow_html=True,
            )

        # EXPORTS
        st.markdown('<div class="section-head">Export</div>', unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "Segment summary (CSV)",
                data=detail.to_csv().encode("utf-8"),
                file_name="segment_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("One row per segment, with figures and recommended action.")

        with e2:
            if models["customers"] is not None:
                out = models["customers"].copy()
                out["Recommended_Action"] = out["Segment"].map(
                    lambda s: SEGMENT_ACTION.get(s, ("Review", ""))[0])
                order = {s: i for i, s in enumerate(rep["Segment"], start=1)}
                out["Priority"] = out["Segment"].map(order)
                out = out.sort_values(["Priority", "Monetary"], ascending=[True, False])
                st.download_button(
                    "Customer action list (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="customer_action_list.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(f"All {len(out):,} customers with segment, action and priority.")
            else:
                st.button("Customer action list (CSV)", disabled=True, use_container_width=True)
                st.caption("Run the notebook cell that saves customer_segments.csv to enable this.")

        st.caption(
            "Derived from the Online Retail II dataset (UCI Machine Learning Repository). "
            "Segments produced by K-Means clustering (k = 4) on log-transformed RFM features. "
            "A further 182 customers (3.1% of the base) were identified by density-based "
            "clustering as behavioural outliers generating 36.7% of revenue; their purchasing "
            "pattern indicates wholesale accounts warranting separate account management."
        )


# VIEW 3: prediction history for the current session
else:
    st.markdown('<div class="section-head">Prediction history</div>', unsafe_allow_html=True)

    hist = st.session_state.history

    if not hist:
        st.info("No predictions yet. Classify a customer to start the log.")
    else:
        # Newest first, so the most recent run is at the top
        df_hist = pd.DataFrame(hist)[::-1].reset_index(drop=True)

        # Summary of the session so far
        seg_counts = df_hist["Segment"].value_counts()
        n_noise = int((df_hist["DBSCAN"] == "Noise").sum())
        n_warned = int((df_hist["Warnings"] > 0).sum())

        s1, s2, s3, s4 = st.columns(4)
        for col, value, label, colour in [
            (s1, f"{len(df_hist)}",  "Predictions run",   "var(--ink)"),
            (s2, f"{seg_counts.index[0]}", "Most frequent segment", "var(--ink)"),
            (s3, f"{n_noise}",       "Flagged as outlier", "#E8B44A"),
            (s4, f"{n_warned}",      "Raised a warning",   "var(--muted)"),
        ]:
            col.markdown(
                f'<div class="stat"><div class="stat-label">{label}</div>'
                f'<div class="stat-value" style="color:{colour};font-size:1.15rem">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Segment tally
        st.markdown('<div class="section-head">Segments assigned this session</div>',
                    unsafe_allow_html=True)
        tally = st.columns(len(seg_counts))
        for col, (seg_name, count) in zip(tally, seg_counts.items()):
            colour = SEGMENT_COLOUR.get(seg_name, "#9AA7C4")
            col.markdown(
                f'<div class="chip" style="border-left:4px solid {colour}">'
                f'<div class="chip-model">{seg_name}</div>'
                f'<div class="chip-value" style="color:{colour}">{count}</div></div>',
                unsafe_allow_html=True,
            )

        # The log itself
        st.markdown('<div class="section-head">Log</div>', unsafe_allow_html=True)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # Export and clear
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download history (CSV)",
                data=df_hist.to_csv(index=False).encode("utf-8"),
                file_name=f"prediction_history_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()   # redraw immediately so the cleared log is shown

        st.caption(
            "The log covers the current session only and is held in memory. "
            "Restarting the application clears it. Download the CSV to keep a record."
        )

# FOOTER
st.markdown(
    '<div style="border-top:1px solid #2A3654;margin-top:2.5rem;padding-top:1rem;text-align:center;'
    'font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#97A6C7;letter-spacing:0.03em">'
    'BMCS2003 Artificial Intelligence Assignment &#183; RFM-Based Customer Segmentation Using Unsupervised '
    'Machine Learning: A Comparative Analysis of K-Means, DBSCAN and Gaussian Mixture Models.'
    '</div>',
    unsafe_allow_html=True,
)