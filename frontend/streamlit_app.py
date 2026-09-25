"""Public-facing demonstration of the historical UFC prediction model."""

import html
import logging
from pathlib import Path

import streamlit as st

from combat_iq.data import load_fighters
from combat_iq.prediction import predict

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Combat IQ · Fight predictions", page_icon="🥊", layout="wide")
st.html(f"<style>{(HERE / 'style.css').read_text()}</style>")
st.html("""<div class="masthead"><span class="brand">COMBAT<span>IQ</span></span>
<span class="edition">THE MATCHUP LAB &nbsp; / &nbsp; VOL. 01</span></div>
<div class="hero"><div class="hero-copy">
<div class="eyebrow">MACHINE LEARNING. MEET THE OCTAGON.</div>
<h1>The fight,<br>by the <em>numbers.</em></h1>
<p>Two fighters. Their track records. One model’s take.<br>
Explore how a historical model reads the matchup.</p></div>
<div class="octagon" aria-hidden="true"><span>VS</span></div></div>""")

try:
    fighters = load_fighters()
except (OSError, ValueError):
    logger.exception("Fighter data unavailable")
    st.error("The fighter roster is temporarily unavailable. Please try again later.")
    st.stop()

names = sorted(fighters.index.tolist())
st.html(f"""<div class="ribbon"><span><b>{len(names):,}</b> fighters in the archive</span>
<span><b>30</b> model inputs</span><span><b>2021</b> historical data cutoff</span></div>""")
st.html(
    '<div class="section-heading"><span>01 / BUILD YOUR MATCHUP</span>'
    "<span>PICK YOUR CORNERS</span></div>"
)

# Selectors rerun freely; inference only runs on submission. Old results are
# hidden as soon as either fighter changes, avoiding a misleading matchup.
red_col, blue_col = st.columns(2, gap="large")
with red_col:
    st.html('<div class="corner red">RED CORNER</div>')
    red = st.selectbox("Red corner fighter", names, index=names.index("Conor McGregor"), key="red")
with blue_col:
    st.html('<div class="corner blue">BLUE CORNER</div>')
    blue = st.selectbox(
        "Blue corner fighter", names, index=names.index("Khabib Nurmagomedov"), key="blue"
    )


def fighter_card(name: str, color: str) -> None:
    stats = fighters.loc[name]
    st.html(f"""<div class="fighter-card {color}-card"><h3>{html.escape(name)}</h3>
<div class="record"><strong>{int(stats.wins)}</strong> W <span>/</span>
<strong>{int(stats.losses)}</strong> L <span>/</span><strong>{int(stats.draw)}</strong> D</div>
<div class="stat-row"><span>KO / TKO wins <b>{int(stats["win_by_KO/TKO"])}</b></span>
<span>Submission wins <b>{int(stats.win_by_Submission)}</b></span></div>
</div>""")


with red_col:
    fighter_card(red, "red")
with blue_col:
    fighter_card(blue, "blue")

st.caption("Historical demo · records through March 2021 · not current fight odds.")

if red == blue:
    st.warning("Choose two different fighters to build a matchup.")

if st.button(
    "Predict this matchup  →",
    type="primary",
    use_container_width=True,
    disabled=red == blue,
):
    with st.spinner("Reading the matchup… The first prediction may take a few seconds."):
        try:
            result = predict(red, blue)
        except Exception:
            logger.exception("Prediction failed")
            st.session_state.pop("result", None)
            st.error("The prediction is temporarily unavailable. Please try again later.")
        else:
            st.session_state.result = {"matchup": (red, blue), "prediction": result}

saved = st.session_state.get("result")
if saved and saved["matchup"] == (red, blue):
    result = saved["prediction"]
    winner = result["winner"]
    confidence = result.get("model_confidence", result["confidence_rate"])
    corner = "red" if winner == red else "blue"
    st.html(f"""<section class="result {corner}-result" aria-label="Prediction result">
<div><div class="eyebrow">THE MODEL’S PICK · {corner.upper()} CORNER</div>
<h2>{html.escape(winner)}</h2><p>Model pick</p></div>
<div class="probability"><strong>{confidence:.1%}</strong>
<span>model confidence</span></div>
</section>""")
    st.progress(confidence, text=f"{winner} · {confidence:.1%} model confidence")
    if corner == "blue":
        st.caption(
            f"This score refers to {red} not winning, which includes a draw. "
            f"{blue} is shown as the model pick; this is not a measured chance of a blue win."
        )
    else:
        st.caption(
            f"This score refers to a red-corner win for {red}. "
            "It has not been calibrated as a real-world win probability."
        )
else:
    st.html("""<div class="empty-result"><span>YOUR NEXT MATCHUP STARTS HERE</span>
<p>Choose two fighters, then let the numbers weigh in.</p></div>""")

st.html(
    '<div class="section-heading"><span>02 / BEHIND THE PREDICTION</span>'
    "<span>CONTEXT MATTERS</span></div>"
)
with st.expander("How does Combat IQ make its prediction?"):
    st.markdown("""The model compares **30 inputs**: fighter names and 14 historical statistics
per corner, including wins, losses, streaks, title bouts, and methods of victory.
A **CatBoost classifier** with 2,500 trees (depth 5, learning rate 0.04)
returns a predicted class and a score, displayed here as **model confidence**.

This demonstration uses a saved model and statistics through **March 2021**. Records
on this page reflect that dataset, not fighters’ current records. Injuries, preparation,
recent fights, and betting markets are not included.

The original training target groups every non-red outcome, including draws, into
one class. This interface presents that class as the blue-corner prediction. The
percentages have not been independently calibrated as real-world win probabilities.
For example, a 60% score has not been shown to correspond to wins in 60% of cases.

**Validation status:** automated tests check software behavior and consistency with
saved-model outputs. They do not measure predictive accuracy. This portfolio version
has not independently verified the original evaluation, audited feature timing for
future-information leakage, or established performance on later, unseen fights.
No validated accuracy score is claimed.
""")
with st.expander("About the project"):
    st.markdown("""**Combat IQ** is a personal machine-learning project by **Clément Guinel**.
It connects historical fight data, feature preparation, model inference, an API,
and this interactive demo.

The app is an independent educational project and is not affiliated with UFC.
""")
st.html("""<footer><span class="brand">COMBAT<span>IQ</span></span>
<span>DATA INTO DECISIONS. &nbsp; A PROJECT PRESENTED BY CLÉMENT GUINEL.</span></footer>""")
