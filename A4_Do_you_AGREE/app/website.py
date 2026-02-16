# app/app.py
import os
import torch
from flask import Flask, render_template, request

from model import load_nli_bundle

# Always resolve checkpoint path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "..", "task2_sbert_snli_softmaxloss.pth")

device = torch.device("cpu")  # keep CPU safe for submission
bundle = load_nli_bundle(CKPT_PATH, device=device)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    premise = ""
    hypothesis = ""

    if request.method == "POST":
        premise = request.form.get("premise", "")
        hypothesis = request.form.get("hypothesis", "")

        if premise.strip() and hypothesis.strip():
            prediction = bundle.predict(premise, hypothesis)

    return render_template(
        "index.html",
        prediction=prediction,
        premise=premise,
        hypothesis=hypothesis
    )

if __name__ == "__main__":
    app.run(debug=True)
