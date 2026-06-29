import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))

from flask import Flask, request, jsonify, render_template
from generate import chat as generate

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        response = generate(
            prompt,
            max_new=data.get("max_new"),
            temp=data.get("temp"),
            top_k=data.get("top_k"),
            top_p=data.get("top_p"),
            rep_pen=data.get("rep_pen"),
        )
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
