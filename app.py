from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

from flask import Flask, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "syndrome.db"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )


class Predictor:
    def __init__(self) -> None:
        self.model_error: Optional[str] = None
        self.logistic_vgg16 = None
        self.lgbm_vgg16 = None
        self.nmf_vgg16 = None
        self.vgg_feature_extractor = None
        self._keras = None
        self._np = None
        self._load_models()

    def _load_models(self) -> None:
        """
        Load ML dependencies lazily so the web app can still run even when
        inference packages/artifacts are unavailable in the host environment.
        """
        try:
            import joblib
            import numpy as np
            from tensorflow.keras.applications.vgg16 import VGG16
            from tensorflow.keras.layers import Flatten
            from tensorflow.keras.models import Model

            self._np = np
            self._keras = {
                "preprocess_input": __import__(
                    "tensorflow.keras.applications.vgg16", fromlist=["preprocess_input"]
                ).preprocess_input,
                "load_img": __import__(
                    "tensorflow.keras.preprocessing.image", fromlist=["load_img"]
                ).load_img,
                "img_to_array": __import__(
                    "tensorflow.keras.preprocessing.image", fromlist=["img_to_array"]
                ).img_to_array,
            }

            self.logistic_vgg16 = joblib.load(BASE_DIR / "logistic_vgg16.joblib")
            self.lgbm_vgg16 = joblib.load(BASE_DIR / "lgbm_vgg16.joblib")
            self.nmf_vgg16 = joblib.load(BASE_DIR / "nmf_vgg16.joblib")

            vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
            self.vgg_feature_extractor = Model(
                inputs=vgg16_base.input,
                outputs=Flatten()(vgg16_base.output),
            )
        except Exception as exc:
            self.model_error = str(exc)

    def predict_vnl_net(self, image_path: Path) -> str:
        if self.model_error:
            return f"Prediction unavailable: {self.model_error}"

        load_img = self._keras["load_img"]
        img_to_array = self._keras["img_to_array"]
        preprocess_input = self._keras["preprocess_input"]
        np = self._np

        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        vgg_features = self.vgg_feature_extractor.predict(img_array, verbose=0)
        nmf_features = self.nmf_vgg16.transform(vgg_features)
        enhanced_features = self.lgbm_vgg16.predict_proba(nmf_features)[:, 1].reshape(-1, 1)
        pred = self.logistic_vgg16.predict(enhanced_features)

        return "Normal Kid" if int(pred[0]) == 1 else "Has Down Syndrome"


predictor = Predictor()
init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        c_password = request.form["c_password"]

        if password != c_password:
            return render_template("register.html", message="Passwords do not match.")

        with get_db() as conn:
            exists = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
            if exists:
                return render_template("register.html", message="Email already exists.")

            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password),
            )

        return render_template("login.html", message="Registered successfully. Please log in.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        with get_db() as conn:
            user = conn.execute(
                "SELECT id, email FROM users WHERE email = ? AND password = ?",
                (email, password),
            ).fetchone()

        if user:
            return redirect(url_for("home"))

        return render_template("login.html", message="Invalid email or password.")

    return render_template("login.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/algorithm", methods=["GET", "POST"])
def algorithm():
    if request.method == "POST":
        file = request.files.get("file")
        algorithm_name = request.form.get("algorithm", "")

        if not file or not file.filename:
            return render_template("algorithm.html", message="Please upload an image.")

        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)

        if algorithm_name != "vnl_net":
            prediction = "Only vnl_net is available with the current repository assets."
        else:
            prediction = predictor.predict_vnl_net(file_path)

        return render_template(
            "algorithm.html",
            prediction=prediction,
            image_name=filename,
            model_error=predictor.model_error,
        )

    return render_template("algorithm.html", model_error=predictor.model_error)


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/graph")
def graph():
    return render_template("graph.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
