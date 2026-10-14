
from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained Naive Bayes model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")
model = joblib.load(MODEL_PATH)

# IMPORTANT:
# These mappings assume LabelEncoder-style alphabetical encoding.
# Change them if your training code used different mappings.
EDUCATION = {
    "0": "Bachelors",
    "1": "Masters",
    "2": "PHD"
}

CITY = {
    "0": "Bangalore",
    "1": "New Delhi",
    "2": "Pune"
}

GENDER = {
    "0": "Female",
    "1": "Male"
}

BENCHED = {
    "0": "No",
    "1": "Yes"
}

FEATURES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain"
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Employee Insights | Naive Bayes</title>

<style>
:root {
    --bg: #f4f7fb;
    --surface: #ffffff;
    --text: #172033;
    --muted: #697586;
    --border: #e1e7f0;
    --primary: #5b5ce2;
    --primary-hover: #4849c9;
    --input: #ffffff;
    --shadow: 0 15px 45px rgba(29, 45, 80, 0.08);
}

body.dark {
    --bg: #0c1220;
    --surface: #151e30;
    --text: #edf2ff;
    --muted: #a5b0c4;
    --border: #2b3850;
    --primary: #9293ff;
    --primary-hover: #7d7eff;
    --input: #101829;
    --shadow: 0 15px 45px rgba(0, 0, 0, 0.22);
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
    transition: background 0.25s, color 0.25s;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 7%;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}

.brand {
    font-size: 19px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.brand span {
    color: var(--primary);
}

.theme-btn {
    background: var(--input);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 14px;
}

.container {
    width: min(1050px, 92%);
    margin: 42px auto;
}

.hero {
    margin-bottom: 28px;
}

.eyebrow {
    display: inline-block;
    color: var(--primary);
    background: color-mix(in srgb, var(--primary) 12%, transparent);
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

h1 {
    margin: 15px 0 10px;
    font-size: clamp(28px, 4vw, 40px);
    letter-spacing: -1.2px;
}

.hero p {
    color: var(--muted);
    line-height: 1.7;
    max-width: 650px;
}

.layout {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 22px;
    align-items: start;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 26px;
    box-shadow: var(--shadow);
}

.card h2 {
    margin: 0 0 7px;
    font-size: 20px;
}

.subtitle {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin: 0 0 24px;
}

.form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 19px;
}

.field {
    min-width: 0;
}

label {
    display: block;
    font-size: 13px;
    font-weight: 650;
    margin-bottom: 8px;
}

input, select {
    width: 100%;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    background: var(--input);
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
}

input:focus, select:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px color-mix(
        in srgb, var(--primary) 15%, transparent
    );
}

.hint {
    color: var(--muted);
    font-size: 11px;
    margin-top: 6px;
    line-height: 1.5;
}

.submit-btn {
    width: 100%;
    margin-top: 24px;
    padding: 14px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 11px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    transition: 0.2s;
}

.submit-btn:hover {
    background: var(--primary-hover);
    transform: translateY(-1px);
}

.result-card {
    position: sticky;
    top: 20px;
}

.result-icon {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    background: color-mix(in srgb, var(--primary) 13%, transparent);
    font-size: 24px;
    margin-bottom: 20px;
}

.result-value {
    font-size: 28px;
    font-weight: 800;
    margin: 14px 0;
    overflow-wrap: anywhere;
}

.result-message {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.metric {
    margin-top: 25px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}

.metric-row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 13px;
    margin-bottom: 10px;
}

.bar {
    height: 8px;
    background: var(--border);
    border-radius: 20px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 20px;
    transition: width 0.3s;
}

.alert {
    margin-bottom: 20px;
    padding: 13px 15px;
    border-radius: 10px;
    background: #fff0f0;
    color: #b42318;
    font-size: 13px;
    overflow-wrap: anywhere;
}

body.dark .alert {
    background: #3a2028;
    color: #ffb4b4;
}

.note {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
    margin-top: 22px;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding: 25px 10px 35px;
}

@media (max-width: 760px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .result-card {
        position: static;
    }

    .container {
        margin-top: 28px;
    }

    .card {
        padding: 20px;
    }
}

@media (max-width: 420px) {
    .form-grid {
        grid-template-columns: 1fr;
    }

    .navbar {
        padding: 17px 5%;
    }
}
</style>
</head>

<body>

<header class="navbar">
    <div class="brand">Employee<span>Insights.</span></div>
    <button class="theme-btn" id="themeToggle"
            type="button" onclick="toggleTheme()">
        🌙 Dark mode
    </button>
</header>

<main class="container">

    <section class="hero">
        <span class="eyebrow">MACHINE LEARNING PROJECT</span>
        <h1>Employee Prediction Dashboard</h1>
        <p>
            Enter employee details to generate a prediction
            using your trained Gaussian Naive Bayes model.
            Explore the interface in either light or dark mode.
        </p>
    </section>

    {% if error %}
        <div class="alert">{{ error }}</div>
    {% endif %}

    <div class="layout">

        <section class="card">
            <h2>Employee Details</h2>
            <p class="subtitle">
                Complete all eight fields to run your prediction.
            </p>

            <form method="POST" action="/predict">

                <div class="form-grid">

                    <div class="field">
                        <label for="Education">Education</label>
                        <select id="Education" name="Education" required>
                            <option value="">Select education</option>
                            {% for key, value in education.items() %}
                            <option value="{{ key }}"
                            {% if values.get('Education') == key %}selected{% endif %}>
                                {{ value }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="field">
                        <label for="JoiningYear">Joining Year</label>
                        <input id="JoiningYear" name="JoiningYear"
                               type="number" min="2000" max="2035"
                               placeholder="e.g. 2018"
                               value="{{ values.get('JoiningYear', '') }}"
                               required>
                    </div>

                    <div class="field">
                        <label for="City">City</label>
                        <select id="City" name="City" required>
                            <option value="">Select city</option>
                            {% for key, value in city.items() %}
                            <option value="{{ key }}"
                            {% if values.get('City') == key %}selected{% endif %}>
                                {{ value }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="field">
                        <label for="PaymentTier">Payment Tier</label>
                        <select id="PaymentTier" name="PaymentTier" required>
                            <option value="">Select tier</option>
                            {% for tier in [1, 2, 3] %}
                            <option value="{{ tier }}"
                            {% if values.get('PaymentTier') == tier|string %}selected{% endif %}>
                                Tier {{ tier }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="field">
                        <label for="Age">Age</label>
                        <input id="Age" name="Age" type="number"
                               min="18" max="100"
                               placeholder="e.g. 27"
                               value="{{ values.get('Age', '') }}"
                               required>
                    </div>

                    <div class="field">
                        <label for="Gender">Gender</label>
                        <select id="Gender" name="Gender" required>
                            <option value="">Select gender</option>
                            {% for key, value in gender.items() %}
                            <option value="{{ key }}"
                            {% if values.get('Gender') == key %}selected{% endif %}>
                                {{ value }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="field">
                        <label for="EverBenched">Ever Benched</label>
                        <select id="EverBenched" name="EverBenched" required>
                            <option value="">Select option</option>
                            {% for key, value in benched.items() %}
                            <option value="{{ key }}"
                            {% if values.get('EverBenched') == key %}selected{% endif %}>
                                {{ value }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="field">
                        <label for="ExperienceInCurrentDomain">
                            Domain Experience
                        </label>
                        <input id="ExperienceInCurrentDomain"
                               name="ExperienceInCurrentDomain"
                               type="number" min="0" max="50"
                               placeholder="Years of experience"
                               value="{{ values.get('ExperienceInCurrentDomain', '') }}"
                               required>
                    </div>

                </div>

                <button type="submit" class="submit-btn">
                    Run Prediction →
                </button>

            </form>

            <p class="note">
                Note: The category encodings must match those used
                when training your model. This interface does not
                retrain the model.
            </p>
        </section>

        <aside class="card result-card">
            <div class="result-icon">✦</div>
            <h2>Prediction Result</h2>

            {% if prediction is not none %}

                <div class="result-value">
                    Class {{ prediction }}
                </div>

                <p class="result-message">
                    The model returned class {{ prediction }}.
                    Interpret this class using the target-label
                    mapping from your original training dataset.
                </p>

                {% if probabilities %}
                <div class="metric">
                    <div class="metric-row">
                        <span>Class {{ probabilities[0][0] }}</span>
                        <strong>{{ probabilities[0][1] }}%</strong>
                    </div>
                    <div class="bar">
                        <div class="bar-fill"
                             style="width: {{ probabilities[0][1] }}%">
                        </div>
                    </div>

                    <div class="metric-row" style="margin-top:20px">
                        <span>Class {{ probabilities[1][0] }}</span>
                        <strong>{{ probabilities[1][1] }}%</strong>
                    </div>
                    <div class="bar">
                        <div class="bar-fill"
                             style="width: {{ probabilities[1][1] }}%">
                        </div>
                    </div>
                </div>
                {% endif %}

            {% else %}

                <div class="result-value">Ready to analyze</div>
                <p class="result-message">
                    Your prediction will appear here after you
                    complete the employee form and click
                    <strong>Run Prediction</strong>.
                </p>

                <div class="metric">
                    <div class="metric-row">
                        <span>Model</span>
                        <strong>GaussianNB</strong>
                    </div>
                    <div class="metric-row">
                        <span>Input features</span>
                        <strong>8</strong>
                    </div>
                    <div class="metric-row">
                        <span>Output classes</span>
                        <strong>2</strong>
                    </div>
                </div>

            {% endif %}
        </aside>

    </div>
</main>

<footer class="footer">
    Employee Insights · Flask & Machine Learning
</footer>

<script>
function applyTheme(theme) {
    document.body.classList.toggle("dark", theme === "dark");

    const button = document.getElementById("themeToggle");
    button.textContent =
        theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode";
}

function toggleTheme() {
    const nextTheme =
        document.body.classList.contains("dark") ? "light" : "dark";

    try {
        localStorage.setItem("employee-theme", nextTheme);
    } catch (error) {
        // Theme still works if browser storage is unavailable.
    }

    applyTheme(nextTheme);
}

try {
    applyTheme(localStorage.getItem("employee-theme") || "light");
} catch (error) {
    applyTheme("light");
}
</script>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML,
        education=EDUCATION,
        city=CITY,
        gender=GENDER,
        benched=BENCHED,
        prediction=None,
        probabilities=None,
        values={},
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()

    try:
        # Validate category selections
        if values["Education"] not in EDUCATION:
            raise ValueError("Please select a valid education level.")

        if values["City"] not in CITY:
            raise ValueError("Please select a valid city.")

        if values["Gender"] not in GENDER:
            raise ValueError("Please select a valid gender.")

        if values["EverBenched"] not in BENCHED:
            raise ValueError("Please select a valid bench status.")

        # Convert inputs into the numeric format expected by the model
        input_data = {
            "Education": int(values["Education"]),
            "JoiningYear": int(values["JoiningYear"]),
            "City": int(values["City"]),
            "PaymentTier": int(values["PaymentTier"]),
            "Age": int(values["Age"]),
            "Gender": int(values["Gender"]),
            "EverBenched": int(values["EverBenched"]),
            "ExperienceInCurrentDomain": int(
                values["ExperienceInCurrentDomain"]
            )
        }

        # Basic input validation
        if not 2000 <= input_data["JoiningYear"] <= 2035:
            raise ValueError("Please enter a valid joining year.")

        if not 18 <= input_data["Age"] <= 100:
            raise ValueError("Age must be between 18 and 100.")

        if input_data["PaymentTier"] not in [1, 2, 3]:
            raise ValueError("Payment tier must be 1, 2, or 3.")

        if not 0 <= input_data["ExperienceInCurrentDomain"] <= 50:
            raise ValueError("Please enter valid experience in years.")

        # Preserve the exact training feature order
        df = pd.DataFrame([input_data], columns=FEATURES)

        result = model.predict(df)[0]
        prediction = result.item() if hasattr(result, "item") else result

        probabilities = None

        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(df)[0]

            probabilities = [
                (
                    cls.item() if hasattr(cls, "item") else cls,
                    round(float(prob) * 100, 2)
                )
                for cls, prob in zip(model.classes_, raw_probabilities)
            ]

        return render_template_string(
            HTML,
            education=EDUCATION,
            city=CITY,
            gender=GENDER,
            benched=BENCHED,
            prediction=prediction,
            probabilities=probabilities,
            values=values,
            error=None
        )

    except (ValueError, TypeError) as e:
        return render_template_string(
            HTML,
            education=EDUCATION,
            city=CITY,
            gender=GENDER,
            benched=BENCHED,
            prediction=None,
            probabilities=None,
            values=values,
            error=str(e)
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template_string(
            HTML,
            education=EDUCATION,
            city=CITY,
            gender=GENDER,
            benched=BENCHED,
            prediction=None,
            probabilities=None,
            values=values,
            error=(
                "Prediction failed. Check that your input encoding "
                "and model features match the training data."
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
