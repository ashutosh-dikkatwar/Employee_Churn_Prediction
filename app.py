
from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "naive_bayes.pkl")
model = joblib.load(MODEL_PATH)

# Input feature order must match model training
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

# Encoded input values and their display labels
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Classification</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg: #f3f6fb;
            --card: #ffffff;
            --text: #172033;
            --muted: #64748b;
            --border: #dce3ed;
            --primary: #2563eb;
            --input: #ffffff;
            --success-bg: #dcfce7;
            --success-text: #166534;
            --warning-bg: #fee2e2;
            --warning-text: #991b1b;
        }

        body.dark {
            --bg: #0f172a;
            --card: #1e293b;
            --text: #f1f5f9;
            --muted: #cbd5e1;
            --border: #475569;
            --primary: #60a5fa;
            --input: #0f172a;
            --success-bg: #14532d;
            --success-text: #dcfce7;
            --warning-bg: #7f1d1d;
            --warning-text: #fee2e2;
        }

        body {
            font-family: Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 30px 16px;
            transition: background 0.2s, color 0.2s;
        }

        .container {
            max-width: 900px;
            margin: auto;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 24px;
        }

        h1 {
            font-size: 28px;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--muted);
            line-height: 1.5;
        }

        button, input, select {
            font: inherit;
        }

        .theme-btn {
            background: var(--card);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 10px;
            cursor: pointer;
            white-space: nowrap;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 26px;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.06);
        }

        h2 {
            font-size: 20px;
            margin-bottom: 20px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 20px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 14px;
            font-weight: 600;
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 9px;
            color: var(--text);
            background: var(--input);
            outline: none;
        }

        input:focus, select:focus {
            border-color: var(--primary);
        }

        .submit-btn {
            width: 100%;
            padding: 14px;
            margin-top: 24px;
            border: none;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            font-weight: bold;
            cursor: pointer;
        }

        .submit-btn:hover {
            opacity: 0.9;
        }

        .result {
            margin-top: 24px;
            padding: 22px;
            border-radius: 12px;
            line-height: 1.6;
        }

        .result.stay {
            background: var(--success-bg);
            color: var(--success-text);
        }

        .result.leave {
            background: var(--warning-bg);
            color: var(--warning-text);
        }

        .result h2 {
            margin-bottom: 8px;
        }

        .result-main {
            font-size: 21px;
            font-weight: bold;
        }

        .details {
            margin-top: 8px;
            font-size: 14px;
        }

        .error {
            margin-top: 20px;
            padding: 14px;
            background: var(--warning-bg);
            color: var(--warning-text);
            border-radius: 10px;
        }

        footer {
            text-align: center;
            color: var(--muted);
            font-size: 13px;
            margin-top: 22px;
        }

        @media (max-width: 600px) {
            body {
                padding: 20px 12px;
            }

            h1 {
                font-size: 23px;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .card {
                padding: 18px;
            }

            .topbar {
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="topbar">
        <div>
            <h1>Employee Classification</h1>
            <p class="subtitle">
                Enter employee details to predict the employee's
                possible retention outcome.
            </p>
        </div>
        <button class="theme-btn" type="button" onclick="toggleTheme()">
            🌙 / ☀️ Theme
        </button>
    </div>

    <div class="card">
        <h2>Employee Information</h2>

        <form method="POST" action="/predict">
            <div class="grid">
                <div class="field">
                    <label for="Education">Education</label>
                    <select id="Education" name="Education" required>
                        <option value="0" {% if form_data.get('Education', '0') == '0' %}selected{% endif %}>Bachelors</option>
                        <option value="1" {% if form_data.get('Education') == '1' %}selected{% endif %}>Masters</option>
                        <option value="2" {% if form_data.get('Education') == '2' %}selected{% endif %}>PhD</option>
                    </select>
                </div>

                <div class="field">
                    <label for="JoiningYear">Joining Year</label>
                    <input type="number" id="JoiningYear" name="JoiningYear"
                           min="2000" max="2100"
                           value="{{ form_data.get('JoiningYear', '2018') }}" required>
                </div>

                <div class="field">
                    <label for="City">City</label>
                    <select id="City" name="City" required>
                        <option value="0" {% if form_data.get('City', '0') == '0' %}selected{% endif %}>Bangalore</option>
                        <option value="1" {% if form_data.get('City') == '1' %}selected{% endif %}>New Delhi</option>
                        <option value="2" {% if form_data.get('City') == '2' %}selected{% endif %}>Pune</option>
                    </select>
                </div>

                <div class="field">
                    <label for="PaymentTier">Payment Tier</label>
                    <select id="PaymentTier" name="PaymentTier" required>
                        <option value="1" {% if form_data.get('PaymentTier', '3') == '1' %}selected{% endif %}>Tier 1</option>
                        <option value="2" {% if form_data.get('PaymentTier') == '2' %}selected{% endif %}>Tier 2</option>
                        <option value="3" {% if form_data.get('PaymentTier', '3') == '3' %}selected{% endif %}>Tier 3</option>
                    </select>
                </div>

                <div class="field">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age"
                           min="18" max="100"
                           value="{{ form_data.get('Age', '25') }}" required>
                </div>

                <div class="field">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="0" {% if form_data.get('Gender', '0') == '0' %}selected{% endif %}>Female</option>
                        <option value="1" {% if form_data.get('Gender') == '1' %}selected{% endif %}>Male</option>
                    </select>
                </div>

                <div class="field">
                    <label for="EverBenched">Ever Benched?</label>
                    <select id="EverBenched" name="EverBenched" required>
                        <option value="0" {% if form_data.get('EverBenched', '0') == '0' %}selected{% endif %}>No</option>
                        <option value="1" {% if form_data.get('EverBenched') == '1' %}selected{% endif %}>Yes</option>
                    </select>
                </div>

                <div class="field">
                    <label for="ExperienceInCurrentDomain">
                        Experience in Current Domain
                    </label>
                    <input type="number" id="ExperienceInCurrentDomain"
                           name="ExperienceInCurrentDomain"
                           min="0" max="60"
                           value="{{ form_data.get('ExperienceInCurrentDomain', '2') }}"
                           required>
                </div>
            </div>

            <button class="submit-btn" type="submit">
                Predict Employee Outcome
            </button>
        </form>

        {% if result %}
        <div class="result {{ result_class }}" role="status">
            <h2>Prediction Result</h2>
            <p class="result-main">{{ result }}</p>
            <p class="details">
                {% if probability is not none %}
                    Model confidence estimate: {{ probability }}%
                {% endif %}
            </p>
            <p class="details">
                This is a machine learning prediction, not a guarantee
                of an employee's future decision.
            </p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error" role="alert">{{ error }}</div>
        {% endif %}
    </div>

    <footer>
        Employee Classification using Machine Learning
    </footer>
</div>

<script>
    function applyTheme(theme) {
        document.body.classList.toggle("dark", theme === "dark");
    }

    function toggleTheme() {
        const dark = !document.body.classList.contains("dark");
        const theme = dark ? "dark" : "light";
        applyTheme(theme);
        try {
            localStorage.setItem("employeeAppTheme", theme);
        } catch (e) {}
    }

    try {
        applyTheme(localStorage.getItem("employeeAppTheme") || "light");
    } catch (e) {
        applyTheme("light");
    }
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TEMPLATE,
        form_data={},
        result=None,
        result_class="",
        probability=None,
        error=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    try:
        # Validate and convert submitted inputs
        education = form_data["Education"]
        city = form_data["City"]
        gender = form_data["Gender"]
        benched = form_data["EverBenched"]

        if education not in EDUCATION:
            raise ValueError("Please select a valid education level.")
        if city not in CITY:
            raise ValueError("Please select a valid city.")
        if gender not in GENDER:
            raise ValueError("Please select a valid gender.")
        if benched not in BENCHED:
            raise ValueError("Please select a valid bench status.")

        joining_year = int(form_data["JoiningYear"])
        payment_tier = int(form_data["PaymentTier"])
        age = int(form_data["Age"])
        experience = int(form_data["ExperienceInCurrentDomain"])

        if not 2000 <= joining_year <= 2100:
            raise ValueError("Please enter a valid joining year.")
        if payment_tier not in (1, 2, 3):
            raise ValueError("Please select a valid payment tier.")
        if not 18 <= age <= 100:
            raise ValueError("Please enter an age between 18 and 100.")
        if not 0 <= experience <= 60:
            raise ValueError("Please enter a valid experience value.")

        # Keep feature names and order consistent with training
        input_data = pd.DataFrame([{
            "Education": int(education),
            "JoiningYear": joining_year,
            "City": int(city),
            "PaymentTier": payment_tier,
            "Age": age,
            "Gender": int(gender),
            "EverBenched": int(benched),
            "ExperienceInCurrentDomain": experience
        }], columns=FEATURES)

        prediction = int(model.predict(input_data)[0])

        # Human-readable result
        # Verify that these labels match your training target encoding.
        if prediction == 1:
            result = "The employee is predicted to leave the company."
            result_class = "leave"
        elif prediction == 0:
            result = "The employee is predicted to stay with the company."
            result_class = "stay"
        else:
            result = f"The model returned an unexpected class: {prediction}"
            result_class = ""

        # Optional confidence estimate, if model supports probabilities
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if prediction in classes:
                probability = round(
                    float(probabilities[classes.index(prediction)]) * 100, 2
                )

        return render_template_string(
            HTML_TEMPLATE,
            form_data=form_data,
            result=result,
            result_class=result_class,
            probability=probability,
            error=None
        )

    except (ValueError, TypeError, KeyError) as exc:
        return render_template_string(
            HTML_TEMPLATE,
            form_data=form_data,
            result=None,
            result_class="",
            probability=None,
            error=str(exc) or "Please check your inputs and try again."
        ), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template_string(
            HTML_TEMPLATE,
            form_data=form_data,
            result=None,
            result_class="",
            probability=None,
            error=(
                "The prediction could not be completed. "
                "Please check the model file and its feature configuration."
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=False)
