from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import csv, os, requests

# =========================
# Flask setup
# =========================
app = Flask(__name__, static_folder='public')
CORS(app)

# -------------------------
# Health check
# -------------------------
@app.get('/api/health')
def health():
    return jsonify(ok=True)

# -------------------------
# Static file serving
# -------------------------
@app.get('/')
def root():
    return send_from_directory('public', 'index.html')

@app.get('/<path:filename>')
def public_files(filename):
    return send_from_directory('public', filename)

# -------------------------
# Teams webhook (optional)
# -------------------------
TEAMS_WEBHOOK = os.environ.get('TEAMS_WEBHOOK', '').strip()

def notify_teams(message: str):
    if not TEAMS_WEBHOOK:
        return
    try:
        requests.post(TEAMS_WEBHOOK, json={"text": message}, timeout=4)
    except Exception as e:
        print('Teams notify failed:', e)

# =========================
# Invoice submission
# =========================
@app.post('/api/submit')
def submit():
    data = request.get_json()

    if not data:
        return jsonify({"ok": False, "error": "No data received"}), 400

    try:
        # Decide CSV filename based on invoice type
        invoice_type = data.get("type", "general").lower()
        if invoice_type == "retail":
            csv_file = os.path.join("data", "retail_invoices.csv")
        elif invoice_type == "business":
            csv_file = os.path.join("data", "business_invoices.csv")
        elif invoice_type == "non-chargeable":
            csv_file = os.path.join("data", "non-chargeable_invoices.csv")
        elif invoice_type == "project":
            csv_file = os.path.join("data", "project_invoices.csv")
        elif invoice_type == "vip":
            csv_file = os.path.join("data", "vip_invoices.csv")
        else:
            csv_file = os.path.join("data", "general_invoices.csv")

        os.makedirs("data", exist_ok=True)

        # Flatten "totals" dict if it exists
        totals = data.pop("totals", {})
        if isinstance(totals, dict):
            for key, val in totals.items():
                data[f"total_{key}"] = val

        # Ensure consistent headers
        fieldnames = list(data.keys())
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)

        # Optional Teams notification
        notify_teams(f"New {invoice_type} invoice submitted by {data.get('engineer','Unknown')}")

        return jsonify({"ok": True, "message": f"Invoice submitted ✓ ({invoice_type})"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# =========================
# Run server
# =========================
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3001, debug=True)


