from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Deferred global reference to avoid importing heavy libraries on startup
_engine = None


def get_engine():
    """Lazily load audit module and initialize the Engine on first use."""
    global _engine
    if _engine is None:
        import audit
        _engine = audit.Engine()
        _engine.warm_up()
    return _engine


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models")
def models():
    import audit
    include_all = request.args.get("all", "0") in ("1", "true", "yes")
    return jsonify({
        "models": audit.model_catalog(include_all=include_all),
        "seeds": audit.SEEDS,
        "lira": audit.lira_models(include_all=include_all)
    })


def _run(single_method_name, average_method_name):
    import audit
    model_id = request.args.get("model", "baseline_e50")
    raw = request.args.get("seed", "1")
    
    if model_id not in {m["id"] for m in audit.model_catalog(include_all=True)}:
        return jsonify({"error": "unknown model"}), 400
        
    try:
        engine = get_engine()
        single = getattr(engine, single_method_name)
        average = getattr(engine, average_method_name)
        
        if raw == "avg":
            return jsonify(average(model_id))
            
        seed = int(raw)
        if seed not in audit.SEEDS:
            return jsonify({"error": "unknown seed"}), 400
            
        return jsonify(single(model_id, seed))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/audit")
def run_audit():
    return _run("audit", "audit_avg")


@app.route("/api/lira")
def run_lira():
    return _run("lira", "lira_avg")


@app.route("/api/summary")
def summary():
    import audit
    include_all = request.args.get("all", "0") in ("1", "true", "yes")
    return jsonify(audit.summary(include_all=include_all))


@app.route("/api/extras")
def extras():
    import audit
    return jsonify(audit.extras())


if __name__ == "__main__":
    print("Initializing engine and warming up attack model...")
    get_engine()
    print("Ready. Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)