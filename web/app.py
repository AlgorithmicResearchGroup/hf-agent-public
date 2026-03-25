import os
import signal
import subprocess
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

current_proc = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    global current_proc
    prompt = request.json["prompt"]

    def stream():
        global current_proc
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["NO_COLOR"] = "1"

        proc = subprocess.Popen(
            ["python", "main.py", prompt],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        current_proc = proc

        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace")
            yield f"data: {text}\n\n"

        exit_code = proc.wait()
        current_proc = None
        yield f"event: done\ndata: exit_code={exit_code}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/stop", methods=["POST"])
def stop():
    global current_proc
    if current_proc and current_proc.poll() is None:
        os.killpg(os.getpgid(current_proc.pid), signal.SIGTERM)
        return jsonify(stopped=True)
    return jsonify(stopped=False)


if __name__ == "__main__":
    app.run(port=5005, debug=True)
