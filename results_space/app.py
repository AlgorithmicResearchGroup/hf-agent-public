import html
import json
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import HfApi


def load_delivery(bucket_id: str, prefix: str) -> dict:
    cleaned_bucket = (bucket_id or "").strip()
    cleaned_prefix = (prefix or "").strip().strip("/")
    if not cleaned_bucket or not cleaned_prefix:
        raise gr.Error("Provide both a bucket and a prefix.")

    delivery_relpath = f"{cleaned_prefix}/delivery.json"
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "delivery.json"
        HfApi().download_bucket_files(
            cleaned_bucket,
            files=[(delivery_relpath, target_path)],
            raise_on_missing_files=True,
        )
        return json.loads(target_path.read_text())


def render_delivery(delivery: dict) -> str:
    title = html.escape(delivery.get("title", "HF Agent Results"))
    status = html.escape(delivery.get("status", "unknown"))
    created_at = html.escape(delivery.get("created_at", ""))

    links = []
    if delivery.get("results_page_url"):
        links.append(("Open Results", delivery["results_page_url"]))
    if delivery.get("report_view_url"):
        links.append(("Open Report", delivery["report_view_url"]))
    if delivery.get("report_download_url"):
        links.append(("Download Report", delivery["report_download_url"]))
    if delivery.get("folder_url"):
        links.append(("Open Run Folder", delivery["folder_url"]))
    if delivery.get("job_url"):
        links.append(("Open Job", delivery["job_url"]))

    buttons = "".join(
        f'<a class="cta" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">{html.escape(label)}</a>'
        for label, url in links
    )

    meta_rows = [
        ("Status", status),
        ("QA Passed", "yes" if delivery.get("qa_passed") else "no"),
        ("Created", created_at or "unknown"),
        ("Bucket", html.escape(delivery.get("bucket_id", ""))),
        ("Prefix", html.escape(delivery.get("bucket_prefix", ""))),
    ]
    meta = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in meta_rows)

    fallback = "".join(
        f"<li><strong>{html.escape(label)}:</strong> <a href=\"{html.escape(url)}\" target=\"_blank\" rel=\"noopener noreferrer\">{html.escape(url)}</a></li>"
        for label, url in links
    )

    return f"""
    <div class="page">
      <div class="hero">
        <p class="eyebrow">Hugging Face Agent Results</p>
        <h1>{title}</h1>
        <p class="lede">One clean page for the final report and the underlying run artifacts.</p>
        <div class="actions">{buttons}</div>
      </div>
      <div class="grid">
        <div class="card">
          <h2>Run Summary</h2>
          <table>{meta}</table>
        </div>
        <div class="card">
          <h2>Fallback Links</h2>
          <ul>{fallback}</ul>
        </div>
      </div>
    </div>
    """


def fetch_and_render(bucket_id: str, prefix: str) -> str:
    return render_delivery(load_delivery(bucket_id, prefix))


def load_from_request(request: gr.Request):
    bucket = ""
    prefix = ""
    if request:
        bucket = request.query_params.get("bucket", "")
        prefix = request.query_params.get("prefix", "")

    if bucket and prefix:
        try:
            return bucket, prefix, fetch_and_render(bucket, prefix)
        except Exception as exc:
            return bucket, prefix, f"<div class='error'>Unable to load delivery.json: {html.escape(str(exc))}</div>"

    return (
        bucket,
        prefix,
        "<div class='empty'>Enter a bucket and prefix, or open this Space with <code>?bucket=&lt;bucket&gt;&amp;prefix=&lt;prefix&gt;</code>.</div>",
    )


CSS = """
body { background: linear-gradient(180deg, #f3f5f7 0%, #e7edf2 100%); }
.page { max-width: 980px; margin: 0 auto; padding: 24px; color: #0f1720; font-family: ui-sans-serif, system-ui, sans-serif; }
.hero { background: white; border-radius: 24px; padding: 28px; box-shadow: 0 12px 30px rgba(15, 23, 32, 0.08); }
.eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: #4f6b7a; margin: 0 0 8px 0; }
.hero h1 { margin: 0 0 12px 0; font-size: 36px; line-height: 1.1; }
.lede { margin: 0; color: #3f5563; font-size: 16px; }
.actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }
.cta { display: inline-block; padding: 12px 16px; border-radius: 999px; text-decoration: none; color: white; background: #0d6efd; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-top: 18px; }
.card { background: white; border-radius: 20px; padding: 20px; box-shadow: 0 10px 24px rgba(15, 23, 32, 0.06); }
.card h2 { margin-top: 0; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 0; text-align: left; vertical-align: top; border-bottom: 1px solid #e8eef2; }
th { width: 120px; color: #4f6b7a; font-weight: 600; }
ul { padding-left: 18px; }
li { margin-bottom: 10px; }
.empty, .error { background: white; border-radius: 18px; padding: 20px; box-shadow: 0 10px 24px rgba(15, 23, 32, 0.06); }
.error { color: #8a1c1c; }
"""


with gr.Blocks(css=CSS, title="HF Agent Results") as demo:
    gr.Markdown("# HF Agent Results")
    gr.Markdown("Open this Space with a `bucket` and `prefix`, or enter them below.")

    with gr.Row():
        bucket_input = gr.Textbox(label="Bucket", placeholder="username/hf-agent")
        prefix_input = gr.Textbox(label="Prefix", placeholder="runs/20260327-084759")

    load_button = gr.Button("Load Results")
    html_output = gr.HTML()

    demo.load(load_from_request, outputs=[bucket_input, prefix_input, html_output])
    load_button.click(fetch_and_render, inputs=[bucket_input, prefix_input], outputs=html_output)


if __name__ == "__main__":
    demo.launch()
