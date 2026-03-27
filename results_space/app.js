function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function bucketResolveUrl(bucket, prefix, path) {
  const bucketParts = bucket.split("/").map(encodeURIComponent).join("/");
  const prefixParts = prefix.split("/").filter(Boolean).map(encodeURIComponent).join("/");
  const fileParts = path.split("/").filter(Boolean).map(encodeURIComponent).join("/");
  return `https://huggingface.co/buckets/${bucketParts}/resolve/${prefixParts}/${fileParts}`;
}

function setStatus(message) {
  document.getElementById("status-message").innerHTML = message;
}

function renderLinks(delivery) {
  const links = [
    ["Open Report", delivery.report_view_url],
    ["Download Report", delivery.report_download_url],
    ["Open Run Folder", delivery.folder_url],
    ["Open Job", delivery.job_url],
  ].filter(([, url]) => url);

  const actions = document.getElementById("actions");
  actions.innerHTML = links
    .map(([label, url]) => `<a class="cta" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`)
    .join("");

  const fallback = document.getElementById("fallback-links");
  fallback.innerHTML = links
    .map(([label, url]) => `<li><strong>${escapeHtml(label)}:</strong> <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a></li>`)
    .join("");
}

function renderSummary(delivery) {
  const rows = [
    ["Status", delivery.status || "unknown"],
    ["QA Passed", delivery.qa_passed ? "yes" : "no"],
    ["Created", delivery.created_at || "unknown"],
    ["Bucket", delivery.bucket_id || ""],
    ["Prefix", delivery.bucket_prefix || ""],
  ];
  document.getElementById("summary").innerHTML = rows
    .map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`)
    .join("");
}

async function loadDelivery() {
  const params = new URLSearchParams(window.location.search);
  const bucket = params.get("bucket");
  const prefix = params.get("prefix");

  if (!bucket || !prefix) {
    document.getElementById("title").textContent = "HF Agent Results";
    setStatus("Open this Space with <code>?bucket=&lt;bucket&gt;&amp;prefix=&lt;prefix&gt;</code> after a run completes.");
    return;
  }

  const deliveryUrl = bucketResolveUrl(bucket, prefix, "delivery.json");
  try {
    setStatus("Fetching delivery manifest…");
    const response = await fetch(deliveryUrl);
    if (!response.ok) {
      throw new Error(`delivery.json not available yet (${response.status})`);
    }

    const delivery = await response.json();
    document.getElementById("title").textContent = delivery.title || "HF Agent Results";
    document.getElementById("lede").textContent = "This report is now available. Use the buttons below to open the final report or the underlying run artifacts.";
    renderLinks(delivery);
    renderSummary(delivery);
    setStatus(`Loaded delivery manifest from ${escapeHtml(deliveryUrl)}`);
  } catch (error) {
    document.getElementById("title").textContent = "Results not ready yet";
    setStatus(
      `The run may still be executing or uploading artifacts. Retry in a moment, or open the bucket folder directly once it exists.<br><br><code>${escapeHtml(deliveryUrl)}</code><br><br>${escapeHtml(error.message)}`
    );
  }
}

loadDelivery();
