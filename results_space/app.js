const DEFAULT_BUCKET = "matthewkenney/hf-agent";

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(message) {
  document.getElementById("status-message").innerHTML = message;
}

function linkButton(label, url) {
  if (!url) {
    return "";
  }
  return `<a class="cta" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
}

function renderReports(reports, highlightedPrefix) {
  const container = document.getElementById("reports-grid");
  if (!reports.length) {
    container.innerHTML = "<article class='card empty-card'><h3>No reports yet</h3><p>Run the job once and the completed report will appear here.</p></article>";
    document.getElementById("report-count").textContent = "0 reports";
    return;
  }

  document.getElementById("report-count").textContent = `${reports.length} report${reports.length === 1 ? "" : "s"}`;
  container.innerHTML = reports
    .map((report) => {
      const highlighted = report.bucket_prefix === highlightedPrefix ? " report-card-highlighted" : "";
      return `
        <article class="card report-card${highlighted}">
          <div class="report-topline">
            <p class="report-prefix">${escapeHtml(report.bucket_prefix || "")}</p>
            <span class="status-pill">${escapeHtml(report.status || "unknown")}</span>
          </div>
          <h3>${escapeHtml(report.title || "Untitled report")}</h3>
          <p class="report-meta">Created: ${escapeHtml(report.created_at || "unknown")}</p>
          <div class="actions">
            ${linkButton("Open Report", report.report_view_url)}
            ${linkButton("Download", report.report_download_url)}
            ${linkButton("Open Folder", report.folder_url)}
            ${linkButton("Job", report.job_url)}
          </div>
        </article>
      `;
    })
    .join("");
}

async function loadReportIndex() {
  const params = new URLSearchParams(window.location.search);
  const bucket = params.get("bucket") || DEFAULT_BUCKET;
  const prefix = params.get("prefix");
  document.getElementById("title").textContent = "Report Index";
  document.getElementById("lede").textContent = `Showing completed reports for ${bucket}.`;

  try {
    setStatus("Loading report index…");
    const response = await fetch("./reports.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`reports.json not available (${response.status})`);
    }

    const payload = await response.json();
    const reports = (payload.reports || []).filter((report) => (report.bucket_id || DEFAULT_BUCKET) === bucket);
    if (prefix) {
      reports.sort((a, b) => {
        if (a.bucket_prefix === prefix) return -1;
        if (b.bucket_prefix === prefix) return 1;
        return (b.created_at || "").localeCompare(a.created_at || "");
      });
    }
    renderReports(reports, prefix);
    setStatus(
      prefix
        ? `Showing reports for ${escapeHtml(bucket)}. Highlighted run: <code>${escapeHtml(prefix)}</code>.`
        : `Showing all reports for ${escapeHtml(bucket)}.`
    );
  } catch (error) {
    renderReports([], prefix);
    setStatus(`Unable to load the report index yet. ${escapeHtml(error.message)}`);
  }
}

loadReportIndex();
