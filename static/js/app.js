/* PlayStore Sentiment Lab: Vanilla ES6+ Frontend (Neo-Brutalism) */

const state = {
  selectedFile: null,
  lastBatchCsvBase64: null,
  batchChart: null,
  scraperChart: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function getSelectedModel() {
  return $("#model-selector").value;
}

const MAX_UPLOAD_BYTES = 1.8 * 1024 * 1024; // Safe under Vercel's 4.5 MB request/response limit
const MAX_BATCH_ROWS = 2000;

async function parseJsonSafe(res) {
  const rawText = await res.text();
  try {
    return JSON.parse(rawText);
  } catch {
    if (res.status === 413) {
      throw new Error(
        "Ukuran file terlalu besar untuk server (batas payload Vercel 4,5 MB). Coba unggah file CSV yang lebih kecil."
      );
    }
    if (res.status === 504) {
      throw new Error(
        "Waktu pemrosesan server habis (timeout). Coba kurangi jumlah baris CSV atau pilih model SVM / Logistic Regression."
      );
    }
    throw new Error(`Server mengembalikan respons error (HTTP ${res.status}).`);
  }
}

async function prepareCsvForUpload(file) {
  if (file.size <= MAX_UPLOAD_BYTES) {
    return { uploadBlob: file, clientTrimmed: false, trimmedRows: null };
  }

  // Read up to 6 MB chunk so very large files do not freeze the browser
  const readLimit = Math.min(file.size, 6 * 1024 * 1024);
  const chunkText = await file.slice(0, readLimit).text();

  const encoder = new TextEncoder();
  const records = [];
  let inQuotes = false;
  let start = 0;
  let totalBytes = 0;

  for (let i = 0; i < chunkText.length; i += 1) {
    const ch = chunkText[i];
    if (ch === '"') {
      if (inQuotes && chunkText[i + 1] === '"') {
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (!inQuotes && ch === "\n") {
      const record = chunkText.slice(start, i + 1);
      const recBytes = encoder.encode(record).length;
      if (records.length > 0 && totalBytes + recBytes > MAX_UPLOAD_BYTES) {
        break;
      }
      records.push(record);
      totalBytes += recBytes;
      start = i + 1;
      if (records.length >= MAX_BATCH_ROWS + 1) {
        break;
      }
    }
  }

  // Need at least header + 1 data row
  if (records.length < 2) {
    throw new Error(
      "File CSV terlalu besar dan baris pertama melebihi batas 1,8 MB. Kurangi kolom atau ukuran file terlebih dahulu."
    );
  }

  const slicedBlob = new Blob([records.join("")], {
    type: "text/csv;charset=utf-8",
  });
  return {
    uploadBlob: slicedBlob,
    clientTrimmed: true,
    trimmedRows: records.length - 1,
  };
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function showToast(message, type = "info") {
  const container = $("#toast-container");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 4500);
}

function stampClassFor(sentiment) {
  if (sentiment === "Positif") return "stamp-positive";
  if (sentiment === "Negatif") return "stamp-negative";
  return "stamp-neutral";
}

function barColorFor(sentiment) {
  if (sentiment === "Positif") return "#00aa5b";
  if (sentiment === "Negatif") return "#ff6b6b";
  return "#e5e0d5";
}

function formatPercent(count, total) {
  if (!total) return "0.0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

function renderDoughnutChart(canvasId, distribution, chartKey) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (state[chartKey]) {
    state[chartKey].destroy();
  }

  state[chartKey] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Positif", "Netral", "Negatif"],
      datasets: [
        {
          data: [
            distribution.Positif || 0,
            distribution.Netral || 0,
            distribution.Negatif || 0,
          ],
          backgroundColor: ["#00aa5b", "#e5e0d5", "#ff6b6b"],
          borderColor: "#111111",
          borderWidth: 2.5,
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "58%",
      animation: { duration: 220 },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 14,
            boxWidth: 14,
            boxHeight: 14,
            color: "#111111",
            font: { family: "IBM Plex Sans", size: 12, weight: "700" },
          },
        },
      },
    },
  });
}

/* =========================================================================
 * Health Check
 * ========================================================================= */
async function checkServerHealth() {
  const box = $("#health-pill");
  const text = $("#health-text");

  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    if (res.ok && data.status === "ok") {
      box.className = "health-box health-ok";
      text.textContent = `${data.models_loaded.length} Model Siap`;
    } else {
      box.className = "health-box health-err";
      text.textContent = "Model Belum Siap";
      showToast("Server menyala, tetapi artefak model belum termuat.", "error");
    }
  } catch {
    box.className = "health-box health-err";
    text.textContent = "Server Offline";
    showToast("Gagal terhubung ke backend API.", "error");
  }
}

/* =========================================================================
 * Tab Navigation (Mouse & Keyboard Arrow Support)
 * ========================================================================= */
function initTabs() {
  const buttons = $$(".tab-btn");
  const panels = $$(".tab-panel");

  function activateTab(target) {
    buttons.forEach((b) => {
      const active = b.dataset.tab === target;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
    });
    panels.forEach((p) => {
      p.classList.toggle("active", p.id === `panel-${target}`);
    });
  }

  buttons.forEach((btn, idx) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    btn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        const next = buttons[(idx + 1) % buttons.length];
        next.focus();
        activateTab(next.dataset.tab);
      } else if (e.key === "ArrowLeft") {
        const prev = buttons[(idx - 1 + buttons.length) % buttons.length];
        prev.focus();
        activateTab(prev.dataset.tab);
      }
    });
  });
}

/* =========================================================================
 * Tab 1: Live Single-Review Analyzer
 * ========================================================================= */
function setSingleState(view, errorMsg = "") {
  $("#single-empty").classList.toggle("hidden", view !== "empty");
  $("#single-loading").classList.toggle("hidden", view !== "loading");
  $("#single-error").classList.toggle("hidden", view !== "error");
  $("#single-result").classList.toggle("hidden", view !== "result");
  if (view === "error") {
    $("#single-error-msg").textContent = errorMsg;
  }
}

function initSingleAnalyzer() {
  const textarea = $("#single-text");
  const btnAnalyze = $("#btn-analyze-single");
  const btnClear = $("#btn-clear-single");

  $$(".chip-btn[data-sample]").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.sample;
      textarea.focus();
    });
  });

  btnClear.addEventListener("click", () => {
    textarea.value = "";
    setSingleState("empty");
    textarea.focus();
  });

  btnAnalyze.addEventListener("click", async () => {
    const text = textarea.value.trim();
    if (!text) {
      setSingleState("error", "Teks ulasan masih kosong. Ketik ulasan atau pilih salah satu contoh terlebih dahulu.");
      showToast("Silakan masukkan teks ulasan terlebih dahulu.", "error");
      return;
    }

    btnAnalyze.disabled = true;
    btnAnalyze.textContent = "Menganalisis...";
    setSingleState("loading");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_type: getSelectedModel() }),
      });

      const data = await parseJsonSafe(res);
      if (!res.ok) {
        throw new Error(data.detail || "Gagal memproses prediksi.");
      }

      const badge = $("#single-badge");
      badge.textContent = data.sentiment;
      badge.className = `sentiment-stamp ${stampClassFor(data.sentiment)}`;

      $("#single-model-tag").textContent = data.model_used;
      $("#single-confidence-text").textContent = `${Number(data.confidence).toFixed(2)}%`;

      const bar = $("#single-confidence-bar");
      bar.style.width = `${Math.min(100, Math.max(0, Number(data.confidence)))}%`;
      bar.style.backgroundColor = barColorFor(data.sentiment);

      $("#single-original-text").textContent = data.original_text;
      $("#single-clean-text").textContent = data.clean_text || "(kosong setelah preprocessing)";

      setSingleState("result");
    } catch (err) {
      setSingleState("error", err.message);
      showToast(err.message, "error");
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.textContent = "Analisis Sentimen";
    }
  });
}

/* =========================================================================
 * Tab 2: Batch CSV Upload
 * ========================================================================= */
function setBatchState(view, errorMsg = "") {
  $("#batch-empty").classList.toggle("hidden", view !== "empty");
  $("#batch-loading").classList.toggle("hidden", view !== "loading");
  $("#batch-error").classList.toggle("hidden", view !== "error");
  $("#batch-result").classList.toggle("hidden", view !== "result");
  if (view === "error") {
    $("#batch-error-msg").textContent = errorMsg;
  }
}

function initBatchAnalyzer() {
  const dropzone = $("#dropzone");
  const fileInput = $("#csv-file-input");
  const selectedBar = $("#selected-file-bar");
  const fileNameEl = $("#selected-file-name");
  const fileSizeEl = $("#selected-file-size");
  const btnRemove = $("#btn-remove-file");
  const btnProcess = $("#btn-analyze-batch");
  const btnDownload = $("#btn-download-csv");

  function setFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showToast("Format file harus .csv", "error");
      return;
    }
    state.selectedFile = file;
    fileNameEl.textContent = file.name;
    fileSizeEl.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
    selectedBar.classList.remove("hidden");
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) setFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  btnRemove.addEventListener("click", () => {
    state.selectedFile = null;
    fileInput.value = "";
    selectedBar.classList.add("hidden");
    setBatchState("empty");
  });

  btnProcess.addEventListener("click", async () => {
    if (!state.selectedFile) {
      setBatchState("error", "Belum ada file CSV yang dipilih. Tarik file ke area unggah atau klik untuk memilih.");
      showToast("Pilih file .csv terlebih dahulu.", "error");
      return;
    }

    const colName = $("#csv-column-input").value.trim() || "content";

    btnProcess.disabled = true;
    btnProcess.textContent = "Memproses CSV...";
    setBatchState("loading");

    try {
      const { uploadBlob, clientTrimmed, trimmedRows } = await prepareCsvForUpload(
        state.selectedFile
      );
      if (clientTrimmed) {
        showToast(
          `File besar terdeteksi: otomatis memproses ${trimmedRows.toLocaleString("id-ID")} baris pertama agar sesuai batas server.`,
          "info"
        );
      }

      const formData = new FormData();
      formData.append("file", uploadBlob, state.selectedFile.name);
      formData.append("text_column", colName);
      formData.append("model_type", getSelectedModel());

      const res = await fetch("/api/predict-batch", {
        method: "POST",
        body: formData,
      });
      const data = await parseJsonSafe(res);
      if (!res.ok) {
        throw new Error(data.detail || "Gagal memproses file CSV.");
      }

      const total = data.total_rows;
      const dist = data.distribution;

      $("#batch-total").textContent = total.toLocaleString("id-ID");
      $("#batch-meta").textContent = data.truncated
        ? `Dibatasi 2.000 baris | Model: ${data.model_used.toUpperCase()}`
        : `Kolom: ${data.text_column_used} | Model: ${data.model_used.toUpperCase()}`;

      const posPct = formatPercent(dist.Positif, total);
      const neuPct = formatPercent(dist.Netral, total);
      const negPct = formatPercent(dist.Negatif, total);

      $("#batch-pos-pct").textContent = posPct;
      $("#batch-pos-count").textContent = `${dist.Positif} ulasan`;
      $("#batch-neu-pct").textContent = neuPct;
      $("#batch-neu-count").textContent = `${dist.Netral} ulasan`;
      $("#batch-neg-pct").textContent = negPct;
      $("#batch-neg-count").textContent = `${dist.Negatif} ulasan`;

      $("#batch-seg-pos").style.width = posPct;
      $("#batch-seg-neu").style.width = neuPct;
      $("#batch-seg-neg").style.width = negPct;

      renderDoughnutChart("batch-chart", dist, "batchChart");

      const tbody = $("#batch-preview-body");
      tbody.innerHTML = data.preview
        .map(
          (row, idx) => `
          <tr>
            <td>${idx + 1}</td>
            <td><span class="sentiment-stamp sm ${stampClassFor(row.sentiment)}">${escapeHtml(row.sentiment)}</span></td>
            <td class="mono-text">${Number(row.confidence).toFixed(2)}%</td>
            <td>${escapeHtml(row.content)}</td>
            <td class="mono-text">${escapeHtml(row.clean_text || "-")}</td>
          </tr>
        `
        )
        .join("");

      state.lastBatchCsvBase64 = data.csv_base64;
      setBatchState("result");
      showToast(`Berhasil menganalisis ${total} ulasan.`, "success");
    } catch (err) {
      setBatchState("error", err.message);
      showToast(err.message, "error");
    } finally {
      btnProcess.disabled = false;
      btnProcess.textContent = "Proses File CSV";
    }
  });

  btnDownload.addEventListener("click", () => {
    if (!state.lastBatchCsvBase64) return;
    const binary = atob(state.lastBatchCsvBase64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `hasil_sentimen_${getSelectedModel()}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });
}

/* =========================================================================
 * Tab 3: Live Play Store Scraper
 * ========================================================================= */
function setScraperState(view, errorMsg = "") {
  $("#scraper-empty").classList.toggle("hidden", view !== "empty");
  $("#scraper-loading").classList.toggle("hidden", view !== "loading");
  $("#scraper-error").classList.toggle("hidden", view !== "error");
  $("#scraper-result").classList.toggle("hidden", view !== "result");
  if (view === "error") {
    $("#scraper-error-msg").textContent = errorMsg;
  }
}

function renderKeywords(containerId, items, tagClass) {
  const el = $(containerId);
  if (!items || items.length === 0) {
    el.innerHTML = `<span class="muted">Tidak ada kata kunci dominan.</span>`;
    return;
  }
  el.innerHTML = items
    .map(
      (item) =>
        `<span class="kw-tag ${tagClass}">${escapeHtml(item.word)} <span class="kw-count">${item.count}</span></span>`
    )
    .join("");
}

function initLiveScraper() {
  const appIdInput = $("#scraper-appid");
  const countSelect = $("#scraper-count");
  const btnScrape = $("#btn-scrape-live");

  $$(".chip-btn[data-appid]").forEach((chip) => {
    chip.addEventListener("click", () => {
      appIdInput.value = chip.dataset.appid;
      appIdInput.focus();
    });
  });

  btnScrape.addEventListener("click", async () => {
    const appId = appIdInput.value.trim();
    const count = Number(countSelect.value);

    if (!appId) {
      setScraperState("error", "App ID kosong. Masukkan App ID seperti com.tokopedia.tkpd lalu coba lagi.");
      showToast("Masukkan App ID Google Play Store terlebih dahulu.", "error");
      return;
    }

    btnScrape.disabled = true;
    btnScrape.textContent = "Menarik Ulasan...";
    setScraperState("loading");

    try {
      const res = await fetch("/api/scrape-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_id: appId,
          count,
          model_type: getSelectedModel(),
        }),
      });

      const data = await parseJsonSafe(res);
      if (!res.ok) {
        throw new Error(data.detail || "Gagal mengambil ulasan dari Play Store.");
      }

      const total = data.total_scraped;
      const dist = data.distribution;

      $("#scraper-total").textContent = total.toLocaleString("id-ID");
      $("#scraper-app-label").textContent = `${data.app_id} | ${data.model_used.toUpperCase()}`;

      $("#scraper-pos-pct").textContent = formatPercent(dist.Positif, total);
      $("#scraper-pos-count").textContent = `${dist.Positif} ulasan`;
      $("#scraper-neu-pct").textContent = formatPercent(dist.Netral, total);
      $("#scraper-neu-count").textContent = `${dist.Netral} ulasan`;
      $("#scraper-neg-pct").textContent = formatPercent(dist.Negatif, total);
      $("#scraper-neg-count").textContent = `${dist.Negatif} ulasan`;

      renderDoughnutChart("scraper-chart", dist, "scraperChart");
      renderKeywords("#kw-positive", data.top_keywords.Positif, "kw-tag-pos");
      renderKeywords("#kw-negative", data.top_keywords.Negatif, "kw-tag-neg");

      const feed = $("#scraper-feed");
      feed.innerHTML = data.reviews
        .map(
          (rev) => `
          <article class="review-item">
            <div class="review-item-header">
              <div class="review-meta">
                <span class="review-author">${escapeHtml(rev.userName)}</span>
                <span class="rating-tag">${Number(rev.score) || 0}/5</span>
                <span class="muted">${escapeHtml(rev.at)}</span>
              </div>
              <div class="review-meta">
                <span class="sentiment-stamp sm ${stampClassFor(rev.sentiment)}">${escapeHtml(rev.sentiment)}</span>
                <span class="mono-text">${Number(rev.confidence).toFixed(1)}%</span>
              </div>
            </div>
            <p class="review-content">${escapeHtml(rev.content)}</p>
          </article>
        `
        )
        .join("");

      setScraperState("result");
      showToast(`Berhasil menarik dan menganalisis ${total} ulasan.`, "success");
    } catch (err) {
      setScraperState("error", err.message);
      showToast(err.message, "error");
    } finally {
      btnScrape.disabled = false;
      btnScrape.textContent = "Tarik & Analisis Live";
    }
  });
}

/* =========================================================================
 * Bootstrap
 * ========================================================================= */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSingleAnalyzer();
  initBatchAnalyzer();
  initLiveScraper();
  checkServerHealth();
});
