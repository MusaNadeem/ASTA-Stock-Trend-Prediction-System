let lineChart = null;
let barChart = null;
let latestChartData = { labels: [], values: [] };
let loadedSymbol = "";
let latestDashboardData = {};
let latestForecastData = null;

const stockSelect = document.getElementById("stockSelect");
const fileInput = document.getElementById("fileInput");
const fileInputData = document.getElementById("fileInputData");
const attentionModeSelect = document.getElementById("attentionModeSelect");
const forecastModeSelect = document.getElementById("forecastModeSelect");
const futureDateInput = document.getElementById("futureDateInput");
const liveStatusText = document.getElementById("liveStatusText");
const modelStatusChip = document.getElementById("modelStatusChip");
const topbarPriceChip = document.getElementById("topbarPriceChip");
const topbarModeChip = document.getElementById("topbarModeChip");
const modelLifecycleText = document.getElementById("modelLifecycleText");
const regimeTextMirror = document.getElementById("regimeTextMirror");
const dataStockSelect = document.getElementById("dataStockSelect");
const dropZone = document.getElementById("dropZone");
const trainingLog = document.getElementById("trainingLog");
const trainingProgressBar = document.getElementById("trainingProgressBar");
const dashboardConfidenceValue = document.getElementById("dashboardConfidenceValue");
const refreshNewsBtn = document.getElementById("refreshNewsBtn");
const newsFeed = document.getElementById("newsFeed");
const loadDataBtn = document.getElementById("loadDataBtn");
const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");
const predictDateBtn = document.getElementById("predictDateBtn");
const trainUploadedBtn = document.getElementById("trainUploadedBtn");
const metricsPanel = document.getElementById("metrics");
const statusBar = document.getElementById("statusBar");
const dataStatusBar = document.getElementById("dataStatusBar");
const datePredictionStatus = document.getElementById("datePredictionStatus");
const trendLabel = document.getElementById("trendLabel");
const confidenceValue = document.getElementById("confidenceValue");
const probabilitiesPanel = document.getElementById("probabilities");
const horizonResultsPanel = document.getElementById("horizonResults");
const analysisGrid = document.getElementById("analysisGrid");
const currentPriceValue = document.getElementById("currentPriceValue");
const trendDirectionValue = document.getElementById("trendDirectionValue");
const tradingVolumeValue = document.getElementById("tradingVolumeValue");
const marketRegimeBadge = document.getElementById("marketRegimeBadge");
const alertPanel = document.getElementById("alertPanel");
const toastContainer = document.getElementById("toastContainer");
const futureForecastLabel = document.getElementById("futureForecastLabel");
const futureConfidenceValue = document.getElementById("futureConfidenceValue");
const futurePriceRange = document.getElementById("futurePriceRange");
const futureRegimeValue = document.getElementById("futureRegimeValue");
const tabLinks = Array.from(document.querySelectorAll(".tab-link"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const sidebarTabLinks = Array.from(document.querySelectorAll("[data-nav-tab]"));

function syncForecastModeUI() {
  if (!forecastModeSelect) return;
  const futureDateGroup = document.getElementById("futureDateGroup");
  const custom = forecastModeSelect.value === "custom_date";
  if (futureDateInput) futureDateInput.disabled = !custom;
  if (futureDateGroup) futureDateGroup.style.display = custom ? "block" : "none";
}

function showToast(message, tone = "info") {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast ${tone}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  window.setTimeout(() => toast.classList.add("visible"), 20);
  window.setTimeout(() => {
    toast.classList.remove("visible");
    window.setTimeout(() => toast.remove(), 250);
  }, 2800);
}

function addAlert(message, tone = "warning") {
  if (!alertPanel) return;
  const empty = alertPanel.querySelector(".alert-empty");
  if (empty) empty.remove();
  const alertItem = document.createElement("div");
  alertItem.className = `alert-item ${tone}`;
  alertItem.textContent = message;
  alertPanel.prepend(alertItem);
  while (alertPanel.children.length > 4) {
    alertPanel.removeChild(alertPanel.lastElementChild);
  }
}

function setMarketBadge(regime) {
  const cleanRegime = regime || "Sideways market";
  if (marketRegimeBadge) {
    marketRegimeBadge.textContent = cleanRegime;
    marketRegimeBadge.className = "regime-badge";
    if (cleanRegime.toLowerCase().includes("bull")) marketRegimeBadge.classList.add("badge-good");
    else if (cleanRegime.toLowerCase().includes("bear")) marketRegimeBadge.classList.add("badge-danger");
    else marketRegimeBadge.classList.add("badge-warning");
  }
  if (regimeTextMirror) {
    regimeTextMirror.textContent = cleanRegime;
  }
}

async function loadPakistanNews() {
  if (!newsFeed) return;
  newsFeed.innerHTML = `<div class="news-empty">Loading latest headlines...</div>`;
  try {
    const response = await fetch("/news?limit=8");
    const data = await response.json();
    const items = Array.isArray(data.news) ? data.news : [];
    if (!items.length) {
      newsFeed.innerHTML = `<div class="news-empty">No headlines available right now.</div>`;
      return;
    }
    newsFeed.innerHTML = items
      .map(
        item => `<article class="news-card">
          <span>${item.source || "News"}</span>
          <h4>${item.title || "Untitled headline"}</h4>
          <p>${item.published_at || "Recently updated"}</p>
          <a href="${item.link || "#"}" target="_blank" rel="noopener noreferrer">Read full story</a>
        </article>`
      )
      .join("");
  } catch (error) {
    console.error(error);
    newsFeed.innerHTML = `<div class="news-empty">Unable to load news at the moment.</div>`;
  }
}

function appendTrainingLog(message, tone = "") {
  if (!trainingLog) return;
  const line = document.createElement("div");
  line.className = `log-line ${tone}`.trim();
  line.innerHTML = `<span class="prompt">Rs </span><span>${message}</span>`;
  trainingLog.appendChild(line);
  trainingLog.scrollTop = trainingLog.scrollHeight;
}

function setTrainingProgress(active) {
  if (!trainingProgressBar) return;
  trainingProgressBar.classList.toggle("active", active);
}

function setDashboardConfidence(percent) {
  const bounded = Math.max(0, Math.min(100, Number(percent) || 0));
  if (dashboardConfidenceValue) {
    dashboardConfidenceValue.textContent = `${bounded.toFixed(0)}%`;
  }
  document.querySelectorAll(".confidence-ring").forEach(ring => {
    ring.style.background = `conic-gradient(var(--accent) ${bounded * 3.6}deg, rgba(255,255,255,0.08) ${bounded * 3.6}deg)`;
  });
}

function setTopbarState({ symbol, price, mode, regime, lifecycle, live = true }) {
  if (liveStatusText) {
    liveStatusText.textContent = live ? "LIVE" : "BUSY";
  }
  if (modelStatusChip) {
    const currentSymbol = symbol || loadedSymbol || "N/A";
    modelStatusChip.textContent = `${currentSymbol} · ${regime || "Sideways"}`;
  }
  if (topbarPriceChip) {
    topbarPriceChip.textContent = Number.isFinite(price) ? `Rs ${Number(price).toFixed(2)}` : "--";
  }
  if (topbarModeChip) {
    topbarModeChip.textContent = mode === "standard" ? "Standard" : "ASTA";
  }
  if (modelLifecycleText && lifecycle) {
    modelLifecycleText.textContent = lifecycle;
  }
}

function setConfidenceRing(percent) {
  setDashboardConfidence(percent);
}

function getSelectedSymbol() {
  return stockSelect.value || loadedSymbol || "";
}

function makeCard(title, value, tone = "") {
  return `<div class="metric-card ${tone}"><span class="metric-label">${title}</span><strong>${value}</strong></div>`;
}

function makeSummaryCard(title, value) {
  return `<div class="summary-card"><span>${title}</span><strong>${value}</strong></div>`;
}

function setStatus(message, tone = "") {
  if (!statusBar) return;
  statusBar.className = `status-bar ${tone}`.trim();
  statusBar.textContent = message;
  if (modelLifecycleText) {
    modelLifecycleText.textContent = tone === "error" ? "Attention needed" : tone === "success" ? "Ready" : "Working";
  }
}

function setDataStatus(message, tone = "") {
  if (!dataStatusBar) return;
  dataStatusBar.className = `status-bar ${tone}`.trim();
  dataStatusBar.textContent = message;
}

function setBusy(isBusy) {
  loadDataBtn.disabled = isBusy;
  trainBtn.disabled = isBusy;
  predictBtn.disabled = isBusy;
  predictDateBtn.disabled = isBusy;
  fileInput.disabled = isBusy;
  fileInputData.disabled = isBusy;
  if (dataStockSelect) dataStockSelect.disabled = isBusy;
  stockSelect.disabled = isBusy;
  attentionModeSelect.disabled = isBusy;
  trainUploadedBtn.disabled = isBusy;
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(currentPriceValue?.textContent || NaN),
    mode: getAttentionMode(),
    regime: marketRegimeBadge?.textContent || "Sideways market",
    lifecycle: isBusy ? "Working" : "Ready",
    live: !isBusy,
  });
}

function getAttentionMode() {
  return attentionModeSelect.value === "standard" ? "standard" : "asta";
}

function activateTab(tabName) {
  tabLinks.forEach(button => {
    const active = button.dataset.tab === tabName;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach(panel => {
    panel.classList.toggle("active", panel.dataset.tab === tabName);
  });
  sidebarTabLinks.forEach(link => {
    link.classList.toggle("active", link.dataset.navTab === tabName);
  });
}

function renderMetrics(result) {
  if (!metricsPanel) return;
  metricsPanel.innerHTML = [
    makeCard("Training Loss", Number(result.train_loss).toFixed(4)),
    makeCard("Validation Loss", Number(result.val_loss).toFixed(4)),
    makeCard("Accuracy", `${(Number(result.accuracy) * 100).toFixed(2)}%`),
    makeCard("ASTA Speedup", `${Number(result.runtime_speedup).toFixed(2)}x`),
  ].join("");
  if (analysisGrid) {
    analysisGrid.innerHTML = [
      `<div class="analysis-item"><span>Standard Attention</span><strong class="value">${Number(result.runtime_standard_ms).toFixed(2)} ms</strong><small>O(T² · d)</small></div>`,
      `<div class="analysis-item"><span>ASTA</span><strong class="value">${Number(result.runtime_asta_ms).toFixed(2)} ms</strong><small>O(T log T · d) approximation</small></div>`,
      `<div class="analysis-item"><span>Sample Count</span><strong class="value">${result.sample_count}</strong><small>Sliding windows with T = 60</small></div>`,
    ].join("");
  }
}

function renderDashboardSummary(data, prediction = null) {
  const currentPrice = Number(data.current_price ?? data.recent_close_value ?? 0);
  const currentVolume = Number(data.current_volume ?? 0);
  const trend = prediction?.label || data.trend_direction || "Neutral";
  if (currentPriceValue) currentPriceValue.textContent = "Rs " + currentPrice.toFixed(2);
  if (trendDirectionValue) trendDirectionValue.textContent = trend;
  if (tradingVolumeValue) tradingVolumeValue.textContent = currentVolume.toLocaleString();
  setDashboardConfidence(Number(prediction?.confidence || latestForecastData?.confidence || 0) * 100);
  setTopbarState({
    symbol: data.symbol || getSelectedSymbol(),
    price: currentPrice,
    mode: getAttentionMode(),
    regime: data.market_regime || data.trend_direction || "Sideways market",
    lifecycle: prediction ? `Forecast: ${trend}` : "Data loaded",
  });
}

function renderPrediction(result) {
  latestForecastData = result;
  if (trendLabel) trendLabel.textContent = result.label;
  if (confidenceValue) confidenceValue.textContent = `${(Number(result.confidence) * 100).toFixed(2)}%`;
  setDashboardConfidence(Number(result.confidence) * 100);
  if (probabilitiesPanel) {
    probabilitiesPanel.innerHTML = ["Downtrend", "Neutral", "Uptrend"].map((label, index) => {
      const value = Number(result.probabilities[index] || 0);
      return `<div class="prob-item"><small>${label}</small><strong>${(value * 100).toFixed(2)}%</strong></div>`;
    }).join("");
  }

  if (barChart) {
    barChart.data.labels = ["Downtrend", "Neutral", "Uptrend"];
    barChart.data.datasets[0].data = result.probabilities.map(value => Number(value) * 100);
    barChart.update();
  }

  const horizonEntries = result.horizon_predictions || {};
  if (horizonResultsPanel) {
    horizonResultsPanel.innerHTML = Object.entries(horizonEntries).map(([key, value]) => {
      const horizonLabel = key.replace("short_term", "Short-term").replace("mid_term", "Mid-term").replace("long_term", "Long-term");
      return `<div class="prediction-row">
        <div>
          <small>${horizonLabel}</small>
          <strong>${value.label}</strong>
        </div>
        <span>${(Number(value.confidence) * 100).toFixed(2)}%</span>
      </div>`;
    }).join("");
  }

  if (trendDirectionValue) trendDirectionValue.textContent = result.label;
  if (futureForecastLabel) futureForecastLabel.textContent = result.label;
  if (futureConfidenceValue) futureConfidenceValue.textContent = `${(Number(result.confidence) * 100).toFixed(0)}%`;
  setConfidenceRing(Number(result.confidence) * 100);
  if (result.price_range) {
    if (futurePriceRange) futurePriceRange.textContent = `Rs ${Number(result.price_range.low).toFixed(2)} - Rs ${Number(result.price_range.high).toFixed(2)}`;
  }
  if (futureRegimeValue) futureRegimeValue.textContent = result.market_regime || "--";
  setMarketBadge(result.market_regime || latestDashboardData.trend_direction || "Sideways market");
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: getAttentionMode(),
    regime: result.market_regime || latestDashboardData.trend_direction || "Sideways market",
    lifecycle: `${result.label} forecast ready`,
  });
  addAlert(`Trend update: ${result.label} (${(Number(result.confidence) * 100).toFixed(1)}%)`, "info");
}

function updateChart(data) {
  if (typeof Chart === "undefined") {
    console.warn("Chart.js not loaded; skipping chart render");
    return;
  }
  const historyValues = data.recent_close_series || data.recent_close || [];
  const historyLabels = data.recent_window_labels || data.recent_dates || historyValues.map((_, i) => `T${i + 1}`);
  const forecastLabels = latestForecastData?.forecast_curve ? latestForecastData.forecast_curve.map(point => `F${point.step}`) : [];
  latestChartData = { labels: historyLabels.concat(forecastLabels), values: historyValues };
  const lineChartCanvas = document.getElementById("lineChart");
  if (lineChartCanvas) {
    const ctx = lineChartCanvas.getContext("2d");
    if (lineChart) {
      lineChart.destroy();
    }
    const closeSeries = latestChartData.labels.map((_, index) => {
      if (index >= historyValues.length) return null;
      const value = Number(historyValues[index]);
      return Number.isFinite(value) ? value : null;
    });

    const forecastSeries = latestChartData.labels.map((_, index) => {
      if (!latestForecastData?.forecast_curve) return null;
      const forecastIndex = index - historyValues.length;
      if (forecastIndex < 0 || forecastIndex >= latestForecastData.forecast_curve.length) return null;
      const value = Number(latestForecastData.forecast_curve[forecastIndex].close);
      return Number.isFinite(value) ? value : null;
    });

    lineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: latestChartData.labels,
        datasets: [
          {
            label: `${data.symbol} Close`,
            data: closeSeries,
            borderColor: "#5eead4",
            backgroundColor: "rgba(94, 234, 212, 0.12)",
            borderWidth: 2,
            tension: 0.28,
            pointRadius: 0,
            fill: true,
          },
          {
            label: "Forecast",
            data: forecastSeries,
            type: "line",
            borderColor: "rgba(251, 191, 36, 0.95)",
            borderDash: [8, 6],
            pointRadius: 4,
            pointBackgroundColor: "rgba(251, 191, 36, 0.95)",
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#e8eef7" } },
        },
        scales: {
          x: { ticks: { color: "#9db0d0", maxRotation: 0 }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
          y: { ticks: { color: "#9db0d0", callback: function(value) { return "Rs " + value; } }, grid: { color: "rgba(148, 163, 184, 0.08)" }, beginAtZero: false },
        },
      },
    });
  }

  const barChartCanvas = document.getElementById("barChart");
  if (barChartCanvas) {
    const barCtx = barChartCanvas.getContext("2d");
    if (barChart) {
      barChart.destroy();
    }
    barChart = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: ["Downtrend", "Neutral", "Uptrend"],
        datasets: [{
          label: "Trend Probability %",
          data: [0, 0, 0],
          backgroundColor: [
            "rgba(251, 113, 133, 0.82)",
            "rgba(255, 205, 86, 0.82)",
            "rgba(134, 239, 172, 0.82)",
          ],
          borderColor: [
            "rgba(251, 113, 133, 1)",
            "rgba(255, 205, 86, 1)",
            "rgba(134, 239, 172, 1)",
          ],
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: "#e8eef7" } },
        },
        scales: {
          x: { ticks: { color: "#9db0d0" }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
          y: { ticks: { color: "#9db0d0", callback: function(value) { return "Rs " + value; } }, grid: { color: "rgba(148, 163, 184, 0.08)" }, beginAtZero: true, max: 100 },
        },
      },
    });
  }
}

function buildFormData(extra = {}) {
  const formData = new FormData();
  const symbol = getSelectedSymbol();
  if (symbol) formData.append("symbol", symbol);
  const selectedFile = fileInputData.files[0] || fileInput.files[0];
  if (selectedFile) formData.append("file", selectedFile);
  formData.append("use_standard_attention", String(getAttentionMode() === "standard"));
  Object.entries(extra).forEach(([key, value]) => formData.append(key, value));
  return formData;
}

function resolveFutureForecastPayload() {
  const mode = forecastModeSelect.value;
  if (mode === "t1") return { steps: 1 };
  if (mode === "t3") return { steps: 3 };
  if (mode === "t7") return { steps: 7 };
  return { future_date: futureDateInput.value };
}

async function loadStocks() {
  try {
    const response = await fetch("/stocks");
    if (!response.ok) throw new Error("Failed to load stocks");
    const data = await response.json();
    stockSelect.innerHTML = data.stocks.map(item => `<option value="${item.symbol}">${item.symbol}</option>`).join("");
    if (dataStockSelect) {
      dataStockSelect.innerHTML = data.stocks.map(item => `<option value="${item.symbol}">${item.symbol}</option>`).join("");
    }
    loadedSymbol = data.stocks[0]?.symbol || "";
    if (stockSelect) stockSelect.value = loadedSymbol;
    if (dataStockSelect) dataStockSelect.value = loadedSymbol;
    setStatus(`Loaded ${data.stocks.length} stocks from the dataset.`, "success");
    appendTrainingLog(`Loaded ${data.stocks.length} symbols from dataset.`, "dim");
  } catch (error) {
    console.error("Error loading stocks:", error);
    showToast("Failed to load stocks.", "error");
  }
}

async function loadData() {
  const symbol = getSelectedSymbol();
  if (!symbol) return;
  setStatus(`Loading processed data for ${symbol}...`);
  appendTrainingLog(`Fetching processed data for ${symbol}...`, "dim");
  try {
    const response = await fetch(`/data?symbol=${encodeURIComponent(symbol)}`);
    if (!response.ok) throw new Error("Failed to load data");
    const data = await response.json();
    latestDashboardData = data;
    renderDashboardSummary(data);
    updateChart(data);
    setMarketBadge(data.market_regime || data.trend_direction || "Sideways market");
    setTopbarState({
      symbol: symbol,
      price: Number(data.current_price ?? data.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: data.market_regime || data.trend_direction || "Sideways market",
      lifecycle: "Dashboard synced",
    });
    setStatus(`Showing processed data for ${symbol}.`, "success");
    appendTrainingLog(`Data ready for ${symbol}.`, "dim");
  } catch (error) {
    console.error("Error loading data:", error);
    setStatus("Failed to load data.", "error");
    showToast("Failed to load data.", "error");
  }
}

async function trainModel() {
  setBusy(true);
  setStatus("Training model... this can take a moment.");
  appendTrainingLog("Starting training cycle...", "dim");
  try {
    setTrainingProgress(true);
    const response = await fetch("/train", {
      method: "POST",
      body: buildFormData({ epochs: 6, batch_size: 32 }),
    });
    const data = await response.json();
    if (!response.ok) {
      setStatus(data.detail || "Training failed.", "error");
      alert(data.detail || "Training failed");
      return;
    }
    renderMetrics(data);
    await loadData();
    setStatus(`Training finished for ${getSelectedSymbol()}.`, "success");
    showToast("Model trained successfully.", "success");
    appendTrainingLog(`Training complete. Accuracy ${(Number(data.accuracy) * 100).toFixed(2)}%.`, "dim");
  } finally {
    setBusy(false);
    setTrainingProgress(false);
  }
}

async function predictTrend() {
  setBusy(true);
  setStatus("Generating prediction...");
  appendTrainingLog("Running trend prediction...", "dim");
  try {
    const mode = forecastModeSelect.value;
    const endpoint = mode === "t1" || mode === "t3" || mode === "t7" ? "/forecast" : mode === "custom_date" ? "/predict-date" : "/predict";
    const extra = mode === "t1" ? { steps: 1 } : mode === "t3" ? { steps: 3 } : mode === "t7" ? { steps: 7 } : mode === "custom_date" ? { future_date: futureDateInput.value } : {};
    const response = await fetch(endpoint, {
      method: "POST",
      body: buildFormData(extra),
    });
    const data = await response.json();
    if (!response.ok) {
      setStatus(data.detail || "Prediction failed.", "error");
      alert(data.detail || "Prediction failed");
      return;
    }
    renderPrediction(data);
    renderDashboardSummary(latestDashboardData, data);
    setStatus(`Prediction complete: ${data.label} (${(Number(data.confidence) * 100).toFixed(1)}%).`, "success");
    if (Number(data.confidence) > 0.6) {
      addAlert(`High-confidence signal detected: ${data.label}`, "success");
    }
    updateChart(latestDashboardData);
    appendTrainingLog(`Prediction complete: ${data.label} (${(Number(data.confidence) * 100).toFixed(1)}%).`, "dim");
  } finally {
    setBusy(false);
  }
}

async function predictFutureDate() {
  setBusy(true);
  if (datePredictionStatus) datePredictionStatus.textContent = "Generating date forecast...";
  appendTrainingLog("Running future-date forecast...", "dim");
  try {
    const mode = forecastModeSelect.value;
    const payload = buildFormData(resolveFutureForecastPayload());
    const endpoint = mode === "t1" || mode === "t3" || mode === "t7" ? "/forecast" : "/predict-date";
    const response = await fetch(endpoint, { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      if (datePredictionStatus) datePredictionStatus.textContent = data.detail || "Date prediction failed.";
      setDataStatus(data.detail || "Date prediction failed.", "error");
      return;
    }
    latestForecastData = data;
    if (futureForecastLabel) futureForecastLabel.textContent = data.label || "Neutral";
    if (futureConfidenceValue) futureConfidenceValue.textContent = `${(Number(data.confidence || 0) * 100).toFixed(0)}%`;
    setConfidenceRing(Number(data.confidence || 0) * 100);
    if (futurePriceRange) futurePriceRange.textContent = data.price_range ? `Rs ${Number(data.price_range.low).toFixed(2)} - Rs ${Number(data.price_range.high).toFixed(2)}` : `--`;
    if (futureRegimeValue) futureRegimeValue.textContent = data.market_regime || "--";
    setMarketBadge(data.market_regime || "Sideways market");
    setTopbarState({
      symbol: getSelectedSymbol(),
      price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: data.market_regime || "Sideways market",
      lifecycle: `Forecast for ${data.future_date || futureDateInput.value}`,
    });
    if (datePredictionStatus) datePredictionStatus.textContent = `Forecast ready for ${data.future_date || futureDateInput.value}.`;
    setDataStatus(`Forecast completed for ${data.future_date || futureDateInput.value}.`, "success");
    addAlert(`Future forecast generated for ${data.future_date || futureDateInput.value}`, "info");
    updateChart(latestDashboardData);
    appendTrainingLog(`Future-date forecast complete for ${data.future_date || futureDateInput.value}.`, "dim");
  } finally {
    setBusy(false);
  }
}

if (loadDataBtn) loadDataBtn.addEventListener("click", loadData);
if (trainBtn) trainBtn.addEventListener("click", trainModel);
if (predictBtn) predictBtn.addEventListener("click", predictTrend);
if (predictDateBtn) predictDateBtn.addEventListener("click", predictFutureDate);
if (trainUploadedBtn) trainUploadedBtn.addEventListener("click", trainModel);

tabLinks.forEach(button => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

// Old sidebar tab logic removed for multi-page architecture

if (forecastModeSelect) {
  forecastModeSelect.addEventListener("change", () => {
    syncForecastModeUI();
  });
}

if (dropZone) {
  dropZone.addEventListener("click", () => fileInputData.click());
  dropZone.addEventListener("dragover", event => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", event => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const file = event.dataTransfer.files?.[0];
    if (file) {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      fileInputData.files = dataTransfer.files;
      fileInputData.dispatchEvent(new Event("change"));
    }
  });
}

if (attentionModeSelect) {
  attentionModeSelect.addEventListener("change", () => {
    setTopbarState({
      symbol: getSelectedSymbol(),
      price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: marketRegimeBadge?.textContent || "Sideways market",
      lifecycle: `${getAttentionMode() === "standard" ? "Standard" : "ASTA"} selected`,
    });
  });
}

if (stockSelect) {
  stockSelect.addEventListener("change", () => {
    if (dataStockSelect && dataStockSelect.value !== stockSelect.value) {
      dataStockSelect.value = stockSelect.value;
    }
    setTopbarState({
      symbol: getSelectedSymbol(),
      price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: marketRegimeBadge?.textContent || "Sideways market",
      lifecycle: `Switched to ${getSelectedSymbol()}`,
    });
    loadData().catch(console.error);
  });
}

if (dataStockSelect) {
  dataStockSelect.addEventListener("change", () => {
    if (stockSelect.value !== dataStockSelect.value) {
      stockSelect.value = dataStockSelect.value;
    }
    setTopbarState({
      symbol: getSelectedSymbol(),
      price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: marketRegimeBadge.textContent || "Sideways market",
      lifecycle: `Switched to ${getSelectedSymbol()}`,
    });
  });
}

if (fileInputData) {
  fileInputData.addEventListener("change", () => {
    const name = fileInputData.files[0]?.name || "No file selected";
    setDataStatus(`Selected: ${name}`, "success");
    appendTrainingLog(`Uploaded file selected: ${name}`, "dim");
  });
}

if (refreshNewsBtn) {
  refreshNewsBtn.addEventListener("click", () => {
    loadPakistanNews().catch(console.error);
  });
}

if (fileInput) {
  fileInput.addEventListener("change", () => {
    const name = fileInput.files[0]?.name || "No file selected";
    setStatus(`Dashboard upload selected: ${name}`, "success");
    appendTrainingLog(`Dashboard upload selected: ${name}`, "dim");
  });
}

loadStocks().then(loadData).catch(error => {
  console.error(error);
  setStatus("Failed to load stock list.", "error");
  alert("Failed to load stock list");
});

loadPakistanNews().catch(console.error);

syncForecastModeUI();
appendTrainingLog("Dashboard initialized.", "dim");
