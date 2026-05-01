let lineChart = null;
let barChart = null;
let portfolioChart = null;
let latestChartData = { labels: [], values: [] };
let loadedSymbol = "";
let latestDashboardData = {};
let latestForecastData = null;
let portfolioState = {
  cash: 10000,
  positions: [],
  history: [],
};

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
const simulateBuyBtn = document.getElementById("simulateBuyBtn");
const resetPortfolioBtn = document.getElementById("resetPortfolioBtn");
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
const focusTimestepsPanel = document.getElementById("focusTimestepsPanel");
const volatilitySpikesPanel = document.getElementById("volatilitySpikesPanel");
const modelRuntimeGrid = document.getElementById("modelRuntimeGrid");
const modelComplexityGrid = document.getElementById("modelComplexityGrid");
const modelStatsGrid = document.getElementById("modelStatsGrid");
const marketRegimeBadge = document.getElementById("marketRegimeBadge");
const alertPanel = document.getElementById("alertPanel");
const toastContainer = document.getElementById("toastContainer");
const futureForecastLabel = document.getElementById("futureForecastLabel");
const futureConfidenceValue = document.getElementById("futureConfidenceValue");
const futurePriceRange = document.getElementById("futurePriceRange");
const futureRegimeValue = document.getElementById("futureRegimeValue");
const whyPredictionPanel = document.getElementById("whyPredictionPanel");
const trendInfluencePanel = document.getElementById("trendInfluencePanel");
const volatilityContributionPanel = document.getElementById("volatilityContributionPanel");
const portfolioQtyInput = document.getElementById("portfolioQtyInput");
const portfolioEntryInput = document.getElementById("portfolioEntryInput");
const portfolioStats = document.getElementById("portfolioStats");
const portfolioCanvas = document.getElementById("portfolioChart");
const tabLinks = Array.from(document.querySelectorAll(".tab-link"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const sidebarTabLinks = Array.from(document.querySelectorAll("[data-nav-tab]"));

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
  line.innerHTML = `<span class="prompt">$</span><span>${message}</span>`;
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
    topbarPriceChip.textContent = Number.isFinite(price) ? `$${Number(price).toFixed(2)}` : "--";
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

function updatePortfolioStats(currentPrice) {
  const positionValue = portfolioState.positions.reduce((total, position) => total + position.qty * currentPrice, 0);
  const invested = portfolioState.positions.reduce((total, position) => total + position.qty * position.entry, 0);
  const pnl = positionValue - invested;
  const returnPct = invested > 0 ? (pnl / invested) * 100 : 0;
  portfolioStats.innerHTML = [
    makeSummaryCard("Cash", `$${portfolioState.cash.toFixed(2)}`),
    makeSummaryCard("Invested", `$${invested.toFixed(2)}`),
    makeSummaryCard("P/L", `${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)}`),
    makeSummaryCard("Return", `${returnPct.toFixed(2)}%`),
  ].join("");

  const ctx = portfolioCanvas.getContext("2d");
  if (portfolioChart) {
    portfolioChart.data.datasets[0].data.push(pnl);
    portfolioChart.data.labels.push(String(portfolioChart.data.labels.length + 1));
    portfolioChart.update();
    return;
  }
  portfolioChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["1"],
      datasets: [{
        label: "Virtual P/L",
        data: [pnl],
        borderColor: "#50e3c2",
        backgroundColor: "rgba(80, 227, 194, 0.12)",
        borderWidth: 2,
        tension: 0.28,
        pointRadius: 0,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: "#e8eef7" } } },
      scales: {
        x: { ticks: { color: "#9db0d0" }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
        y: { ticks: { color: "#9db0d0" }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
      },
    },
  });
}

function resetPortfolio() {
  portfolioState = { cash: 10000, positions: [], history: [] };
  updatePortfolioStats(Number(latestDashboardData.current_price || latestDashboardData.recent_close_value || 0));
  showToast("Portfolio reset.", "info");
}

function simulateBuy() {
  const qty = Math.max(1, Number(portfolioQtyInput.value || 1));
  const currentPrice = Number(portfolioEntryInput.value || latestDashboardData.current_price || latestDashboardData.recent_close_value || 0);
  if (!currentPrice) {
    showToast("Load data before simulating a buy.", "warning");
    return;
  }
  const cost = qty * currentPrice;
  if (cost > portfolioState.cash) {
    showToast("Not enough simulated cash.", "error");
    return;
  }
  portfolioState.cash -= cost;
  portfolioState.positions.push({ qty, entry: currentPrice, symbol: getSelectedSymbol() });
  updatePortfolioStats(currentPrice);
  addAlert(`Portfolio buy simulated: ${qty} shares at ${currentPrice.toFixed(2)}`, "info");
  showToast("Trade simulated.", "success");
}

function renderExplainability(result, data) {
  const focus = (data?.focus_timesteps || result?.focus_timesteps || []).slice(0, 6);
  const volatility = (data?.volatility_scores || result?.volatility_scores || []).slice(-6);
  const trendScore = result?.probabilities ? (Number(result.probabilities[2] || 0) - Number(result.probabilities[0] || 0)) : 0;
  const volatilityContribution = volatility.length ? volatility.reduce((sum, value) => sum + Number(value), 0) / volatility.length : 0;
  whyPredictionPanel.textContent = `ASTA concentrated attention on ${focus.length} recent and volatility-heavy timesteps, using market-aware positional encoding to bias the forecast.`;
  trendInfluencePanel.textContent = `Trend influence score: ${trendScore.toFixed(4)}. Positive values bias the model toward upward continuation.`;
  volatilityContributionPanel.textContent = `Volatility contribution: ${volatilityContribution.toFixed(4)}. Higher values signal uncertainty and can widen the estimated range.`;
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
  analysisGrid.innerHTML = [
    `<div class="analysis-item"><span>Standard Attention</span><strong class="value">${Number(result.runtime_standard_ms).toFixed(2)} ms</strong><small>O(T² · d)</small></div>`,
    `<div class="analysis-item"><span>ASTA</span><strong class="value">${Number(result.runtime_asta_ms).toFixed(2)} ms</strong><small>O(T log T · d) approximation</small></div>`,
    `<div class="analysis-item"><span>Sample Count</span><strong class="value">${result.sample_count}</strong><small>Sliding windows with T = 60</small></div>`,
  ].join("");
}

function renderDashboardSummary(data, prediction = null) {
  const currentPrice = Number(data.current_price ?? data.recent_close_value ?? 0);
  const currentVolume = Number(data.current_volume ?? 0);
  const trend = prediction?.label || data.trend_direction || "Neutral";
  currentPriceValue.textContent = currentPrice.toFixed(2);
  trendDirectionValue.textContent = trend;
  tradingVolumeValue.textContent = currentVolume.toLocaleString();
  setDashboardConfidence(Number(prediction?.confidence || latestForecastData?.confidence || 0) * 100);
  setTopbarState({
    symbol: data.symbol || getSelectedSymbol(),
    price: currentPrice,
    mode: getAttentionMode(),
    regime: data.market_regime || data.trend_direction || "Sideways market",
    lifecycle: prediction ? `Forecast: ${trend}` : "Data loaded",
  });
  updatePortfolioStats(currentPrice);
}

function renderFocusHighlights(data) {
  const focusTimesteps = Array.isArray(data.focus_timesteps) ? data.focus_timesteps : [];
  const volatilitySpikes = Array.isArray(data.volatility_spikes) ? data.volatility_spikes : [];
  const dates = data.recent_window_labels || data.recent_dates || [];
  const closeSeries = data.recent_close_series || data.recent_close || [];
  const volatilityScores = data.volatility_scores || [];

  focusTimestepsPanel.innerHTML = focusTimesteps.length
    ? focusTimesteps.map(index => {
        const label = dates[index] || `T${index}`;
        const value = Number(closeSeries[index] ?? 0).toFixed(2);
        return `<div class="focus-pill" data-tip="ASTA focus point"><strong>${label}</strong><span>${value}</span></div>`;
      }).join("")
    : `<div class="focus-empty">No focus points available yet.</div>`;

  volatilitySpikesPanel.innerHTML = volatilitySpikes.length
    ? volatilitySpikes.map(index => {
        const label = dates[index] || `T${index}`;
        const value = Number(volatilityScores[index] ?? 0).toFixed(4);
        return `<div class="focus-pill volatility" data-tip="Volatility spike"><strong>${label}</strong><span>${value}</span></div>`;
      }).join("")
    : `<div class="focus-empty">No volatility spikes available yet.</div>`;
}

function renderModelStats(stats) {
  modelRuntimeGrid.innerHTML = [
    `<div class="analysis-item"><span>Standard Attention</span><strong class="value">${Number(stats.runtime.standard_ms).toFixed(2)} ms</strong><small>Dense baseline</small></div>`,
    `<div class="analysis-item"><span>ASTA</span><strong class="value">${Number(stats.runtime.asta_ms).toFixed(2)} ms</strong><small>Sparse candidate attention</small></div>`,
    `<div class="analysis-item"><span>Speedup</span><strong class="value">${Number(stats.runtime.speedup).toFixed(2)}x</strong><small>Lower is better for runtime</small></div>`,
  ].join("");

  modelComplexityGrid.innerHTML = [
    `<div class="analysis-item"><span>Standard</span><strong class="value">${stats.complexity.standard}</strong><small>O(T² × d)</small></div>`,
    `<div class="analysis-item"><span>ASTA</span><strong class="value">${stats.complexity.asta}</strong><small>O(T log T × d)</small></div>`,
    `<div class="analysis-item"><span>Mode</span><strong class="value">${stats.attention_mode}</strong><small>Current toggle</small></div>`,
  ].join("");

  modelStatsGrid.innerHTML = [
    makeSummaryCard("Symbol", stats.symbol || loadedSymbol || "N/A"),
    makeSummaryCard("Local", stats.selection_strategy.local),
    makeSummaryCard("Log Sparse", stats.selection_strategy.log_sparse),
    makeSummaryCard("Volatility", stats.selection_strategy.volatility),
  ].join("");
}

function renderPrediction(result) {
  latestForecastData = result;
  trendLabel.textContent = result.label;
  confidenceValue.textContent = `${(Number(result.confidence) * 100).toFixed(2)}%`;
  setDashboardConfidence(Number(result.confidence) * 100);
  probabilitiesPanel.innerHTML = ["Downtrend", "Neutral", "Uptrend"].map((label, index) => {
    const value = Number(result.probabilities[index] || 0);
    return `<div class="prob-item"><small>${label}</small><strong>${(value * 100).toFixed(2)}%</strong></div>`;
  }).join("");

  if (barChart) {
    barChart.data.labels = ["Downtrend", "Neutral", "Uptrend"];
    barChart.data.datasets[0].data = result.probabilities.map(value => Number(value) * 100);
    barChart.update();
  }

  const horizonEntries = result.horizon_predictions || {};
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

  trendDirectionValue.textContent = result.label;
  futureForecastLabel.textContent = result.label;
  futureConfidenceValue.textContent = `${(Number(result.confidence) * 100).toFixed(0)}%`;
  setConfidenceRing(Number(result.confidence) * 100);
  if (result.price_range) {
    futurePriceRange.textContent = `$${Number(result.price_range.low).toFixed(2)} - $${Number(result.price_range.high).toFixed(2)}`;
  }
  futureRegimeValue.textContent = result.market_regime || "--";
  setMarketBadge(result.market_regime || latestDashboardData.trend_direction || "Sideways market");
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: getAttentionMode(),
    regime: result.market_regime || latestDashboardData.trend_direction || "Sideways market",
    lifecycle: `${result.label} forecast ready`,
  });
  renderExplainability(result, latestDashboardData);
  addAlert(`Trend update: ${result.label} (${(Number(result.confidence) * 100).toFixed(1)}%)`, "info");
}

function updateChart(data) {
  const historyLabels = data.recent_window_labels || data.recent_dates || [];
  const historyValues = data.recent_close_series || data.recent_close || [];
  const forecastLabels = latestForecastData?.forecast_curve ? latestForecastData.forecast_curve.map(point => `F${point.step}`) : [];
  latestChartData = { labels: historyLabels.concat(forecastLabels), values: historyValues };
  const ctx = document.getElementById("lineChart").getContext("2d");
  if (lineChart) {
    lineChart.destroy();
  }
  const closeSeries = latestChartData.labels.map((_, index) => (index < historyValues.length ? Number(historyValues[index]) : null));
  const forecastSeries = latestChartData.labels.map((_, index) => {
    if (!latestForecastData?.forecast_curve) return null;
    const forecastIndex = index - historyValues.length;
    if (forecastIndex < 0 || forecastIndex >= latestForecastData.forecast_curve.length) return null;
    return Number(latestForecastData.forecast_curve[forecastIndex].close);
  });
  const volatilitySeries = latestChartData.labels.map((_, index) => (index < (data.volatility_scores || []).length ? Number((data.volatility_scores || [])[index]) * 100 : null));
  const focusSeries = latestChartData.labels.map((_, index) => (data.focus_timesteps || []).includes(index) && index < historyValues.length ? Number(historyValues[index]) : null);
  lineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: latestChartData.labels,
      datasets: [{
        label: `${data.symbol} Close`,
        data: closeSeries,
        borderColor: "#5eead4",
        backgroundColor: "rgba(94, 234, 212, 0.12)",
        borderWidth: 2,
        tension: 0.28,
        pointRadius: 0,
        fill: true,
      }, {
        label: "ASTA Focus",
        data: focusSeries,
        type: "scatter",
        backgroundColor: "rgba(255, 205, 86, 0.95)",
        borderColor: "rgba(255, 205, 86, 1)",
        pointRadius: 6,
        showLine: false,
        yAxisID: "y",
      }, {
        label: "Forecast",
        data: forecastSeries,
        type: "line",
        borderColor: "rgba(251, 191, 36, 0.95)",
        borderDash: [8, 6],
        pointRadius: 4,
        pointBackgroundColor: "rgba(251, 191, 36, 0.95)",
        fill: false,
        yAxisID: "y",
      }, {
        label: "Volatility",
        data: volatilitySeries,
        borderColor: "rgba(96, 165, 250, 0.95)",
        backgroundColor: "rgba(96, 165, 250, 0.12)",
        borderWidth: 1,
        tension: 0.28,
        pointRadius: 0,
        fill: false,
        yAxisID: "y1",
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#e8eef7" } },
      },
      scales: {
        x: { ticks: { color: "#9db0d0", maxRotation: 0 }, grid: { color: "rgba(148, 163, 184, 0.08)" } },
        y: { ticks: { color: "#9db0d0" }, grid: { color: "rgba(148, 163, 184, 0.08)" }, position: "left" },
        y1: { ticks: { color: "#9db0d0" }, grid: { drawOnChartArea: false }, position: "right" },
      },
    },
  });

  const barCtx = document.getElementById("barChart").getContext("2d");
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
        y: { ticks: { color: "#9db0d0" }, grid: { color: "rgba(148, 163, 184, 0.08)" }, beginAtZero: true },
      },
    },
  });
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
  const response = await fetch("/stocks");
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
}

async function loadData() {
  const symbol = getSelectedSymbol();
  if (!symbol) return;
  setStatus(`Loading processed data for ${symbol}...`);
  appendTrainingLog(`Fetching processed data for ${symbol}...`, "dim");
  const response = await fetch(`/data?symbol=${encodeURIComponent(symbol)}`);
  const data = await response.json();
  latestDashboardData = data;
  renderDashboardSummary(data);
  renderFocusHighlights(data);
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
  await loadModelStats();
  await updateMarketRegime();
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
    await loadModelStats();
    updateChart(latestDashboardData);
    appendTrainingLog(`Prediction complete: ${data.label} (${(Number(data.confidence) * 100).toFixed(1)}%).`, "dim");
  } finally {
    setBusy(false);
  }
}

async function predictFutureDate() {
  setBusy(true);
  datePredictionStatus.textContent = "Generating date forecast...";
  appendTrainingLog("Running future-date forecast...", "dim");
  try {
    const mode = forecastModeSelect.value;
    const payload = buildFormData(resolveFutureForecastPayload());
    const endpoint = mode === "t1" || mode === "t3" || mode === "t7" ? "/forecast" : "/predict-date";
    const response = await fetch(endpoint, { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      datePredictionStatus.textContent = data.detail || "Date prediction failed.";
      setDataStatus(data.detail || "Date prediction failed.", "error");
      return;
    }
    latestForecastData = data;
    futureForecastLabel.textContent = data.label || "Neutral";
    futureConfidenceValue.textContent = `${(Number(data.confidence || 0) * 100).toFixed(0)}%`;
    setConfidenceRing(Number(data.confidence || 0) * 100);
    futurePriceRange.textContent = data.price_range ? `$${Number(data.price_range.low).toFixed(2)} - $${Number(data.price_range.high).toFixed(2)}` : `--`;
    futureRegimeValue.textContent = data.market_regime || "--";
    setMarketBadge(data.market_regime || "Sideways market");
    setTopbarState({
      symbol: getSelectedSymbol(),
      price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
      mode: getAttentionMode(),
      regime: data.market_regime || "Sideways market",
      lifecycle: `Forecast for ${data.future_date || futureDateInput.value}`,
    });
    renderExplainability(data, latestDashboardData);
    datePredictionStatus.textContent = `Forecast ready for ${data.future_date || futureDateInput.value}.`;
    setDataStatus(`Forecast completed for ${data.future_date || futureDateInput.value}.`, "success");
    addAlert(`Future forecast generated for ${data.future_date || futureDateInput.value}`, "info");
    updateChart({ ...latestDashboardData, focus_timesteps: data.focus_timesteps, volatility_scores: data.volatility_scores });
    appendTrainingLog(`Future-date forecast complete for ${data.future_date || futureDateInput.value}.`, "dim");
  } finally {
    setBusy(false);
  }
}

async function loadModelStats() {
  const symbol = getSelectedSymbol();
  if (!symbol) return;
  const response = await fetch(`/model-stats?symbol=${encodeURIComponent(symbol)}&use_standard_attention=${getAttentionMode() === "standard"}`);
  if (!response.ok) return;
  const stats = await response.json();
  renderModelStats(stats);
  setTopbarState({
    symbol: stats.symbol || getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: stats.attention_mode || getAttentionMode(),
    regime: marketRegimeBadge.textContent || "Sideways market",
    lifecycle: `${stats.attention_mode || getAttentionMode()} ready`,
  });
}

async function updateMarketRegime() {
  const symbol = getSelectedSymbol();
  if (!symbol) return;
  const response = await fetch(`/market-regime?symbol=${encodeURIComponent(symbol)}&use_standard_attention=${getAttentionMode() === "standard"}`);
  if (!response.ok) return;
  const data = await response.json();
  setMarketBadge(data.market_regime || "Sideways market");
  addAlert(`Market regime: ${data.market_regime}`, data.badge || "warning");
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: getAttentionMode(),
    regime: data.market_regime || "Sideways market",
    lifecycle: data.market_regime || "Regime updated",
  });
}

loadDataBtn.addEventListener("click", loadData);
trainBtn.addEventListener("click", trainModel);
predictBtn.addEventListener("click", predictTrend);
predictDateBtn.addEventListener("click", predictFutureDate);
trainUploadedBtn.addEventListener("click", trainModel);
simulateBuyBtn.addEventListener("click", simulateBuy);
resetPortfolioBtn.addEventListener("click", resetPortfolio);

tabLinks.forEach(button => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
});

// Old sidebar tab logic removed for multi-page architecture

forecastModeSelect.addEventListener("change", () => {
  const custom = forecastModeSelect.value === "custom_date";
  futureDateInput.disabled = !custom;
});

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

attentionModeSelect.addEventListener("change", () => {
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: getAttentionMode(),
    regime: marketRegimeBadge.textContent || "Sideways market",
    lifecycle: `${getAttentionMode() === "standard" ? "Standard" : "ASTA"} selected`,
  });
  loadModelStats().catch(console.error);
  updateMarketRegime().catch(console.error);
});

stockSelect.addEventListener("change", () => {
  if (dataStockSelect && dataStockSelect.value !== stockSelect.value) {
    dataStockSelect.value = stockSelect.value;
  }
  setTopbarState({
    symbol: getSelectedSymbol(),
    price: Number(latestDashboardData.current_price ?? latestDashboardData.recent_close_value ?? 0),
    mode: getAttentionMode(),
    regime: marketRegimeBadge.textContent || "Sideways market",
    lifecycle: `Switched to ${getSelectedSymbol()}`,
  });
  loadData().catch(console.error);
});

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

fileInputData.addEventListener("change", () => {
  const name = fileInputData.files[0]?.name || "No file selected";
  setDataStatus(`Selected: ${name}`, "success");
  appendTrainingLog(`Uploaded file selected: ${name}`, "dim");
});

if (refreshNewsBtn) {
  refreshNewsBtn.addEventListener("click", () => {
    loadPakistanNews().catch(console.error);
  });
}

fileInput.addEventListener("change", () => {
  const name = fileInput.files[0]?.name || "No file selected";
  setStatus(`Dashboard upload selected: ${name}`, "success");
  appendTrainingLog(`Dashboard upload selected: ${name}`, "dim");
});

loadStocks().then(loadData).catch(error => {
  console.error(error);
  setStatus("Failed to load stock list.", "error");
  alert("Failed to load stock list");
});

loadPakistanNews().catch(console.error);

resetPortfolio();
futureDateInput.disabled = true;
appendTrainingLog("Dashboard initialized.", "dim");
