/* dashboard.js – AI Intrusion Detection Dashboard */
let lineChart, pieChart, agreeChart, xaiChart;
let historyScores = [];
let heatmapChart;

// Canvas contexts
const scoreCtx = () => document.getElementById('scoreLine').getContext('2d');
const pieCtx = () => document.getElementById('threatPie').getContext('2d');
const agreeCtx = () => document.getElementById('agreementBar').getContext('2d');
const xaiCtx = () => document.getElementById('xaiBar').getContext('2d');
const heatmapCtx = () => document.getElementById('heatmap').getContext('2d');

function ensureChart(instance, type, ctx, data, options) {
    if (instance) instance.destroy();
    return new Chart(ctx, { type, data, options });
}

// -----------------------------------------------------------------
// Real‑time data fetcher
// -----------------------------------------------------------------
async function refresh() {
    try {
        const res = await fetch('/predict-stream');
        const out = await res.json();
        if (out.error) {
            console.error('API Error:', out.error);
            return;
        }
        // Update visualisations
        updateScoreLine(out.if.scores);
        updateThreatPie(out.categories);
        updateAgreement(out);
        updateXAI(out);
        updateHeatmap(out.heatmap);
        updateStatistics(out.statistics);
        // Alerts
        fetchAlertStatus();
        const timeEl = document.getElementById('lastUpdate');
        if (timeEl) timeEl.textContent = 'Last update: ' + new Date().toLocaleTimeString();
    } catch (e) {
        console.error('Refresh error:', e);
    }
}

// -----------------------------------------------------------------
// Score line chart
// -----------------------------------------------------------------
function updateScoreLine(scores) {
    historyScores = scores;
    const labels = scores.map((_, i) => i + 1);
    lineChart = ensureChart(lineChart, 'line', scoreCtx(), {
        labels,
        datasets: [{
            label: 'Anomaly Score (Auto-Encoder)',
            data: scores,
            borderColor: 'rgba(239, 68, 68, 1)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 5
        }]
    }, {
        responsive: true,
        maintainAspectRatio: true,
        animation: false,
        scales: {
            y: { ticks: { color: '#a1a1a1' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
            x: { ticks: { color: '#a1a1a1' }, grid: { display: false } }
        },
        plugins: { legend: { labels: { color: '#a1a1a1' } } }
    });
}

// -----------------------------------------------------------------
// Threat pie chart
// -----------------------------------------------------------------
function updateThreatPie(categories) {
    const count = {};
    categories.forEach(c => {
        const k = c?.label || 'Other';
        count[k] = (count[k] || 0) + 1;
    });
    pieChart = ensureChart(pieChart, 'doughnut', pieCtx(), {
        labels: Object.keys(count),
        datasets: [{
            data: Object.values(count),
            backgroundColor: [
                'rgba(239, 68, 68, 0.8)',   // DDoS
                'rgba(59, 130, 246, 0.8)',  // Normal
                'rgba(251, 191, 36, 0.8)',  // SQL Injection
                'rgba(16, 185, 129, 0.8)'   // Brute Force
            ]
        }]
    }, {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { position: 'right', labels: { color: '#a1a1a1' } } }
    });
}

// -----------------------------------------------------------------
// Model agreement bar chart (SVM / Isolation‑Forest / Auto‑Encoder)
// -----------------------------------------------------------------
function updateAgreement(out) {
    const modelPreds = out.model_predictions || {};
    const agreement = modelPreds.agreement_percentage || 0;
    const agreementEl = document.getElementById('agreementPercent');
    if (agreementEl) agreementEl.textContent = `${agreement.toFixed(1)}%`;

    const svmRate = modelPreds.svm ? (modelPreds.svm.filter(x => x === 1).length / modelPreds.svm.length) : 0;
    const ifRate = modelPreds.isolation_forest ? (modelPreds.isolation_forest.filter(x => x === 1).length / modelPreds.isolation_forest.length) : 0;
    const autoencoderRate = modelPreds.autoencoder ? (modelPreds.autoencoder.filter(x => x === 1).length / modelPreds.autoencoder.length) : 0;

    agreeChart = ensureChart(agreeChart, 'bar', agreeCtx(), {
        labels: ['SVM', 'Isolation Forest', 'Auto-Encoder'],
        datasets: [{
            label: 'Anomaly Rate',
            data: [svmRate, ifRate, autoencoderRate],
            backgroundColor: [
                'rgba(239, 68, 68, 0.8)',
                'rgba(59, 130, 246, 0.8)',
                'rgba(16, 185, 129, 0.8)'
            ]
        }]
    }, {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
            y: { min: 0, max: 1, ticks: { color: '#a1a1a1' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
            x: { ticks: { color: '#a1a1a1' }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
    });
}

// -----------------------------------------------------------------
// Explainable‑AI bar chart
// -----------------------------------------------------------------
function updateXAI(out) {
    const feats = out.features || [];
    const imps = out.xai_proxy?.feature_importances || [];
    xaiChart = ensureChart(xaiChart, 'bar', xaiCtx(), {
        labels: feats,
        datasets: [{
            label: 'Feature Importance',
            data: imps,
            backgroundColor: 'rgba(147, 51, 234, 0.8)'
        }]
    }, {
        responsive: true,
        maintainAspectRatio: true,
        indexAxis: 'y',
        scales: {
            x: { beginAtZero: true, ticks: { color: '#a1a1a1' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
            y: { ticks: { color: '#a1a1a1' }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
    });
}

// -----------------------------------------------------------------
// Statistics grid
// -----------------------------------------------------------------
function updateStatistics(stats) {
    if (!stats) return;
    const statsEl = document.getElementById('statsDisplay');
    if (!statsEl) return;
    const anomalyRateClass = stats.anomaly_percentage > 5 ? 'warning' : '';
    statsEl.innerHTML = `
        <div class="stat-item"><div class="stat-label">Total Packets:</div><div class="stat-value">${stats.total_packets}</div></div>
        <div class="stat-item"><div class="stat-label">Anomalies:</div><div class="stat-value anomaly">${stats.anomaly_count}</div></div>
        <div class="stat-item"><div class="stat-label">Normal:</div><div class="stat-value normal">${stats.normal_count}</div></div>
        <div class="stat-item"><div class="stat-label">Anomaly Rate:</div><div class="stat-value ${anomalyRateClass}">${stats.anomaly_percentage.toFixed(0)}%</div></div>
    `;
}

// -----------------------------------------------------------------
// Alert status handling
// -----------------------------------------------------------------
async function fetchAlertStatus() {
    try {
        const res = await fetch('/alert-status');
        const data = await res.json();
        updateAlertStatus(data);
    } catch (e) {
        console.error('Alert status error:', e);
    }
}

function updateAlertStatus(data) {
    const alertEl = document.getElementById('alertStatus');
    if (!alertEl) return;
    if (!data.last_alert) {
        alertEl.innerHTML = `
            <div class="stat-item"><div class="stat-label">Total Alerts:</div><div class="stat-value">0</div></div>
            <div class="stat-item" style="grid-column: span 2;"><div class="stat-label">Last Alert:</div><div class="stat-value" style="font-size: 1rem;">None yet</div></div>
        `;
        return;
    }
    let recentHtml = '';
    if (data.recent_alerts && data.recent_alerts.length > 0) {
        recentHtml = `
            <div style="grid-column: span 2; margin-top: 8px;">
                <strong style="color: var(--text-secondary);">Recent Alerts:</strong>
                <div class="recent-alerts" style="margin-top: 8px;">
                    ${data.recent_alerts.map(alert => `<div class="alert-item">${alert.time} - ${alert.anomalies}/${alert.total} (${alert.rate}%)</div>`).join('')}
                </div>
            </div>
        `;
    }
    alertEl.innerHTML = `
        <div class="stat-item"><div class="stat-label">Total Alerts:</div><div class="stat-value">${data.alert_count}</div></div>
        <div class="stat-item" style="grid-column: span 2;"><div class="stat-label">Last Alert:</div><div class="stat-value" style="font-size: 1rem;">${data.last_alert}</div></div>
        ${recentHtml}
    `;
}

// -----------------------------------------------------------------
// Density heatmap
// -----------------------------------------------------------------
function updateHeatmap(data) {
    const maxValue = Math.max(...data.map(d => d.v));
    heatmapChart = ensureChart(heatmapChart, 'matrix', heatmapCtx(), {
        datasets: [{
            label: 'Density Heatmap',
            data: data,
            backgroundColor: ctx => {
                const value = ctx.raw.v;
                if (value > maxValue * 0.6) return 'rgba(255,60,60,0.9)'; // hot
                if (value > maxValue * 0.3) return 'rgba(255,180,0,0.9)'; // mid
                return 'rgba(0,255,157,0.6)'; // normal
            },
            width: () => 16,
            height: () => 16
        }]
    }, {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
            x: { ticks: { color: '#aaa' }, grid: { display: false } },
            y: { ticks: { color: '#aaa' }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
    });
}

// -----------------------------------------------------------------
// Auto‑refresh toggle
// -----------------------------------------------------------------
let timer = null;
function toggleAuto(on) {
    if (timer) clearInterval(timer);
    if (on) timer = setInterval(refresh, 3000);
}
document.getElementById('autoRefresh').addEventListener('change', e => toggleAuto(e.target.checked));

// -----------------------------------------------------------------
// Gemini summary loader
// -----------------------------------------------------------------
async function loadGeminiSummary() {
    try {
        const res = await fetch('/ai-summary');
        const out = await res.json();
        document.getElementById('aiSummary').innerHTML = out.summary || 'No summary generated.';
    } catch (e) {
        document.getElementById('aiSummary').innerHTML = 'Failed to load AI summary.';
    }
}

// -----------------------------------------------------------------
// Initialise dashboard
// -----------------------------------------------------------------
toggleAuto(true);
refresh();
setInterval(loadGeminiSummary, 15000);
loadGeminiSummary();