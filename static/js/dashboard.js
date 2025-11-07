// === Live AI Intrusion Dashboard ===
// ⚡ Real-time visualization for Isolation Forest + AI threat models

let lineChart, pieChart, agreeChart, xaiChart;
let historyScores = [];

// === Chart.js Global Neon Theme ===
Chart.defaults.color = "#e4e6eb";
Chart.defaults.borderColor = "rgba(255,255,255,0.1)";
Chart.defaults.font.family = "Fira Code";
Chart.defaults.plugins.legend.labels.color = "#b0b3b8";
Chart.defaults.plugins.tooltip.backgroundColor = "rgba(0,0,0,0.85)";
Chart.defaults.plugins.tooltip.titleColor = "#00ffc8";
Chart.defaults.plugins.tooltip.bodyColor = "#e4e6eb";

// === Context Accessors ===
const scoreCtx = () => document.getElementById("scoreLine").getContext("2d");
const pieCtx = () => document.getElementById("threatPie").getContext("2d");
const agreeCtx = () => document.getElementById("agreementBar").getContext("2d");
const xaiCtx = () => document.getElementById("xaiBar").getContext("2d");

// === Chart Utility ===
function ensureChart(instance, type, ctx, data, options) {
    if (instance) instance.destroy();
    return new Chart(ctx, { type, data, options });
}

// === Backend Fetch ===
async function fetchStream() {
    const res = await fetch("/predict-stream");
    return await res.json();
}

// === LIVE Anomaly Line Chart ===
function ScoreLine(scores) {
    if (!scores || scores.length === 0) return;

    // Keep only last 200 points
    historyScores = historyScores.concat(scores);
    if (historyScores.length > 200) historyScores = historyScores.slice(-200);

    // Create chart if not already
    if (!lineChart) {
        lineChart = new Chart(scoreCtx(), {
            type: "line",
            data: {
                labels: historyScores.map((_, i) => i + 1),
                datasets: [{
                    label: "Anomaly Score (IF)",
                    data: historyScores,
                    borderWidth: 2,
                    borderColor: "#00ffc8",
                    backgroundColor: "rgba(0, 255, 200, 0.08)",
                    tension: 0.35,
                    pointRadius: 0,
                    fill: true,
                }],
            },
            options: {
                responsive: true,
                animation: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: "rgba(255,255,255,0.05)" },
                    },
                    x: {
                        grid: { color: "rgba(255,255,255,0.05)" },
                        display: false,
                    },
                },
            },
        });
    } else {
        // Smooth live scrolling
        const latest = scores[scores.length - 1];
        lineChart.data.labels.push(lineChart.data.labels.length + 1);
        lineChart.data.datasets[0].data.push(latest);

        if (lineChart.data.labels.length > 200) {
            lineChart.data.labels.shift();
            lineChart.data.datasets[0].data.shift();
        }

        lineChart.update("none");
    }
}

// === Threat Distribution (Pie) ===
function updateThreatPie(categories) {
    const count = {};
    categories.forEach(c => {
        const k = c?.label || "Normal";
        count[k] = (count[k] || 0) + 1;
    });

    const labels = Object.keys(count);
    const data = Object.values(count);
    const total = data.reduce((a, b) => a + b, 0);
    const percents = data.map(v => ((v / total) * 100).toFixed(1));

    const neonColors = [
        "#00fff7", "#ff00f7", "#00ff9f",
        "#ffc300", "#ff5c5c", "#6c63ff",
        "#00e5ff", "#ff8c00"
    ];

    pieChart = ensureChart(
        pieChart,
        "doughnut",
        pieCtx(),
        {
            labels,
            datasets: [{
                data,
                backgroundColor: neonColors.slice(0, labels.length),
                borderColor: "#0affea",
                borderWidth: 2,
                hoverOffset: 12,
            }],
        },
        {
            responsive: true,
            cutout: "60%",
            animation: { animateRotate: true, duration: 1200 },
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#b8fff4", font: { size: 13 } },
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${percents[ctx.dataIndex]}% (${data[ctx.dataIndex]})`,
                    },
                },
                title: {
                    display: true,
                    text: "Threat Distribution (AI-Detected)",
                    color: "#00ffcc",
                    font: { size: 16, weight: "bold" },
                },
            },
        }
    );
}

// === Model Agreement (Bar) ===
function updateAgreement(out) {
    const models = ["ae", "if", "lof", "svm"];
    const labels = [];
    const values = [];

    models.forEach(m => {
        if (out[m] && out[m].labels) {
            const arr = out[m].labels;
            const frac = arr.length
                ? arr.reduce((a, b) => a + (b === 1 ? 1 : 0), 0) / arr.length
                : 0;
            labels.push(m.toUpperCase());
            values.push(frac);
        }
    });

    agreeChart = ensureChart(
        agreeChart,
        "bar",
        agreeCtx(),
        {
            labels,
            datasets: [{
                label: "Anomaly Rate",
                data: values,
                backgroundColor: [
                    "rgba(0,255,200,0.6)",
                    "rgba(0,150,255,0.6)",
                    "rgba(255,180,0,0.6)",
                    "rgba(255,0,80,0.6)",
                ],
            }],
        },
        {
            responsive: true,
            animation: { duration: 1000, easing: "easeOutElastic" },
            scales: {
                y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { grid: { color: "rgba(255,255,255,0.05)" } },
            },
        }
    );
}

// === XAI Feature Importance (Bar) ===
function updateXAI(out) {
    const proxy = out.xai_proxy;
    const feats = out.features || [];
    const imps = proxy?.feature_importances || [];
    const list = document.getElementById("xaiList");
    list.innerHTML = "";

    feats.forEach((f, i) => {
        const v = imps[i] || 0;
        const row = document.createElement("div");
        row.className = "xai-row";
        row.innerHTML = `<span>${f}</span><span>${v.toFixed(2)}</span>`;
        list.appendChild(row);
    });

    xaiChart = ensureChart(
        xaiChart,
        "bar",
        xaiCtx(),
        {
            labels: feats,
            datasets: [{
                label: "Feature Importance",
                data: imps,
                backgroundColor: "rgba(0,255,200,0.6)",
            }],
        },
        {
            responsive: true,
            animation: { duration: 800, easing: "easeInOutSine" },
            scales: {
                y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.05)" } },
                x: { grid: { color: "rgba(255,255,255,0.05)" } },
            },
        }
    );
}

// === Predictive Threat Intel ===
function updateIntel(intel) {
    const div = document.getElementById("intelList");
    if (!div) return;

    div.innerHTML = "";
    if (!intel || intel.length === 0) {
        div.innerHTML = "<p>No suspicious IPs detected 🚀</p>";
        return;
    }

    intel.forEach(i => {
        const row = document.createElement("div");
        row.className = "intel-row";

        let riskClass = "low";
        let riskLabel = "🟢 Low";
        if (i.threat_score > 70) {
            riskClass = "high";
            riskLabel = "🔴 High";
        } else if (i.threat_score > 40) {
            riskClass = "medium";
            riskLabel = "🟡 Medium";
        }

        row.innerHTML = `
          <span class="ip">${i.ip}</span>
          <span class="score ${riskClass}">${riskLabel} (${i.threat_score.toFixed(1)}%)</span>
        `;
        div.appendChild(row);
    });
}

// === Auto-Refresh Cycle ===
async function refresh() {
    try {
        const out = await fetchStream();
        if (!out) return;

        updateScoreLine(out.if?.scores || []);
        updateThreatPie(out.categories || []);
        updateAgreement(out);
        updateXAI(out);
        updateIntel(out.intel || []);

        const timeEl = document.getElementById("lastUpdate");
        if (timeEl)
            timeEl.textContent = "Last update: " + new Date().toLocaleTimeString();
    } catch (e) {
        console.error("Refresh error:", e);
    }
}

// === Voice Alert ===
document.getElementById("voiceBtn").addEventListener("click", async () => {
    const res = await fetch("/alert-voice", { method: "POST" });
    const out = await res.json();
    if (out.ok) new Audio("/" + out.path).play();
    else alert("Voice alert failed: " + out.error);
});

// === Auto Refresh ===
let timer = null;
function toggleAuto(on) {
    if (timer) clearInterval(timer);
    if (on) timer = setInterval(refresh, 3000);
}
document.getElementById("autoRefresh").addEventListener("change", e => toggleAuto(e.target.checked));

// === Gemini Summary Loader ===
async function loadGeminiSummary() {
    try {
        const res = await fetch("/ai-summary");
        const out = await res.json();
        document.getElementById("aiSummary").innerHTML = out.summary || "No summary generated.";
    } catch (e) {
        document.getElementById("aiSummary").innerHTML = "Failed to load AI summary.";
    }
}

// === Init Dashboard ===
toggleAuto(true);
refresh();
setInterval(loadGeminiSummary, 15000);
loadGeminiSummary();
