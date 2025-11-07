let lineChart, pieChart, agreeChart, xaiChart;
let historyScores = [];

const scoreCtx = () => document.getElementById("scoreLine").getContext("2d");
const pieCtx = () => document.getElementById("threatPie").getContext("2d");
const agreeCtx = () => document.getElementById("agreementBar").getContext("2d");
const xaiCtx = () => document.getElementById("xaiBar").getContext("2d");

function ensureChart(instance, type, ctx, data, options) {
    if (instance) instance.destroy();
    return new Chart(ctx, { type, data, options });
}

// Temporary mock data generator (for testing)
async function refresh() {
    try {
        const scores = Array.from({ length: 50 }, () => Math.random());
        const categories = [
            { label: "DDoS", score: Math.random() },
            { label: "SQL Injection", score: Math.random() },
            { label: "Brute Force", score: Math.random() },
        ];

        const out = {
            if: { scores },
            categories,
            xai_proxy: { feature_importances: [0.4, 0.25, 0.35] },
            features: ["src_port", "dst_port", "length"],
        };

        updateScoreLine(out.if.scores);
        updateThreatPie(out.categories);
        updateAgreement(out);
        updateXAI(out);

        const timeEl = document.getElementById("lastUpdate");
        if (timeEl) timeEl.textContent = "Last update: " + new Date().toLocaleTimeString();

    } catch (e) {
        console.error("Refresh error:", e);
    }
}

function updateScoreLine(scores) {
    historyScores = scores;
    const labels = scores.map((_, i) => i + 1);
    lineChart = ensureChart(lineChart, "line", scoreCtx(), {
        labels,
        datasets: [{ label: "Anomaly Score (IF)", data: scores, borderWidth: 2, tension: 0.3 }]
    }, { responsive: true, animation: false });
}

function updateThreatPie(categories) {
    const count = {};
    categories.forEach(c => {
        const k = c?.label || "Other";
        count[k] = (count[k] || 0) + 1;
    });
    pieChart = ensureChart(pieChart, "doughnut", pieCtx(), {
        labels: Object.keys(count),
        datasets: [{ data: Object.values(count) }]
    }, { responsive: true });
}

function updateAgreement(out) {
    const models = ["ae", "if", "lof", "svm"];
    const labels = models.map(m => m.toUpperCase());
    const values = [Math.random(), Math.random(), Math.random(), Math.random()];
    agreeChart = ensureChart(agreeChart, "bar", agreeCtx(), {
        labels,
        datasets: [{ label: "Anomaly Rate", data: values }]
    }, { responsive: true, scales: { y: { min: 0, max: 1 } } });
}

function updateXAI(out) {
    const feats = out.features || [];
    const imps = out.xai_proxy?.feature_importances || [];
    xaiChart = ensureChart(xaiChart, "bar", xaiCtx(), {
        labels: feats,
        datasets: [{ label: "Feature Importance", data: imps }]
    }, { responsive: true, scales: { y: { beginAtZero: true } } });
}


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
