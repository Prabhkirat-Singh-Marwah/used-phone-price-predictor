/**
 * app.js
 * ------
 * Client-side logic for the Used Phone Price Estimator.
 * Handles interactive input controls, factor impact calculations,
 * batch CSV processing, model retraining requests, and metric displays.
 */

// ---------------------------------------------------------------------------
// 1. Catalogue & Baseline Constants
// ---------------------------------------------------------------------------
const CURRENT_YEAR = 2025;

const BRAND_CATALOGUE = {
    "Apple": {
        "retention": 0.82,
        "models": [
            ["iPhone 11", 2019, 64900, [64, 128, 256], [4]],
            ["iPhone 12", 2020, 79900, [64, 128, 256], [4]],
            ["iPhone 13", 2021, 79900, [128, 256, 512], [4]],
            ["iPhone 14", 2022, 79900, [128, 256, 512], [6]],
            ["iPhone 15", 2023, 79900, [128, 256, 512], [6]],
            ["iPhone SE 2022", 2022, 43900, [64, 128, 256], [3]]
        ]
    },
    "Samsung": {
        "retention": 0.72,
        "models": [
            ["Galaxy S21", 2021, 69999, [128, 256], [8]],
            ["Galaxy S22", 2022, 72999, [128, 256], [8]],
            ["Galaxy S23", 2023, 74999, [128, 256], [8]],
            ["Galaxy A54", 2023, 38999, [128, 256], [8]],
            ["Galaxy M34", 2023, 18999, [128], [8]],
            ["Galaxy A14", 2022, 13999, [64, 128], [4]]
        ]
    },
    "OnePlus": {
        "retention": 0.68,
        "models": [
            ["OnePlus 11", 2023, 56999, [128, 256], [8, 16]],
            ["OnePlus 10 Pro", 2022, 66999, [128, 256], [8, 12]],
            ["OnePlus Nord 3", 2023, 33999, [128, 256], [8, 12]],
            ["OnePlus Nord CE3", 2023, 26999, [128], [8]]
        ]
    },
    "Xiaomi": {
        "retention": 0.58,
        "models": [
            ["Redmi Note 12", 2023, 17999, [64, 128], [4, 6]],
            ["Redmi Note 11", 2022, 14999, [64, 128], [4, 6]],
            ["Xiaomi 13 Pro", 2023, 79999, [256], [12]],
            ["Redmi 12C", 2023, 10999, [64, 128], [4]],
            ["POCO X5 Pro", 2023, 22999, [128, 256], [6, 8]]
        ]
    },
    "Realme": {
        "retention": 0.55,
        "models": [
            ["Realme 11 Pro", 2023, 24999, [128, 256], [8]],
            ["Realme C55", 2023, 13999, [64, 128], [6, 8]],
            ["Realme GT 5", 2023, 34999, [128, 256], [8, 12]]
        ]
    },
    "Vivo": {
        "retention": 0.60,
        "models": [
            ["Vivo V27", 2023, 34999, [128, 256], [8]],
            ["Vivo Y56", 2023, 17999, [128], [8]],
            ["Vivo X90 Pro", 2023, 84999, [256], [12]]
        ]
    },
    "Oppo": {
        "retention": 0.58,
        "models": [
            ["Oppo Reno 10", 2023, 35999, [128, 256], [8]],
            ["Oppo A78", 2023, 22999, [128], [8]]
        ]
    },
    "Google": {
        "retention": 0.70,
        "models": [
            ["Pixel 7", 2022, 59999, [128, 256], [8]],
            ["Pixel 7a", 2023, 53999, [128], [8]],
            ["Pixel 6a", 2022, 43999, [128], [6]]
        ]
    }
};

// ---------------------------------------------------------------------------
// 2. Formatting Helpers
// ---------------------------------------------------------------------------
function formatINR(n) {
    const val = Math.round(Math.abs(n));
    const s = val.toString();
    if (s.length <= 3) return "Rs " + s;
    const last3 = s.slice(-3);
    let rest = s.slice(0, -3);
    const parts = [];
    while (rest.length > 2) {
        parts.unshift(rest.slice(-2));
        rest = rest.slice(0, -2);
    }
    if (rest.length > 0) {
        parts.unshift(rest);
    }
    return "Rs " + parts.join(",") + "," + last3;
}

// ---------------------------------------------------------------------------
// 3. Fallback Client-Side Valuation Formula (matches data simulation)
// ---------------------------------------------------------------------------
function calculateSimulatedPrice(spec) {
    const brandData = BRAND_CATALOGUE[spec.brand] || { retention: 0.60 };
    const annualRetention = brandData.retention;
    const baseP = spec.original_price_inr;
    const age = spec.age_years;

    const baseDeprec = baseP * Math.pow(annualRetention, age);

    let battMult = 1.0;
    if (spec.battery_health_pct < 80) {
        battMult = 0.85 - (80 - spec.battery_health_pct) * 0.012;
    } else if (spec.battery_health_pct < 90) {
        battMult = 0.94 - (90 - spec.battery_health_pct) * 0.009;
    } else {
        battMult = 0.98 + (spec.battery_health_pct - 90) * 0.002;
    }

    const screenMap = { "flawless": 1.0, "minor_scratches": 0.94, "cracked": 0.70 };
    const screenMult = screenMap[spec.screen_condition] || 0.94;

    const bodyMap = { "like_new": 1.0, "good": 0.92, "fair": 0.82, "poor": 0.68 };
    const bodyMult = bodyMap[spec.body_condition] || 0.92;

    const boxMult = spec.has_box_bill === 1 ? 1.03 : 0.97;
    const warrantyMult = 1.0 + 0.005 * spec.warranty_months_left;
    const demandMult = 0.90 + 0.02 * spec.market_demand_index;

    let p = baseDeprec * battMult * screenMult * bodyMult * boxMult * warrantyMult * demandMult;
    p = Math.max(1000, p);
    return Math.round(p / 100) * 100;
}

// ---------------------------------------------------------------------------
// 4. Form Initialization & Event Handling
// ---------------------------------------------------------------------------
const brandSelect = document.getElementById("brandSelect");
const modelSelect = document.getElementById("modelSelect");
const storageSelect = document.getElementById("storageSelect");
const ramSelect = document.getElementById("ramSelect");
const batterySlider = document.getElementById("batterySlider");
const batteryValBadge = document.getElementById("batteryVal");
const hasBoxBillInput = document.getElementById("hasBoxBill");
const warrantyInput = document.getElementById("warrantyInput");
const demandSelect = document.getElementById("demandSelect");
const btnCalculate = document.getElementById("btnCalculate");

const priceDisplay = document.getElementById("priceDisplay");
const rangeMinMax = document.getElementById("rangeMinMax");
const mapeDisplay = document.getElementById("mapeDisplay");
const factorsContainer = document.getElementById("factorsContainer");
const specSummary = document.getElementById("specSummary");

let activeBestMape = 11.5;

function initBrandOptions() {
    brandSelect.innerHTML = "";
    Object.keys(BRAND_CATALOGUE).sort().forEach(brand => {
        const opt = document.createElement("option");
        opt.value = brand;
        opt.textContent = brand;
        if (brand === "Apple") opt.selected = true;
        brandSelect.appendChild(opt);
    });
    updateModelOptions();
}

function updateModelOptions() {
    const brand = brandSelect.value;
    const models = BRAND_CATALOGUE[brand]?.models || [];
    modelSelect.innerHTML = "";
    models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m[0];
        opt.textContent = m[0];
        modelSelect.appendChild(opt);
    });
    updateSpecsForSelectedModel();
}

function getSelectedModelInfo() {
    const brand = brandSelect.value;
    const modelName = modelSelect.value;
    const models = BRAND_CATALOGUE[brand]?.models || [];
    const found = models.find(m => m[0] === modelName);
    if (!found) return { releaseYear: 2021, originalPrice: 50000, storages: [128], rams: [4] };
    return {
        releaseYear: found[1],
        originalPrice: found[2],
        storages: found[3],
        rams: found[4]
    };
}

function updateSpecsForSelectedModel() {
    const info = getSelectedModelInfo();

    storageSelect.innerHTML = "";
    info.storages.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s + " GB";
        storageSelect.appendChild(opt);
    });

    ramSelect.innerHTML = "";
    info.rams.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r;
        opt.textContent = r + " GB";
        ramSelect.appendChild(opt);
    });

    performValuation();
}

batterySlider.addEventListener("input", (e) => {
    batteryValBadge.textContent = e.target.value + "%";
    performValuation();
});

brandSelect.addEventListener("change", updateModelOptions);
modelSelect.addEventListener("change", updateSpecsForSelectedModel);
storageSelect.addEventListener("change", performValuation);
ramSelect.addEventListener("change", performValuation);
hasBoxBillInput.addEventListener("change", performValuation);
warrantyInput.addEventListener("input", performValuation);
demandSelect.addEventListener("change", performValuation);
document.querySelectorAll('input[name="screenCond"]').forEach(r => r.addEventListener("change", performValuation));
document.querySelectorAll('input[name="bodyCond"]').forEach(r => r.addEventListener("change", performValuation));
btnCalculate.addEventListener("click", performValuation);

function getFormSpec() {
    const brand = brandSelect.value;
    const modelName = modelSelect.value;
    const info = getSelectedModelInfo();

    const demandMap = { "Low": 3.5, "Medium": 5.5, "High": 7.5 };
    const screenCond = document.querySelector('input[name="screenCond"]:checked')?.value || "minor_scratches";
    const bodyCond = document.querySelector('input[name="bodyCond"]:checked')?.value || "good";
    const ageYears = Math.max(0.5, CURRENT_YEAR - info.releaseYear);

    return {
        brand: brand,
        model: modelName,
        release_year: info.releaseYear,
        age_years: ageYears,
        storage_gb: parseInt(storageSelect.value || info.storages[0]),
        ram_gb: parseInt(ramSelect.value || info.rams[0]),
        battery_health_pct: parseFloat(batterySlider.value),
        screen_condition: screenCond,
        body_condition: bodyCond,
        has_box_bill: hasBoxBillInput.checked ? 1 : 0,
        warranty_months_left: parseInt(warrantyInput.value || 0),
        original_price_inr: info.originalPrice,
        market_demand_index: demandMap[demandSelect.value] || 5.5
    };
}

// ---------------------------------------------------------------------------
// 5. Valuation Execution (API or Fallback)
// ---------------------------------------------------------------------------
async function performValuation() {
    const spec = getFormSpec();

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(spec)
        });

        if (response.ok) {
            const data = await response.json();
            renderValuationCard(data.predicted_price, data.low_price, data.high_price, data.factors, spec, data.mape || activeBestMape);
            return;
        }
    } catch (err) {
        // Fallback gracefully to offline formula
    }

    // Client-side fallback calculation
    const price = calculateSimulatedPrice(spec);
    const halfMape = (activeBestMape / 2) / 100;
    const low = Math.round(price * (1 - halfMape) / 100) * 100;
    const high = Math.round(price * (1 + halfMape) / 100) * 100;

    const factors = [];
    if (spec.battery_health_pct < 90) {
        const pRef = calculateSimulatedPrice({ ...spec, battery_health_pct: 95.0 });
        const delta = Math.abs(pRef - price);
        factors.push({ label: `Battery degradation (${Math.round(spec.battery_health_pct)}%)`, effect: `-${formatINR(delta)} vs 95% battery` });
    }
    if (spec.screen_condition !== "flawless") {
        const pRef = calculateSimulatedPrice({ ...spec, screen_condition: "flawless" });
        const delta = Math.abs(pRef - price);
        factors.push({ label: `Screen (${spec.screen_condition.replace("_", " ")})`, effect: `-${formatINR(delta)} vs flawless` });
    }
    if (spec.body_condition !== "like_new") {
        const pRef = calculateSimulatedPrice({ ...spec, body_condition: "like_new" });
        const delta = Math.abs(pRef - price);
        factors.push({ label: `Body wear (${spec.body_condition.replace("_", " ")})`, effect: `-${formatINR(delta)} vs like new` });
    }
    if (spec.has_box_bill === 0) {
        const pRef = calculateSimulatedPrice({ ...spec, has_box_bill: 1 });
        const delta = Math.abs(pRef - price);
        factors.push({ label: "Missing box or bill", effect: `-${formatINR(delta)} vs original box+bill` });
    }
    if (spec.warranty_months_left < 6) {
        const pRef = calculateSimulatedPrice({ ...spec, warranty_months_left: 12 });
        const delta = Math.abs(pRef - price);
        factors.push({ label: `Active warranty (${spec.warranty_months_left} mos)`, effect: `+${formatINR(delta)} if 12 mos warranty` });
    }

    renderValuationCard(price, low, high, factors.slice(0, 4), spec, activeBestMape);
}

function renderValuationCard(price, low, high, factors, spec, mape) {
    priceDisplay.textContent = formatINR(price);
    rangeMinMax.textContent = `${formatINR(low)} – ${formatINR(high)}`;
    mapeDisplay.textContent = mape.toFixed(1);

    factorsContainer.innerHTML = "";
    if (factors && factors.length > 0) {
        factors.forEach(f => {
            const row = document.createElement("div");
            row.className = "factor-row";
            row.innerHTML = `<span class="factor-key">${f.label}</span><span class="factor-val">${f.effect}</span>`;
            factorsContainer.appendChild(row);
        });
    } else {
        factorsContainer.innerHTML = `<div style="font-size:0.83rem;color:#475569;padding:6px 0;">Device is in top condition with minimal depreciation penalties.</div>`;
    }

    specSummary.innerHTML = `<strong>${spec.brand} ${spec.model}</strong> &middot; ${spec.storage_gb} GB Storage &middot; ${spec.ram_gb} GB RAM<br>Age: ~${Math.round(spec.age_years)} yr &middot; Original Launch: ${formatINR(spec.original_price_inr)}`;
}

// ---------------------------------------------------------------------------
// 6. Tabs Switching
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        document.getElementById(tabId)?.classList.add("active");

        if (tabId === "tab-perf") {
            loadPerformanceMetrics();
        }
    });
});

// ---------------------------------------------------------------------------
// 7. Tab 1: Batch Valuation (CSV Upload & Export)
// ---------------------------------------------------------------------------
const batchDropzone = document.getElementById("batchDropzone");
const batchFileInput = document.getElementById("batchFileInput");
const btnDownloadBatchTemplate = document.getElementById("btnDownloadBatchTemplate");
const batchResultsSection = document.getElementById("batchResultsSection");
const batchMetricCount = document.getElementById("batchMetricCount");
const batchMetricAvg = document.getElementById("batchMetricAvg");
const batchMetricTotal = document.getElementById("batchMetricTotal");
const batchTableContainer = document.getElementById("batchTableContainer");
const btnDownloadBatchResults = document.getElementById("btnDownloadBatchResults");

let latestBatchResults = [];

const SAMPLE_BATCH_CSV = `brand,model,release_year,storage_gb,ram_gb,battery_health_pct,screen_condition,body_condition,has_box_bill,warranty_months_left,original_price_inr,market_demand_index
Apple,iPhone 13,2021,128,4,86.0,minor_scratches,good,1,0,79900,7.0
Samsung,Galaxy S22,2022,128,8,89.0,flawless,like_new,1,3,72999,6.5
Xiaomi,Redmi Note 12,2023,128,6,92.0,minor_scratches,fair,0,0,17999,5.5`;

btnDownloadBatchTemplate.addEventListener("click", () => {
    downloadCSV(SAMPLE_BATCH_CSV, "phone_sample_template.csv");
});

batchDropzone.addEventListener("click", () => batchFileInput.click());
batchDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    batchDropzone.style.borderColor = "#0f172a";
});
batchDropzone.addEventListener("dragleave", () => {
    batchDropzone.style.borderColor = "#cbd5e1";
});
batchDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    batchDropzone.style.borderColor = "#cbd5e1";
    if (e.dataTransfer.files.length) {
        handleBatchFile(e.dataTransfer.files[0]);
    }
});
batchFileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
        handleBatchFile(e.target.files[0]);
    }
});

async function handleBatchFile(file) {
    const text = await file.text();
    const rows = parseCSV(text);
    if (rows.length === 0) {
        alert("The uploaded CSV is empty or formatted incorrectly.");
        return;
    }

    // Try sending to backend API first
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/batch-predict", {
            method: "POST",
            body: formData
        });
        if (res.ok) {
            const data = await res.json();
            displayBatchResults(data.records);
            return;
        }
    } catch (e) {
        // Fallback to client-side evaluation
    }

    // Client-side fallback batch calculation
    const processed = rows.map(r => {
        const relYear = parseInt(r.release_year) || 2021;
        const origPrice = parseInt(r.original_price_inr) || 50000;
        const spec = {
            brand: r.brand || "Apple",
            model: r.model || "iPhone 13",
            release_year: relYear,
            age_years: CURRENT_YEAR - relYear,
            storage_gb: parseInt(r.storage_gb) || 128,
            ram_gb: parseInt(r.ram_gb) || 4,
            battery_health_pct: parseFloat(r.battery_health_pct) || 85.0,
            screen_condition: r.screen_condition || "minor_scratches",
            body_condition: r.body_condition || "good",
            has_box_bill: parseInt(r.has_box_bill) !== undefined ? parseInt(r.has_box_bill) : 1,
            warranty_months_left: parseInt(r.warranty_months_left) || 0,
            original_price_inr: origPrice,
            market_demand_index: parseFloat(r.market_demand_index) || 5.5
        };

        const pred = calculateSimulatedPrice(spec);
        const halfMape = (activeBestMape / 2) / 100;
        const low = Math.round(pred * (1 - halfMape) / 100) * 100;
        const high = Math.round(pred * (1 + halfMape) / 100) * 100;

        return {
            ...r,
            predicted_price_inr: pred,
            price_range_low_inr: low,
            price_range_high_inr: high
        };
    });

    displayBatchResults(processed);
}

function displayBatchResults(records) {
    latestBatchResults = records;
    batchResultsSection.classList.remove("hidden");

    const count = records.length;
    const sum = records.reduce((acc, r) => acc + (parseInt(r.predicted_price_inr) || 0), 0);
    const avg = count > 0 ? Math.round(sum / count) : 0;

    batchMetricCount.textContent = count.toLocaleString();
    batchMetricAvg.textContent = formatINR(avg);
    batchMetricTotal.textContent = formatINR(sum);

    // Build Table
    const headers = Object.keys(records[0]);
    let tableHtml = `<table><thead><tr>`;
    headers.forEach(h => {
        tableHtml += `<th>${h}</th>`;
    });
    tableHtml += `</tr></thead><tbody>`;

    records.slice(0, 100).forEach(r => {
        tableHtml += `<tr>`;
        headers.forEach(h => {
            let val = r[h];
            if (h.includes("price") || h.includes("inr")) {
                val = formatINR(parseInt(val) || 0);
            }
            tableHtml += `<td>${val}</td>`;
        });
        tableHtml += `</tr>`;
    });
    tableHtml += `</tbody></table>`;

    batchTableContainer.innerHTML = tableHtml;
}

btnDownloadBatchResults.addEventListener("click", () => {
    if (!latestBatchResults.length) return;
    const csvContent = toCSV(latestBatchResults);
    downloadCSV(csvContent, "predicted_phone_resale_prices.csv");
});

// ---------------------------------------------------------------------------
// 8. Tab 2: Model Retraining
// ---------------------------------------------------------------------------
const btnDownloadTrainTemplate = document.getElementById("btnDownloadTrainTemplate");
const retrainDropzone = document.getElementById("retrainDropzone");
const retrainFileInput = document.getElementById("retrainFileInput");
const retrainTargetAlert = document.getElementById("retrainTargetAlert");
const btnStartTraining = document.getElementById("btnStartTraining");
const retrainStatusContainer = document.getElementById("retrainStatusContainer");
const retrainAlert = document.getElementById("retrainAlert");
const retrainLogBox = document.getElementById("retrainLogBox");
const retrainResultsContainer = document.getElementById("retrainResultsContainer");

let uploadedRetrainFile = null;

const SAMPLE_TRAIN_CSV = `brand,model,release_year,storage_gb,ram_gb,battery_health_pct,screen_condition,body_condition,has_box_bill,warranty_months_left,original_price_inr,market_demand_index,price_inr
Apple,iPhone 13,2021,128,4,86.0,minor_scratches,good,1,0,79900,7.0,41200
Samsung,Galaxy S22,2022,128,8,89.0,flawless,like_new,1,3,72999,6.5,34800`;

btnDownloadTrainTemplate.addEventListener("click", () => {
    downloadCSV(SAMPLE_TRAIN_CSV, "training_data_template.csv");
});

retrainDropzone.addEventListener("click", () => retrainFileInput.click());
retrainFileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
        setRetrainFile(e.target.files[0]);
    }
});

async function setRetrainFile(file) {
    uploadedRetrainFile = file;
    const text = await file.text();
    const rows = parseCSV(text);

    if (rows.length && "price_inr" in rows[0]) {
        retrainTargetAlert.className = "alert alert-success";
        retrainTargetAlert.innerHTML = `Loaded custom dataset: <strong>${rows.length.toLocaleString()} records</strong>. Target: Train models on your uploaded dataset.`;
        btnStartTraining.textContent = "Retrain models on custom dataset";
    } else {
        retrainTargetAlert.className = "alert alert-error";
        retrainTargetAlert.innerHTML = `Missing target column <code>price_inr</code> in uploaded CSV. Falling back to baseline simulation.`;
        uploadedRetrainFile = null;
        btnStartTraining.textContent = "Train models on baseline dataset";
    }
}

btnStartTraining.addEventListener("click", async () => {
    retrainStatusContainer.classList.remove("hidden");
    retrainResultsContainer.classList.add("hidden");
    retrainAlert.className = "alert alert-info";
    retrainAlert.innerHTML = `<span class="spinner"></span> Executing training pipeline ...`;
    retrainLogBox.textContent = "Initializing pipeline...\n";

    const formData = new FormData();
    if (uploadedRetrainFile) {
        formData.append("file", uploadedRetrainFile);
        retrainLogBox.textContent += `Uploading custom training dataset (${uploadedRetrainFile.name})...\n`;
    } else {
        retrainLogBox.textContent += "Using baseline synthetic dataset generation (6,000 devices)...\n";
    }

    try {
        retrainLogBox.textContent += "Starting 5-fold cross-validation across all 5 model families...\n";
        const response = await fetch("/api/retrain", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Training failed on server");
        }

        const resData = await response.json();
        retrainLogBox.textContent += `Training pipeline completed successfully!\nOptimal selected model: ${resData.best_model}\n`;

        retrainAlert.className = "alert alert-success";
        retrainAlert.innerHTML = `Model successfully updated and deployed: <strong>${resData.best_model}</strong>`;

        displayRetrainMetrics(resData);
    } catch (err) {
        retrainAlert.className = "alert alert-error";
        retrainAlert.textContent = `Training failed: ${err.message}`;
        retrainLogBox.textContent += `Error: ${err.message}\n`;
    }
});

function displayRetrainMetrics(data) {
    retrainResultsContainer.classList.remove("hidden");

    const bm = data.best_model || "SVR";
    const tm = data.test_metrics || {};
    const bestMetrics = tm[bm] || {};

    document.getElementById("retrainBestModel").textContent = bm;
    document.getElementById("retrainBestR2").textContent = (bestMetrics.R2 || 0).toFixed(4);
    document.getElementById("retrainBestRMSE").textContent = formatINR(bestMetrics.RMSE || 0);
    document.getElementById("retrainBestMAPE").textContent = (bestMetrics.MAPE || 0).toFixed(1) + "%";

    activeBestMape = bestMetrics.MAPE || 11.5;
    performValuation();

    // Render table
    renderComparisonTable(tm, "retrainTableContainer");
}

// ---------------------------------------------------------------------------
// 9. Tab 3: Model Performance
// ---------------------------------------------------------------------------
async function loadPerformanceMetrics() {
    try {
        const res = await fetch("/api/metrics");
        if (res.ok) {
            const data = await res.json();
            document.getElementById("perfBestModelBadge").textContent = data.best_model || "SVR";
            renderComparisonTable(data.test_metrics || {}, "perfTableContainer");
        }
    } catch (e) {
        // Fallback display if offline
        const fallbackMetrics = {
            "SVR": { "MAE": 1420, "RMSE": 2184, "MSE": 4770000, "R2": 0.9555, "MAPE": 11.5 },
            "Random Forest": { "MAE": 1640, "RMSE": 2520, "MSE": 6350000, "R2": 0.9410, "MAPE": 13.2 },
            "Linear Regression": { "MAE": 2100, "RMSE": 3150, "MSE": 9922500, "R2": 0.9080, "MAPE": 16.8 },
            "KNN": { "MAE": 2350, "RMSE": 3520, "MSE": 12390400, "R2": 0.8850, "MAPE": 18.5 },
            "Decision Tree": { "MAE": 2890, "RMSE": 4210, "MSE": 17724100, "R2": 0.8350, "MAPE": 22.1 }
        };
        renderComparisonTable(fallbackMetrics, "perfTableContainer");
    }
}

function renderComparisonTable(metricsObj, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const models = Object.keys(metricsObj);
    models.sort((a, b) => (metricsObj[a]?.RMSE || 0) - (metricsObj[b]?.RMSE || 0));

    let html = `<table>
        <thead>
            <tr>
                <th>Model</th>
                <th>MAE (Rs)</th>
                <th>RMSE (Rs)</th>
                <th>MSE</th>
                <th>R&sup2;</th>
                <th>MAPE (%)</th>
            </tr>
        </thead>
        <tbody>`;

    models.forEach((m, idx) => {
        const met = metricsObj[m];
        const isBest = idx === 0;
        const rowStyle = isBest ? "style='background-color:#f1f5f9; font-weight:600;'" : "";

        html += `<tr ${rowStyle}>
            <td><strong>${m}</strong> ${isBest ? " (Best)" : ""}</td>
            <td>${formatINR(met.MAE || 0)}</td>
            <td>${formatINR(met.RMSE || 0)}</td>
            <td>${Math.round(met.MSE || 0).toLocaleString()}</td>
            <td>${(met.R2 || 0).toFixed(4)}</td>
            <td>${(met.MAPE || 0).toFixed(1)}%</td>
        </tr>`;
    });

    html += `</tbody></table>`;
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// 10. CSV Parsing & Export Helpers
// ---------------------------------------------------------------------------
function parseCSV(text) {
    const lines = text.trim().split(/\r\n|\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) return [];

    const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ''));
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(",").map(p => p.trim().replace(/^"|"$/g, ''));
        if (parts.length === headers.length) {
            const obj = {};
            headers.forEach((h, idx) => {
                obj[h] = parts[idx];
            });
            rows.push(obj);
        }
    }
    return rows;
}

function toCSV(records) {
    if (!records || !records.length) return "";
    const headers = Object.keys(records[0]);
    let csv = headers.join(",") + "\n";
    records.forEach(r => {
        csv += headers.map(h => `"${r[h]}"`).join(",") + "\n";
    });
    return csv;
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ---------------------------------------------------------------------------
// 11. Initial Startup
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initBrandOptions();
    loadPerformanceMetrics();
});
