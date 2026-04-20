const API_URL = "/predict";
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const modelSelect = document.getElementById("modelSelect");
const btn = document.getElementById("analyzeBtn");
const selectedModelLabel = document.getElementById("selectedModel");
const modelNameLabel = document.getElementById("modelName");
const errorEl = document.getElementById("error");
const card = document.getElementById("card");
let selectedFile = null;

function riskColor(prob) {
    if (prob > 0.75) return "#00c97d";
    if (prob > 0.45) return "#ffaa00";
    return "#ff4d4d";
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover",  (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", ()  => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  loadFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
modelSelect.addEventListener("change", () => updateSelectionLabels());

function updateSelectionLabels() {
  const selectedText = modelSelect.options[modelSelect.selectedIndex].text;
  selectedModelLabel.innerHTML = `Selected model: <strong>${selectedText}</strong>`;
  modelNameLabel.textContent = selectedText;
}

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  errorEl.style.display = "none";
  card.style.display = "none";

  const reader = new FileReader();
  reader.onload = (e) => {
    dropzone.innerHTML = `<img src="${e.target.result}" alt="preview" />`;
    dropzone.classList.add("has-image");
    btn.style.display = "block";
  };
  reader.readAsDataURL(file);
}

btn.addEventListener("click", async () => {
  if (!selectedFile) return;
  btn.disabled = true;
  btn.textContent = "Analysing…";
  errorEl.style.display = "none";
  card.style.display = "none";

  try {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("model_type", modelSelect.value);
    const res = await fetch(API_URL, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    errorEl.textContent = "⚠ " + err.message;
    errorEl.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse";
  }
});

function renderResult(data) {
  const color = riskColor(data.probability);
  const pct = (data.probability * 100).toFixed(1);

  document.getElementById("diagnosis").textContent   = data.prediction;
  document.getElementById("diagnosis").style.color   = color;
  document.getElementById("topBar").style.width = pct + "%";
  document.getElementById("topBar").style.background = color;
  document.getElementById("topPct").textContent = pct + "%";

  const list = document.getElementById("classList");
  list.innerHTML = "";
  data.all_classes.forEach(({ label, probability }) => {
    const p = (probability * 100).toFixed(1);
    const c = riskColor(probability);
    const row = document.createElement("div");
    row.className = "class-row";
    row.innerHTML = `
 <span class="class-label">${label}</span>
 <div class="mini-track">
   <div class="mini-fill" style="width:${p}%; background:${c}; opacity:${label === data.prediction ? 1 : 0.45}"></div>
 </div>
 <span class="class-pct">${p}%</span>
    `;
    list.appendChild(row);
  });

  modelNameLabel.textContent = modelSelect.options[modelSelect.selectedIndex].text;
  card.style.display = "flex";
}
