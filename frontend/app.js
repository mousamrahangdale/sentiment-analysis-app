const analyzeBtn = document.getElementById("analyzeBtn");
const textInput = document.getElementById("textInput");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultReason = document.getElementById("resultReason");
const resultMeta = document.getElementById("resultMeta");

analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  const source = document.querySelector('input[name="source"]:checked').value;

  errorBox.hidden = true;
  resultBox.hidden = true;

  if (!text) {
    showError("Please enter some text first.");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Checking...";

  try {
    const res = await fetch("/api/v1/sentiment/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    showResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Check Sentiment";
  }
});

function showResult(data) {
  resultLabel.textContent = data.label;
  resultLabel.className = "label " + data.label;
  resultConfidence.textContent = (data.confidence * 100).toFixed(1) + "%";
  resultReason.textContent = data.reason ? `"${data.reason}"` : "";
  resultMeta.textContent = `${data.engine} · ${data.latency_ms.toFixed(1)} ms`;
  resultBox.hidden = false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
