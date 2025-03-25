// Load transcription from local storage
const transcriptionText = localStorage.getItem("transcription");
const transcriptionElement = document.getElementById("transcription");

// Sample corrections and improvements
const corrections = [
  { error: "vey", suggestion: "very", type: "error" },
  { error: "happy", suggestion: "joyful, cheerful, content", type: "improvement" }
];

function highlightText(text) {
  corrections.forEach(({ error, suggestion, type }) => {
    const regex = new RegExp(`\\b${error}\\b`, "gi");
    const spanClass = type === "error" ? "highlight error" : "highlight improvement";
    const title = type === "error"
      ? `Correct: ${suggestion}`
      : `Suggestion: ${suggestion}`;
    text = text.replace(regex, `<span class="${spanClass}" title="${title}">${error}</span>`);
  });
  return text;
}

// Apply highlights to transcription
transcriptionElement.innerHTML = highlightText(transcriptionText);

document.addEventListener("mouseover", (event) => {
    if (event.target.classList.contains("highlight")) {
      const tooltip = document.createElement("div");
      tooltip.className = "tooltip";
      tooltip.innerText = event.target.getAttribute("title");
      document.body.appendChild(tooltip);
  
      const rect = event.target.getBoundingClientRect();
      tooltip.style.top = `${rect.top - 30}px`;
      tooltip.style.left = `${rect.left}px`;
  
      event.target.addEventListener("mouseleave", () => tooltip.remove());
    }
  });
  
