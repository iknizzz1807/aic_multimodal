const API_BASE_URL = "http://127.0.0.1:8000";
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

// DOM Elements
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const topKSelect = document.getElementById("topK");
const statusElement = document.getElementById("status");
const resultsContainer = document.getElementById("results");

// Modal Elements
const modal = document.getElementById("mediaModal");
const modalTitle = document.getElementById("modalTitle");
const modalMediaContainer = document.getElementById("modalMediaContainer");
const modalDetailsContent = document.getElementById("modalDetailsContent");

// State
let currentResults = [];

// Event Listeners
searchInput.addEventListener("keypress", (e) => e.key === "Enter" && search());
window.addEventListener("load", () => {
  checkApiStatus();
  addSampleQueries();
});
window.onclick = (event) => {
  if (event.target == modal) {
    closeModal();
  }
};

// --- Core Functions ---

async function search() {
  const query = searchInput.value.trim();
  if (!query) {
    showStatus("Please enter a search query.", "error");
    return;
  }

  const topK = parseInt(topKSelect.value);
  const endpoint = "/search"; // The API now uses a single, unified endpoint

  setLoadingState(true, query);
  resultsContainer.innerHTML = "";

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: query, top_k: topK }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(
        `HTTP ${response.status}: ${errData?.detail || response.statusText}`
      );
    }

    const data = await response.json();
    currentResults = data.results;
    displayResults(currentResults, query);
  } catch (error) {
    console.error("Search error:", error);
    showError(`Failed to perform search. ${error.message}`);
  } finally {
    setLoadingState(false);
  }
}

function displayResults(results, query) {
  if (!results || results.length === 0) {
    showNoResults(query);
    return;
  }

  showStatus(`Found ${results.length} results.`, "success");

  const resultsHtml = results
    .map((result, index) => renderResultItem(result, index))
    .join("");
  resultsContainer.innerHTML = resultsHtml;
}

// --- Result Rendering Functions ---

const placeholderImage =
  "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPlRodW1ibmFpbCBsb2FkaW5n...PC90ZXh0Pjwvc3ZnPg==";

function renderResultItem(r, i) {
  const isVideo = VIDEO_EXTENSIONS.some((ext) =>
    r.id.toLowerCase().endsWith(ext)
  );
  return `
    <div class="result-item" style="animation-delay: ${
      i * 0.05
    }s;" onclick="openModal(${i})">
      <img class="result-image" src="${r.image || placeholderImage}" alt="${
    r.id
  }" loading="lazy" onerror="this.src='${placeholderImage}'"/>
      ${isVideo ? '<div class="video-indicator">▶</div>' : ""}
      <div class="result-info">
        <div>
          <div class="result-path">${r.id}</div>
          <div class="reason-tags">
            ${r.reason
              .map(
                (reason) =>
                  `<span class="reason-tag ${reason.replace(
                    "_",
                    "-"
                  )}">${reason}</span>`
              )
              .join("")}
          </div>
        </div>
        <span class="result-score">Score: ${r.score.toFixed(3)}</span>
      </div>
    </div>`;
}

// --- Modal Functions ---

function openModal(resultIndex) {
  const result = currentResults[resultIndex];
  if (!result) return;

  const mediaUrl = `${API_BASE_URL}/media/${result.id}`;
  modalTitle.textContent = result.id;
  modalMediaContainer.innerHTML = "";
  modalDetailsContent.innerHTML = "";

  const isVideo = VIDEO_EXTENSIONS.some((ext) =>
    result.id.toLowerCase().endsWith(ext)
  );

  if (isVideo) {
    modalMediaContainer.innerHTML = `<video id="modalVideoPlayer" src="${mediaUrl}" controls autoplay muted loop></video>`;
  } else {
    modalMediaContainer.innerHTML = `<img src="${mediaUrl}" alt="${result.id}" />`;
  }

  // Populate matching moments from the new 'details' structure
  if (result.details && result.details.length > 0) {
    modalDetailsContent.innerHTML = result.details
      .map((detail) => {
        let timeInfo,
          textInfo = "";
        const seekTime =
          detail.type === "visual" ? detail.timestamp : detail.start_time;

        if (detail.type === "visual") {
          timeInfo = `@ ${detail.timestamp.toFixed(1)}s`;
          textInfo = `<span class="detail-item-text">Visual match</span>`;
        } else {
          // 'audio' or 'audio_event'
          timeInfo = `${detail.start_time.toFixed(
            1
          )}s - ${detail.end_time.toFixed(1)}s`;
          if (detail.match_content) {
            textInfo = `<span class="detail-item-text">"${detail.match_content}"</span>`;
          }
        }

        return `
          <div class="detail-item" onclick="seekMedia(${seekTime})">
            <span class="reason-tag ${detail.type.replace(
              "_",
              "-"
            )}">${detail.type.replace("_", " ")}</span>
            <span class="detail-item-time">${timeInfo}</span>
            ${textInfo}
          </div>`;
      })
      .join("");
  } else {
    modalDetailsContent.innerHTML =
      "<p>No specific match details available for this item.</p>";
  }

  modal.style.display = "flex";
  document.body.style.overflow = "hidden";
}

function closeModal() {
  modal.style.display = "none";
  document.body.style.overflow = "auto";
  // Pause and clean up media to stop background loading/playing
  const video = document.getElementById("modalVideoPlayer");
  if (video) video.pause();
  modalMediaContainer.innerHTML = "";
}

function seekMedia(time) {
  const videoPlayer = document.getElementById("modalVideoPlayer");
  if (videoPlayer) {
    // Ensure time is a valid number before seeking
    if (typeof time === "number" && !isNaN(time)) {
      videoPlayer.currentTime = time;
      videoPlayer.play();
    }
  }
}

// --- UI & UX Helper Functions ---

function addSampleQueries() {
  const sampleQueries = [
    "a cat sleeping",
    "a person walking on the beach",
    "someone talking about technology",
    "red car on a street",
    "dog playing in a park",
    "sound of a car engine",
  ];
  const searchSection = document.querySelector(".search-section");
  const samplesDiv = document.createElement("div");
  samplesDiv.className = "sample-queries";
  samplesDiv.innerHTML = `
    <p>Try these examples:</p>
    <div>${sampleQueries
      .map(
        (q) =>
          `<button class="sample-btn" onclick="searchSample('${q}')">${q}</button>`
      )
      .join("")}</div>`;
  searchSection.appendChild(samplesDiv);
}

function searchSample(query) {
  searchInput.value = query;
  search();
}

function setLoadingState(isLoading, query = "") {
  searchBtn.disabled = isLoading;
  if (isLoading) {
    searchBtn.textContent = "Searching...";
    showStatus(`Searching for "${query}"...`, "loading");
  } else {
    searchBtn.textContent = "Search";
  }
}

function showNoResults(query) {
  resultsContainer.innerHTML = `
    <div class="no-results">
      <h3>No results found for "${query}"</h3>
      <p>Try different keywords, or check if your media has been processed and indexed.</p>
    </div>`;
  showStatus("No results found.", "error");
}

function showError(message) {
  resultsContainer.innerHTML = `
    <div class="error-message">
      <h3>⚠️ An Error Occurred</h3>
      <p>${message}</p>
      <p>Please ensure the backend server is running at <strong>${API_BASE_URL}</strong> and is accessible.</p>
    </div>`;
  showStatus("Search failed.", "error");
}

function showStatus(message, type) {
  statusElement.textContent = message;
  statusElement.className = `status ${type}`;
  // Clear status after a few seconds unless it's a loading message
  if (type !== "loading") {
    setTimeout(() => {
      if (statusElement.textContent === message) {
        statusElement.textContent = "";
        statusElement.className = "status";
      }
    }, 5000);
  }
}

async function checkApiStatus() {
  try {
    // The new backend has a GET / health check endpoint
    const response = await fetch(`${API_BASE_URL}/`);
    if (response.ok) {
      showStatus("API Connected", "success");
    } else {
      throw new Error(`Server responded with status: ${response.status}`);
    }
  } catch (error) {
    console.error("API Status Check Error:", error);
    showStatus("API Connection Failed", "error");
    showError(
      "Could not connect to the backend API. Please make sure the server is running and there are no CORS issues."
    );
  }
}
