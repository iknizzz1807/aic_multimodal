// File: src/frontend/script.js (MODIFIED)
const API_BASE_URL = "http://127.0.0.1:8000";
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"];

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const topKSelect = document.getElementById("topK");
const statusElement = document.getElementById("status");
const resultsContainer = document.getElementById("results");

// NEW: Modal elements
const modal = document.getElementById("mediaModal");
const modalTitle = document.getElementById("modalTitle");
const modalMediaContainer = document.getElementById("modalMediaContainer");
const modalDetails = document.getElementById("modalDetails");
let currentResults = []; // Store current search results

searchInput.addEventListener("keypress", (e) => e.key === "Enter" && search());
window.addEventListener("load", () => {
  checkApiStatus();
  addSampleQueries();
});
// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
  if (event.target == modal) {
    closeModal();
  }
};

const sampleQueries = [
  "a cat sleeping",
  "a person walking on the beach",
  "someone talking about technology",
  "red car on a street",
  "dog playing in a park",
];

function addSampleQueries() {
  const searchSection = document.querySelector(".search-section");
  const samplesDiv = document.createElement("div");
  samplesDiv.className = "sample-queries";
  samplesDiv.innerHTML = `
    <p style="margin-bottom: 10px; color: #666; font-size: 14px;">Try these examples:</p>
    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
      ${sampleQueries
        .map(
          (query) =>
            `<button class="sample-btn" onclick="searchSample('${query}')">${query}</button>`
        )
        .join("")}
    </div>`;
  searchSection.appendChild(samplesDiv);
}

function searchSample(query) {
  searchInput.value = query;
  search();
}

async function search() {
  const query = searchInput.value.trim();
  if (!query) {
    showStatus("Please enter a search query", "error");
    return;
  }

  const topK = parseInt(topKSelect.value);
  const searchType = document.querySelector(
    'input[name="searchType"]:checked'
  ).value;

  const endpointMap = {
    unified: "/unified_search",
    visual: "/search_visual",
    audio: "/search_audio",
  };
  const endpoint = endpointMap[searchType];

  searchBtn.disabled = true;
  searchBtn.textContent = "Searching...";
  showStatus(`Searching for "${query}"...`, "loading");
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
    currentResults = data.results; // Store results
    displayResults(currentResults, query, searchType);
  } catch (error) {
    console.error("Search error:", error);
    showError(`Failed to search: ${error.message}`);
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }
}

function displayResults(results, query, searchType) {
  if (!results || results.length === 0) {
    showNoResults(query);
    return;
  }

  showStatus(`Found ${results.length} results`, "success");

  const displayFunctionMap = {
    unified: displayUnifiedResults,
    visual: displayVisualResults,
    audio: displayAudioResults,
  };

  displayFunctionMap[searchType](results);
}

const placeholderImage =
  "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=";

function createFadeInAnimation() {
  if (!document.querySelector("#animations")) {
    const style = document.createElement("style");
    style.id = "animations";
    style.textContent = `
      @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
      }`;
    document.head.appendChild(style);
  }
}

function displayUnifiedResults(results) {
  createFadeInAnimation();
  resultsContainer.innerHTML = results
    .map(
      (r, i) => `
    <div class="result-item" style="animation: fadeInUp 0.3s ease ${
      i * 0.05
    }s both;" onclick="openModal(${i})">
      <img class="result-image" src="${r.image || placeholderImage}" alt="${
        r.id
      }" loading="lazy"/>
      <div class="result-info">
        <div>
          <div class="result-path">${r.id}</div>
          <div class="reason-tags">
            ${r.reason
              .map(
                (reason) =>
                  `<span class="reason-tag ${reason}">${reason}</span>`
              )
              .join("")}
          </div>
        </div>
        <span class="result-score">Score: ${(r.score * 100).toFixed(1)}%</span>
      </div>
    </div>
  `
    )
    .join("");
}

function displayVisualResults(results) {
  createFadeInAnimation();
  resultsContainer.innerHTML = results
    .map(
      (r, i) => `
    <div class="result-item" style="animation: fadeInUp 0.3s ease ${
      i * 0.05
    }s both;">
      <img class="result-image" src="${r.image || placeholderImage}" alt="${
        r.id
      }" loading="lazy"/>
      <div class="result-info">
        <div class="result-path">${r.id} ${
        r.timestamp ? `<span>@ ${r.timestamp.toFixed(1)}s</span>` : ""
      }</div>
        <span class="result-score">Similarity: ${r.score.toFixed(3)}</span>
      </div>
    </div>
  `
    )
    .join("");
}

function displayAudioResults(results) {
  createFadeInAnimation();
  resultsContainer.innerHTML = results
    .map(
      (r, i) => `
    <div class="result-item audio-result-item" style="animation: fadeInUp 0.3s ease ${
      i * 0.05
    }s both;">
      <div class="result-path">${r.id}</div>
      <p class="audio-match-text">"${r.match}"</p>
      <div class="audio-time">Time: ${r.start.toFixed(1)}s - ${r.end.toFixed(
        1
      )}s</div>
    </div>
  `
    )
    .join("");
}

// --- NEW MODAL FUNCTIONS ---

function openModal(resultIndex) {
  const result = currentResults[resultIndex];
  if (!result) return;

  const mediaUrl = `${API_BASE_URL}/media/${result.id}`;
  modalTitle.textContent = result.id;
  modalMediaContainer.innerHTML = ""; // Clear previous content
  document.querySelector("#modalDetails > div")?.remove(); // Clear previous details

  const isVideo = VIDEO_EXTENSIONS.some((ext) =>
    result.id.toLowerCase().endsWith(ext)
  );

  if (isVideo) {
    modalMediaContainer.innerHTML = `<video id="modalVideoPlayer" src="${mediaUrl}" controls autoplay muted></video>`;
  } else {
    modalMediaContainer.innerHTML = `<img src="${mediaUrl}" alt="${result.id}" />`;
  }

  const detailsContainer = document.createElement("div");
  modalDetails.appendChild(detailsContainer);

  if (result.details && result.details.length > 0) {
    detailsContainer.innerHTML = result.details
      .map((detail) => {
        let timeInfo,
          textInfo = "";
        if (detail.type === "visual") {
          timeInfo = `@ ${detail.timestamp.toFixed(1)}s`;
        } else {
          timeInfo = `${detail.start_time.toFixed(
            1
          )}s - ${detail.end_time.toFixed(1)}s`;
          textInfo = `<span class="detail-item-text">"${detail.text_match}"</span>`;
        }

        const seekTime =
          detail.type === "visual" ? detail.timestamp : detail.start_time;

        return `
          <div class="detail-item" onclick="seekMedia(${seekTime})">
            <span class="reason-tag ${detail.type}">${detail.type}</span>
            <span class="detail-item-time">${timeInfo}</span>
            ${textInfo}
          </div>
        `;
      })
      .join("");
  } else {
    detailsContainer.innerHTML = "<p>No specific match details available.</p>";
  }

  modal.style.display = "block";
}

function closeModal() {
  modal.style.display = "none";
  modalMediaContainer.innerHTML = ""; // Stop video/image loading
}

function seekMedia(time) {
  const videoPlayer = document.getElementById("modalVideoPlayer");
  if (videoPlayer) {
    videoPlayer.currentTime = time;
    videoPlayer.play();
  }
}

// --- END NEW MODAL FUNCTIONS ---

function showNoResults(query) {
  resultsContainer.innerHTML = `
    <div class="no-results">
      <h3>No results found for "${query}"</h3>
      <p>Try different keywords or check if media has been indexed.</p>
    </div>`;
  showStatus("No results found", "error");
}

function showError(message) {
  resultsContainer.innerHTML = `
    <div class="error-message">
      <h3>⚠️ Error</h3>
      <p>${message}</p>
      <p style="margin-top: 10px; font-size: 14px;">Make sure the API server is running at ${API_BASE_URL}</p>
    </div>`;
  showStatus("Search failed", "error");
}

function showStatus(message, type) {
  statusElement.textContent = message;
  statusElement.className = `status ${type}`;
  if (type !== "loading") {
    setTimeout(() => {
      if (statusElement.textContent === message) {
        statusElement.textContent = "";
        statusElement.className = "status";
      }
    }, 4000);
  }
}

async function checkApiStatus() {
  try {
    const response = await fetch(API_BASE_URL);
    if (response.ok) {
      showStatus("API connected", "success");
    } else {
      throw new Error("Server responded but not OK");
    }
  } catch (error) {
    showStatus("API connection failed", "error");
    showError("Could not connect to the backend API.");
  }
}
