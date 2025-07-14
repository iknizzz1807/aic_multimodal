const API_BASE_URL = "http://127.0.0.1:8000";

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const topKSelect = document.getElementById("topK");
const statusElement = document.getElementById("status");
const resultsContainer = document.getElementById("results");

searchInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    search();
  }
});

const sampleQueries = [
  "a cat sleeping",
  "red car on street",
  "person eating food",
  "beautiful landscape",
  "dog playing in park",
];

window.addEventListener("load", function () {
  addSampleQueries();
});

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
        </div>
    `;

  const style = document.createElement("style");
  style.textContent = `
        .sample-queries {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }
        .sample-btn {
            padding: 6px 12px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .sample-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
    `;
  document.head.appendChild(style);

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

  // Update UI
  searchBtn.disabled = true;
  searchBtn.textContent = "Searching...";
  showStatus("Searching for images...", "loading");
  resultsContainer.innerHTML = "";

  try {
    const response = await fetch(`${API_BASE_URL}/search_visual`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: query,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    displayResults(data.results, query);
  } catch (error) {
    console.error("Search error:", error);
    showError(`Failed to search: ${error.message}`);
  } finally {
    // Reset UI
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }
}

function displayResults(results, query) {
  if (!results || results.length === 0) {
    showNoResults(query);
    showStatus("No results found", "error");
    return;
  }

  showStatus(`Found ${results.length} results`, "success");

  resultsContainer.innerHTML = results
    .map((result, index) => {
      const imageId = result.id;
      const score = (result.score * 100).toFixed(1);
      const imageData =
        result.image ||
        "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBmb3VuZDwvdGV4dD48L3N2Zz4=";

      return `
                <div class="result-item" style="animation: fadeInUp 0.3s ease ${
                  index * 0.1
                }s both;">
                    <img 
                        class="result-image" 
                        src="${imageData}" 
                        alt="${imageId}"
                        loading="lazy"
                    />
                    <div class="result-info">
                        <div class="result-path">${imageId}</div>
                        <span class="result-score">Score: ${score}%</span>
                    </div>
                </div>
            `;
    })
    .join("");

  // Animation CSS
  if (!document.querySelector("#animations")) {
    const animationStyle = document.createElement("style");
    animationStyle.id = "animations";
    animationStyle.textContent = `
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
    document.head.appendChild(animationStyle);
  }
}

function showNoResults(query) {
  resultsContainer.innerHTML = `
        <div class="no-results">
            <h3>No results found</h3>
            <p>No images match "${query}"</p>
            <p style="margin-top: 10px; font-size: 14px; color: #888;">
                Try different keywords or check if images are indexed
            </p>
        </div>
    `;
}

function showError(message) {
  resultsContainer.innerHTML = `
        <div class="error-message">
            <h3>⚠️ Error</h3>
            <p>${message}</p>
            <p style="margin-top: 10px; font-size: 14px;">
                Make sure the API server is running at ${API_BASE_URL}
            </p>
        </div>
    `;
  showStatus("Search failed", "error");
}

function showStatus(message, type) {
  statusElement.textContent = message;
  statusElement.className = `status ${type}`;

  // Auto hide success/error status after 3 seconds
  if (type !== "loading") {
    setTimeout(() => {
      statusElement.textContent = "";
      statusElement.className = "status";
    }, 3000);
  }
}

window.addEventListener("load", async function () {
  try {
    const response = await fetch(`${API_BASE_URL}/docs`);
    if (response.ok) {
      showStatus("API connected", "success");
    }
  } catch (error) {
    showStatus("API not accessible - make sure server is running", "error");
  }
});
