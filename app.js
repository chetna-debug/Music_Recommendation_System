async function searchTrack() {
  const query = document.getElementById("searchInput").value;
  if (!query) return;

  const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  const results = await res.json();

  const list = document.getElementById("searchResults");
  list.innerHTML = "";
  results.forEach(track => {
    const li = document.createElement("li");
    li.className = "p-2 border rounded cursor-pointer hover:bg-gray-100";
    li.textContent = `${track.track_name} — ${track.artists}`;
    li.onclick = () => recommendByTrack(track.track_name);
    list.appendChild(li);
  });
}

async function recommendByTrack(trackName) {
  const res = await fetch(`/api/recommend/by-track?track=${encodeURIComponent(trackName)}&k=10`);
  const recs = await res.json();
  renderResults(recs);
}

async function recommendByMood() {
  const params = new URLSearchParams({
    valence: document.getElementById("valence").value,
    energy: document.getElementById("energy").value,
    danceability: document.getElementById("danceability").value,
    tempo: document.getElementById("tempo").value,
    k: 10
  });

  const res = await fetch(`/api/recommend/by-mood?${params.toString()}`);
  const recs = await res.json();
  renderResults(recs);
}

function renderResults(recs) {
  const list = document.getElementById("recommendations");
  list.innerHTML = "";
  recs.forEach(track => {
    const li = document.createElement("li");
    li.className = "p-2 border rounded bg-gray-50";
    li.textContent = `${track.track_name} — ${track.artists}`;
    list.appendChild(li);
  });
}

function updateSlider(id) {
  document.getElementById(id + "Val").textContent = document.getElementById(id).value;
}
