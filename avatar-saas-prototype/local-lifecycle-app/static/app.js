const State = {
    DISCONNECTED: "disconnected",
    CONNECTING: "connecting",
    READY: "avatar_ready",
    STOPPING: "stopping",
    ERROR: "error",
};

let currentState = State.DISCONNECTED;

function setState(state, detail) {
    currentState = state;
    const bar = document.getElementById("status-bar");
    const text = document.getElementById("status-text");
    const btnConnect = document.getElementById("btn-connect");
    const btnStop = document.getElementById("btn-stop");
    const btnDebug = document.getElementById("btn-debug");

    bar.className = "status-bar " + state;
    text.textContent = detail || state;

    btnConnect.disabled = state !== State.DISCONNECTED;
    btnStop.disabled = state !== State.READY;
    btnDebug.disabled = state !== State.READY;
}

function showMetadata(data) {
    const el = document.getElementById("metadata");
    el.textContent = JSON.stringify(data, null, 2);
}

function showError(msg) {
    const bar = document.getElementById("error-bar");
    const text = document.getElementById("error-text");
    bar.style.display = "flex";
    text.textContent = msg;
}

function hideError() {
    document.getElementById("error-bar").style.display = "none";
}

function showPlaceholder(title, hint) {
    const img = document.getElementById("avatar-img");
    const placeholder = document.getElementById("avatar-placeholder");
    img.style.display = "none";
    placeholder.style.display = "block";
    placeholder.innerHTML = "<p>" + title + "</p>" +
        (hint ? '<p class="hint">' + hint + "</p>" : "");
}

function showFrame(dataBase64, mimeType, width, height) {
    const img = document.getElementById("avatar-img");
    const placeholder = document.getElementById("avatar-placeholder");
    placeholder.style.display = "none";
    img.src = "data:" + mimeType + ";base64," + dataBase64;
    img.width = Math.min(width, 400);
    img.height = Math.round(img.width * (height / width));
    img.style.display = "block";
}

async function postJSON(url, body) {
    const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : "{}",
    });
    return resp.json();
}

async function doConnect() {
    hideError();
    setState(State.CONNECTING, "Connecting...");
    showPlaceholder("Waking avatar runtime...", "This may take a few seconds on first call (snapshot restore).");

    try {
        const result = await postJSON("/api/connect");

        if (result.status === "avatar_ready") {
            setState(State.READY, "Avatar ready");
            showMetadata(result);
            doDebugFrame();
        } else {
            setState(State.ERROR, "Error");
            const msg = result.error || JSON.stringify(result);
            showError("Connect failed: " + msg);
            showPlaceholder("Connection failed", msg);
            showMetadata(result);
        }
    } catch (err) {
        setState(State.ERROR, "Network error");
        showError("Network error: " + err.message);
        showPlaceholder("Network error", "Is the local server running?");
    }
}

async function doDebugFrame() {
    if (currentState !== State.READY) return;

    try {
        const result = await postJSON("/api/debug-frame");

        if (result.available && result.dataBase64) {
            showFrame(result.dataBase64, result.mimeType, result.width || 1280, result.height || 720);
        } else {
            showPlaceholder(
                "Avatar runtime ready",
                result.note || "No debug frame available. The runtime is connected and ready for audio."
            );
        }
        showMetadata(result);
    } catch (err) {
        showError("Debug frame error: " + err.message);
    }
}

async function doStop() {
    hideError();
    setState(State.STOPPING, "Stopping...");

    try {
        const result = await postJSON("/api/stop");
        setState(State.DISCONNECTED, "Disconnected");
        showPlaceholder("No active session.", "Click Connect to start.");
        showMetadata(result);
    } catch (err) {
        setState(State.DISCONNECTED, "Disconnected");
        showError("Stop error: " + err.message);
    }
}
