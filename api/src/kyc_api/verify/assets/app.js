const app = document.querySelector("#app");
const MAX_DOCUMENT_BYTES = 12 * 1024 * 1024;
const LIVENESS_FRAME_COUNT = 3;
let sessionId = "";
let token = "";
let current = null;
let preview = null;
let cameraStream = null;
let livenessFrames = [];
let pollTimer = null;
let pollDelay = 2000;

function storageKey(id) { return `kyc.verify.token.${id}`; }
function escapeText(value) { return String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c])); }
function clearAccess() { sessionStorage.removeItem(storageKey(sessionId)); token = ""; stopCamera(); }
function stopCamera() { if (cameraStream) { cameraStream.getTracks().forEach(track => track.stop()); cameraStream = null; } }

function initializeAccess() {
  const parts = location.pathname.split("/").filter(Boolean);
  sessionId = parts.length === 2 && parts[0] === "verify" ? parts[1] : "";
  if (!/^[A-Za-z0-9_-]{8,128}$/.test(sessionId)) return false;
  const fragment = location.hash.slice(1);
  if (fragment) {
    if (!/^bt_[A-Za-z0-9_-]{20,}$/.test(fragment)) return false;
    sessionStorage.setItem(storageKey(sessionId), fragment);
    history.replaceState(null, "", `/verify/${encodeURIComponent(sessionId)}`);
  }
  token = sessionStorage.getItem(storageKey(sessionId)) || "";
  return Boolean(token);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Authorization: `Bearer ${token}`, ...(options.headers || {}) },
    credentials: "omit",
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    /** @type {Error & { status?: number, code?: string, retryAfter?: string | null }} */
    const error = new Error(body.error?.message || "The request could not be completed.");
    error.status = response.status; error.code = body.error?.code; error.retryAfter = response.headers.get("Retry-After");
    throw error;
  }
  return response.json();
}

async function refresh() {
  try {
    current = await api(`/v1/sessions/${encodeURIComponent(sessionId)}`);
    pollDelay = 2000;
    render();
  } catch (error) { renderError(error); }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (!current || current.next_action !== "wait") return;
  pollTimer = setTimeout(refresh, pollDelay);
  pollDelay = Math.min(5000, pollDelay + 1000);
}

function renderError(error) {
  clearTimeout(pollTimer); stopCamera();
  const expired = error.status === 401 || error.status === 410;
  if (expired) clearAccess();
  app.innerHTML = `<section class="card"><h1>${expired ? "This verification link has expired" : "We could not continue"}</h1><p>${expired ? "Ask the organisation that sent this link for a new verification link." : escapeText(error.message)}</p></section>`;
}

function render() {
  clearTimeout(pollTimer);
  if (!current) return;
  if (current.status === "completed") return terminal("Verification steps complete", "The organisation that sent this link will continue from here.", true);
  if (current.status === "failed") return terminal("Verification could not complete", "Please contact the organisation that sent this link to start another check.", false);
  if (current.next_action === "submit_document_front") return captureScreen("front");
  if (current.next_action === "submit_document_back") return captureScreen("back");
  if (current.next_action === "submit_liveness") return livenessScreen();
  waitingScreen(); schedulePoll();
}

function terminal(title, message, success) {
  clearAccess();
  app.innerHTML = `<section class="card"><h1>${title}</h1><p class="${success ? "success" : "error"}">${message}</p></section>`;
}

function captureScreen(side) {
  stopCamera(); preview = null;
  const name = side === "front" ? "front" : "back";
  app.innerHTML = `<section class="card"><h1>Capture the ${name} of your document</h1><p class="hint">Use even light, avoid glare, and make the document fill the frame.</p><label class="file-label">Choose photo<input id="file" type="file" accept="image/jpeg,image/png" capture="environment"></label><button id="camera" class="secondary">Use camera</button><div id="capture"></div><p id="message" class="error"></p></section>`;
  document.querySelector("#file").addEventListener("change", event => chooseFile(/** @type {HTMLInputElement} */ (event.target).files?.[0], side));
  document.querySelector("#camera").addEventListener("click", () => openCamera(side, "environment"));
}

async function chooseFile(file, side) {
  if (!file) return;
  try { preview = await normalize(file, MAX_DOCUMENT_BYTES, 2048, .9); showPreview(side); }
  catch (error) { document.querySelector("#message").textContent = error.message; }
}

async function openCamera(side, facingMode) {
  try {
    stopCamera(); cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: facingMode }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    const area = document.querySelector("#capture"); area.innerHTML = `<video id="video" autoplay playsinline></video><button id="take">Take photo</button>`;
    /** @type {HTMLVideoElement} */ (document.querySelector("#video")).srcObject = cameraStream;
    document.querySelector("#take").addEventListener("click", async () => { preview = await frameBlob(document.querySelector("#video"), 2048, .9); stopCamera(); showPreview(side); });
  } catch { document.querySelector("#message").textContent = "Camera access was not available. Choose a photo instead."; }
}

function showPreview(side) {
  const area = document.querySelector("#capture");
  const url = URL.createObjectURL(preview);
  area.innerHTML = `<img class="preview" alt="Document photo preview" src="${url}"><button id="submit">Use photo</button><button id="retake" class="secondary">Retake</button>`;
  document.querySelector("#submit").addEventListener("click", () => uploadDocument(side));
  document.querySelector("#retake").addEventListener("click", () => captureScreen(side));
}

async function uploadDocument(side) {
  const form = new FormData(); form.append("image", preview, `${side}.jpg`);
  try {
    const response = await api(`/v1/sessions/${encodeURIComponent(sessionId)}/images/${side}`, { method: "POST", body: form });
    current = response.session;
    if (!response.accepted) { captureScreen(side); document.querySelector("#message").textContent = response.issues.map(issue => issue.message).join(" ") || "Use a clearer image and try again."; return; }
    render();
  } catch (error) { if (error.status === 409) return refresh(); renderError(error); }
}

function livenessScreen() {
  preview = null; livenessFrames = [];
  app.innerHTML = `<section class="card"><h1>Confirm your presence</h1><p class="hint">Use the front camera. Keep your face centred and well lit.</p><div class="progress">${Array.from({length:LIVENESS_FRAME_COUNT}, () => "<span></span>").join("")}</div><button id="start">Start camera</button><div id="capture"></div><p id="message" class="error"></p></section>`;
  document.querySelector("#start").addEventListener("click", startLivenessCamera);
}

async function startLivenessCamera() {
  try {
    stopCamera(); cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    document.querySelector("#capture").innerHTML = `<video id="video" autoplay playsinline></video><button id="frame">Capture frame</button>`;
    /** @type {HTMLVideoElement} */ (document.querySelector("#video")).srcObject = cameraStream;
    document.querySelector("#frame").addEventListener("click", takeLivenessFrame);
  } catch { document.querySelector("#message").textContent = "Camera access is required for this step."; }
}

async function takeLivenessFrame() {
  const button = /** @type {HTMLButtonElement} */ (document.querySelector("#frame")); button.disabled = true;
  livenessFrames.push(await frameBlob(/** @type {HTMLVideoElement} */ (document.querySelector("#video")), 1280, .85));
  document.querySelectorAll(".progress span")[livenessFrames.length - 1].classList.add("done");
  if (livenessFrames.length === LIVENESS_FRAME_COUNT) { stopCamera(); await uploadLiveness(); return; }
  button.disabled = false; button.textContent = `Capture frame ${livenessFrames.length + 1}`;
}

async function uploadLiveness() {
  const form = new FormData(); livenessFrames.forEach((frame, index) => form.append("frames", frame, `frame-${index + 1}.jpg`));
  try { current = (await api(`/v1/sessions/${encodeURIComponent(sessionId)}/liveness`, { method: "POST", body: form })).session; render(); }
  catch (error) { if (error.status === 409) return refresh(); renderError(error); }
}

function waitingScreen() {
  stopCamera();
  const documentDone = current.document.status === "completed";
  app.innerHTML = `<section class="card"><h1>${documentDone ? "Final checks in progress" : "Document processing in progress"}</h1><p>Please keep this page open. This usually takes a short time.</p></section>`;
}

async function normalize(file, maxBytes, maxEdge, quality) {
  if (!/^image\/(jpeg|png)$/.test(file.type)) throw new Error("Choose a JPEG or PNG image.");
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas"); canvas.width = Math.round(bitmap.width * scale); canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height); bitmap.close();
  let q = quality; let blob = await canvasBlob(canvas, q);
  while (blob.size > maxBytes && q > .7) { q -= .05; blob = await canvasBlob(canvas, q); }
  if (blob.size > maxBytes) throw new Error("This image is too large. Choose a smaller, clear photo.");
  return blob;
}

async function frameBlob(video, maxEdge, quality) {
  const canvas = document.createElement("canvas"); const scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale)); canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvasBlob(canvas, quality);
}
function canvasBlob(canvas, quality) { return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not prepare image.")), "image/jpeg", quality)); }

if (!initializeAccess()) renderError({ status: 401, message: "This verification link is invalid." }); else refresh();
