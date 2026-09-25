document.addEventListener("dragover", function (event) {
  if (event.target.closest("[data-drop-zone]")) event.preventDefault();
});

document.addEventListener("dragenter", function (event) {
  const zone = event.target.closest("[data-drop-zone]");
  if (zone) zone.closest(".upload-card").classList.add("drag-over");
});

document.addEventListener("dragleave", function (event) {
  const zone = event.target.closest("[data-drop-zone]");
  if (zone) zone.closest(".upload-card").classList.remove("drag-over");
});

document.addEventListener("drop", function (event) {
  const zone = event.target.closest("[data-drop-zone]");
  if (!zone) return;
  event.preventDefault();
  const form = zone.closest(".upload-card");
  form.classList.remove("drag-over");
  const input = form.querySelector("input[type=file]");
  if (input.disabled || !event.dataTransfer.files.length) return;
  input.files = event.dataTransfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
});

(function () {
  var stream = null;
  var retry = null;

  function refresh() {
    if (!document.getElementById("console")) return;
    htmx.ajax("GET", "/checks/status", { target: "#console", swap: "outerHTML" });
  }

  function close() {
    if (stream) {
      stream.close();
      stream = null;
    }
  }

  function sync() {
    var live = !!document.querySelector("#console[data-live]");
    if (!live) {
      close();
      if (retry) {
        clearTimeout(retry);
        retry = null;
      }
      return;
    }
    if (stream) return;
    stream = new EventSource("/events/stream");
    stream.addEventListener("kyc.session.updated", refresh);
    stream.onerror = function () {
      if (stream && stream.readyState === EventSource.CLOSED) {
        stream = null;
        retry = setTimeout(function () {
          retry = null;
          sync();
        }, 3000);
      }
    };
  }

  document.addEventListener("DOMContentLoaded", sync);
  document.addEventListener("htmx:afterSwap", sync);
})();
