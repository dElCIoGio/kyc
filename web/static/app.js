(function () {
  var stream = null;
  var retry = null;

  function refresh() {
    var consoleNode = document.getElementById("console");
    if (!consoleNode || !consoleNode.dataset.live) return;
    var activeId = consoleNode.dataset.sessionId;
    if (activeId) htmx.ajax("GET", "/sandbox/sessions/" + encodeURIComponent(activeId) + "/status", { target: "#console", swap: "outerHTML" });
  }

  function sync() {
    var consoleNode = document.getElementById("console");
    if (!consoleNode || !consoleNode.dataset.live) {
      if (stream) { stream.close(); stream = null; }
      return;
    }
    if (stream) return;
    stream = new EventSource("/events/stream");
    stream.addEventListener("kyc.session.updated", refresh);
    stream.onerror = function () {
      if (stream && stream.readyState === EventSource.CLOSED) {
        stream = null;
        retry = setTimeout(function () { retry = null; sync(); }, 3000);
      }
    };
  }

  async function copyVerificationLink(button) {
    var value = document.querySelector("[data-verification-url]");
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value.textContent);
      var original = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = original; }, 1500);
    } catch (_) {
      button.textContent = "Copy unavailable";
    }
  }

  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-copy-link]");
    if (button) copyVerificationLink(button);
  });
  document.addEventListener("DOMContentLoaded", sync);
  document.addEventListener("htmx:afterSwap", sync);
})();
