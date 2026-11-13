// Service worker for the trial page. Two jobs:
//
// 1. Privacy: every request the page or its workers make passes through here, and
//    anything that is not one of this site's own files is refused.
// 2. Speed: it marks the page "cross-origin isolated" (two response headers), which
//    browsers require before the models may use several processor cores. Static hosts
//    such as GitHub Pages cannot send those headers themselves.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) {
    e.respondWith(new Response("blocked: this page only loads its own files", { status: 403 }));
    return;
  }
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    })
  );
});
