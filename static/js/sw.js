const CACHE_NAME = "qr-scanner-v1";
const ASSETS = [
    "/",
    "/index.html",
    "/static/css/index.css",
    "/static/js/main.js",
    "/static/img/logo.png"
    ];

    self.addEventListener("install", (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
    });

    self.addEventListener("fetch", (e) => {
    e.respondWith(
        fetch(e.request).catch(() => caches.match(e.request))
    );
});