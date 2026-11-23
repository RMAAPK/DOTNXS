#include "third_party/httplib.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace {

std::unordered_set<std::string> split_origins(const char *value) {
    std::unordered_set<std::string> origins;
    std::stringstream input(value ? value : "");
    std::string origin;
    while (std::getline(input, origin, ',')) {
        const auto first = origin.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const auto last = origin.find_last_not_of(" \t\r\n");
        origins.insert(origin.substr(first, last - first + 1));
    }
    return origins;
}

void add_cors_headers(const httplib::Request &request, httplib::Response &response,
                      const std::unordered_set<std::string> &allowed_origins) {
    const auto origin = request.get_header_value("Origin");
    if (!origin.empty() && allowed_origins.count(origin)) {
        response.set_header("Access-Control-Allow-Origin", origin);
        response.set_header("Vary", "Origin");
        response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        response.set_header("Access-Control-Allow-Headers", "Content-Type");
        response.set_header("Access-Control-Max-Age", "600");
    }
}

int env_port(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    try {
        const int port = std::stoi(value);
        return port > 0 && port <= 65535 ? port : fallback;
    } catch (...) {
        return fallback;
    }
}

void proxy_to_worker(const httplib::Request &request, httplib::Response &response,
                     int worker_port, const std::string &path,
                     const std::unordered_set<std::string> &allowed_origins) {
    add_cors_headers(request, response, allowed_origins);
    httplib::Client worker("127.0.0.1", worker_port);
    worker.set_connection_timeout(5, 0);
    // CPU generation can take a while while NXS chunks are fetched and processed.
    worker.set_read_timeout(3600, 0);
    worker.set_write_timeout(60, 0);

    httplib::Result upstream;
    if (request.method == "POST") {
        upstream = worker.Post(path, request.body, "application/json");
    } else {
        upstream = worker.Get(path);
    }
    if (!upstream) {
        if (path == "/health") {
            response.status = 200;
            response.set_content(R"({"status":"loading"})", "application/json");
        } else {
            response.status = 503;
            response.set_content(R"({"detail":"The local model worker is unavailable."})", "application/json");
        }
        return;
    }
    response.status = upstream->status;
    std::string content_type = upstream->get_header_value("Content-Type");
    if (content_type.empty()) content_type = "application/json";
    response.set_content(upstream->body, content_type);
    response.set_header("Cache-Control", "no-store");
}

} // namespace

int main() {
    const int port = env_port("PORT", 10000);
    const int worker_port = env_port("DOTNXS_WORKER_PORT", 8001);
    const std::string default_origins =
        "https://dotnxs.rmaa.pk,https://dotnxs-website.onrender.com,"
        "https://rmaaorg-dotnxs-demo.static.hf.space,https://rmaaorg-dotnxs-demo.hf.space,"
        "http://localhost:8000,http://127.0.0.1:8000";
    const char *cors_value = std::getenv("CORS_ORIGINS");
    const auto allowed_origins = split_origins(cors_value ? cors_value : default_origins.c_str());

    httplib::Server server;
    server.set_payload_max_length(64 * 1024);

    server.Options(R"(/.*)", [&allowed_origins](const httplib::Request &request, httplib::Response &response) {
        add_cors_headers(request, response, allowed_origins);
        response.status = 204;
    });

    server.Get("/ping", [&allowed_origins](const httplib::Request &request, httplib::Response &response) {
        add_cors_headers(request, response, allowed_origins);
        response.set_header("Cache-Control", "no-store");
        response.set_content("pong", "text/plain; charset=utf-8");
    });

    server.Get("/health", [worker_port, &allowed_origins](const httplib::Request &request,
                                                           httplib::Response &response) {
        proxy_to_worker(request, response, worker_port, "/health", allowed_origins);
    });

    server.Post("/v1/chat", [worker_port, &allowed_origins](const httplib::Request &request,
                                                             httplib::Response &response) {
        proxy_to_worker(request, response, worker_port, "/v1/chat", allowed_origins);
    });

    std::cout << "DOTNXS C++ API listening on 0.0.0.0:" << port
              << ", forwarding model requests to 127.0.0.1:" << worker_port << std::endl;
    if (!server.listen("0.0.0.0", port)) {
        std::cerr << "Failed to bind Render PORT " << port << std::endl;
        return 1;
    }
    return 0;
}
