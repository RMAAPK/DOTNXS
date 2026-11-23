#include "third_party/httplib.h"

#include <atomic>
#include <cctype>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

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

std::string url_encode(const std::string &value) {
    static const char hex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(value.size() * 3);
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back(static_cast<char>(ch));
        } else {
            encoded.push_back('%');
            encoded.push_back(hex[ch >> 4]);
            encoded.push_back(hex[ch & 0x0F]);
        }
    }
    return encoded;
}

struct StreamBuffer {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> chunks;
    std::atomic<int> upstream_status{0};
    bool finished = false;
    bool cancelled = false;
    std::thread producer;

    bool push(const char *data, size_t size) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return chunks.size() < 8 || cancelled; });
        if (cancelled) return false;
        chunks.emplace_back(data, size);
        changed.notify_all();
        return true;
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        changed.notify_all();
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        changed.notify_all();
    }

    bool is_cancelled() {
        std::lock_guard<std::mutex> lock(mutex);
        return cancelled;
    }

    bool write_next(httplib::DataSink &sink) {
        std::unique_lock<std::mutex> lock(mutex);
        while (chunks.empty() && !finished && !cancelled) {
            changed.wait_for(lock, std::chrono::milliseconds(250));
            if (!sink.is_writable()) {
                cancelled = true;
                changed.notify_all();
                return false;
            }
        }
        if (cancelled) return false;
        if (chunks.empty()) {
            if (finished) sink.done();
            return finished;
        }
        std::string chunk = std::move(chunks.front());
        chunks.pop_front();
        changed.notify_all();
        lock.unlock();
        if (!sink.write(chunk.data(), chunk.size())) {
            cancel();
            return false;
        }
        return true;
    }
};

std::atomic<bool> generation_active{false};

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
        add_cors_headers(request, response, allowed_origins);
        if (generation_active.exchange(true)) {
            response.status = 429;
            response.set_content(R"({"detail":"The CPU model is busy; retry shortly."})", "application/json");
            return;
        }

        httplib::Client health_client("127.0.0.1", worker_port);
        health_client.set_connection_timeout(5, 0);
        health_client.set_read_timeout(5, 0);
        const auto health = health_client.Get("/health");
        if (!health || health->status != 200 || health->body.find("\"status\":\"ready\"") == std::string::npos) {
            generation_active.store(false);
            response.status = 503;
            response.set_header("Retry-After", "30");
            response.set_content(R"({"detail":"The model worker is still warming up."})", "application/json");
            return;
        }

        auto stream = std::make_shared<StreamBuffer>();
        const std::string path = "/v1/chat-stream?payload=" + url_encode(request.body);
        stream->producer = std::thread([stream, worker_port, path]() {
            try {
                httplib::Client worker("127.0.0.1", worker_port);
                worker.set_connection_timeout(5, 0);
                worker.set_read_timeout(3600, 0);
                worker.set_write_timeout(60, 0);
                const auto upstream = worker.Get(
                    path,
                    [stream](const httplib::Response &headers) {
                        stream->upstream_status.store(headers.status);
                        return headers.status == 200;
                    },
                    [stream](const char *data, size_t size) { return stream->push(data, size); });
                if (!upstream && !stream->is_cancelled()) {
                    const int status = stream->upstream_status.load();
                    const std::string message = "event: error\ndata: {\"message\":\"Model worker returned HTTP " +
                                                std::to_string(status) + "\"}\n\n";
                    stream->push(message.data(), message.size());
                }
            } catch (const std::exception &error) {
                const std::string message = std::string("event: error\ndata: {\"message\":\"") +
                                            error.what() + "\"}\n\n";
                stream->push(message.data(), message.size());
            }
            stream->finish();
        });

        response.status = 200;
        response.set_header("Cache-Control", "no-cache, no-transform");
        response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider(
            "text/event-stream",
            [stream](size_t, httplib::DataSink &sink) { return stream->write_next(sink); },
            [stream](bool) {
                stream->cancel();
                if (stream->producer.joinable()) stream->producer.join();
                generation_active.store(false);
            });
    });

    std::cout << "DOTNXS C++ API listening on 0.0.0.0:" << port
              << ", forwarding model requests to 127.0.0.1:" << worker_port << std::endl;
    if (!server.listen("0.0.0.0", port)) {
        std::cerr << "Failed to bind Render PORT " << port << std::endl;
        return 1;
    }
    return 0;
}
