#include "nxs_remote_archive.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <regex>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace dotnxs {
namespace {

constexpr std::uint64_t kHeaderSize = 80;
constexpr std::uint64_t kAlignment = 4096;
constexpr std::uint64_t kMaxManifestBytes = 256ULL * 1024ULL * 1024ULL;
constexpr std::array<unsigned char, 20> kMagic = {
    0x4E, 0x58, 0x53, 0x00, 0x62, 0x62, 0x96, 0x6D, 0x2F, 0x13,
    0x49, 0xC9, 0x86, 0x91, 0xC5, 0x22, 0xDC, 0xE3, 0x08, 0x43};

class CurlGlobal {
public:
    CurlGlobal() {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("libcurl global initialization failed");
        }
    }
    ~CurlGlobal() { curl_global_cleanup(); }
};

CurlGlobal &curl_global() {
    static CurlGlobal instance;
    return instance;
}

struct CurlBuffer {
    std::string body;
    std::string content_range;
};

size_t write_body(char *data, size_t item_size, size_t item_count, void *user_data) {
    const size_t length = item_size * item_count;
    auto *buffer = static_cast<CurlBuffer *>(user_data);
    buffer->body.append(data, length);
    return length;
}

size_t write_header(char *data, size_t item_size, size_t item_count, void *user_data) {
    const size_t length = item_size * item_count;
    auto *buffer = static_cast<CurlBuffer *>(user_data);
    std::string line(data, length);
    constexpr char prefix[] = "Content-Range:";
    if (line.size() >= sizeof(prefix) - 1 &&
        std::equal(prefix, prefix + sizeof(prefix) - 1, line.begin(),
                   [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) ==
                                              std::tolower(static_cast<unsigned char>(b)); })) {
        const auto first = line.find_first_not_of(" \t", sizeof(prefix) - 1);
        const auto last = line.find_last_not_of(" \t\r\n");
        buffer->content_range = first == std::string::npos ? "" : line.substr(first, last - first + 1);
    }
    return length;
}

std::uint64_t read_le(const unsigned char *data, size_t width) {
    std::uint64_t value = 0;
    for (size_t index = 0; index < width; ++index) {
        value |= static_cast<std::uint64_t>(data[index]) << (8 * index);
    }
    return value;
}

std::string sha256_hex(const std::string &data) {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    if (!context) throw std::runtime_error("could not allocate SHA-256 context");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_length = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
                    EVP_DigestFinal_ex(context, digest.data(), &digest_length) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_length != 32) throw std::runtime_error("SHA-256 calculation failed");
    static const char hex[] = "0123456789abcdef";
    std::string output;
    output.reserve(64);
    for (unsigned int index = 0; index < digest_length; ++index) {
        output.push_back(hex[digest[index] >> 4]);
        output.push_back(hex[digest[index] & 0x0F]);
    }
    return output;
}

std::string encode_path(const std::string &value) {
    static const char hex[] = "0123456789ABCDEF";
    std::string output;
    output.reserve(value.size());
    for (unsigned char byte : value) {
        if (std::isalnum(byte) || byte == '-' || byte == '_' || byte == '.' || byte == '~' || byte == '/') {
            output.push_back(static_cast<char>(byte));
        } else {
            output.push_back('%');
            output.push_back(hex[byte >> 4]);
            output.push_back(hex[byte & 0x0F]);
        }
    }
    return output;
}

} // namespace

NxsRemoteArchive::NxsRemoteArchive(std::string url, std::uint64_t file_size,
                                   std::uint64_t cache_bytes)
    : url_(std::move(url)), file_size_(file_size), cache_limit_bytes_(cache_bytes) {
    (void)curl_global();
    if (url_.empty()) throw std::invalid_argument("NXS URL must not be empty");
    if (file_size_ < kHeaderSize) throw std::invalid_argument("NXS file size is smaller than its header");
    ReadHeaderAndManifest();
}

NxsRemoteArchive::~NxsRemoteArchive() { ClearCache(); }

NxsRemoteArchive NxsRemoteArchive::OpenHf(const std::string &repo_id,
                                          const std::string &filename,
                                          const std::string &revision,
                                          std::uint64_t cache_bytes) {
    const std::string url = "https://huggingface.co/" + repo_id + "/resolve/" +
                            encode_path(revision) + "/" + encode_path(filename);
    return NxsRemoteArchive(url, ProbeSize(url), cache_bytes);
}

std::uint64_t NxsRemoteArchive::ProbeSize(const std::string &url) {
    (void)curl_global();
    CURL *curl = curl_easy_init();
    if (!curl) throw std::runtime_error("libcurl could not create a HEAD request");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "dotnxs-cpp-range/0.3");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    CURLcode result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_off_t length = -1;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK) {
        throw std::runtime_error(std::string("NXS HEAD request failed: ") + curl_easy_strerror(result));
    }
    if (status < 200 || status >= 300 || length <= 0) {
        throw std::runtime_error("Hugging Face did not return the NXS archive size");
    }
    return static_cast<std::uint64_t>(length);
}

std::string NxsRemoteArchive::FetchRange(std::uint64_t start, std::uint64_t length) {
    if (length == 0 || start > file_size_ || length > file_size_ - start) {
        throw std::out_of_range("NXS HTTP range is outside the archive");
    }
    CURL *curl = curl_easy_init();
    if (!curl) throw std::runtime_error("libcurl could not create a range request");
    CurlBuffer buffer;
    const std::string range = std::to_string(start) + "-" + std::to_string(start + length - 1);
    curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
    curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "dotnxs-cpp-range/0.3");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_body);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, write_header);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &buffer);
    const CURLcode result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK) {
        throw std::runtime_error(std::string("NXS HTTP range failed: ") + curl_easy_strerror(result));
    }
    if (status != 206) throw std::runtime_error("NXS host did not honor an HTTP Range request");
    if (buffer.body.size() != length) throw std::runtime_error("NXS range returned an unexpected byte count");
    static const std::regex content_range(R"(^bytes (\d+)-(\d+)/(\d+|\*)$)");
    std::smatch match;
    if (!std::regex_match(buffer.content_range, match, content_range) ||
        std::stoull(match[1].str()) != start || std::stoull(match[2].str()) != start + length - 1) {
        throw std::runtime_error("NXS host returned an invalid Content-Range header");
    }
    if (match[3].str() != "*" && std::stoull(match[3].str()) != file_size_) {
        throw std::runtime_error("NXS archive size changed during ranged reading");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        network_bytes_ += length;
        network_ranges_ += 1;
    }
    return buffer.body;
}

void NxsRemoteArchive::ReadHeaderAndManifest() {
    const std::string header = FetchRange(0, kHeaderSize);
    if (header.size() != kHeaderSize) throw std::runtime_error("truncated NXS header");
    for (size_t index = 0; index < kMagic.size(); ++index) {
        if (static_cast<unsigned char>(header[index]) != kMagic[index]) {
            throw std::runtime_error("remote file has the wrong DOTNXS magic");
        }
    }
    const auto *bytes = reinterpret_cast<const unsigned char *>(header.data());
    const auto major = read_le(bytes + 20, 2);
    const auto minor = read_le(bytes + 22, 2);
    const auto header_size = read_le(bytes + 24, 4);
    const auto flags = read_le(bytes + 28, 4);
    const auto manifest_offset = read_le(bytes + 32, 8);
    const auto manifest_length = read_le(bytes + 40, 8);
    std::string expected_hash;
    static const char hex[] = "0123456789abcdef";
    for (size_t index = 48; index < 80; ++index) {
        const auto byte = static_cast<unsigned char>(header[index]);
        expected_hash.push_back(hex[byte >> 4]);
        expected_hash.push_back(hex[byte & 0x0F]);
    }
    if (major != 0 || minor != 2 || header_size != kHeaderSize || flags != 0) {
        throw std::runtime_error("unsupported DOTNXS header version or flags");
    }
    if (manifest_length == 0 || manifest_length > kMaxManifestBytes ||
        manifest_offset < kHeaderSize || manifest_offset > file_size_ ||
        manifest_length != file_size_ - manifest_offset) {
        throw std::runtime_error("remote NXS manifest range is invalid");
    }
    manifest_offset_ = manifest_offset;
    const std::string manifest_text = FetchRange(manifest_offset, manifest_length);
    if (sha256_hex(manifest_text) != expected_hash) {
        throw std::runtime_error("NXS manifest SHA-256 mismatch");
    }

    const auto manifest = nlohmann::json::parse(manifest_text);
    if (manifest.value("format", std::string()) != "DOTNXS" ||
        manifest.at("version") != nlohmann::json::array({0, 2})) {
        throw std::runtime_error("NXS manifest format/version mismatch");
    }
    for (const auto &item : manifest.at("entries")) {
        NxsEntry entry;
        entry.name = item.at("name").get<std::string>();
        entry.kind = item.at("kind").get<std::string>();
        entry.nbytes = item.at("nbytes").get<std::uint64_t>();
        entry.dtype = item.value("dtype", std::string());
        if (item.contains("shape")) entry.shape = item.at("shape").get<std::vector<std::int64_t>>();
        for (const auto &chunk_json : item.at("chunks")) {
            entry.chunks.push_back({
                chunk_json.at("offset").get<std::uint64_t>(),
                chunk_json.at("length").get<std::uint64_t>(),
                chunk_json.at("logical_offset").get<std::uint64_t>(),
                chunk_json.at("sha256").get<std::string>(),
            });
        }
        if (!entry_index_.emplace(entry.name, entries_.size()).second) {
            throw std::runtime_error("duplicate NXS entry name");
        }
        entries_.push_back(std::move(entry));
    }
    ValidateEntries();
}

void NxsRemoteArchive::ValidateEntries() {
    std::vector<std::tuple<std::uint64_t, std::uint64_t, std::string>> payload_ranges;
    for (const auto &entry : entries_) {
        std::uint64_t logical = 0;
        for (const auto &chunk : entry.chunks) {
            if (chunk.length == 0 || chunk.logical_offset != logical || chunk.offset < kHeaderSize ||
                chunk.offset % kAlignment != 0 || chunk.offset > manifest_offset_ ||
                chunk.length > manifest_offset_ - chunk.offset) {
                throw std::runtime_error("invalid NXS payload chunk bounds");
            }
            logical += chunk.length;
            payload_ranges.emplace_back(chunk.offset, chunk.offset + chunk.length, entry.name);
        }
        if (logical != entry.nbytes) throw std::runtime_error("NXS chunks do not cover entry bytes");
    }
    std::sort(payload_ranges.begin(), payload_ranges.end());
    std::uint64_t previous_end = kHeaderSize;
    for (const auto &range : payload_ranges) {
        if (std::get<0>(range) < previous_end) throw std::runtime_error("overlapping NXS payload chunks");
        previous_end = std::get<1>(range);
    }
}

const NxsEntry &NxsRemoteArchive::entry(const std::string &name) const {
    const auto found = entry_index_.find(name);
    if (found == entry_index_.end()) throw std::out_of_range("NXS entry not found: " + name);
    return entries_[found->second];
}

std::string NxsRemoteArchive::CacheKey(const std::string &name, std::size_t chunk_index) const {
    return name + "\n" + std::to_string(chunk_index);
}

void NxsRemoteArchive::StoreChunk(const std::string &key, std::string bytes) {
    if (bytes.size() > cache_limit_bytes_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto old = cache_.find(key);
    if (old != cache_.end()) {
        cache_used_bytes_ -= old->second.bytes.size();
        lru_.erase(old->second.lru_position);
        cache_.erase(old);
    }
    while (!lru_.empty() && cache_used_bytes_ + bytes.size() > cache_limit_bytes_) {
        const std::string oldest_key = lru_.back();
        const auto found = cache_.find(oldest_key);
        if (found != cache_.end()) {
            cache_used_bytes_ -= found->second.bytes.size();
            cache_.erase(found);
        }
        lru_.pop_back();
        cache_evictions_ += 1;
    }
    lru_.push_front(key);
    cache_used_bytes_ += bytes.size();
    cache_.emplace(key, CacheValue{std::move(bytes), lru_.begin()});
}

std::string NxsRemoteArchive::FetchChunk(const std::string &name, std::size_t index,
                                         const NxsChunk &chunk) {
    const std::string key = CacheKey(name, index);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = cache_.find(key);
        if (found != cache_.end()) {
            lru_.erase(found->second.lru_position);
            lru_.push_front(key);
            found->second.lru_position = lru_.begin();
            cache_hits_ += 1;
            return found->second.bytes;
        }
        cache_misses_ += 1;
    }
    std::string bytes = FetchRange(chunk.offset, chunk.length);
    if (sha256_hex(bytes) != chunk.sha256) throw std::runtime_error("NXS payload chunk SHA-256 mismatch: " + name);
    StoreChunk(key, bytes);
    return bytes;
}

std::string NxsRemoteArchive::ReadEntryRange(const std::string &name, std::uint64_t start,
                                            std::uint64_t length) {
    const auto &source = entry(name);
    if (start > source.nbytes || length > source.nbytes - start) {
        throw std::out_of_range("NXS entry range is outside the entry");
    }
    std::string output;
    output.reserve(static_cast<std::size_t>(length));
    const std::uint64_t end = start + length;
    for (std::size_t index = 0; index < source.chunks.size(); ++index) {
        const auto &chunk = source.chunks[index];
        const auto chunk_start = chunk.logical_offset;
        const auto chunk_end = chunk.logical_offset + chunk.length;
        const auto overlap_start = std::max(start, chunk_start);
        const auto overlap_end = std::min(end, chunk_end);
        if (overlap_start >= overlap_end) continue;
        const std::string bytes = FetchChunk(name, index, chunk);
        const auto local_start = static_cast<std::size_t>(overlap_start - chunk_start);
        const auto local_length = static_cast<std::size_t>(overlap_end - overlap_start);
        output.append(bytes.data() + local_start, local_length);
    }
    return output;
}

std::string NxsRemoteArchive::ReadEntry(const std::string &name, std::uint64_t max_bytes) {
    const auto &source = entry(name);
    if (source.nbytes > max_bytes) throw std::length_error("NXS entry exceeds bounded read limit");
    return ReadEntryRange(name, 0, source.nbytes);
}

std::string NxsRemoteArchive::ReadTensorExpert(const std::string &name, std::uint32_t expert_id,
                                               std::uint32_t expert_count) {
    const auto &source = entry(name);
    if (source.kind != "tensor" || source.shape.empty() || source.shape[0] != expert_count ||
        expert_id >= expert_count || source.nbytes % expert_count != 0) {
        throw std::invalid_argument("NXS tensor does not match the requested expert layout");
    }
    const auto bytes_per_expert = source.nbytes / expert_count;
    return ReadEntryRange(name, expert_id * bytes_per_expert, bytes_per_expert);
}

NxsCacheStats NxsRemoteArchive::cache_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {cache_limit_bytes_, cache_used_bytes_, cache_hits_, cache_misses_, cache_evictions_,
            network_bytes_, network_ranges_};
}

void NxsRemoteArchive::SetCacheLimit(std::uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_limit_bytes_ = bytes;
    while (!lru_.empty() && cache_used_bytes_ > cache_limit_bytes_) {
        const std::string oldest_key = lru_.back();
        const auto found = cache_.find(oldest_key);
        if (found != cache_.end()) {
            cache_used_bytes_ -= found->second.bytes.size();
            cache_.erase(found);
        }
        lru_.pop_back();
        cache_evictions_ += 1;
    }
}

void NxsRemoteArchive::ClearCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_.clear();
    cache_used_bytes_ = 0;
}

} // namespace dotnxs
