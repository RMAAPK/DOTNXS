#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dotnxs {

struct NxsChunk {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t logical_offset = 0;
    std::string sha256;
};

struct NxsEntry {
    std::string name;
    std::string kind;
    std::string dtype;
    std::uint64_t nbytes = 0;
    std::vector<std::int64_t> shape;
    std::vector<NxsChunk> chunks;
};

struct NxsCacheStats {
    std::uint64_t limit_bytes = 0;
    std::uint64_t used_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t network_bytes = 0;
    std::uint64_t network_ranges = 0;
};

class NxsRemoteArchive {
public:
    static constexpr std::uint64_t kDefaultCacheBytes = 200ULL * 1024ULL * 1024ULL;

    NxsRemoteArchive(std::string url, std::uint64_t file_size,
                     std::uint64_t cache_bytes = kDefaultCacheBytes);
    ~NxsRemoteArchive();

    NxsRemoteArchive(const NxsRemoteArchive &) = delete;
    NxsRemoteArchive &operator=(const NxsRemoteArchive &) = delete;

    static NxsRemoteArchive OpenHf(const std::string &repo_id,
                                   const std::string &filename,
                                   const std::string &revision = "main",
                                   std::uint64_t cache_bytes = kDefaultCacheBytes);
    static std::uint64_t ProbeSize(const std::string &url);

    const std::string &url() const noexcept { return url_; }
    std::uint64_t file_size() const noexcept { return file_size_; }
    const std::vector<NxsEntry> &entries() const noexcept { return entries_; }
    const NxsEntry &entry(const std::string &name) const;
    std::string ReadEntryRange(const std::string &name, std::uint64_t start,
                               std::uint64_t length);
    std::string ReadEntry(const std::string &name,
                          std::uint64_t max_bytes = 64ULL * 1024ULL * 1024ULL);
    std::string ReadTensorExpert(const std::string &name, std::uint32_t expert_id,
                                 std::uint32_t expert_count);
    NxsCacheStats cache_stats() const;
    void SetCacheLimit(std::uint64_t bytes);
    void ClearCache();

private:
    struct CacheValue {
        std::string bytes;
        std::list<std::string>::iterator lru_position;
    };

    std::string CacheKey(const std::string &name, std::size_t chunk_index) const;
    std::string FetchRange(std::uint64_t start, std::uint64_t length);
    std::string FetchChunk(const std::string &name, std::size_t index,
                           const NxsChunk &chunk);
    void StoreChunk(const std::string &key, std::string bytes);
    void ReadHeaderAndManifest();
    void ValidateEntries();

    std::string url_;
    std::uint64_t file_size_ = 0;
    std::uint64_t cache_limit_bytes_ = 0;
    std::uint64_t cache_used_bytes_ = 0;
    std::uint64_t cache_hits_ = 0;
    std::uint64_t cache_misses_ = 0;
    std::uint64_t cache_evictions_ = 0;
    std::uint64_t network_bytes_ = 0;
    std::uint64_t network_ranges_ = 0;
    std::uint64_t manifest_offset_ = 0;
    std::vector<NxsEntry> entries_;
    std::unordered_map<std::string, std::size_t> entry_index_;
    std::unordered_map<std::string, CacheValue> cache_;
    std::list<std::string> lru_;
    mutable std::mutex mutex_;
};

} // namespace dotnxs
