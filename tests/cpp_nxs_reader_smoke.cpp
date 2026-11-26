#include "nxs_remote_archive.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: nxs-reader-smoke URL\n";
        return 2;
    }
    try {
        const std::string url = argv[1];
        dotnxs::NxsRemoteArchive archive(url, dotnxs::NxsRemoteArchive::ProbeSize(url), 8);
        if (archive.ReadEntry("payload") != "abcdefgh") return 3;
        if (archive.ReadTensorExpert("tensor", 1, 2) != "89abcdef") return 4;
        const auto stats = archive.cache_stats();
        if (stats.used_bytes > stats.limit_bytes || stats.limit_bytes != 8) return 5;
        archive.ClearCache();
        if (archive.cache_stats().used_bytes != 0) return 6;
        std::cout << "C++ NXS reader verified ranges, expert slicing, hashes, and LRU cache.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
