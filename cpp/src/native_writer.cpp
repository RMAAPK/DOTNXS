#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {
constexpr std::array<unsigned char, 20> kMagic = {
    0x4E, 0x58, 0x53, 0x00, 0x62, 0x62, 0x96, 0x6D, 0x2F, 0x13,
    0x49, 0xC9, 0x86, 0x91, 0xC5, 0x22, 0xDC, 0xE3, 0x08, 0x43};
constexpr std::uint64_t kHeaderSize = 80;
constexpr std::uint64_t kAlignment = 4096;
constexpr std::uint64_t kCopyBufferBytes = 1024 * 1024;
constexpr std::uint64_t kMaxManifestBytes = 256ULL * 1024ULL * 1024ULL;

void validate_name(const std::string& name) {
    if (name.empty() || name.front() == '/' || name.front() == '\\') {
        throw std::invalid_argument("entry name must be non-empty and relative");
    }
    std::string component;
    for (char ch : name) {
        if (ch == '/' || ch == '\\') {
            if (component == "..") {
                throw std::invalid_argument("entry name may not contain '..' path components");
            }
            component.clear();
        } else {
            component.push_back(ch);
        }
    }
    if (component == "..") {
        throw std::invalid_argument("entry name may not contain '..' path components");
    }
}

py::object new_sha256() {
    return py::module_::import("hashlib").attr("sha256")();
}

void update_sha256(const py::object& hasher, const char* bytes, std::size_t size) {
    hasher.attr("update")(py::bytes(bytes, size));
}

std::string hexdigest(const py::object& hasher) {
    return py::cast<std::string>(hasher.attr("hexdigest")());
}

std::string raw_digest(const py::object& hasher) {
    return py::cast<std::string>(hasher.attr("digest")());
}

void store_le(unsigned char* target, std::uint64_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        target[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFF);
    }
}

}  // namespace

class NativeWriter {
public:
    NativeWriter(const std::string& output_path, std::uint64_t chunk_bytes)
        : final_path_(fs::u8path(output_path)), temp_path_(final_path_),
          chunk_bytes_(chunk_bytes) {
        temp_path_ += ".part";
        if (chunk_bytes_ == 0) {
            throw std::invalid_argument("chunk_bytes must be greater than zero");
        }
        fs::path parent = final_path_.parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        std::error_code error;
        fs::remove(temp_path_, error);
        output_.open(temp_path_, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!output_) {
            throw std::runtime_error("could not create temporary NXS archive");
        }
        std::array<char, kHeaderSize> empty_header{};
        output_.write(empty_header.data(), static_cast<std::streamsize>(empty_header.size()));
        if (!output_) {
            throw std::runtime_error("could not reserve NXS header");
        }
    }

    NativeWriter(const NativeWriter&) = delete;
    NativeWriter& operator=(const NativeWriter&) = delete;

    ~NativeWriter() { abort(); }

    py::dict add_file_range(const std::string& name,
                            const std::string& source_path,
                            std::uint64_t source_offset,
                            std::uint64_t size,
                            const std::string& kind,
                            const std::string& dtype,
                            const std::vector<std::int64_t>& shape,
                            const py::dict& metadata) {
        if (finished_) {
            throw std::runtime_error("NXS writer is already finalized");
        }
        validate_name(name);
        if (kind.empty()) {
            throw std::invalid_argument("entry kind must be non-empty");
        }
        if (names_.count(name)) {
            throw std::invalid_argument("duplicate NXS entry name: " + name);
        }
        for (auto dimension : shape) {
            if (dimension < 0) {
                throw std::invalid_argument("tensor shape dimensions must be non-negative");
            }
        }

        std::ifstream source(fs::u8path(source_path), std::ios::binary);
        if (!source) {
            throw std::runtime_error("could not open source file: " + source_path);
        }
        source.seekg(0, std::ios::end);
        const auto end_position = source.tellg();
        if (end_position == std::streampos(-1)) {
            throw std::runtime_error("could not determine source file size");
        }
        const auto source_end = static_cast<std::streamoff>(end_position);
        if (source_end < 0) {
            throw std::runtime_error("could not determine source file size");
        }
        const auto source_size = static_cast<std::uint64_t>(source_end);
        if (source_offset > source_size || size > source_size - source_offset) {
            throw std::invalid_argument("source range extends past end of file");
        }
        source.seekg(static_cast<std::streamoff>(source_offset), std::ios::beg);
        if (!source) {
            throw std::runtime_error("could not seek to source range");
        }
        py::dict entry = write_entry(name, source, size, kind, dtype, shape, metadata);
        names_.insert(name);
        return entry;
    }

    py::dict add_bytes(const std::string& name,
                       const py::bytes& input,
                       const std::string& kind,
                       const std::string& dtype,
                       const std::vector<std::int64_t>& shape,
                       const py::dict& metadata) {
        if (finished_) {
            throw std::runtime_error("NXS writer is already finalized");
        }
        const std::string data = py::cast<std::string>(input);
        std::istringstream source(data, std::ios::in | std::ios::binary);
        py::dict entry = write_entry(name, source, data.size(), kind, dtype, shape, metadata);
        names_.insert(name);
        return entry;
    }

    void finalize(const py::bytes& manifest_bytes) {
        if (finished_) {
            throw std::runtime_error("NXS writer is already finalized");
        }
        try {
            const std::string manifest = py::cast<std::string>(manifest_bytes);
            if (manifest.empty() || manifest.size() > kMaxManifestBytes) {
                throw std::invalid_argument("manifest size is outside the v0.2 limit");
            }
            align_output();
            const auto manifest_offset = tell_output();
            output_.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
            if (!output_) {
                throw std::runtime_error("failed writing NXS manifest");
            }
            auto manifest_hash = new_sha256();
            update_sha256(manifest_hash, manifest.data(), manifest.size());
            const std::string digest = raw_digest(manifest_hash);
            if (digest.size() != 32) {
                throw std::runtime_error("SHA-256 implementation returned an invalid digest");
            }

            std::array<unsigned char, kHeaderSize> header{};
            std::copy(kMagic.begin(), kMagic.end(), header.begin());
            store_le(header.data() + 20, 0, 2);  // major
            store_le(header.data() + 22, 2, 2);  // minor
            store_le(header.data() + 24, kHeaderSize, 4);
            store_le(header.data() + 28, 0, 4);  // reserved flags
            store_le(header.data() + 32, manifest_offset, 8);
            store_le(header.data() + 40, manifest.size(), 8);
            std::copy(digest.begin(), digest.end(), header.begin() + 48);

            output_.seekp(0, std::ios::beg);
            output_.write(reinterpret_cast<const char*>(header.data()),
                          static_cast<std::streamsize>(header.size()));
            output_.flush();
            if (!output_) {
                throw std::runtime_error("failed finalizing NXS header");
            }
            output_.close();
            std::error_code error;
            fs::remove(final_path_, error);
            error.clear();
            fs::rename(temp_path_, final_path_, error);
            if (error) {
                throw std::runtime_error("could not move completed archive into place: " + error.message());
            }
            finished_ = true;
        } catch (...) {
            abort();
            throw;
        }
    }

    void abort() noexcept {
        if (finished_) {
            return;
        }
        if (output_.is_open()) {
            output_.close();
        }
        std::error_code error;
        fs::remove(temp_path_, error);
        finished_ = true;
    }

private:
    std::uint64_t tell_output() {
        const auto position = output_.tellp();
        if (position == std::streampos(-1)) {
            throw std::runtime_error("could not determine NXS output position");
        }
        const auto byte_position = static_cast<std::streamoff>(position);
        if (byte_position < 0) {
            throw std::runtime_error("could not determine NXS output position");
        }
        return static_cast<std::uint64_t>(byte_position);
    }

    void align_output() {
        const auto position = tell_output();
        const auto padding = (kAlignment - (position % kAlignment)) % kAlignment;
        if (padding) {
            std::array<char, kAlignment> zeros{};
            output_.write(zeros.data(), static_cast<std::streamsize>(padding));
            if (!output_) {
                throw std::runtime_error("failed writing NXS alignment padding");
            }
        }
    }

    py::dict write_entry(const std::string& name,
                         std::istream& source,
                         std::uint64_t size,
                         const std::string& kind,
                         const std::string& dtype,
                         const std::vector<std::int64_t>& shape,
                         const py::dict& metadata) {
        validate_name(name);
        if (kind.empty()) {
            throw std::invalid_argument("entry kind must be non-empty");
        }
        if (names_.count(name)) {
            throw std::invalid_argument("duplicate NXS entry name: " + name);
        }
        for (auto dimension : shape) {
            if (dimension < 0) {
                throw std::invalid_argument("tensor shape dimensions must be non-negative");
            }
        }
        py::object whole_hash = new_sha256();
        py::list chunks;
        std::uint64_t logical_offset = 0;
        std::vector<char> buffer(static_cast<std::size_t>(
            std::min(kCopyBufferBytes, chunk_bytes_)));

        while (logical_offset < size) {
            const auto chunk_length = std::min(chunk_bytes_, size - logical_offset);
            align_output();
            const auto file_offset = tell_output();
            py::object chunk_hash = new_sha256();
            std::uint64_t remaining = chunk_length;
            while (remaining > 0) {
                const auto request = static_cast<std::streamsize>(
                    std::min<std::uint64_t>(remaining, buffer.size()));
                source.read(buffer.data(), request);
                const auto got = source.gcount();
                if (got <= 0) {
                    throw std::runtime_error("source file ended before the requested tensor range");
                }
                output_.write(buffer.data(), got);
                if (!output_) {
                    throw std::runtime_error("failed writing NXS tensor payload");
                }
                update_sha256(chunk_hash, buffer.data(), static_cast<std::size_t>(got));
                update_sha256(whole_hash, buffer.data(), static_cast<std::size_t>(got));
                remaining -= static_cast<std::uint64_t>(got);
            }
            py::dict chunk;
            chunk["offset"] = file_offset;
            chunk["length"] = chunk_length;
            chunk["logical_offset"] = logical_offset;
            chunk["sha256"] = hexdigest(chunk_hash);
            chunks.append(std::move(chunk));
            logical_offset += chunk_length;
        }

        py::dict entry;
        entry["name"] = name;
        entry["kind"] = kind;
        entry["nbytes"] = size;
        entry["sha256"] = hexdigest(whole_hash);
        entry["chunks"] = std::move(chunks);
        if (!dtype.empty()) {
            entry["dtype"] = dtype;
        }
        if (kind == "tensor" || !shape.empty()) {
            entry["shape"] = shape;
        }
        if (metadata.size() > 0) {
            entry["metadata"] = metadata;
        }
        return entry;
    }

    fs::path final_path_;
    fs::path temp_path_;
    std::uint64_t chunk_bytes_;
    std::fstream output_;
    std::set<std::string> names_;
    bool finished_ = false;
};

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native DOTNXS archive writer";
    py::class_<NativeWriter>(module, "NativeWriter")
        .def(py::init<const std::string&, std::uint64_t>(),
             py::arg("output_path"), py::arg("chunk_bytes"))
        .def("add_file_range", &NativeWriter::add_file_range,
             py::arg("name"), py::arg("source_path"), py::arg("source_offset"),
             py::arg("size"), py::arg("kind"), py::arg("dtype"), py::arg("shape"),
             py::arg("metadata"))
        .def("add_bytes", &NativeWriter::add_bytes,
             py::arg("name"), py::arg("data"), py::arg("kind"), py::arg("dtype"),
             py::arg("shape"), py::arg("metadata"))
        .def("finalize", &NativeWriter::finalize, py::arg("manifest_bytes"))
        .def("abort", &NativeWriter::abort);
}
