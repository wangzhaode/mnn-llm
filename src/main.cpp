//
//  main.cpp
//
//  Created by MNN on 2024/12/25.
//  ZhaodeWang
//

#include <unistd.h>
#include <regex>
#include <iomanip>
#include <csignal>
#include <iostream>
#include <filesystem>

#include "progress.h"
#include "httplib.h"
#include "json.hpp"
#include "CLI11.hpp"
#include "spdlog/async.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "core/MNNFileUtils.h"
#include "llm/llm.hpp"

using namespace MNN::Transformer;
namespace fs = std::filesystem;

static const char* MLLM_VERSION = "0.0.1";
static const char* MLLM_LOGO = R"(
     __    __   __       __       __    __
    /\ \- /  \ /\ \     /\ \     /\ \- /  \
    \ \ \- /\ \\ \ \____\ \ \____\ \ \- /\ \
     \ \_\ \ \_\\ \_____\\ \_____\\ \_\ \ \_\
      \/_/  \/_/ \/_____/ \/_____/ \/_/  \/_/
)";

void download_file(const std::string& host, const std::string& path, const fs::path& file_name, const std::string& sha256 = "") {
    std::ofstream ofs(file_name, std::ios::binary);
    auto content_receiver = [&ofs](const char *data, size_t data_length) {
        ofs.write(data, data_length);
        return true;
    };
    progress::tqdm bar;
    bar.set_label("pulling " + sha256.substr(0, 10) + "...");
    bar.disable_colors();
    uint64_t last_progress_percent = 0;
    uint64_t update_increment = 1;
    auto progress = [&](uint64_t current, uint64_t total) {
        uint64_t current_percent = current * 100 / total;
        if (current_percent >= last_progress_percent + update_increment) {
            bar.progress(current, total);
            last_progress_percent = current_percent;
        }
        return true;
    };
    httplib::Client cli(host);
    cli.Get(path, content_receiver, progress);
    bar.finish();
}

std::string fetch_content(const std::string& host, const std::string& path) {
    auto res = httplib::Client(host).Get(path);
    if (res && res->status == 200) { return res->body; }
    return "";
}

static fs::path get_home_dir() {
#ifdef _WIN32
    const char* homeDir = std::getenv("USERPROFILE");
    if (!homeDir) {
        throw std::runtime_error("USERPROFILE environment variable is not set.");
    }
    return fs::path(homeDir);
#else
    const char* homeDir = std::getenv("HOME");
    if (!homeDir) {
        throw std::runtime_error("HOME environment variable is not set.");
    }
    return fs::path(homeDir);
#endif
}

static void create_dir(const fs::path& dir) {
    if (!fs::exists(dir)) { fs::create_directories(dir); }
}

std::string dir_size(const fs::path& directoryPath) {
    uintmax_t total_size = 0;
    for (const auto& entry : fs::recursive_directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry.status())) {
            total_size += fs::file_size(entry.path());
        }
    }
    std::ostringstream oss;
    if (total_size < 1073741824) {
        float size_in_mb = total_size / 1048576.0;
        oss << std::fixed << std::setprecision(1) << size_in_mb << " MB";
    } else {
        float size_in_gb = total_size / 1073741824.0;
        oss << std::fixed << std::setprecision(1) << size_in_gb << " GB";
    }
    return oss.str();
}

std::string time_ago(const fs::path& filePath) {
    auto ftime = fs::last_write_time(filePath);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
    auto file_time = std::chrono::system_clock::from_time_t(std::chrono::system_clock::to_time_t(sctp));
    auto now = std::chrono::system_clock::now();
    auto duration = now - file_time;
    auto hours_ago = std::chrono::duration_cast<std::chrono::hours>(duration).count();
    if (hours_ago < 24) {
        return std::to_string(hours_ago) + " hours ago";
    }
    if (hours_ago < 24 * 30) {
        return std::to_string(hours_ago / 24) + " days ago";
    }
    if (hours_ago < 24 * 365) {
        return std::to_string(hours_ago / (24 * 30)) + " months ago";
    }
    return std::to_string(hours_ago / (24 * 365)) + " years ago";
}

std::string human_byte(size_t size) {
    std::ostringstream oss;
    if (size < 1024) {
        oss << size << " B";
    } else if (size < 1048576) {
        oss << std::fixed << std::setprecision(1) << size / 1024.0 << " KB";
    } else if (size < 1073741824) {
        oss << std::fixed << std::setprecision(1) << size / 1048576.0 << " MB";
    } else {
        oss << std::fixed << std::setprecision(1) << size / 1073741824.0 << " GB";
    }
    return oss.str();
}

std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto local_time = std::gmtime(&time_t_now);
    std::ostringstream oss;
    oss << std::put_time(local_time, "%Y-%m-%dT%H:%M:%S");
    auto duration = now.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration) -
                        std::chrono::duration_cast<std::chrono::microseconds>(seconds);
    oss << "." << std::setfill('0') << std::setw(6) << microseconds.count() << "Z";
    return oss.str();
}

void create_daemon(spdlog::logger* logger) {
    pid_t pid = fork();

    if (pid < 0) {
        logger->error("Fork failed!");
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        logger->info("Daemon process created with PID: {}", pid);
        exit(EXIT_SUCCESS);
    }

    if (setsid() < 0) {
        logger->error("setsid failed!");
        exit(EXIT_FAILURE);
    }

    if (chdir("/") < 0) {
        logger->error("chdir failed!");
        exit(EXIT_FAILURE);
    }

    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    open("/dev/null", O_RDONLY); // stdin
    open("/dev/null", O_RDWR);   // stdout
    open("/dev/null", O_RDWR);   // stderr
}

class LlmStreamBuffer : public std::streambuf {
public:
    using CallBack = std::function<void(const char* str, size_t len)>;;
    LlmStreamBuffer(CallBack callback) : callback_(callback) {}

protected:
    virtual std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (callback_) {
            callback_(s, n);
        }
        return n;
    }
private:
    CallBack callback_ = nullptr;
};

class Runner {
public:
    int port = 8000;
    const std::string name = "mllm";
    std::string end_point = "modelscope.cn";
    bool version = false, help = false;
    fs::path root_dir, models_dir, server_dir,
             manifests_file, pid_file, log_file;
    std::shared_ptr<spdlog::logger> logger;
public:
    Runner() { init(); }
    ~Runner() {}

    void init() {
        root_dir = get_home_dir() / ".mllm";
        models_dir = root_dir / "models";
        server_dir = root_dir / "server";
        create_dir(models_dir.c_str());
        create_dir(server_dir.c_str());
        manifests_file = models_dir / "manifests";
        pid_file = server_dir / "server.pid";
        log_file = server_dir / "server.log";
        // logger
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::warn);
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file);
        file_sink->set_level(spdlog::level::trace);
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
        logger->flush_on(spdlog::level::info);
    }

    void registry(const std::string& type) {
        if (type == "HF") {
        } else if (type == "MS") {
        }
    }

    void pull_manifests() {
        auto content = fetch_content(end_point, "/datasets/zhaode/manifests/resolve/master/manifests");
        auto body = nlohmann::json::parse(content);
        std::ofstream file(manifests_file);
        if (file.is_open()) {
            file << body.dump(4);
            file.close();
        }
    }

    nlohmann::json pull_manifest(const std::string& model) {
        if (!fs::exists(manifests_file)) { pull_manifests(); }
        auto manifests_json = nlohmann::json::parse(std::ifstream(manifests_file));
        if (manifests_json.find(model) == manifests_json.end()) {
            return nlohmann::json();
        }
        progress::spinner spinner;
        spinner.set_label("pulling manifest ");
        spinner.start();
        std::string model_id = manifests_json[model];
        auto content = fetch_content(end_point, "/api/v1/models/MNN/" + model_id + "/repo/files");
        auto body = nlohmann::json::parse(content);
        std::string shortId = body["Data"]["LatestCommitter"]["ShortId"];
        size_t total_size = 0;
        nlohmann::json files_array = nlohmann::json::array();
        for (const auto& file : body["Data"]["Files"]) {
            std::string path = file["Path"];
            if (path == ".gitattributes" || path == "README.md" ||
                path == "configuration.json" || path == "llm.mnn.json") {
                continue;
            }
            uint64_t size = file["Size"];
            std::string sha256 = file["Sha256"];
            total_size += size;
            nlohmann::json file_json;
            file_json["path"] = path;
            file_json["size"] = size;
            file_json["sha256"] = sha256;
            files_array.push_back(file_json);
        }
        nlohmann::json manifest;
        manifest["model"] = model;
        manifest["model_id"] = model_id;
        manifest["id"] = shortId;
        manifest["size"] = total_size;
        manifest["files"] = files_array;
        spinner.stop();
        return manifest;
    }

    void pull(const std::string& model) {
        auto model_dir = models_dir / model;
        create_dir(model_dir);
        auto model_manifest = model_dir / "manifest";
        auto manifest = pull_manifest(model);
        if (manifest.empty()) { return; }
        // dump manifest to file
        std::ofstream file(model_manifest);
        if (file.is_open()) {
            file << manifest.dump(4);
            file.close();
        }
        // download model files
        auto base_path = "/api/v1/models/MNN/" + manifest["model_id"].get<std::string>() + "/repo?FilePath=";
        for (const auto& file : manifest["files"]) {
            std::string path = file["path"];
            size_t size = file["size"];
            std::string sha256 = file["sha256"];
            auto file_path = model_dir / path;
            if (fs::exists(file_path) && fs::file_size(file_path) == size) {
                continue;
            }
            download_file(end_point, base_path + path, file_path, sha256);
        }
    }

    void run(const std::string& model) {
        auto model_dir = MNNFilePathConcat(models_dir, model);
        auto config = MNNFilePathConcat(model_dir, "config.json");
        std::unique_ptr<Llm> llm(Llm::createLLM(config));
        llm->load();
        while (true) {
            std::cout << "\nQ: ";
            std::string user_str;
            std::cin >> user_str;
            if (user_str == "/exit") {
                break;
            }
            std::cout << "\nA: " << std::flush;
            llm->response(user_str);
            std::cout << std::endl;
        }
    }

    void list(bool all) {
        if (all) {
            std::cout << std::left
                << std::setw(20) << "NAME"
                << std::setw(50) << "URL" << std::endl;
            std::cout << std::setfill('-') << std::setw(70) << "" << std::setfill(' ') << std::endl;
            auto manifests_json = nlohmann::json::parse(std::ifstream(manifests_file));
            for (const auto& [key, value] : manifests_json.items()) {
                std::cout << std::left
                    << std::setw(20) << key
                    << std::setw(36) << end_point + "/models/MNN/" + value.get<std::string>() << std::endl;
            }
            return;
        }
        const int widths[] = {20, 12, 12, 12};
        const char* headers[] = {"NAME", "ID", "SIZE", "MODIFIED"};
        std::cout << std::left
            << std::setw(widths[0]) << headers[0]
            << std::setw(widths[1]) << headers[1]
            << std::setw(widths[2]) << headers[2]
            << std::setw(widths[3]) << headers[3]
            << std::endl;
        std::cout << std::setfill('-') << std::setw(std::accumulate(widths, widths + 4, 0)) << "" << std::setfill(' ') << std::endl;
        for (const auto& entry : fs::directory_iterator(models_dir)) {
            if (fs::is_directory(entry.status())) {
                auto manifest = nlohmann::json::parse(std::ifstream(entry.path() / "manifest"));
                std::cout << std::left
                    << std::setw(widths[0]) << manifest["model"].get<std::string>()
                    << std::setw(widths[1]) << manifest["id"].get<std::string>()
                    << std::setw(widths[2]) << human_byte(manifest["size"])
                    << std::setw(widths[3]) << time_ago(entry.path())
                    << std::endl;
            }
        }
    }

    void handle_generate(const httplib::Request &req, httplib::Response &res) {
        nlohmann::json request_json;
        try {
            request_json = nlohmann::json::parse(req.body);
        } catch (nlohmann::json::parse_error &e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
            return;
        }

        std::string model = request_json.value("model", "");
        std::string prompt = request_json.value("prompt", "");
        bool stream = request_json.value("stream", false);
        logger->info("[/api/generate] request: {}", request_json.dump());
        std::unique_ptr<Llm> llm(Llm::createLLM(models_dir / model / "config.json"));
        llm->load();
        if (stream && false) {
            // TODO: stream response
            res.set_chunked_content_provider(
                "application/json",
            [&](size_t offset, httplib::DataSink& sink) {
                // sink.write(str, len);
                return false;
            });
        } else {
            auto response = llm->response(prompt);
            nlohmann::json response_json;
            response_json["model"] = model;
            response_json["created_at"] = current_timestamp();
            response_json["response"] = response;
            response_json["done"] = true;
            response_json["done_reason"] = "stop";
            res.set_content(response_json.dump(), "application/json");
            logger->info("[/api/generate] response: {}", response_json.dump());
        }
    }

    void handle_embed(const httplib::Request &req, httplib::Response &res) {
        nlohmann::json request_json;
        try {
            request_json = nlohmann::json::parse(req.body);
        } catch (nlohmann::json::parse_error &e) {
            res.status = 400;
            res.set_content("Invalid JSON", "text/plain");
            return;
        }

        std::string model = request_json.value("model", "");
        std::string input = request_json.value("input", "");
        logger->info("[/api/embed] request: {}", request_json.dump());
        std::unique_ptr<Embedding> embed(Embedding::createEmbedding(models_dir / model / "config.json"));
        embed->load();
        auto embedding_var = embed->txt_embedding(input);
        std::vector<float> embeddings_vec(embedding_var->getInfo()->size);
        for (int i = 0; i < embedding_var->getInfo()->size; i++) {
            embeddings_vec[i] = embedding_var->readMap<float>()[i];
        }
        nlohmann::json response_json;
        response_json["model"] = model;
        response_json["created_at"] = current_timestamp();
        response_json["response"] = embeddings_vec;
        response_json["done"] = true;
        response_json["done_reason"] = "stop";
        res.set_content(response_json.dump(), "application/json");
        logger->info("[/api/embed] response: {}", response_json.dump());
    }

    void serve() {
        create_daemon(logger.get());
        httplib::Server server;
        server.Get("/", [this](const httplib::Request &req, httplib::Response &res) {
            res.set_content("mllm is running\n", "text/plain");
        });
        server.Post("/api/generate", [this](const httplib::Request &req, httplib::Response &res) {
            this->handle_generate(req, res);
        });
        server.Post("/api/embed", [this](const httplib::Request &req, httplib::Response &res) {
            this->handle_embed(req, res);
        });
        std::ofstream pidfile(pid_file, std::ios::trunc);
        if (pidfile.is_open()) {
            pidfile << getpid();
            pidfile.close();
        }
        logger->info("{} server is running at http://localhost:{}", name, port);
        server.listen("127.0.0.1", port);
    }

    void stop() {
        // get pid from pid_file and stop it
        std::ifstream pidfile(pid_file);
        if (pidfile.is_open()) {
            int pid;
            pidfile >> pid;
            pidfile.close();
            if (pid > 0) {
                if (kill(pid, SIGTERM) == 0) {
                    std::cout << pid << " mllm server is stopped." << std::endl;
                    std::remove(pid_file.c_str());
                }
            }
        }
    }
};

void start_server() {
    create_daemon(nullptr);
    auto runner = Runner();
    runner.serve();
}

int main(int argc, char **argv) {
    auto runner = Runner();
    CLI::App app{MLLM_LOGO};
    bool version = false;
    app.add_flag("-v,--version", version, "Show version information");
    app.require_subcommand(0, 1);
    std::string arg;
    {
        auto registry = app.add_subcommand("registry", "Set remote registry");
        registry->add_option("type", arg, "The type of the registry, `HF`(huggingface) or `MS`(modelscope)")->required();
        registry->callback([&arg, &runner]() { runner.registry(arg); });
    }
    {
        auto run = app.add_subcommand("pull", "Pull a model");
        run->add_option("model", arg, "The name of the model to pull")->required();
        run->callback([&arg, &runner]() { runner.pull(arg); });
    }
    {
        auto run = app.add_subcommand("run", "Run a model");
        run->add_option("model", arg, "The name of the model to run")->required();
        run->callback([&arg, &runner]() { runner.run(arg); });
    }
    {
        bool flag = false;
        auto list = app.add_subcommand("list", "List models");
        list->add_flag("-a,--all", flag, "List all models in remote registry");
        list->callback([&flag, &runner]() { runner.list(flag); });
    }
    {
        auto serve = app.add_subcommand("serve", "Start mllm server");
        serve->callback([&arg, &runner]() { runner.serve(); });
        // serve->callback([&arg]() { start_server(); });
    }
    {
        auto serve = app.add_subcommand("stop", "Stop mllm server");
        serve->callback([&runner]() { runner.stop(); });
    }
    CLI11_PARSE(app, argc, argv);
    if (version) {
        std::cout << "mllm version is " << MLLM_VERSION << std::endl;
    }
    if (argc == 1) {
        std::cout << app.help() << std::endl;
    }
    return 0;
}