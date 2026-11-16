#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace riichi::json {
inline std::string value(bool x) { return x ? "true" : "false"; }
inline std::string value(int x) { return std::to_string(x); }
// All strings emitted here are fixed engine identifiers or field names.
inline std::string value(std::string_view x) { return "\"" + std::string(x) + "\""; }
inline std::string value(const std::string& x) { return value(std::string_view(x)); }
template<class T> std::string value(const std::optional<T>& x) { return x ? value(*x) : "null"; }
template<class Range> std::string array(const Range& xs) {
    std::string out = "[";
    for (const auto& x : xs) { if (out.size() > 1) out += ','; out += value(x); }
    return out + ']';
}
struct Object {
    std::string text = "{";
    void raw(std::string_view key, const std::string& v) {
        if (text.size() > 1) text += ',';
        text += value(key) + ':' + v;
    }
    template<class T> void add(std::string_view key, const T& v) { raw(key, value(v)); }
    template<class T> void list(std::string_view key, const T& v) { raw(key, array(v)); }
    std::string str() const { return text + '}'; }
};
inline std::string objects(const std::vector<std::string>& xs) {
    std::string out = "[";
    for (const auto& x : xs) { if (out.size() > 1) out += ','; out += x; }
    return out + ']';
}
}
