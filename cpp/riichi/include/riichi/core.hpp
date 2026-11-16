#pragma once
#include <exception>
#include "riichi/rng.hpp"
#include "riichi/tiles.hpp"
#include "riichi/rules.hpp"
#include "riichi/agari.hpp"
#include "riichi/shanten.hpp"
#include "riichi/yaku.hpp"
#include "riichi/scoring.hpp"

namespace riichi::abi {
// Shared by the core and future game ABI translation units. Error reporting must
// remain allocation-free, including when an allocation inside the engine fails.
void set_error(const char* message) noexcept;

template<class F> int32_t guarded(F&& function) noexcept {
    set_error("");
    try {
        return function();
    } catch (const NoYakuError& error) {
        set_error(error.what());
        return -2;
    } catch (const std::exception& error) {
        set_error(error.what());
        return -1;
    } catch (...) {
        set_error("unknown native error");
        return -1;
    }
}
}
