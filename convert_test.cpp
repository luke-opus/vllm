#include "float_convert.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>

static uint16_t reference(uint32_t bits) {
  const uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) return 0x7fc0U;
  const uint16_t sign = static_cast<uint16_t>(bits >> 16) & 0x8000U;
  if (magnitude == 0x7f800000U) return sign | 0x7f80U;
  const uint16_t lower = static_cast<uint16_t>(magnitude >> 16);
  const long double value = std::fabs(static_cast<long double>(
      std::bit_cast<float>(bits)));
  const long double low_value = bf16_to_float(lower);
  const long double high_value = lower == 0x7f7fU
                                     ? std::ldexp(1.0L, 128)
                                     : bf16_to_float(lower + 1);
  const auto below = value - low_value;
  const auto above = high_value - value;
  const bool round_up = below > above || (below == above && (lower & 1U));
  return sign | static_cast<uint16_t>(lower + round_up);
}

int main() {
  uint64_t total = 0, failed = 0;
  auto check = [&](uint32_t bits) {
    const auto actual = float_to_bf16(std::bit_cast<float>(bits));
    const auto expected = reference(bits);
    ++total;
    if (actual != expected) {
      if (failed < 12)
        std::printf("bits=%08x actual=%04x expected=%04x\n", bits, actual,
                    expected);
      ++failed;
    }
  };
  constexpr std::array<uint32_t, 6> low_words = {
      0, 1, 0x7fff, 0x8000, 0x8001, 0xffff};
  for (uint32_t upper = 0; upper <= 0xffffU; ++upper)
    for (const auto low : low_words) check((upper << 16) | low);
  uint32_t state = 0x12345678U;
  for (int i = 0; i < 1000000; ++i) {
    state = state * 1664525U + 1013904223U;
    check(state);
  }
  for (uint32_t value = 0; value <= 0xffffU; ++value) {
    const auto actual = float_to_bf16(bf16_to_float(value));
    const auto expected = reference(value << 16);
    ++total;
    if (actual != expected) ++failed;
  }
  const float silu = 0.125f / (1.0f + std::exp(-0.125f));
  const auto silu_bf16 = float_to_bf16(silu);
  ++total;
  if (silu_bf16 != 0x3d88U) ++failed;
  std::printf("SiLU(0.125): fp32=%.10g bf16=%04x expected=3d88\n", silu,
              silu_bf16);
  std::printf("Checked %llu conversions; mismatches=%llu\n",
              static_cast<unsigned long long>(total),
              static_cast<unsigned long long>(failed));
  return failed != 0;
}
