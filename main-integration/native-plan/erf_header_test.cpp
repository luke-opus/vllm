#include "cpu_types_riscv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

static_assert(__riscv_v_min_vlen == 256);

struct Counts {
  unsigned checks = 0;
  unsigned numerical = 0;
  unsigned signs = 0;
  unsigned nans = 0;
};

template <typename V>
void check(const float* input, size_t count, Counts& counts) {
  std::array<float, V::VEC_ELEM_NUM> output;
  for (size_t offset = 0; offset < count; offset += V::VEC_ELEM_NUM) {
    const V values(input + offset);
    values.er().save(output.data());
    for (size_t lane = 0; lane < output.size(); ++lane) {
      const float x = input[offset + lane];
      const float actual = output[lane];
      const float expected = std::erf(x);
      ++counts.checks;
      if (std::isnan(expected)) {
        if (!std::isnan(actual)) ++counts.numerical;
      } else if (!std::isfinite(actual) ||
                 std::abs(actual - expected) >
                     1e-6f + 1e-5f * std::abs(expected)) {
        ++counts.numerical;
        if (std::isnan(actual)) ++counts.nans;
      }
      if (x == 0.0f && std::signbit(actual) != std::signbit(x)) {
        ++counts.signs;
      }
    }
  }
}

int main() {
  unsigned long vlenb;
  asm volatile("csrr %0, vlenb" : "=r"(vlenb));
  std::printf("vlen_bits=%lu\n", vlenb * 8);
  if (vlenb != 32) return 2;

  // Execute half-vector arithmetic; metadata alone does not verify Zvfh.
  std::array<_Float16, 8> half;
  half.fill(static_cast<_Float16>(1.0f));
  const auto one = __riscv_vle16_v_f16m1(half.data(), half.size());
  const auto two = __riscv_vfadd_vv_f16m1(one, one, half.size());
  __riscv_vse16_v_f16m1(half.data(), two, half.size());
  for (const auto value : half) {
    if (std::bit_cast<uint16_t>(value) != 0x4000) return 2;
  }
  std::puts("zvfh_vector_arithmetic=passed");

  std::array<float, 256> values;
  constexpr std::array<float, 26> edge = {
      0.0f, -0.0f, 0.0001f, -0.0001f, 0.125f, -0.125f, 0.25f,
      -0.25f, 0.5f, -0.5f, 0.75f, -0.75f, 1.0f, -1.0f, 1.5f,
      -1.5f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 6.0f,
      -6.0f, 9.0f, -9.0f};
  std::copy(edge.begin(), edge.end(), values.begin());
  for (size_t i = edge.size(); i < values.size(); ++i) {
    const int index = static_cast<int>(i - edge.size());
    values[i] = static_cast<float>((index * 73) % 257 - 128) / 16.0f;
  }
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 16> special = {
      0.0f, -0.0f, inf, -inf, nan, -nan, 1.0f, -1.0f,
      0.0f, -0.0f, inf, -inf, nan, -nan, 1.0f, -1.0f};
  Counts counts;
  check<vec_op::FP32Vec8>(values.data(), values.size(), counts);
  check<vec_op::FP32Vec16>(values.data(), values.size(), counts);
  check<vec_op::FP32Vec8>(special.data(), special.size(), counts);
  check<vec_op::FP32Vec16>(special.data(), special.size(), counts);
  std::printf("checks=%u numerical_failures=%u nan_failures=%u sign_failures=%u\n",
              counts.checks, counts.numerical, counts.nans, counts.signs);
  return counts.numerical || counts.signs ? 1 : 0;
}
