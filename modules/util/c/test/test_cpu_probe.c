// SPDX-FileCopyrightText: 2026 Alin-Petru Roșu <rosualinpetru@gmail.com>
// SPDX-License-Identifier: Apache-2.0
// The probe's contract, which tools/test/engine-run.sh and the engine picker
// both decide from: "absent" and "cannot judge" must stay distinguishable, or
// a mistyped capability looks like a CPU that merely lacks the feature.

#include <unity.h>
#include <vfhe_cpu.h>

void setUp(void) {}
void tearDown(void) {}

static void no_requirement_is_always_satisfied(void)
{
    TEST_ASSERT_TRUE(vfhe_cpu_supports(""));
    TEST_ASSERT_TRUE(vfhe_cpu_knows(""));
    TEST_ASSERT_TRUE(vfhe_cpu_supports(NULL));
}

static void an_unteachable_name_is_unjudgeable_not_absent(void)
{
    // Both false, and `knows` is the one that separates a typo from a CPU
    // without the feature: the runner exits 1 on this, never 77.
    TEST_ASSERT_FALSE(vfhe_cpu_knows("nonsense"));
    TEST_ASSERT_FALSE(vfhe_cpu_supports("nonsense"));
}

static void this_architectures_names_are_judgeable(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    TEST_ASSERT_TRUE(vfhe_cpu_knows("avx512ifma"));
    TEST_ASSERT_TRUE(vfhe_cpu_knows("avx512f"));
    TEST_ASSERT_TRUE(vfhe_cpu_knows("avx2"));
#elif defined(__aarch64__) || defined(_M_ARM64)
    TEST_ASSERT_TRUE(vfhe_cpu_knows("neon"));
    TEST_ASSERT_TRUE(vfhe_cpu_supports("neon")); // baseline on arm64
    // An x86 name is not judgeable here, which is why the engine facts are
    // architecture-gated rather than probed on every host.
    TEST_ASSERT_FALSE(vfhe_cpu_knows("avx512ifma"));
#endif
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(no_requirement_is_always_satisfied);
    RUN_TEST(an_unteachable_name_is_unjudgeable_not_absent);
    RUN_TEST(this_architectures_names_are_judgeable);
    return UNITY_END();
}
