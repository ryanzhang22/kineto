/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cupti_pmsampling.h>

#include "src/CuptiPMSamplingApi.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono_literals;

TEST(CuptiPMSamplingApiTest, RequiresConfigurationForSamplingOperations) {
  CuptiPMSamplingApi api;
  std::vector<CuptiPMSample> samples;

  EXPECT_THROW(api.start(), std::runtime_error);
  EXPECT_THROW(api.decode(samples), std::runtime_error);
  EXPECT_THROW(api.stop(), std::runtime_error);
  EXPECT_TRUE(samples.empty());
}

TEST(CuptiPMSamplingApiTest, UsesFixedSysclkIntervalOnGa100) {
  const CUpti_PmSampling_SetConfig_Params params =
      detail::makeCuptiPMSamplingSetConfigParams(8, 0, 0ns);

  EXPECT_EQ(
      params.triggerMode, CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL);
  EXPECT_EQ(params.samplingInterval, 1'000'000);
}

TEST(CuptiPMSamplingApiTest, UsesRequestedTimeIntervalOnGa10xAndNewer) {
  for (const auto& [major, minor] :
       {std::pair{8, 6}, std::pair{8, 9}, std::pair{9, 0}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    const CUpti_PmSampling_SetConfig_Params params =
        detail::makeCuptiPMSamplingSetConfigParams(major, minor, 500us);

    EXPECT_EQ(
        params.triggerMode, CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
    EXPECT_EQ(params.samplingInterval, 500'000);
  }
}

TEST(CuptiPMSamplingApiTest, RejectsUnsupportedComputeCapabilities) {
  for (const auto& [major, minor] : {std::pair{7, 5}, std::pair{8, 5}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    EXPECT_THROW(
        static_cast<void>(
            detail::makeCuptiPMSamplingSetConfigParams(major, minor, 500us)),
        std::runtime_error);
  }
}

TEST(CuptiPMSamplingApiTest, RejectsNonpositiveTimeIntervals) {
  EXPECT_THROW(
      static_cast<void>(detail::makeCuptiPMSamplingSetConfigParams(8, 6, 0ns)),
      std::runtime_error);
  EXPECT_THROW(
      static_cast<void>(detail::makeCuptiPMSamplingSetConfigParams(9, 0, -1ns)),
      std::runtime_error);
}
