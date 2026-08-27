/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include "src/CuptiPMSamplingApi.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono_literals;

namespace {

// CuptiPMSamplingApi calls CUDA and CUPTI directly. This test overrides those C
// symbols and uses shared state to script results and record calls without
// touching a GPU.
struct FakeSample {
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  std::vector<double> values;
};

struct FakeCuptiState {
  int major{9};
  int minor{0};
  cudaError_t devicePropertiesResult{cudaSuccess};
  bool hasChipName{true};
  std::string chipName{"mock-chip"};
  size_t counterAvailabilityImageSize{3};
  size_t configImageSize{4};
  size_t numPasses{1};
  size_t counterDataSize{8};
  CUpti_PmSampling_DecodeStopReason decodeStopReason{
      CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS};
  std::vector<FakeSample> decodedSamples;

  std::vector<std::string> calls;
  std::unordered_map<std::string, CUptiResult> results;

  std::vector<std::string> configMetricNames;
  size_t enabledDeviceIndex{0};
  bool setConfigCalled{false};
  CUpti_PmSampling_SetConfig_Params setConfig{};
  std::vector<std::string> counterDataMetricNames;
  uint32_t maxSamples{0};

  std::byte hostObject{};
  std::byte samplingObject{};
};

FakeCuptiState& fakeCupti() {
  static FakeCuptiState state;
  return state;
}

CUpti_Profiler_Host_Object* fakeHostObject() {
  return reinterpret_cast<CUpti_Profiler_Host_Object*>(&fakeCupti().hostObject);
}

CUpti_PmSampling_Object* fakeSamplingObject() {
  return reinterpret_cast<CUpti_PmSampling_Object*>(
      &fakeCupti().samplingObject);
}

void recordCall(std::string call) {
  fakeCupti().calls.push_back(std::move(call));
}

void clearCalls() {
  fakeCupti().calls.clear();
}

size_t callCount(std::string_view call) {
  return static_cast<size_t>(
      std::count(fakeCupti().calls.begin(), fakeCupti().calls.end(), call));
}

CUptiResult resultFor(std::string_view call) {
  const auto result = fakeCupti().results.find(std::string(call));
  return result == fakeCupti().results.end() ? CUPTI_SUCCESS : result->second;
}

void setResult(std::string call, CUptiResult result) {
  fakeCupti().results.insert_or_assign(std::move(call), result);
}

void expectCalls(std::initializer_list<std::string> expected) {
  EXPECT_EQ(fakeCupti().calls, std::vector<std::string>(expected));
}

std::vector<std::string> copyMetricNames(
    const char* const* metricNames,
    size_t count) {
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(metricNames[i]);
  }
  return result;
}

CuptiPMSamplingConfig makeConfig(
    std::chrono::nanoseconds samplingInterval = 500us,
    int32_t deviceId = 0,
    std::vector<std::string> metricNames = {"sm__cycles_active.avg"}) {
  return CuptiPMSamplingConfig{
      deviceId, std::move(metricNames), samplingInterval};
}

void configureForDevice(
    int major,
    int minor,
    std::chrono::nanoseconds samplingInterval) {
  fakeCupti().major = major;
  fakeCupti().minor = minor;
  CuptiPMSamplingApi api;
  api.configure(makeConfig(samplingInterval));
  api.disable();
}

class CuptiPMSamplingApiTest : public testing::Test {
 protected:
  void SetUp() override {
    fakeCupti() = FakeCuptiState{};
  }
};

} // namespace

// CUDA/CUPTI symbol overrides used by the tests below.
extern "C" {

CUptiResult CUPTIAPI
cuptiGetResultString(CUptiResult result, const char** resultString) {
  static_cast<void>(result);
  *resultString = "mock CUPTI error";
  return CUPTI_SUCCESS;
}

// Record every call, return a scripted error if present, then fill outputs.
#define DEFINE_CUPTI_FAKE(function, call, paramsType, ...) \
  CUptiResult CUPTIAPI function(paramsType* params) {      \
    static_cast<void>(params);                             \
    recordCall(call);                                      \
    const auto result = resultFor(call);                   \
    if (result != CUPTI_SUCCESS) {                         \
      return result;                                       \
    }                                                      \
    __VA_ARGS__                                            \
    return CUPTI_SUCCESS;                                  \
  }

DEFINE_CUPTI_FAKE(
    cuptiProfilerInitialize,
    "profilerInitialize",
    CUpti_Profiler_Initialize_Params)
DEFINE_CUPTI_FAKE(
    cuptiDeviceGetChipName,
    "deviceGetChipName",
    CUpti_Device_GetChipName_Params,
    {
      params->pChipName =
          fakeCupti().hasChipName ? fakeCupti().chipName.c_str() : nullptr;
    })

cudaError_t CUDARTAPI
cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  recordCall("cudaGetDeviceProperties");
  static_cast<void>(device);
  if (fakeCupti().devicePropertiesResult == cudaSuccess) {
    *properties = cudaDeviceProp{};
    properties->major = fakeCupti().major;
    properties->minor = fakeCupti().minor;
  }
  return fakeCupti().devicePropertiesResult;
}

const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  static_cast<void>(error);
  return "mock CUDA error";
}

DEFINE_CUPTI_FAKE(
    cuptiPmSamplingGetCounterAvailability,
    "getCounterAvailability",
    CUpti_PmSampling_GetCounterAvailability_Params,
    {
      if (params->pCounterAvailabilityImage == nullptr) {
        params->counterAvailabilityImageSize =
            fakeCupti().counterAvailabilityImageSize;
      } else {
        std::fill_n(
            params->pCounterAvailabilityImage,
            params->counterAvailabilityImageSize,
            uint8_t{0});
      }
    })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostInitialize,
    "hostInitialize",
    CUpti_Profiler_Host_Initialize_Params,
    { params->pHostObject = fakeHostObject(); })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostConfigAddMetrics,
    "hostConfigAddMetrics",
    CUpti_Profiler_Host_ConfigAddMetrics_Params,
    {
      fakeCupti().configMetricNames =
          copyMetricNames(params->ppMetricNames, params->numMetrics);
    })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostGetConfigImageSize,
    "hostGetConfigImageSize",
    CUpti_Profiler_Host_GetConfigImageSize_Params,
    { params->configImageSize = fakeCupti().configImageSize; })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostGetConfigImage,
    "hostGetConfigImage",
    CUpti_Profiler_Host_GetConfigImage_Params,
    { std::fill_n(params->pConfigImage, params->configImageSize, uint8_t{0}); })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostGetNumOfPasses,
    "hostGetNumOfPasses",
    CUpti_Profiler_Host_GetNumOfPasses_Params,
    { params->numOfPasses = fakeCupti().numPasses; })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingEnable,
    "samplingEnable",
    CUpti_PmSampling_Enable_Params,
    {
      fakeCupti().enabledDeviceIndex = params->deviceIndex;
      params->pPmSamplingObject = fakeSamplingObject();
    })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingSetConfig,
    "samplingSetConfig",
    CUpti_PmSampling_SetConfig_Params,
    {
      fakeCupti().setConfig = *params;
      fakeCupti().setConfigCalled = true;
    })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingGetCounterDataSize,
    "getCounterDataSize",
    CUpti_PmSampling_GetCounterDataSize_Params,
    {
      fakeCupti().counterDataMetricNames =
          copyMetricNames(params->pMetricNames, params->numMetrics);
      fakeCupti().maxSamples = params->maxSamples;
      params->counterDataSize = fakeCupti().counterDataSize;
    })

DEFINE_CUPTI_FAKE(
    cuptiPmSamplingCounterDataImageInitialize,
    "counterDataImageInitialize",
    CUpti_PmSampling_CounterDataImage_Initialize_Params)
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingStart,
    "samplingStart",
    CUpti_PmSampling_Start_Params)
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingDecodeData,
    "samplingDecodeData",
    CUpti_PmSampling_DecodeData_Params,
    {
      params->decodeStopReason = fakeCupti().decodeStopReason;
      params->overflow = 0;
    })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingGetCounterDataInfo,
    "getCounterDataInfo",
    CUpti_PmSampling_GetCounterDataInfo_Params,
    {
      params->numTotalSamples = fakeCupti().decodedSamples.size();
      params->numPopulatedSamples = fakeCupti().decodedSamples.size();
      params->numCompletedSamples = fakeCupti().decodedSamples.size();
    })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingCounterDataGetSampleInfo,
    "getSampleInfo",
    CUpti_PmSampling_CounterData_GetSampleInfo_Params,
    {
      if (params->sampleIndex >= fakeCupti().decodedSamples.size()) {
        return CUPTI_ERROR_INVALID_PARAMETER;
      }
      const auto& sample = fakeCupti().decodedSamples[params->sampleIndex];
      params->startTimestamp = sample.startTimestamp;
      params->endTimestamp = sample.endTimestamp;
    })
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostEvaluateToGpuValues,
    "hostEvaluateToGpuValues",
    CUpti_Profiler_Host_EvaluateToGpuValues_Params,
    {
      if (params->rangeIndex >= fakeCupti().decodedSamples.size()) {
        return CUPTI_ERROR_INVALID_PARAMETER;
      }
      const auto& values =
          fakeCupti().decodedSamples[params->rangeIndex].values;
      if (values.size() != params->numMetrics) {
        return CUPTI_ERROR_INVALID_PARAMETER;
      }
      std::copy(values.begin(), values.end(), params->pMetricValues);
    })
DEFINE_CUPTI_FAKE(
    cuptiPmSamplingStop,
    "samplingStop",
    CUpti_PmSampling_Stop_Params)

DEFINE_CUPTI_FAKE(
    cuptiPmSamplingDisable,
    "samplingDisable",
    CUpti_PmSampling_Disable_Params)
DEFINE_CUPTI_FAKE(
    cuptiProfilerHostDeinitialize,
    "hostDeinitialize",
    CUpti_Profiler_Host_Deinitialize_Params)
DEFINE_CUPTI_FAKE(
    cuptiProfilerDeInitialize,
    "profilerDeInitialize",
    CUpti_Profiler_DeInitialize_Params)

#undef DEFINE_CUPTI_FAKE

} // extern "C"

TEST_F(CuptiPMSamplingApiTest, RequiresConfigurationForSamplingOperations) {
  CuptiPMSamplingApi api;
  std::vector<CuptiPMSample> samples;

  EXPECT_THROW(api.start(), std::runtime_error);
  EXPECT_THROW(api.decode(samples), std::runtime_error);
  EXPECT_THROW(api.stop(), std::runtime_error);
  EXPECT_NO_THROW(api.disable());
  EXPECT_TRUE(samples.empty());
  EXPECT_TRUE(fakeCupti().calls.empty());
}

TEST_F(CuptiPMSamplingApiTest, ConfiguresDeviceMetricsAndBuffers) {
  const auto config = makeConfig(
      250us,
      /*deviceId=*/2,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"});
  CuptiPMSamplingApi api;

  api.configure(config);

  expectCalls({
      "profilerInitialize",
      "deviceGetChipName",
      "cudaGetDeviceProperties",
      "getCounterAvailability",
      "getCounterAvailability",
      "hostInitialize",
      "hostConfigAddMetrics",
      "hostGetConfigImageSize",
      "hostGetConfigImage",
      "hostGetNumOfPasses",
      "samplingEnable",
      "samplingSetConfig",
      "getCounterDataSize",
      "counterDataImageInitialize",
  });
  EXPECT_EQ(fakeCupti().configMetricNames, config.metricNames);
  EXPECT_EQ(fakeCupti().enabledDeviceIndex, 2);

  ASSERT_TRUE(fakeCupti().setConfigCalled);
  EXPECT_EQ(
      fakeCupti().setConfig.structSize,
      CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE);
  EXPECT_EQ(fakeCupti().setConfig.pPriv, nullptr);
  EXPECT_EQ(fakeCupti().setConfig.pPmSamplingObject, fakeSamplingObject());
  EXPECT_EQ(fakeCupti().setConfig.configSize, fakeCupti().configImageSize);
  EXPECT_NE(fakeCupti().setConfig.pConfig, nullptr);
  EXPECT_EQ(fakeCupti().setConfig.hardwareBufferSize, 64 * 1024 * 1024);
  EXPECT_EQ(fakeCupti().setConfig.samplingInterval, 250'000);
  EXPECT_EQ(
      fakeCupti().setConfig.triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
  EXPECT_EQ(
      fakeCupti().setConfig.hwBufferAppendMode,
      CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST);

  EXPECT_EQ(fakeCupti().counterDataMetricNames, config.metricNames);
  EXPECT_EQ(fakeCupti().maxSamples, 1024);
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, UsesFixedSysclkIntervalOnGa100) {
  configureForDevice(8, 0, 0ns);

  ASSERT_TRUE(fakeCupti().setConfigCalled);
  EXPECT_EQ(
      fakeCupti().setConfig.triggerMode,
      CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL);
  EXPECT_EQ(fakeCupti().setConfig.samplingInterval, 1'000'000);
}

TEST_F(CuptiPMSamplingApiTest, UsesRequestedTimeIntervalOnGa10xAndNewer) {
  for (const auto& [major, minor] :
       {std::pair{8, 6}, std::pair{8, 9}, std::pair{9, 0}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    fakeCupti().setConfigCalled = false;
    configureForDevice(major, minor, 500us);

    ASSERT_TRUE(fakeCupti().setConfigCalled);
    EXPECT_EQ(
        fakeCupti().setConfig.triggerMode,
        CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL);
    EXPECT_EQ(fakeCupti().setConfig.samplingInterval, 500'000);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsUnsupportedComputeCapabilities) {
  for (const auto& [major, minor] : {std::pair{7, 5}, std::pair{8, 5}}) {
    SCOPED_TRACE(testing::Message() << major << "." << minor);
    EXPECT_THROW(configureForDevice(major, minor, 500us), std::runtime_error);
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsNonpositiveTimeIntervals) {
  EXPECT_THROW(configureForDevice(8, 6, 0ns), std::runtime_error);
  EXPECT_THROW(configureForDevice(9, 0, -1ns), std::runtime_error);
}

TEST_F(CuptiPMSamplingApiTest, ReportsDeviceDiscoveryFailures) {
  fakeCupti().hasChipName = false;
  {
    CuptiPMSamplingApi api;
    EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);
    EXPECT_EQ(callCount("cudaGetDeviceProperties"), 0);
    api.disable();
  }

  fakeCupti().hasChipName = true;
  fakeCupti().devicePropertiesResult = cudaErrorInvalidDevice;
  clearCalls();
  {
    CuptiPMSamplingApi api;
    EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);
    EXPECT_EQ(callCount("hostInitialize"), 0);
    api.disable();
  }
}

TEST_F(CuptiPMSamplingApiTest, RejectsMultipassConfigurationBeforeEnabling) {
  fakeCupti().numPasses = 2;
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);
  EXPECT_EQ(callCount("samplingEnable"), 0);

  clearCalls();
  api.disable();
  expectCalls({"hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, CleansUpAfterCuptiConfigurationFailure) {
  setResult("samplingSetConfig", CUPTI_ERROR_UNKNOWN);
  CuptiPMSamplingApi api;

  EXPECT_THROW(api.configure(makeConfig()), std::runtime_error);
  EXPECT_EQ(callCount("getCounterDataSize"), 0);

  clearCalls();
  api.disable();
  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}

TEST_F(CuptiPMSamplingApiTest, RejectsReconfigurationUntilDisabled) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  const size_t initializeCalls = callCount("profilerInitialize");

  EXPECT_THROW(
      api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"})),
      std::runtime_error);
  EXPECT_EQ(callCount("profilerInitialize"), initializeCalls);

  api.disable();
  api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"}));
  EXPECT_EQ(callCount("profilerInitialize"), initializeCalls + 1);
  EXPECT_EQ(
      fakeCupti().configMetricNames,
      (std::vector<std::string>{"dram__bytes_read.sum"}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, StartsAndStopsConfiguredSampler) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.start();
  api.stop();

  expectCalls({"samplingStart", "samplingStop"});
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DecodesCompletedSamplesAndAppendsThem) {
  const auto config = makeConfig(
      500us,
      /*deviceId=*/0,
      {"sm__cycles_active.avg", "dram__bytes_read.sum"});
  fakeCupti().decodedSamples = {
      FakeSample{100, 120, {1.25, 2048.0}},
      FakeSample{130, 170, {2.5, 4096.0}},
  };
  CuptiPMSamplingApi api;
  api.configure(config);
  std::vector<CuptiPMSample> samples{
      CuptiPMSample{1, 2, std::vector<double>{3.0}}};
  clearCalls();

  EXPECT_TRUE(api.decode(samples));

  expectCalls({
      "samplingDecodeData",
      "getCounterDataInfo",
      "getSampleInfo",
      "hostEvaluateToGpuValues",
      "getSampleInfo",
      "hostEvaluateToGpuValues",
      "counterDataImageInitialize",
  });
  ASSERT_EQ(samples.size(), 3);
  EXPECT_EQ(samples[0].rawStartTimestamp, 1);
  EXPECT_EQ(samples[0].rawEndTimestamp, 2);
  EXPECT_EQ(samples[0].values, (std::vector<double>{3.0}));
  EXPECT_EQ(samples[1].rawStartTimestamp, 100);
  EXPECT_EQ(samples[1].rawEndTimestamp, 120);
  EXPECT_EQ(samples[1].values, (std::vector<double>{1.25, 2048.0}));
  EXPECT_EQ(samples[2].rawStartTimestamp, 130);
  EXPECT_EQ(samples[2].rawEndTimestamp, 170);
  EXPECT_EQ(samples[2].values, (std::vector<double>{2.5, 4096.0}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, ReportsUndrainedDecodeStopReasons) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  std::vector<CuptiPMSample> samples;

  for (const auto reason :
       {CUPTI_PM_SAMPLING_DECODE_STOP_REASON_OTHER,
        CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL}) {
    SCOPED_TRACE(static_cast<int>(reason));
    fakeCupti().decodeStopReason = reason;
    clearCalls();

    EXPECT_FALSE(api.decode(samples));
    EXPECT_EQ(callCount("counterDataImageInitialize"), 1);
  }
  EXPECT_TRUE(samples.empty());
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, RejectsUnexpectedDecodeStopReason) {
  fakeCupti().decodeStopReason = CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNT;
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  std::vector<CuptiPMSample> samples;
  clearCalls();

  EXPECT_THROW(api.decode(samples), std::runtime_error);

  expectCalls({"samplingDecodeData"});
  EXPECT_TRUE(samples.empty());
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, PropagatesSamplingOperationFailures) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  std::vector<CuptiPMSample> samples;
  clearCalls();

  setResult("samplingStart", CUPTI_ERROR_UNKNOWN);
  EXPECT_THROW(api.start(), std::runtime_error);
  setResult("samplingDecodeData", CUPTI_ERROR_UNKNOWN);
  EXPECT_THROW(api.decode(samples), std::runtime_error);
  setResult("samplingStop", CUPTI_ERROR_UNKNOWN);
  EXPECT_THROW(api.stop(), std::runtime_error);

  expectCalls({"samplingStart", "samplingDecodeData", "samplingStop"});
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DisableIsOrderedIdempotentAndAllowsReuse) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  api.disable();

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
  clearCalls();
  api.disable();
  EXPECT_TRUE(fakeCupti().calls.empty());
  EXPECT_THROW(api.start(), std::runtime_error);

  api.configure(makeConfig(750us, 1, {"dram__bytes_read.sum"}));
  EXPECT_EQ(
      fakeCupti().configMetricNames,
      (std::vector<std::string>{"dram__bytes_read.sum"}));
  api.disable();
}

TEST_F(CuptiPMSamplingApiTest, DisableRetriesPartialTeardownFailures) {
  CuptiPMSamplingApi api;
  api.configure(makeConfig());
  clearCalls();

  setResult("samplingDisable", CUPTI_ERROR_UNKNOWN);
  api.disable();
  expectCalls({"samplingDisable"});

  setResult("samplingDisable", CUPTI_SUCCESS);
  setResult("hostDeinitialize", CUPTI_ERROR_UNKNOWN);
  clearCalls();
  api.disable();
  expectCalls({"samplingDisable", "hostDeinitialize"});

  setResult("hostDeinitialize", CUPTI_SUCCESS);
  setResult("profilerDeInitialize", CUPTI_ERROR_UNKNOWN);
  clearCalls();
  api.disable();
  expectCalls({"hostDeinitialize", "profilerDeInitialize"});

  setResult("profilerDeInitialize", CUPTI_SUCCESS);
  clearCalls();
  api.disable();
  expectCalls({"profilerDeInitialize"});

  clearCalls();
  api.disable();
  EXPECT_TRUE(fakeCupti().calls.empty());
}

TEST_F(CuptiPMSamplingApiTest, DestructorDisablesConfiguredSampler) {
  {
    CuptiPMSamplingApi api;
    api.configure(makeConfig());
    clearCalls();
  }

  expectCalls({"samplingDisable", "hostDeinitialize", "profilerDeInitialize"});
}
