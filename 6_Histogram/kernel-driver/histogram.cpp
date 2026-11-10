#include "histogram.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pFacesExamples {

namespace {

bool isPowerOfTwo(size_t v) {
  return v > 0U && ((v & (v - 1U)) == 0U);
}

size_t parsePositiveSize(
  const std::shared_ptr<pfacesConfigurationReader>& spCfg,
  const char* key,
  size_t defaultVal) {
  std::string raw = spCfg->readConfigValueString(key);
  if (raw.empty()) {
    return defaultVal;
  }
  const long long parsed = std::stoll(raw);
  if (parsed <= 0) {
    throw std::runtime_error(std::string("Configuration key '") + key + "' must be positive.");
  }
  return static_cast<size_t>(parsed);
}

uint32_t generateSampleByte(size_t idx) {
  uint32_t x = static_cast<uint32_t>(idx) * 747796405U + 2891336453U;
  x = ((x >> ((x >> 28U) + 4U)) ^ x) * 277803737U;
  x = (x >> 22U) ^ x;
  return x & 0xFFU;
}

size_t verifyHistogramOutput(
  const pfaces2DKernel& /*kernel*/,
  const pfacesParallelProgram& program,
  std::vector<std::shared_ptr<void>>& params) {
  if (params.empty()) {
    throw std::runtime_error("[FAIL] 6_Histogram: missing verification state.");
  }

  const auto state = std::static_pointer_cast<HistogramState>(params[0]);
  const auto* inData = reinterpret_cast<const uint32_t*>(program.m_dataPool[state->inDataBufIdx].first);
  const auto* devBins = reinterpret_cast<const uint32_t*>(program.m_dataPool[state->binsBufIdx].first);
  if (inData == nullptr || devBins == nullptr) {
    throw std::runtime_error("[FAIL] 6_Histogram: data pool buffers are missing.");
  }

  std::array<uint32_t, HistogramState::kNumBins> cpuBins{};
  for (size_t i = 0; i < state->numElements; i++) {
    cpuBins[inData[i] & 0xFFU]++;
  }

  size_t mismatchedBins = 0U;
  size_t firstBadBin = 0U;
  uint64_t totalDevCount = 0U;
  for (size_t b = 0; b < HistogramState::kNumBins; b++) {
    totalDevCount += devBins[b];
    if (devBins[b] != cpuBins[b]) {
      if (mismatchedBins == 0U) {
        firstBadBin = b;
      }
      mismatchedBins++;
    }
  }

  std::ostringstream oss;
  if (mismatchedBins != 0U || totalDevCount != state->numElements) {
    oss << "[FAIL] 6_Histogram: N=" << state->numElements
        << ", mismatched_bins=" << mismatchedBins
        << ", total_dev_count=" << totalDevCount
        << ", first_bad_bin=" << firstBadBin
        << " (device=" << devBins[firstBadBin]
        << ", cpu=" << cpuBins[firstBadBin] << ")";
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  oss << "[PASS] 6_Histogram: N=" << state->numElements
      << ", bins=" << HistogramState::kNumBins
      << ", total_count=" << totalDevCount
      << ", bin0=" << devBins[0]
      << ", mismatched_bins=0";
  std::cout << oss.str() << std::endl;
  return 0U;
}

}  // namespace

HistogramKernel::HistogramKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("histogram"), 1U, 1U, spCfg),
    m_state(std::make_shared<HistogramState>()) {

  if (spCfg) {
    m_state->numElements = parsePositiveSize(spCfg, "num_elements", m_state->numElements);
    m_state->wgSize = parsePositiveSize(spCfg, "wg_size", m_state->wgSize);
    m_state->numGroups = parsePositiveSize(spCfg, "num_groups", m_state->numGroups);
  }

  if (!isPowerOfTwo(m_state->wgSize) || m_state->wgSize > HistogramState::kNumBins) {
    throw std::runtime_error("wg_size must be a power of 2 in [1, 256].");
  }

  const size_t totalThreads = m_state->numGroups * m_state->wgSize;
  setDimensions(totalThreads, 1U);

  const std::string memFile = spLaunchState->getKernelPackPath() + "histogram.mem";

  auto argsZero = pfacesKernelFunctionArguments::loadFromFile(memFile, "zeroBins", {"bins"});
  argsZero.m_baseTypeMultiple[0] = HistogramState::kNumBins;
  addKernelFunction(pfacesKernelFunction("zeroBins", argsZero));

  auto argsHist = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "histogramCompute", {"in_data", "bins", "local_bins"});
  argsHist.m_baseTypeMultiple[0] = m_state->numElements;
  argsHist.m_baseTypeMultiple[2] = HistogramState::kNumBins;
  addKernelFunction(pfacesKernelFunction("histogramCompute", argsHist));

  updateParameters({"@@HIST_N@@"}, {std::to_string(m_state->numElements)});

  std::vector<std::shared_ptr<void>> verifyParams{m_state};
  registerPostExecuteFunction(
    verifyHistogramOutput,
    std::string("Verify 256-bin histogram exact counts against CPU reference"),
    verifyParams);
}

void HistogramKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "6_Histogram expects a single target OpenCL device (-d 1).");

  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];

  const size_t maxWg = thisDevice.get().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (m_state->wgSize > maxWg) {
    throw std::runtime_error(
      "wg_size (" + std::to_string(m_state->wgSize) +
      ") exceeds device max work-group size (" + std::to_string(maxWg) + ").");
  }

  const cl::NDRange ndGlobalZero(HistogramState::kNumBins, 1U);
  const cl::NDRange ndGlobalHist(m_state->numGroups * m_state->wgSize, 1U);
  const cl::NDRange ndLocal(m_state->wgSize, 1U);
  const cl::NDRange ndOffset(0U, 0U);

  const auto memReport = allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, true);
  m_state->inDataBufIdx = getBufferIndex("histogramCompute", "in_data", memReport);
  m_state->binsBufIdx = getBufferIndex("zeroBins", "bins", memReport);

  std::vector<uint32_t> initInput(m_state->numElements);
  for (size_t i = 0; i < m_state->numElements; i++) {
    initInput[i] = generateSampleByte(i);
  }
  memoryWriteArgument(
    parallelProgram.m_dataPool, 1U, 0U,
    reinterpret_cast<const char*>(initInput.data()), initInput.size() * sizeof(uint32_t), 0U);

  if (!parallelProgram.m_useHostMemory) {
    auto writeInputJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 1U, 3U, 0U);
    parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeInputJob);
  }

  auto zeroJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  zeroJob->setKernelFunctionIdx(0U, 1U);
  zeroJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobalZero, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(zeroJob);

  auto histJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  histJob->setKernelFunctionIdx(1U, 3U);
  histJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobalHist, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(histJob);

  auto readBinsJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 1U, 0U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readBinsJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobalHist;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobalHist;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void HistogramKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::HistogramKernel)
