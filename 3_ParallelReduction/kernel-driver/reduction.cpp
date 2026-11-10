#include "reduction.h"

#include <cmath>
#include <iomanip>
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

size_t verifyReductionOutput(
  const pfaces2DKernel& kernel,
  const pfacesParallelProgram& program,
  std::vector<std::shared_ptr<void>>& params) {
  std::ignore = kernel;
  if (params.empty())
    throw std::runtime_error("[FAIL] reduction: missing verification state.");

  const auto state = std::static_pointer_cast<ReductionState>(params[0]);
  const auto* inData = reinterpret_cast<const float*>(program.m_dataPool[state->inDataBufIdx].first);
  const auto* finalSum = reinterpret_cast<const float*>(program.m_dataPool[state->finalSumBufIdx].first);
  if (inData == nullptr || finalSum == nullptr)
    throw std::runtime_error("[FAIL] reduction: data pool buffers are missing.");

  // Kahan summation on host in double precision as high-accuracy ground truth.
  double cpuSum = 0.0;
  double comp = 0.0;
  for (size_t i = 0; i < state->numElements; i++) {
    const double y = static_cast<double>(inData[i]) - comp;
    const double t = cpuSum + y;
    comp = (t - cpuSum) - y;
    cpuSum = t;
  }

  const double devSum = static_cast<double>(finalSum[0]);
  const double relErr = std::fabs(devSum - cpuSum) / std::fmax(1.0, std::fabs(cpuSum));

  if (!(relErr <= 1e-4) || !std::isfinite(relErr)) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "[FAIL] 3_ParallelReduction: N=" << state->numElements
        << ", device_sum=" << devSum
        << ", cpu_sum=" << cpuSum
        << ", rel_err=" << std::scientific << std::setprecision(3) << relErr;
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  std::cout << std::fixed << std::setprecision(4)
            << "[PASS] 3_ParallelReduction: N=" << state->numElements
            << ", device_sum=" << devSum
            << ", cpu_sum=" << cpuSum
            << ", rel_err=" << std::scientific << std::setprecision(3) << relErr
            << std::defaultfloat << std::endl;
  return 0U;
}

}  // namespace

ReductionKernel::ReductionKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("reduction"), 1U, 1U, spCfg),
    m_state(std::make_shared<ReductionState>()) {

  if (spCfg) {
    m_state->numElements = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("num_elements")));
    m_state->wgSize = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("wg_size")));
    m_state->numGroups = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("num_groups")));
  }

  if (m_state->numElements == 0U || !isPowerOfTwo(m_state->wgSize))
    throw std::runtime_error("num_elements must be > 0 and wg_size must be a power of 2.");
  if (m_state->numGroups == 0U || m_state->numGroups > m_state->wgSize)
    throw std::runtime_error("num_groups must be in [1, wg_size] for two-stage reduction.");

  const size_t totalThreads = m_state->numGroups * m_state->wgSize;
  setDimensions(totalThreads, 1U);

  const std::string memFile = spLaunchState->getKernelPackPath() + "reduction.mem";

  auto argsPartial = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "reducePartial", {"in_data", "partial_sums", "scratch"});
  argsPartial.m_baseTypeMultiple[0] = m_state->numElements;
  argsPartial.m_baseTypeMultiple[1] = m_state->numGroups;
  argsPartial.m_baseTypeMultiple[2] = m_state->wgSize;
  addKernelFunction(pfacesKernelFunction("reducePartial", argsPartial));

  auto argsFinal = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "reduceFinal", {"partial_sums", "final_sum", "scratch"});
  argsFinal.m_baseTypeMultiple[1] = 1U;
  argsFinal.m_baseTypeMultiple[2] = m_state->wgSize;
  addKernelFunction(pfacesKernelFunction("reduceFinal", argsFinal));

  updateParameters(
    {"@@RED_N@@", "@@NUM_GROUPS@@"},
    {std::to_string(m_state->numElements), std::to_string(m_state->numGroups)});

  std::vector<std::shared_ptr<void>> verifyParams{m_state};
  registerPostExecuteFunction(verifyReductionOutput, "Verify two-stage reduction sum against Kahan CPU sum", verifyParams);
}

void ReductionKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "3_ParallelReduction targets a single OpenCL device.");

  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];

  const size_t maxWg = thisDevice.get().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (m_state->wgSize > maxWg) {
    throw std::runtime_error(
      "wg_size (" + std::to_string(m_state->wgSize) + ") exceeds max work-group size (" +
      std::to_string(maxWg) + ").");
  }

  const cl::NDRange ndGlobalStage1(m_state->numGroups * m_state->wgSize, 1U);
  const cl::NDRange ndGlobalStage2(m_state->wgSize, 1U);
  const cl::NDRange ndLocal(m_state->wgSize, 1U);
  const cl::NDRange ndOffset(0U, 0U);

  const auto memReport = allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, true);
  m_state->inDataBufIdx = getBufferIndex("reducePartial", "in_data", memReport);
  m_state->finalSumBufIdx = getBufferIndex("reduceFinal", "final_sum", memReport);

  std::vector<float> initData(m_state->numElements);
  for (size_t i = 0; i < m_state->numElements; i++) {
    initData[i] = 0.25f + 0.01f * static_cast<float>(i % 97U);
  }
  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 0U,
    reinterpret_cast<const char*>(initData.data()), initData.size() * sizeof(float), 0U);

  if (!parallelProgram.m_useHostMemory) {
    auto writeJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 3U, 0U);
    parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeJob);
  }

  auto stage1Job = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stage1Job->setKernelFunctionIdx(0, 3U);
  stage1Job->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobalStage1, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stage1Job);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  auto stage2Job = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stage2Job->setKernelFunctionIdx(1, 3U);
  stage2Job->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobalStage2, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stage2Job);

  auto readFinalJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 1U, 3U, 1U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readFinalJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobalStage1;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobalStage1;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void ReductionKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::ReductionKernel)
