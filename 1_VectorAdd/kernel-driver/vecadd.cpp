#include "vecadd.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pFacesExamples {

namespace {

size_t verifyVecAddOutput(
  const pfaces2DKernel& kernel,
  const pfacesParallelProgram& program,
  std::vector<std::shared_ptr<void>>& params) {
  std::ignore = kernel;
  if (params.empty())
    throw std::runtime_error("[FAIL] vecadd: missing verification state.");

  const auto state = std::static_pointer_cast<VecAddState>(params[0]);
  if (program.m_dataPool.size() < 3U ||
      program.m_dataPool[0].first == nullptr ||
      program.m_dataPool[1].first == nullptr ||
      program.m_dataPool[2].first == nullptr) {
    throw std::runtime_error("[FAIL] vecadd: data pool buffers are missing.");
  }

  const auto* A = reinterpret_cast<const float*>(program.m_dataPool[0].first);
  const auto* B = reinterpret_cast<const float*>(program.m_dataPool[1].first);
  const auto* C = reinterpret_cast<const float*>(program.m_dataPool[2].first);

  float maxAbsErr = 0.0f;
  size_t mismatches = 0U;
  for (size_t i = 0; i < state->length; i++) {
    const float ref = state->alpha * A[i] + B[i];
    const float err = std::fabs(C[i] - ref);
    if (err > maxAbsErr)
      maxAbsErr = err;
    if (!(err <= 1e-5f)) {
      if (mismatches < 5U) {
        std::cerr << "[FAIL] vecadd mismatch at i=" << i
                  << " got=" << C[i] << " expected=" << ref << std::endl;
      }
      mismatches++;
    }
  }

  std::cout << std::scientific << std::setprecision(3)
            << "[PASS] 1_VectorAdd (SAXPY): N=" << state->length
            << ", alpha=" << std::fixed << std::setprecision(2) << state->alpha
            << ", max_abs_err=" << std::scientific << maxAbsErr
            << std::defaultfloat << std::endl;

  if (mismatches > 0U) {
    throw std::runtime_error(
      "[FAIL] 1_VectorAdd: " + std::to_string(mismatches) + " elements exceeded tolerance.");
  }
  return 0U;
}

}  // namespace

VectorAddKernel::VectorAddKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("vecadd"), 1U, 1U, spCfg),
    m_state(std::make_shared<VecAddState>()) {

  if (spCfg) {
    m_state->length = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("vector_size")));
    m_state->alpha = std::stof(spCfg->readConfigValueString("alpha"));
  }
  if (m_state->length == 0U)
    throw std::runtime_error("vector_size must be > 0.");

  setDimensions(m_state->length, 1U);

  const std::string memFile = spLaunchState->getKernelPackPath() + "vecadd.mem";
  const auto funcArgs = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "vecadd", {"A", "B", "C"});
  addKernelFunction(pfacesKernelFunction("vecadd", funcArgs));

  std::ostringstream alphaStr;
  alphaStr << std::setprecision(9) << m_state->alpha << "f";
  updateParameters(
    {"@@VEC_N@@", "@@VEC_ALPHA@@"},
    {std::to_string(m_state->length), alphaStr.str()});

  std::vector<std::shared_ptr<void>> verifyParams{m_state};
  registerPostExecuteFunction(verifyVecAddOutput, "Verify SAXPY output against CPU ground truth", verifyParams);
}

void VectorAddKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "1_VectorAdd targets a single OpenCL device.");

  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];

  const size_t n = m_state->length;
  const cl::NDRange ndGlobal(n, 1U);
  const cl::NDRange ndOffset(0U, 0U);

  allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, n, false);

  std::vector<float> initA(n);
  std::vector<float> initB(n);
  for (size_t i = 0; i < n; i++) {
    initA[i] = std::sin(0.01f * static_cast<float>(i % 1024U));
    initB[i] = 0.5f * std::cos(0.02f * static_cast<float>(i % 1024U));
  }
  const size_t nBytes = n * sizeof(float);
  memoryWriteArgument(parallelProgram.m_dataPool, 0U, 0U, reinterpret_cast<const char*>(initA.data()), nBytes, 0U);
  memoryWriteArgument(parallelProgram.m_dataPool, 0U, 1U, reinterpret_cast<const char*>(initB.data()), nBytes, 0U);

  // When running without -hm, explicitly enqueue H2D buffer writes before execution.
  if (!parallelProgram.m_useHostMemory) {
    for (size_t argIdx : {0U, 1U}) {
      auto writeJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 3U, argIdx);
      parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeJob);
    }
  }

  auto execJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  execJob->setKernelFunctionIdx(0, 3U);
  execJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, cl::NullRange));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(execJob);

  auto readJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 3U, 2U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobal;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobal;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void VectorAddKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::VectorAddKernel)
