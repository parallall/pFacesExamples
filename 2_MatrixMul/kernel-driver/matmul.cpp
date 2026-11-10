#include "matmul.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pFacesExamples {

namespace {

size_t roundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1U) / multiple) * multiple;
}

size_t verifyMatMulOutput(
  const pfaces2DKernel& kernel,
  const pfacesParallelProgram& program,
  std::vector<std::shared_ptr<void>>& params) {
  std::ignore = kernel;
  if (params.empty())
    throw std::runtime_error("[FAIL] matmul: missing verification state.");

  const auto state = std::static_pointer_cast<MatMulState>(params[0]);
  if (program.m_dataPool.size() < 3U ||
      program.m_dataPool[0].first == nullptr ||
      program.m_dataPool[1].first == nullptr ||
      program.m_dataPool[2].first == nullptr) {
    throw std::runtime_error("[FAIL] matmul: data pool buffers are missing.");
  }

  const auto* A = reinterpret_cast<const float*>(program.m_dataPool[0].first);
  const auto* B = reinterpret_cast<const float*>(program.m_dataPool[1].first);
  const auto* C = reinterpret_cast<const float*>(program.m_dataPool[2].first);

  const float tol = 1e-3f * (1.0f + 0.01f * static_cast<float>(state->k));
  float maxAbsErr = 0.0f;
  size_t mismatches = 0U;

  for (size_t r = 0; r < state->m; r++) {
    for (size_t c = 0; c < state->n; c++) {
      float ref = 0.0f;
      for (size_t kk = 0; kk < state->k; kk++) {
        ref += A[r * state->k + kk] * B[kk * state->n + c];
      }
      const float err = std::fabs(C[r * state->n + c] - ref);
      if (err > maxAbsErr)
        maxAbsErr = err;
      if (!(err <= tol)) {
        if (mismatches < 5U) {
          std::cerr << "[FAIL] matmul mismatch at (" << r << "," << c
                    << ") got=" << C[r * state->n + c] << " expected=" << ref << std::endl;
        }
        mismatches++;
      }
    }
  }

  std::cout << std::scientific << std::setprecision(3)
            << "[PASS] 2_MatrixMul: (" << state->m << "x" << state->k << ") * ("
            << state->k << "x" << state->n << "), tile=" << state->tileSize
            << "x" << state->tileSize << ", max_abs_err=" << maxAbsErr
            << std::defaultfloat << std::endl;

  if (mismatches > 0U) {
    throw std::runtime_error(
      "[FAIL] 2_MatrixMul: " + std::to_string(mismatches) + " elements exceeded tolerance.");
  }
  return 0U;
}

}  // namespace

MatrixMulKernel::MatrixMulKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("matmul"), 1U, 1U, spCfg),
    m_state(std::make_shared<MatMulState>()) {

  if (spCfg) {
    m_state->m = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("mat_m")));
    m_state->k = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("mat_k")));
    m_state->n = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("mat_n")));
    m_state->tileSize = static_cast<size_t>(std::stoul(spCfg->readConfigValueString("tile_size")));
  }
  if (m_state->m == 0U || m_state->k == 0U || m_state->n == 0U || m_state->tileSize == 0U)
    throw std::runtime_error("Matrix dimensions and tile_size must be > 0.");

  m_state->padM = roundUp(m_state->m, m_state->tileSize);
  m_state->padN = roundUp(m_state->n, m_state->tileSize);
  setDimensions(m_state->padN, m_state->padM);

  const std::string memFile = spLaunchState->getKernelPackPath() + "matmul.mem";
  auto funcArgs = pfacesKernelFunctionArguments::loadFromFile(memFile, "matmul", {"A", "B", "C"});
  funcArgs.m_baseTypeMultiple[0] = m_state->m * m_state->k;
  funcArgs.m_baseTypeMultiple[1] = m_state->k * m_state->n;
  funcArgs.m_baseTypeMultiple[2] = m_state->m * m_state->n;
  addKernelFunction(pfacesKernelFunction("matmul", funcArgs));

  updateParameters(
    {"@@MAT_M@@", "@@MAT_K@@", "@@MAT_N@@", "@@TILE_SIZE@@"},
    {
      std::to_string(m_state->m),
      std::to_string(m_state->k),
      std::to_string(m_state->n),
      std::to_string(m_state->tileSize),
    });

  std::vector<std::shared_ptr<void>> verifyParams{m_state};
  registerPostExecuteFunction(verifyMatMulOutput, "Verify matrix product against CPU ground truth", verifyParams);
}

void MatrixMulKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "2_MatrixMul targets a single OpenCL device.");

  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];

  const size_t maxWg = thisDevice.get().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (m_state->tileSize * m_state->tileSize > maxWg) {
    throw std::runtime_error(
      "tile_size^2 (" + std::to_string(m_state->tileSize * m_state->tileSize) +
      ") exceeds device max work-group size (" + std::to_string(maxWg) + ").");
  }

  const cl::NDRange ndGlobal(m_state->padN, m_state->padM);
  const cl::NDRange ndLocal(m_state->tileSize, m_state->tileSize);
  const cl::NDRange ndOffset(0U, 0U);

  allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, false);

  std::vector<float> initA(m_state->m * m_state->k);
  std::vector<float> initB(m_state->k * m_state->n);
  for (size_t r = 0; r < m_state->m; r++) {
    for (size_t kk = 0; kk < m_state->k; kk++) {
      initA[r * m_state->k + kk] = 0.01f * static_cast<float>((r * 7U + kk * 3U) % 29U) - 0.14f;
    }
  }
  for (size_t kk = 0; kk < m_state->k; kk++) {
    for (size_t c = 0; c < m_state->n; c++) {
      initB[kk * m_state->n + c] = 0.01f * static_cast<float>((kk * 5U + c * 11U) % 23U) - 0.11f;
    }
  }

  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 0U, reinterpret_cast<const char*>(initA.data()), initA.size() * sizeof(float), 0U);
  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 1U, reinterpret_cast<const char*>(initB.data()), initB.size() * sizeof(float), 0U);

  if (!parallelProgram.m_useHostMemory) {
    for (size_t argIdx : {0U, 1U}) {
      auto writeJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 3U, argIdx);
      parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeJob);
    }
  }

  auto execJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  execJob->setKernelFunctionIdx(0, 3U);
  execJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(execJob);

  auto readJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 3U, 2U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobal;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobal;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void MatrixMulKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = targetFunctionIdx;
  const auto& thisMachine = tuneParallelProgram.getMachine();
  const auto& thisDevice = tuneParallelProgram.getTargetDevices()[0];

  const cl::NDRange ndGlobal(m_state->padN, m_state->padM);
  const cl::NDRange ndLocal(m_state->tileSize, m_state->tileSize);
  const cl::NDRange ndOffset(0U, 0U);

  allocateMemory(
    tuneParallelProgram.m_dataPool, thisMachine, tuneParallelProgram.getTargetDevicesIndicies(), 1U, true);

  auto tuneJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  tuneJob->setKernelFunctionIdx(0, 3U);
  tuneJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));

  tuneParallelProgram.m_spInstructionList.clear();
  tuneParallelProgram.addNetInstruction()->setAsDeviceExecute(tuneJob);
  tuneParallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  tuneParallelProgram.m_Universal_globalNDRange = ndGlobal;
  tuneParallelProgram.m_Universal_offsetNDRange = ndOffset;
  tuneParallelProgram.m_Process_globalNDRange = ndGlobal;
  tuneParallelProgram.m_Process_offsetNDRange = ndOffset;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::MatrixMulKernel)
