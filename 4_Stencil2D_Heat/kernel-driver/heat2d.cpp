#include "heat2d.h"

#include <algorithm>
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

constexpr float kTopDirichletTemp = 100.0f;
constexpr float kMaxAllowedAbsError = 1e-4f;

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

float parseFloatValue(
  const std::shared_ptr<pfacesConfigurationReader>& spCfg,
  const char* key,
  float defaultVal) {
  std::string raw = spCfg->readConfigValueString(key);
  if (raw.empty()) {
    return defaultVal;
  }
  return std::stof(raw);
}

size_t roundUpMultiple(size_t value, size_t multiple) {
  return ((value + multiple - 1U) / multiple) * multiple;
}

std::string formatFloatLiteral(float value) {
  std::ostringstream oss;
  oss << std::setprecision(9) << value << "f";
  return oss.str();
}

void initializeHeatGrid(std::vector<float>& grid, size_t width, size_t height) {
  grid.assign(width * height, 0.0f);
  for (size_t x = 0; x < width; x++) {
    grid[x] = kTopDirichletTemp;
  }
}

void computeCpuReferenceHeat2D(
  size_t width,
  size_t height,
  size_t iterations,
  float r,
  std::vector<float>& gridA) {
  initializeHeatGrid(gridA, width, height);
  std::vector<float> gridB = gridA;
  const float centerWeight = 1.0f - 4.0f * r;

  for (size_t step = 0; step < iterations; step++) {
    const std::vector<float>& src = (step % 2U == 0U) ? gridA : gridB;
    std::vector<float>& dst = (step % 2U == 0U) ? gridB : gridA;

    for (size_t y = 0; y < height; y++) {
      const size_t rowOffset = y * width;
      if (y == 0U || y == height - 1U) {
        std::copy_n(&src[rowOffset], width, &dst[rowOffset]);
        continue;
      }
      dst[rowOffset] = src[rowOffset];
      for (size_t x = 1U; x < width - 1U; x++) {
        const size_t idx = rowOffset + x;
        dst[idx] = centerWeight * src[idx] +
                   r * (src[idx - width] + src[idx + width] + src[idx - 1U] + src[idx + 1U]);
      }
      dst[rowOffset + (width - 1U)] = src[rowOffset + (width - 1U)];
    }
  }
}

size_t hostStepLoopCheck(void* pPackedKernel, void* /*pPackedParallelProgram*/) {
  auto* kernel = static_cast<Heat2DKernel*>(static_cast<pfaces2DKernel*>(pPackedKernel));
  return kernel->advanceLoopPair();
}

size_t verifyHeat2DOutput(
  const pfaces2DKernel& /*kernel*/,
  const pfacesParallelProgram& parallelProgram,
  std::vector<std::shared_ptr<void>>& params) {
  const auto state = std::static_pointer_cast<Heat2DState>(params[0]);
  const size_t totalCells = state->width * state->height;

  const auto& bufPair = parallelProgram.m_dataPool[state->gridABufIdx];
  if (bufPair.first == nullptr || bufPair.second < totalCells * sizeof(float)) {
    throw std::runtime_error("4_Stencil2D_Heat: output buffer missing or undersized.");
  }

  std::vector<float> devOut(totalCells);
  std::memcpy(devOut.data(), bufPair.first, totalCells * sizeof(float));

  std::vector<float> cpuRef;
  computeCpuReferenceHeat2D(state->width, state->height, state->iterations, state->diffusion, cpuRef);

  float maxAbsErr = 0.0f;
  size_t worstIdx = 0U;
  double sumInterior = 0.0;
  for (size_t i = 0; i < totalCells; i++) {
    const float err = std::fabs(devOut[i] - cpuRef[i]);
    if (err > maxAbsErr) {
      maxAbsErr = err;
      worstIdx = i;
    }
    sumInterior += static_cast<double>(devOut[i]);
  }

  const float centerVal = devOut[(state->height / 2U) * state->width + (state->width / 2U)];
  const float centerRef = cpuRef[(state->height / 2U) * state->width + (state->width / 2U)];

  std::ostringstream oss;
  if (!(maxAbsErr <= kMaxAllowedAbsError) || !std::isfinite(maxAbsErr)) {
    oss << "[FAIL] 4_Stencil2D_Heat: width=" << state->width << ", height=" << state->height
        << ", iters=" << state->iterations << ", max_abs_err=" << std::scientific
        << std::setprecision(3) << maxAbsErr << " at idx=" << worstIdx
        << " (device=" << std::fixed << std::setprecision(5) << devOut[worstIdx]
        << ", cpu=" << cpuRef[worstIdx] << ")";
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  oss << "[PASS] 4_Stencil2D_Heat: grid=" << state->width << "x" << state->height
      << ", iters=" << state->iterations << ", center=(" << std::fixed << std::setprecision(4)
      << centerVal << " vs " << centerRef << "), mean=" << (sumInterior / static_cast<double>(totalCells))
      << ", max_abs_err=" << std::scientific << std::setprecision(3) << maxAbsErr;
  std::cout << oss.str() << std::endl;
  return 0;
}

}  // namespace

Heat2DKernel::Heat2DKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("heat2d"), 1U, 1U, spCfg),
    m_state(std::make_shared<Heat2DState>()) {
  const std::string kernelPackDir = spLaunchState->getKernelPackPath();

  if (spCfg) {
    m_state->width = parsePositiveSize(spCfg, "width", m_state->width);
    m_state->height = parsePositiveSize(spCfg, "height", m_state->height);
    m_state->iterations = parsePositiveSize(spCfg, "iterations", m_state->iterations);
    m_state->diffusion = parseFloatValue(spCfg, "diffusion", m_state->diffusion);
    m_state->wgX = parsePositiveSize(spCfg, "wg_x", m_state->wgX);
    m_state->wgY = parsePositiveSize(spCfg, "wg_y", m_state->wgY);
  }

  if (m_state->width < 3U || m_state->height < 3U) {
    throw std::runtime_error("4_Stencil2D_Heat requires width >= 3 and height >= 3.");
  }
  if (m_state->iterations < 2U || (m_state->iterations % 2U) != 0U) {
    throw std::runtime_error("4_Stencil2D_Heat requires an even iterations count >= 2.");
  }
  if (m_state->diffusion <= 0.0f || m_state->diffusion > 0.25f) {
    throw std::runtime_error("4_Stencil2D_Heat requires 0 < diffusion <= 0.25.");
  }

  setDimensions(m_state->width, m_state->height);

  m_state->numPairs = m_state->iterations / 2U;
  const size_t totalCells = m_state->width * m_state->height;
  const std::string memFile = kernelPackDir + "heat2d.mem";

  pfacesKernelFunctionArguments argsAB = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "jacobiStepAB", {"grid_in", "grid_out"});
  argsAB.m_baseTypeMultiple[0] = totalCells;
  argsAB.m_baseTypeMultiple[1] = totalCells;

  pfacesKernelFunctionArguments argsBA = pfacesKernelFunctionArguments::loadFromFile(
    memFile, "jacobiStepBA", {"grid_in", "grid_out"});

  addKernelFunction(pfacesKernelFunction("jacobiStepAB", argsAB));
  addKernelFunction(pfacesKernelFunction("jacobiStepBA", argsBA));

  updateParameters(
    {"@@HEAT_WIDTH@@", "@@HEAT_HEIGHT@@", "@@HEAT_R@@"},
    {std::to_string(m_state->width), std::to_string(m_state->height), formatFloatLiteral(m_state->diffusion)});

  std::vector<std::shared_ptr<void>> postParams = {m_state};
  registerPostExecuteFunction(
    verifyHeat2DOutput,
    std::string("Verify 2D Jacobi heat-diffusion solution against CPU reference"),
    postParams);
}

size_t Heat2DKernel::advanceLoopPair() {
  m_state->completedPairs++;
  return (m_state->completedPairs < m_state->numPairs) ? 1U : 0U;
}

void Heat2DKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "4_Stencil2D_Heat expects a single target OpenCL device (-d 1).");

  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];
  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t maxWg = thisDevice.get().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (m_state->wgX * m_state->wgY > maxWg) {
    throw std::runtime_error(
      "wg_x * wg_y (" + std::to_string(m_state->wgX * m_state->wgY) +
      ") exceeds device max work-group size (" + std::to_string(maxWg) + ").");
  }

  const size_t globalX = roundUpMultiple(m_state->width, m_state->wgX);
  const size_t globalY = roundUpMultiple(m_state->height, m_state->wgY);
  const cl::NDRange ndGlobal(globalX, globalY);
  const cl::NDRange ndLocal(m_state->wgX, m_state->wgY);
  const cl::NDRange ndOffset(0U, 0U);

  const auto memReport = allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, true);
  m_state->gridABufIdx = getBufferIndex("jacobiStepAB", "grid_in", memReport);
  m_state->completedPairs = 0U;

  std::vector<float> initGrid(m_state->width * m_state->height);
  initializeHeatGrid(initGrid, m_state->width, m_state->height);
  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 0U,
    reinterpret_cast<const char*>(initGrid.data()), initGrid.size() * sizeof(float), 0U);

  if (!parallelProgram.m_useHostMemory) {
    auto writeGridJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 2U, 0U);
    parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeGridJob);
  }

  const size_t loopStartIdx = parallelProgram.m_spInstructionList.size();

  auto stepABJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stepABJob->setKernelFunctionIdx(0, 2U);
  stepABJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stepABJob);

  auto stepBAJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stepBAJob->setKernelFunctionIdx(1, 2U);
  stepBAJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stepBAJob);

  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();
  parallelProgram.addNetInstruction()->setAsHostFunction(hostStepLoopCheck, "hostStepLoopCheck");
  parallelProgram.addNetInstruction()->setAsJumpNe(loopStartIdx);

  auto readGridJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 2U, 0U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readGridJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobal;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobal;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void Heat2DKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::Heat2DKernel)
