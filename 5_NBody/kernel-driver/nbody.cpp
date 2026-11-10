#include "nbody.h"

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

constexpr float kMaxAllowedPosError = 1e-4f;
constexpr float kMaxAllowedVelError = 1e-4f;

struct BodyFloat4 {
  float x;
  float y;
  float z;
  float w;
};

static_assert(sizeof(BodyFloat4) == 16U, "BodyFloat4 must match OpenCL float4 size.");

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

float parsePositiveFloat(
  const std::shared_ptr<pfacesConfigurationReader>& spCfg,
  const char* key,
  float defaultVal) {
  std::string raw = spCfg->readConfigValueString(key);
  if (raw.empty()) {
    return defaultVal;
  }
  const float val = std::stof(raw);
  if (!(val > 0.0f) || !std::isfinite(val)) {
    throw std::runtime_error(std::string("Configuration key '") + key + "' must be > 0.");
  }
  return val;
}

size_t roundUpMultiple(size_t value, size_t multiple) {
  return ((value + multiple - 1U) / multiple) * multiple;
}

std::string formatFloatLiteral(float value) {
  std::ostringstream oss;
  oss << std::setprecision(9) << value << "f";
  return oss.str();
}

void initializeGalaxyBodies(
  size_t numBodies,
  std::vector<BodyFloat4>& pos,
  std::vector<BodyFloat4>& vel) {
  pos.resize(numBodies);
  vel.resize(numBodies);

  constexpr float kTwoPi = 6.28318530718f;
  for (size_t i = 0; i < numBodies; i++) {
    const float frac = static_cast<float>(i + 1U) / static_cast<float>(numBodies);
    const float angle = 13.0f * kTwoPi * frac;
    const float radius = 0.5f + 2.5f * std::sqrt(frac);
    const float height = 0.15f * std::sin(5.0f * angle);
    const float mass = 0.2f + 0.05f * static_cast<float>(i % 7U);
    const float speed = 0.65f / std::sqrt(radius + 0.1f);

    pos[i] = {
      radius * std::cos(angle),
      radius * std::sin(angle),
      height,
      mass
    };
    vel[i] = {
      -speed * std::sin(angle),
      speed * std::cos(angle),
      0.02f * std::cos(3.0f * angle),
      0.0f
    };
  }
}

void computeCpuReferenceNBody(
  size_t numBodies,
  size_t steps,
  float dt,
  float eps2,
  std::vector<BodyFloat4>& posA,
  std::vector<BodyFloat4>& vel) {
  initializeGalaxyBodies(numBodies, posA, vel);
  std::vector<BodyFloat4> posB = posA;

  for (size_t step = 0; step < steps; step++) {
    const std::vector<BodyFloat4>& src = (step % 2U == 0U) ? posA : posB;
    std::vector<BodyFloat4>& dst = (step % 2U == 0U) ? posB : posA;

    for (size_t i = 0; i < numBodies; i++) {
      const BodyFloat4 pi = src[i];
      float ax = 0.0f;
      float ay = 0.0f;
      float az = 0.0f;

      for (size_t j = 0; j < numBodies; j++) {
        const BodyFloat4 pj = src[j];
        const float rx = pj.x - pi.x;
        const float ry = pj.y - pi.y;
        const float rz = pj.z - pi.z;
        const float distSqr = rx * rx + ry * ry + rz * rz + eps2;
        const float invDist = 1.0f / std::sqrt(distSqr);
        const float scale = pj.w * invDist * invDist * invDist;
        ax += scale * rx;
        ay += scale * ry;
        az += scale * rz;
      }

      vel[i].x += ax * dt;
      vel[i].y += ay * dt;
      vel[i].z += az * dt;

      dst[i].x = pi.x + vel[i].x * dt;
      dst[i].y = pi.y + vel[i].y * dt;
      dst[i].z = pi.z + vel[i].z * dt;
      dst[i].w = pi.w;
    }
  }
}

size_t hostNBodyLoopCheck(void* pPackedKernel, void* /*pPackedParallelProgram*/) {
  auto* kernel = static_cast<NBodyKernel*>(static_cast<pfaces2DKernel*>(pPackedKernel));
  return kernel->advanceLoopPair();
}

size_t verifyNBodyOutput(
  const pfaces2DKernel& /*kernel*/,
  const pfacesParallelProgram& parallelProgram,
  std::vector<std::shared_ptr<void>>& params) {
  const auto state = std::static_pointer_cast<NBodyState>(params[0]);
  const size_t expectedBytes = state->numBodies * sizeof(BodyFloat4);

  const auto& posPair = parallelProgram.m_dataPool[state->posABufIdx];
  const auto& velPair = parallelProgram.m_dataPool[state->velBufIdx];
  if (posPair.first == nullptr || posPair.second < expectedBytes ||
      velPair.first == nullptr || velPair.second < expectedBytes) {
    throw std::runtime_error("5_NBody: output buffer missing or undersized.");
  }

  std::vector<BodyFloat4> devPos(state->numBodies);
  std::vector<BodyFloat4> devVel(state->numBodies);
  std::memcpy(devPos.data(), posPair.first, expectedBytes);
  std::memcpy(devVel.data(), velPair.first, expectedBytes);

  std::vector<BodyFloat4> cpuPos;
  std::vector<BodyFloat4> cpuVel;
  const float eps2 = state->softening * state->softening;
  computeCpuReferenceNBody(state->numBodies, state->steps, state->dt, eps2, cpuPos, cpuVel);

  float maxPosErr = 0.0f;
  float maxVelErr = 0.0f;
  size_t worstIdx = 0U;
  for (size_t i = 0; i < state->numBodies; i++) {
    const float pe = std::max(
      std::fabs(devPos[i].x - cpuPos[i].x),
      std::max(std::fabs(devPos[i].y - cpuPos[i].y), std::fabs(devPos[i].z - cpuPos[i].z)));
    const float ve = std::max(
      std::fabs(devVel[i].x - cpuVel[i].x),
      std::max(std::fabs(devVel[i].y - cpuVel[i].y), std::fabs(devVel[i].z - cpuVel[i].z)));
    if (pe > maxPosErr || ve > maxVelErr) {
      worstIdx = i;
    }
    maxPosErr = std::max(maxPosErr, pe);
    maxVelErr = std::max(maxVelErr, ve);
  }

  std::ostringstream oss;
  const bool ok =
    std::isfinite(maxPosErr) && std::isfinite(maxVelErr) &&
    (maxPosErr <= kMaxAllowedPosError) && (maxVelErr <= kMaxAllowedVelError);
  if (!ok) {
    oss << "[FAIL] 5_NBody: N=" << state->numBodies << ", steps=" << state->steps
        << ", worst_idx=" << worstIdx
        << ", max_pos_err=" << std::scientific << std::setprecision(3) << maxPosErr
        << ", max_vel_err=" << maxVelErr;
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  oss << "[PASS] 5_NBody: N=" << state->numBodies << ", steps=" << state->steps
      << ", max_pos_err=" << std::scientific << std::setprecision(3) << maxPosErr
      << ", max_vel_err=" << maxVelErr;
  std::cout << oss.str() << std::endl;
  return 0;
}

}  // namespace

NBodyKernel::NBodyKernel(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("nbody"), 1U, 1U, spCfg),
    m_state(std::make_shared<NBodyState>()) {
  const std::string kernelPackDir = spLaunchState->getKernelPackPath();

  if (spCfg) {
    m_state->numBodies = parsePositiveSize(spCfg, "num_bodies", m_state->numBodies);
    m_state->steps = parsePositiveSize(spCfg, "steps", m_state->steps);
    m_state->dt = parsePositiveFloat(spCfg, "dt", m_state->dt);
    m_state->softening = parsePositiveFloat(spCfg, "softening", m_state->softening);
    m_state->wgSize = parsePositiveSize(spCfg, "wg_size", m_state->wgSize);
  }

  if (m_state->steps < 2U || (m_state->steps % 2U) != 0U) {
    throw std::runtime_error("5_NBody requires an even steps count >= 2.");
  }

  setDimensions(m_state->numBodies, 1U);
  m_state->numPairs = m_state->steps / 2U;

  const std::string memFile = kernelPackDir + "nbody.mem";
  const std::vector<std::string> argNames = {"pos_in", "pos_out", "vel", "tile_pos"};

  pfacesKernelFunctionArguments argsAB =
    pfacesKernelFunctionArguments::loadFromFile(memFile, "nbodyStepAB", argNames);
  argsAB.m_baseTypeMultiple[0] = m_state->numBodies;
  argsAB.m_baseTypeMultiple[1] = m_state->numBodies;
  argsAB.m_baseTypeMultiple[2] = m_state->numBodies;
  argsAB.m_baseTypeMultiple[3] = m_state->wgSize;

  pfacesKernelFunctionArguments argsBA =
    pfacesKernelFunctionArguments::loadFromFile(memFile, "nbodyStepBA", argNames);
  argsBA.m_baseTypeMultiple[3] = m_state->wgSize;

  addKernelFunction(pfacesKernelFunction("nbodyStepAB", argsAB));
  addKernelFunction(pfacesKernelFunction("nbodyStepBA", argsBA));

  const float eps2 = m_state->softening * m_state->softening;
  updateParameters(
    {"@@NBODY_COUNT@@", "@@NBODY_DT@@", "@@NBODY_EPS2@@"},
    {std::to_string(m_state->numBodies), formatFloatLiteral(m_state->dt), formatFloatLiteral(eps2)});

  std::vector<std::shared_ptr<void>> postParams = {m_state};
  registerPostExecuteFunction(
    verifyNBodyOutput,
    std::string("Verify 3D N-body trajectories against CPU reference"),
    postParams);
}

size_t NBodyKernel::advanceLoopPair() {
  m_state->completedPairs++;
  return (m_state->completedPairs < m_state->numPairs) ? 1U : 0U;
}

void NBodyKernel::configureParallelProgram(pfacesParallelProgram& parallelProgram) {
  PFACES_ASSERT_CONTEXT(
    parallelProgram.countTargetDevices() == 1U,
    "5_NBody expects a single target OpenCL device (-d 1).");

  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];
  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t maxWg = thisDevice.get().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (m_state->wgSize > maxWg) {
    throw std::runtime_error(
      "wg_size (" + std::to_string(m_state->wgSize) +
      ") exceeds device max work-group size (" + std::to_string(maxWg) + ").");
  }

  const size_t globalN = roundUpMultiple(m_state->numBodies, m_state->wgSize);
  const cl::NDRange ndGlobal(globalN, 1U);
  const cl::NDRange ndLocal(m_state->wgSize, 1U);
  const cl::NDRange ndOffset(0U, 0U);

  const auto memReport = allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, true);
  m_state->posABufIdx = getBufferIndex("nbodyStepAB", "pos_in", memReport);
  m_state->velBufIdx = getBufferIndex("nbodyStepAB", "vel", memReport);
  m_state->completedPairs = 0U;

  std::vector<BodyFloat4> initPos;
  std::vector<BodyFloat4> initVel;
  initializeGalaxyBodies(m_state->numBodies, initPos, initVel);

  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 0U,
    reinterpret_cast<const char*>(initPos.data()), initPos.size() * sizeof(BodyFloat4), 0U);
  memoryWriteArgument(
    parallelProgram.m_dataPool, 0U, 2U,
    reinterpret_cast<const char*>(initVel.data()), initVel.size() * sizeof(BodyFloat4), 0U);

  if (!parallelProgram.m_useHostMemory) {
    auto writePosJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 4U, 0U);
    auto writeVelJob = std::make_shared<pfacesDeviceWriteJob>(thisDevice, 0U, 4U, 2U);
    parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writePosJob);
    parallelProgram.addNetInstruction()->setAsWriteDeviceBuffer(writeVelJob);
  }

  const size_t loopStartIdx = parallelProgram.m_spInstructionList.size();

  auto stepABJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stepABJob->setKernelFunctionIdx(0, 4U);
  stepABJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stepABJob);

  auto stepBAJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  stepBAJob->setKernelFunctionIdx(1, 4U);
  stepBAJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndOffset, ndGlobal, ndLocal));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(stepBAJob);

  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();
  parallelProgram.addNetInstruction()->setAsHostFunction(hostNBodyLoopCheck, "hostNBodyLoopCheck");
  parallelProgram.addNetInstruction()->setAsJumpNe(loopStartIdx);

  auto readPosJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 4U, 0U);
  auto readVelJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 4U, 2U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readPosJob);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readVelJob);
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  parallelProgram.m_Universal_globalNDRange = ndGlobal;
  parallelProgram.m_Universal_offsetNDRange = ndOffset;
  parallelProgram.m_Process_globalNDRange = ndGlobal;
  parallelProgram.m_Process_offsetNDRange = ndOffset;
}

void NBodyKernel::configureTuneParallelProgram(
  pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}

}  // namespace pFacesExamples

PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::NBodyKernel)
