#include "example.h"

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pFacesExamples {

namespace {

constexpr std::array<char, 4> kExpectedToken = {'p', 'F', 'c', 's'};

size_t verifyHelloWorldOutput(
  const pfaces2DKernel& /*kernel*/,
  const pfacesParallelProgram& program,
  std::vector<std::shared_ptr<void>>& params) {
  if (params.empty()) {
    throw std::runtime_error("[FAIL] 0_HelloWorld: missing verification state.");
  }

  const auto state = std::static_pointer_cast<HelloWorldState>(params[0]);
  const auto& bufPair = program.m_dataPool[state->dataOutBufIdx];
  if (bufPair.first == nullptr || bufPair.second < kExpectedToken.size()) {
    throw std::runtime_error("[FAIL] 0_HelloWorld: output buffer missing or undersized.");
  }

  std::string actual(bufPair.first, bufPair.first + kExpectedToken.size());
  std::string expected(kExpectedToken.data(), kExpectedToken.size());
  if (actual != expected) {
    std::ostringstream oss;
    oss << "[FAIL] 0_HelloWorld: device handshake mismatch (got '" << actual
        << "', expected '" << expected << "').";
    std::cerr << oss.str() << std::endl;
    throw std::runtime_error(oss.str());
  }

  std::cout << "[PASS] 0_HelloWorld: device->host handshake verified (\"" << actual << "\")"
            << std::endl;
  return 0U;
}

}  // namespace

/// Some constants for the ND-Range.
constexpr size_t thread_grid_x{1U};
constexpr size_t thread_grid_y{1U};

/// Constructor for the example class.
pFacesExample::pFacesExample(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(spLaunchState->getDefaultSourceFilePath("example"), thread_grid_x, thread_grid_y, spCfg),
    m_state(std::make_shared<HelloWorldState>()) {

  // Add a kernel function after loading its memory fingerprints
  const std::string memFile = spLaunchState->getKernelPackPath() + "example.mem";
  const std::string funcName = "exampleKernelFunction";
  const auto funcArgs = pfacesKernelFunctionArguments::loadFromFile(memFile, funcName, {"data_in"});
  addKernelFunction(pfacesKernelFunction(funcName, funcArgs));

  // Add a parameter that can be accessed from the .cl file.
  updateParameters({"@@param_message@@"}, {"\"Hello from the driver side!\""});

  std::vector<std::shared_ptr<void>> verifyParams{m_state};
  registerPostExecuteFunction(
    verifyHelloWorldOutput,
    std::string("Verify Hello World device handshake token against ground truth"),
    verifyParams);
}

/// Provides an implementation of the virtual method @c configureParallelProgram.
void pFacesExample::configureParallelProgram(pfacesParallelProgram& parallelProgram) {

  // Check we run within only one device.
  PFACES_ASSERT_CONTEXT(parallelProgram.countTargetDevices() == 1, "This example supports only one target device.");

  // Get target machine/device.
  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];
  const size_t thisDeviceIdx = parallelProgram.getTargetDevicesIndicies()[0];

  // Define the range/offset for this kernel.
  const cl::NDRange ndKernelRange(thread_grid_x, thread_grid_y);
  const cl::NDRange ndKernelOffset(0U, 0U);

  // Allocate device memory zero-initialized and record buffer index.
  const auto memReport = allocateMemory(parallelProgram.m_dataPool, thisMachine, {thisDeviceIdx}, 1U, true);
  m_state->dataOutBufIdx = getBufferIndex("exampleKernelFunction", "data_in", memReport);

  // An instruction to show a simple message from the CPU side.
  parallelProgram.addNetInstruction()->setAsMessage("Hello World from host side !");

  // An instruction to launch an execute job with single parallel task in the targeted device.
  auto execJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  execJob->setKernelFunctionIdx(0U, 1U);
  execJob->addTask(std::make_shared<pfacesDeviceExecuteTask>(ndKernelOffset, ndKernelRange, ndKernelRange));
  parallelProgram.addNetInstruction()->setAsDeviceExecute(execJob);

  // Read back the device handshake token for host verification.
  auto readJob = std::make_shared<pfacesDeviceReadJob>(thisDevice, 0U, 1U, 0U);
  parallelProgram.addNetInstruction()->setAsReadDeviceBuffer(readJob);

  // An instruction for synchronization.
  parallelProgram.addNetInstruction()->setAsBlockingSyncPoint();

  // Set some required settings in the parallel program.
  parallelProgram.m_Universal_globalNDRange = ndKernelRange;
  parallelProgram.m_Universal_offsetNDRange = ndKernelOffset;
  parallelProgram.m_Process_globalNDRange = ndKernelRange;
  parallelProgram.m_Process_offsetNDRange = ndKernelOffset;
}

/// provides an implementation of the virtual method @c configureTuneParallelProgram.
void pFacesExample::configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}
}  // namespace pFacesExamples

/// register the kernel
PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::pFacesExample)
