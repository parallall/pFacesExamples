#include "example.h"

namespace pFacesExamples {

/// Some constants
// The name of the kernel source file without the .cl extension.
const std::string kernel_source = "example";
// The name of the kernel memory footprint file.
const std::string mem_cfg_file = "example.mem";
// The 2-dim size of the ND Range.
constexpr size_t thread_grid_x{1U};
constexpr size_t thread_grid_y{1U};

/// Constructor for the example class.
pFacesExample::pFacesExample(
  const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
  const std::shared_ptr<pfacesConfigurationReader>& spCfg)
  : pfaces2DKernel(
      spLaunchState->getDefaultSourceFilePath(kernel_source, spLaunchState->kernelScope, spLaunchState->kernelPackPath),
      thread_grid_x,
      thread_grid_y,
      spCfg) {

  // Full path of the memory config file.
  const std::string memFile = spLaunchState->kernelPackPath + mem_cfg_file;

  // Create a kernel function and load its memory fingerprints
  const std::string funcName = "exampleKernelFunction";
  const std::vector<std::string> argNames{"data_in"};
  const auto funcArgs = pfacesKernelFunctionArguments::loadFromFile(memFile, funcName, argNames);
  const pfacesKernelFunction theKernelFunction(funcName, funcArgs);
  addKernelFunction(theKernelFunction);

  // Add a parameter that can be accessed from the .cl file.
  updateParameters({"@@param_message@@"}, {"\"Hello from the driver side!\""});
}

/// Provides an implementation of the virtual method @c configureParallelProgram.
void pFacesExample::configureParallelProgram(pfacesParallelProgram& parallelProgram) {

  // Check we run within only one device.
  PFACES_ASSERT_CONTEXT(parallelProgram.countTargetDevices() > 1, "This example supports only one target device.");

  // Get target machine/device.
  const auto& thisMachine = parallelProgram.getMachine();
  const auto& thisDevice = parallelProgram.getTargetDevices()[0];

  // Define the range/offset for this kernel.
  const cl::NDRange ndKernelRange(thread_grid_x, thread_grid_y);
  const cl::NDRange ndKernelOffset(0U, 0U);

  /* allocate memory */
  std::vector<std::pair<char*, size_t>> dataPool;
  pFacesMemoryAllocationReport memReport;
  memReport = allocateMemory(dataPool, thisMachine, parallelProgram.getTargetDevicesIndicies(), 1, false);

  /* the parallel program needs a alist of instructions */
  std::vector<std::shared_ptr<pfacesInstruction>> instrList;

  /* an instruction to show a simple message from the CPU side */
  std::shared_ptr<pfacesInstruction> instrMsg_start = std::make_shared<pfacesInstruction>();
  instrMsg_start->setAsMessage("Hello World from host side !");

  /* an instruction to lauch parallel threadss in the device */
  std::shared_ptr<pfacesDeviceExecuteTask> singleTask =
    std::make_shared<pfacesDeviceExecuteTask>(ndKernelOffset, ndKernelRange, ndKernelRange);
  std::shared_ptr<pfacesDeviceExecuteJob> exampleFunctionJob = std::make_shared<pfacesDeviceExecuteJob>(thisDevice);
  exampleFunctionJob->addTask(singleTask);
  exampleFunctionJob->setKernelFunctionIdx(
    0, /* index of the kernel function we're to write data to */
    1  /* how many arguments it has */
  );
  std::shared_ptr<pfacesInstruction> devFunctionInstruction = std::make_shared<pfacesInstruction>();
  devFunctionInstruction->setAsDeviceExecute(exampleFunctionJob);

  /* a synchronization instruction */
  std::shared_ptr<pfacesInstruction> instrSyncPoint = std::make_shared<pfacesInstruction>();
  instrSyncPoint->setAsBlockingSyncPoint();

  /* build the parallel program */
  instrList.push_back(instrMsg_start);
  instrList.push_back(devFunctionInstruction);
  instrList.push_back(instrSyncPoint);

  /* complete the required settings in the parallel program */
  parallelProgram.m_Universal_globalNDRange = ndKernelRange;
  parallelProgram.m_Universal_offsetNDRange = ndKernelOffset;
  parallelProgram.m_dataPool = dataPool;
  parallelProgram.m_spInstructionList = instrList;
}

/// provides an implementation of the virtual method @c configureTuneParallelProgram.
void pFacesExample::configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) {
  std::ignore = tuneParallelProgram;
  std::ignore = targetFunctionIdx;
}
}  // namespace pFacesExamples

/// register the kernel
PFACES_REGISTER_LOADABLE_KERNEL(pFacesExamples::pFacesExample)
