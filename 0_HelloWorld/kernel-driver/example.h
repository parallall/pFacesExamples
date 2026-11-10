/*
 * example.h
 *
 *  Created on: 01.04.2020
 *      Author: M. Khaled
 */

#ifndef PFACES_KERNEL_EXAMPLE_H_
#define PFACES_KERNEL_EXAMPLE_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

struct HelloWorldState {
  size_t dataOutBufIdx = 0U;
};

/// A class to represent the example kernel driver.
class pFacesExample : public pfaces2DKernel {
public:
  pFacesExample(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~pFacesExample() = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram);

  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx);

private:
  std::shared_ptr<HelloWorldState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_KERNEL_EXAMPLE_H_ */
