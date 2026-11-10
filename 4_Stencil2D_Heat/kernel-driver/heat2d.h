#ifndef PFACES_EXAMPLE_HEAT2D_H_
#define PFACES_EXAMPLE_HEAT2D_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

/** State shared between Heat2DKernel setup, VM loop callback, and verifier. */
struct Heat2DState {
  size_t width = 128U;
  size_t height = 128U;
  size_t iterations = 100U;
  size_t numPairs = 50U;
  size_t completedPairs = 0U;
  float diffusion = 0.2f;
  size_t wgX = 16U;
  size_t wgY = 16U;
  size_t gridABufIdx = 0U;
};

/** 2D 5-point Jacobi heat-diffusion kernel using VM ping-pong loop control. */
class Heat2DKernel : public pfaces2DKernel {
public:
  Heat2DKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~Heat2DKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

  /** Advances the VM loop pair counter; returns 1 to repeat or 0 to finish. */
  size_t advanceLoopPair();

private:
  std::shared_ptr<Heat2DState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_HEAT2D_H_ */
