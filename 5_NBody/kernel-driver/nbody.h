#ifndef PFACES_EXAMPLE_NBODY_H_
#define PFACES_EXAMPLE_NBODY_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

/** State shared between NBodyKernel setup, VM loop callback, and verifier. */
struct NBodyState {
  size_t numBodies = 1024U;
  size_t steps = 10U;
  size_t numPairs = 5U;
  size_t completedPairs = 0U;
  float dt = 0.01f;
  float softening = 0.2f;
  size_t wgSize = 64U;
  size_t posABufIdx = 0U;
  size_t velBufIdx = 0U;
};

/** Tiled all-pairs 3D gravitational N-body kernel with VM ping-pong loop. */
class NBodyKernel : public pfaces2DKernel {
public:
  NBodyKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~NBodyKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

  /** Advances the VM loop pair counter; returns 1 to repeat or 0 to finish. */
  size_t advanceLoopPair();

private:
  std::shared_ptr<NBodyState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_NBODY_H_ */
