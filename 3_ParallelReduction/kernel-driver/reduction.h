#ifndef PFACES_EXAMPLE_REDUCTION_H_
#define PFACES_EXAMPLE_REDUCTION_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

struct ReductionState {
  size_t numElements = 65536U;
  size_t wgSize = 64U;
  size_t numGroups = 64U;
  size_t inDataBufIdx = 0U;
  size_t finalSumBufIdx = 0U;
};

class ReductionKernel : public pfaces2DKernel {
public:
  ReductionKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~ReductionKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

private:
  std::shared_ptr<ReductionState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_REDUCTION_H_ */
