#ifndef PFACES_EXAMPLE_HISTOGRAM_H_
#define PFACES_EXAMPLE_HISTOGRAM_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

/** State shared between HistogramKernel configuration and verifier. */
struct HistogramState {
  static constexpr size_t kNumBins = 256U;
  size_t numElements = 262144U;
  size_t wgSize = 64U;
  size_t numGroups = 64U;
  size_t inDataBufIdx = 0U;
  size_t binsBufIdx = 0U;
};

/** 256-bin parallel histogram kernel using __local privatization and atomics. */
class HistogramKernel : public pfaces2DKernel {
public:
  HistogramKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~HistogramKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

private:
  std::shared_ptr<HistogramState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_HISTOGRAM_H_ */
