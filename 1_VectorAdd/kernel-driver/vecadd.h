#ifndef PFACES_EXAMPLE_VECADD_H_
#define PFACES_EXAMPLE_VECADD_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

struct VecAddState {
  size_t length = 32768U;
  float alpha = 1.75f;
};

class VectorAddKernel : public pfaces2DKernel {
public:
  VectorAddKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~VectorAddKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

private:
  std::shared_ptr<VecAddState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_VECADD_H_ */
