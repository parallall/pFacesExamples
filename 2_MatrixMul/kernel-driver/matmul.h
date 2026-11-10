#ifndef PFACES_EXAMPLE_MATMUL_H_
#define PFACES_EXAMPLE_MATMUL_H_

#include <pfaces-sdk.h>
#include <cstddef>
#include <memory>

namespace pFacesExamples {

struct MatMulState {
  size_t m = 64U;
  size_t k = 64U;
  size_t n = 64U;
  size_t tileSize = 16U;
  size_t padM = 64U;
  size_t padN = 64U;
};

class MatrixMulKernel : public pfaces2DKernel {
public:
  MatrixMulKernel(
    const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState,
    const std::shared_ptr<pfacesConfigurationReader>& spCfg);

  ~MatrixMulKernel() override = default;

  void configureParallelProgram(pfacesParallelProgram& parallelProgram) override;
  void configureTuneParallelProgram(pfacesParallelProgram& tuneParallelProgram, size_t targetFunctionIdx) override;

private:
  std::shared_ptr<MatMulState> m_state;
};

}  // namespace pFacesExamples

#endif /* PFACES_EXAMPLE_MATMUL_H_ */
