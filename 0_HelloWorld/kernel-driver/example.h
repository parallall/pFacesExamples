/*
 * example.h
 *
 *  Created on: 01.04.2020
 *      Author: M. Khaled
 */

#ifndef PFACES_KERNEL_EXAMPLE_H_
#define PFACES_KERNEL_EXAMPLE_H_

#include <pfaces-sdk.h>

namespace pFacesExamples{

	// class: pFacesExample
	class pFacesExample : public pfaces2DKernel {
	private:
		const std::shared_ptr<pfacesConfigurationReader> m_spCfg;
	
	public:
		pFacesExample(
			const std::shared_ptr<pfacesKernelLaunchState>& spLaunchState, 
			const std::shared_ptr<pfacesConfigurationReader>& spCfg);
			
		~pFacesExample() = default;

		void configureParallelProgram(
			pfacesParallelProgram& parallelProgram);

		void configureTuneParallelProgram(
			pfacesParallelProgram& tuneParallelProgram, 
			size_t targetFunctionIdx);

	};

}

#endif /* PFACES_KERNEL_EXAMPLE_H_ */
