SUBDIRS = kernel-driver

.PHONY: all clean

all: check-pfaces-sdk
	for dir in $(SUBDIRS); do $(MAKE) all -C $$dir $@; done

clean:
	rm -f ./kernel-pack/*.log
	rm -f ./kernel-pack/*.render
	rm -f ./kernel-pack/*.driver
	rm -f ./kernel-pack/*.dll
	rm -f ./kernel-pack/*.so
	rm -f ./kernel-pack/*.exp
	rm -f ./kernel-pack/*.ilk
	rm -f ./kernel-pack/*.lib
	rm -f ./kernel-pack/*.pdb
	rm -f ./kernel-pack/*.ipdb
	rm -f ./kernel-pack/*.iobj
	rm -rf x64
	rm -rf Release
	rm -rf Debug
	find . -name "*.raw" -type f -delete
	find . -name "*.asv" -type f -delete
	find . -name "*.exe" -type f -delete
	find . -name "*.prism" -type f -delete
	for dir in $(SUBDIRS); do $(MAKE) clean -C $$dir $@; done


check-pfaces-sdk:
ifeq ($(PFACES_SDK_ROOT),)
	$(error PFACES_SDK_ROOT was not found. Please assign PFACES_SDK_ROOT to the root directory of pFaces SDK)
endif
