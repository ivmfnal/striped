
BUILD_DIR = $(HOME)/build/striped
TARDIR = /tmp/$(USER)
TARFILE = $(TARDIR)/striped_$(VERSION).tar

all:
	make VERSION=`python version.py` all_with_version_defined
        
all_with_version_defined: tarball

build: $(BUILD_DIR) build_striped_tools striped
	cd data_server; make DSTDIR=$(BUILD_DIR)/data_server build
	cd job_server; make DSTDIR=$(BUILD_DIR)/job_server build
	cd worker; make DSTDIR=$(BUILD_DIR)/worker build
	cd striped; make DSTDIR=$(BUILD_DIR)/striped build
        
tarball: clean $(TARDIR) build
	cd $(BUILD_DIR); tar cf $(TARFILE) *
	@echo
	@echo Tar file $(TARFILE) is ready
	@echo

clean:
	rm -rf $(BUILD_DIR)
        
$(BUILD_DIR):
	mkdir -p $@
        
$(TARDIR):
	mkdir -p $@

build_striped_tools: stripe_tools/stripe_tools.c
	python build_stripe_tools.py build --build-lib $(BUILD_DIR)

$(DSTDIR):
	mkdir -p $@

	
