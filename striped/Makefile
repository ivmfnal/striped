build: $(DSTDIR)
	cd ml; make DSTDIR=$(DSTDIR)/ml build
	cd job; make DSTDIR=$(DSTDIR)/job build
	cd hist; make DSTDIR=$(DSTDIR)/hist build
	cd common; make DSTDIR=$(DSTDIR)/common build
	cd client; make DSTDIR=$(DSTDIR)/client build
	cd ingestion; make DSTDIR=$(DSTDIR)/ingestion build
	cp -R pythreader $(DSTDIR)
	cp version.py __init__.py $(DSTDIR)
	
$(DSTDIR):
	mkdir -p $@
	
	
	
