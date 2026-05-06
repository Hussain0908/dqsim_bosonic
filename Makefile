.PHONY: build publish test

DIST_DIR := dist
REPOSITORY ?= testpypi

ifeq ($(REPOSITORY),testpypi)
PUBLISH_URL := https://test.pypi.org/legacy/
else ifeq ($(REPOSITORY),pypi)
PUBLISH_URL := https://upload.pypi.org/legacy/
else
$(error Unsupported REPOSITORY '$(REPOSITORY)'; use REPOSITORY=testpypi or REPOSITORY=pypi)
endif

test:
	uv run --python 3.11 --extra test pytest tests/ -v -s

perf:
	uv run --python 3.11 --extra test pytest examples/test_performance.py -v -s

build:
	rm -rf $(DIST_DIR)
	uvx --from build pyproject-build --outdir $(DIST_DIR)
	uvx twine check $(DIST_DIR)/*

publish: build
	uvx twine upload --repository-url $(PUBLISH_URL) $(DIST_DIR)/*
