SHELL := /bin/bash

.PHONY: visualize
visualize: ## Visualize inference predictions
	streamlit run ./scripts/dashbord.py

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ \
	{ printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }\
	 ' $(MAKEFILE_LIST)