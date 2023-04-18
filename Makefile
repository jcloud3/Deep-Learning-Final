

# Looks like some packages can't be installed via conda
.PHONY: conda_env
conda_env: # Build conda environment
conda_env: environment.yaml
	conda env create -f environment.yaml


