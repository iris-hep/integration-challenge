# IRIS-HEP ATLAS integration challenge

## Step 1: TopCPToolkit

This step produces CP algorithm ntuples with TopCPToolkit from ATLAS PHYSLITE samples.
See the `ntuple_production/` folder for more information.
This step runs on the grid.
Once completed, `collect_file_metadata.py` aggregates all relevant metadata for subsequent processing.


## Step 2: coffea for ntuple processing

The second step produces histograms from the ntuples.
This uses coffea and Dask-distributed processing at an Analysis Facility.
Use `analysis.ipynb` for this step, which is instrumented with utilities to extract metrics for processing.

### Register the pixi env as a Jupyter kernel:

To register the pixi env as a local kernel to be selected from the notebook (top-rigth corner)

```bash
pixi run python -m ipykernel install --user \
    --name some-name \
    --display-name "What you see in the notebook"
```

### Start Triton server locally

To run a template version of Triton locally (running on the CPU), the following command may be used:

```
apptainer run \
    -B triton-models:/models  \
    /data/milescb/apptainer/triton_sandbox  \
    tritonserver --model-repository=/models
```

To ensure everything is working while running inference, try `curl -s localhost:8002/metrics | grep -E "nv_inference_request_success"`