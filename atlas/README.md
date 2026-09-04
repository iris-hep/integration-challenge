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

To register the pixi env as a local kernel to be selected from the notebook (top-right corner)

```bash
pixi run python -m ipykernel install --user \
    --name some-name \
    --display-name "What you see in the notebook"
```

### Triton Server

If running inference on the Triton server, this is already installed on the [AF Triton deployment](https://usatlas.github.io/af-docs/uchicago/triton/?h=Triton#4-request-model-activation). The model installed here is called `jet_network_batch` and corresponds directly to the model used in the `onnx` deployment. 

Loading the model is done via `MLModel(use_triton=True).load()` which loads the Triton model and prepares it for inference. Inference is run simply with `MLModel(use_triton=True).run(events)`. Once inference is complete, unload the model with `MLModel(use_triton=True).cleanup_triton()`. 

During inference, the Triton server can be monitored via the [AF Grafana instance](https://grafana.af.uchicago.edu/d/02cfc53d-de9b-4c53-84ee-e60b29a0b76e/triton-inference-server?orgId=1&from=now-1h&to=now&timezone=utc&var-datasource=beken5g2xymm8c&var-cluster=af&refresh=10s).