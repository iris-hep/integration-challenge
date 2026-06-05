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


# Optional: Run the pipeline in Reana
At some point, we want to be able to run the pipeline in Reana.
This is a work in progress.

1. Create an account on the [U Chicago Reana platform](https://reana.af.uchicago.edu/) and obtain a personal access token.
2. [Install Reana](https://docs.reana.io/getting-started/first-example/) in a virtual environment.
3. Create a custom `.servicex` file in the `reana` folder.
```yaml
default_endpoint: servicex-uc-af
api_endpoints:
  - endpoint: https://servicex.af.uchicago.edu
    name: servicex-uc-af
    token: <<Obtain token from AF ServiceX Dashboard>>
cache_path: ./cache-dir
```
4. Install this file as a secret in your Reana workspace.
5. `reana-client secrets-add --file .servicex`
6. Run a workflow instance with `reana-client run -w IC`
7. Monitor the workflow with `reana-client status -w IC`
8. View the workflow output with `reana-client logs -w IC`