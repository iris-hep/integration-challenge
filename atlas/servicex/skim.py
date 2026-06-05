from servicex import deliver, query
import gzip
import json
from servicex_analysis_utils import ds_type_resolver, to_awk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils

fname = "ntuple_production/file_metadata_v2.json.gz"
with gzip.open(fname) as f:
    dataset_info = json.loads(f.read().decode())


total_sig=0
signal_containers={}
for container, metadata in dataset_info["Hplus_cb"].items():
    evts = metadata["nevts_input"]
    total_sig+=evts
    mass = utils.hplus_signal_mass(container)
    _ , _, campaign = utils.dsid_rtag_campaign(container)
    signal_containers[f"{mass}_{campaign}"]={'DSID' : container, 'Events': evts }
print(f"H+ -> cb signal in {len(signal_containers)} DSIDs for {total_sig:.0e} total events")

sample = ds_type_resolver(signal_containers['20GeV_mc20e']['DSID'])
print(sample)

config_path = "ntuple_production/reco.yaml"
request = query.TopCP(reco=config_path, max_events = 200_000)
print(request)

spec = {
        "General": {"OutputDirectory": "."},
        "Sample": [{
            "Name": '20GeV_mc20e',
            "Dataset": sample,
            "Query": request,
        }]
    }
files = deliver(spec, config_path='/etc/reana/secrets/.servicex')
print(files)
