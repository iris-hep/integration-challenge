import GridSubmission.grid
import GridSubmission.ami

import input_containers

config = GridSubmission.grid.Config()
config.code = "runTop_el.py -t integration-challenge" # --no-systematics
config.outputName = "output"
config.gridUsername = "alheld"
config.suffix = "test-v1"
config.excludedSites = ""
config.noSubmit = False
config.mergeType = "Default"
config.destSE = "NET2_LOCALGROUPDISK"
# config.otherOptions = ""
# config.maxNFilesPerJob = "4"

names = ["mctest", "datatest"]
samples = GridSubmission.grid.Samples(names)
GridSubmission.ami.check_sample_status(
    samples, False
)  # (samples, True) to halt on error
GridSubmission.grid.submit(config, samples)
