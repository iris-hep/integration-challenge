import glob

import GridSubmission.grid
import GridSubmission.ami

import input_containers

config = GridSubmission.grid.Config()
config.outputName = "output"
config.gridUsername = "alheld"
config.suffix = "IC-v1.2"
config.excludedSites = ""
config.noSubmit = False
config.mergeType = "Default"
config.destSE = "NET2_LOCALGROUPDISK"
config.reuseTarBall = True
# config.otherOptions = ""

# data submission
# config.code = "runTop_el.py -t integration-challenge --no-systematics"
# categories = GridSubmission.grid.Samples(["data"])
# GridSubmission.grid.submit(config, categories)

if glob.glob("top-el.tar.gz"):
    input("\nINFO: reusing tarball, enter to confirm")

# MC submission
config.code = "runTop_el.py -t integration-challenge"
config.maxNFilesPerJob = "4"
names = [k for k in input_containers.containers.keys() if k != "data" and k in ["rare_top"]]

categories = GridSubmission.grid.Samples(names)
GridSubmission.ami.check_sample_status(categories, False)  # stop on error if True
GridSubmission.grid.submit(config, categories)
