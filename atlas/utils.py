# saving preprocessing output
import base64
import dataclasses
import re

import coffea


##################################################
### fileset saving / loading
##################################################

def preprocess_to_json(samples):
    # encode bytes
    serializable = []
    for s in samples:
        chunk = dataclasses.asdict(s)
        chunk["fileuuid"] = base64.b64encode(chunk["fileuuid"]).decode("ascii")
        serializable.append(chunk)

    return serializable


def json_to_preprocess(samples):
    # decode bytes
    for i in range(len(samples)):
        samples[i]["fileuuid"] = base64.b64decode(samples[i]["fileuuid"])
        samples[i] = coffea.processor.executor.WorkItem(**samples[i])

    return samples


##################################################
### container and sample handling
##################################################

def dsid_rtag_campaign(name: str) -> tuple[str, str, str]:
    """extract information from container name"""
    data_tag = re.findall(r":(data\d+)_", name)
    if data_tag:
        return None, None, data_tag[0]

    dsid = re.findall(r".(\d{6}).", name)[0]
    rtag = re.findall(r"_(r\d+)_", name)[0]

    if rtag in ["r13167", "r14859", "r13297", "r14862"]:
        campaign = "mc20a"
    elif rtag in ["r13144", "r14860", "r13298", "r14863"]:
        campaign = "mc20d"
    elif rtag in ["r13145", "r14861", "r13299", "r14864"]:
        campaign = "mc20e"
    elif rtag in ["r14622", "r15540"]:
        campaign = "mc23a"
    elif rtag in ["r15224", "r15530"]:
        campaign = "mc23d"
    elif rtag in ["r16083"]:
        campaign = "mc23e"
    else:
        print("cannot classify", name)
        campaign = None

    return dsid, rtag, campaign


def integrated_luminosity(campaign: str) -> float:
    """get integrated luminosity in pb for each MC campaign"""
    lumi = {
        "mc20a": 3244.54 + 33402.2,
        "mc20d": 44630.6,
        "mc20e": 58791.6,
        "mc23a": 26328.8,
        "mc23d": 25204.3,
        "mc23e": 109376.0
    }[campaign]
    return lumi


# cache for large x-sec information dicts
MC16_XSEC_DICT = None
MC23_XSEC_DICT = None


def sample_xs(campaign: str, dsid: str) -> float:
    """get product of sample cross-section, filter efficiency and k-factor"""
    global MC16_XSEC_DICT
    global MC23_XSEC_DICT

    # extracting this information is expensive, so do it once and cache
    if "mc20" in campaign:
        if MC16_XSEC_DICT is None:
            # in case of no cvmfs: https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/PMGxsecDB_mc16.txt
            with open("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt") as f:
                content = f.readlines()

            MC16_XSEC_DICT = dict([(line.split("\t\t")[0], (line.split("\t\t")[2:5])) for line in content[1:]])

        xsec_dict = MC16_XSEC_DICT

    elif "mc23" in campaign:
        if MC23_XSEC_DICT is None:
            # in case of no cvmfs: https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/PMGxsecDB_mc23.txt
            with open("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc23.txt") as f:
                content = f.readlines()

            MC23_XSEC_DICT = dict([(line.split("\t\t")[0], (line.split("\t\t")[2:5])) for line in content[1:]])

        xsec_dict = MC23_XSEC_DICT

    else:
        raise ValueError("cannot parse campaign")

    # return x-sec [pb] * filter efficiency * k-factor
    return float(xsec_dict[dsid][0]) * float(xsec_dict[dsid][1]) * float(xsec_dict[dsid][2])
