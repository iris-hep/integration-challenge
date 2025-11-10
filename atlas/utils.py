# saving preprocessing output
import asyncio
import base64
import dataclasses
import datetime
import re
import urllib.request

import coffea
import matplotlib.pyplot as plt


##################################################
### Dask task tracking
##################################################

def start_tracking(dask_scheduler) -> None:
    """"run on scheduler to track worker count"""
    dask_scheduler.worker_counts = {}
    dask_scheduler.track_count = True

    async def track_count() -> None:
        while dask_scheduler.track_count:
            dask_scheduler.worker_counts[datetime.datetime.now()] = len(dask_scheduler.workers)
            await asyncio.sleep(1)

    asyncio.create_task(track_count())


def stop_tracking(dask_scheduler) -> dict:
    """obtain worker count and stop tracking"""
    dask_scheduler.track_count = False
    return dask_scheduler.worker_counts


def get_avg_num_workers(worker_count_dict: dict) -> float:
    """get time-averaged worker count"""
    worker_info = list(worker_count_dict.items())
    nworker_dt = 0
    for (t0, nw0), (t1, nw1) in zip(worker_info[:-1], worker_info[1:]):
        nworker_dt += (nw1 + nw0) / 2 * (t1 - t0).total_seconds()
    return nworker_dt / (worker_info[-1][0] - worker_info[0][0]).total_seconds()


def plot_worker_count(worker_count_dict: dict):
    """plot worker count over time"""
    fig, ax = plt.subplots()
    ax.plot(worker_count_dict.keys(), worker_count_dict.values())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel("time")
    ax.set_ylabel("number of workers")
    return fig, ax


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
        return "data", None, data_tag[0]

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


def integrated_luminosity(campaign: str, total=False) -> float:
    """get integrated luminosity in pb for each MC campaign"""
    if "data" in campaign:
        return 1.0

    lumi_dict = {
        "mc20a": 3244.54 + 33402.2,
        "mc20d": 44630.6,
        "mc20e": 58791.6,
        "mc23a": 26328.8,
        "mc23d": 25204.3,
        "mc23e": 109376.0,
    }

    if total:
        return sum(lumi_dict.values())

    return lumi_dict[campaign]


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
            try:
                with open("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc16.txt") as f:
                    content = f.readlines()
            except FileNotFoundError:
                print("falling back to reading cross-section information over https")
                content = urllib.request.urlopen("https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/PMGxsecDB_mc16.txt").read().decode().split("\n")

            MC16_XSEC_DICT = dict([(line.strip().split("\t\t")[0], (line.split("\t\t")[2:5])) for line in content[1:]])

        xsec_dict = MC16_XSEC_DICT

    elif "mc23" in campaign:
        if MC23_XSEC_DICT is None:
            try:
                with open("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/dev/PMGTools/PMGxsecDB_mc23.txt") as f:
                    content = f.readlines()
            except FileNotFoundError:
                print("falling back to reading cross-section information over https")
                content = urllib.request.urlopen("https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/PMGTools/PMGxsecDB_mc23.txt").read().decode().split("\n")

            MC23_XSEC_DICT = dict([(line.strip().split("\t\t")[0], (line.split("\t\t")[2:5])) for line in content[1:]])

        xsec_dict = MC23_XSEC_DICT

    elif "data" in campaign:
        return 1.0

    else:
        raise ValueError(f"cannot parse campaign {campaign}")

    # return x-sec [pb] * filter efficiency * k-factor
    return float(xsec_dict[dsid][0]) * float(xsec_dict[dsid][1]) * float(xsec_dict[dsid][2])
