# saving preprocessing output
import asyncio
import base64
import dataclasses
import datetime
import functools
import gzip
import json
import math
import os
import re
import time
from typing import Optional
import urllib.request

import coffea.nanoevents
import coffea.processor
import dask.bag
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tqdm.notebook
import uproot


##################################################
### output folder
##################################################

def _create_output_path():
    if not os.path.exists("output"):
        os.makedirs("output")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"output/results-{timestamp}"
    if not os.path.exists(name):
        os.makedirs(name)
    print(f"output stored in {name}")
    return name


_OUTPUT_PATH = _create_output_path()


def new_output_path():
    global _OUTPUT_PATH
    _OUTPUT_PATH = _create_output_path()
    return _OUTPUT_PATH


def get_output_path():
    return _OUTPUT_PATH


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


def calculate_instantaneous_rates(t0: float, t1: float, report: dict, num_samples: int = 10):
    """calculate chunk-aggregated data rates in Gbps over time"""
    if "chunk_info" not in report:
        return None, None  # supported only for custom processing

    chunk_info = np.asarray(list(report["chunk_info"].values()))
    starts = chunk_info[:, 0]
    ends = chunk_info[:, 1]
    bytesread = chunk_info[:, 2]
    rates_per_chunk = bytesread / (ends - starts)

    t_samples = np.linspace(t0, t1, num_samples)
    rate_samples = []
    for t in t_samples:
        mask = np.logical_and((starts <= t), (t < ends))
        rate_samples.append(float(sum(rates_per_chunk[mask]) / 1000**3 * 8))

    print(f"[INFO] total data read from data rate integral: {sum((t_samples[1] - t_samples[0]) * np.asarray(rate_samples)) / 8:.2f} GB")
    return [datetime.datetime.fromtimestamp(t) for t in t_samples.tolist()], rate_samples


def plot_worker_count(worker_count_dict: dict, timestamps: Optional[list[float]], datarates: Optional[list[float]]):
    """plot worker count over time and data rate samples in Gbps"""
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(worker_count_dict.keys(), worker_count_dict.values(), linewidth=2, color="C0")
    ax1.set_title("worker count and data rate over time")
    ax1.set_xlabel("time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.set_ylabel("number of workers", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim([0, ax1.get_ylim()[1] * 1.1])

    if datarates is not None:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, datarates, marker="v", linewidth=0, color="C2")
        ax2.set_ylabel("data rate [Gbps]", color="C2")
        ax2.set_ylim([0, ax2.get_ylim()[1] * 1.1])
        ax2.tick_params(axis="y", labelcolor="C2")

    fig.savefig(f"{get_output_path()}/datarate.pdf")
    return fig, ax1


def plot_taskstream(ts: dict):
    """simplified version of Dask html report task stream"""
    fig, ax = plt.subplots(constrained_layout=True)
    t0 = min(min(t["start"] for t in ts_["startstops"]) for ts_ in ts)
    tmax = max(max(t["start"] for t in ts_["startstops"]) for ts_ in ts) - t0
    y_next = 0
    worker_pos = {}
    for task in ts:
        # get y position for worker or create new one for new worker
        y_pos = worker_pos.get(task["worker"], None)
        if y_pos is None:
            y_pos = y_next
            worker_pos[task["worker"]] = y_pos
            y_next += 1

        for subtask in task["startstops"]:
            if subtask["action"] != "compute":
                continue
            start = subtask["start"] - t0
            stop = subtask["stop"] - t0
            c = "yellow" if subtask["action"] == "compute" else "red"
            patch = mpl.patches.Rectangle((start, y_pos - 0.4), stop - start, 0.8, facecolor=c, edgecolor="black")
            ax.add_patch(patch)

    ax.set_xlim(0, tmax)
    ax.set_ylim(-0.5, y_next - 0.5)
    ax.set_xlabel("time [sec]")
    ax.set_ylabel("unique workers")
    fig.savefig(f"{get_output_path()}/taskstream.pdf")
    return fig, ax


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

def hplus_signal_mass(name: str) -> str:
    """get the mass from these signal samples"""
    m = re.search(r'(?i)mhc(\d+)(?=\.)', name)  # case-insensitive, stop at the dot
    return str(m.group(1)+"GeV") if m else None


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


##################################################
### fileset handling
##################################################

def get_fileset(campaign_filter: list | None = None, dsid_filter: list | None = None, max_files_per_sample: int | None = None, version: str = "v2"):
    """prepare fileset, with possibility to only include subset of input files"""
    # load metadata from file
    fname = f"ntuple_production/file_metadata_{version}.json.gz"
    with gzip.open(fname) as f:
        dataset_info = json.loads(f.read().decode())

    # construct fileset
    fileset = {}
    input_size_GB = 0
    for category, containers_for_category in dataset_info.items():
        for container, metadata in containers_for_category.items():
            if metadata["files_output"] is None:
                # print(f"[DEBUG] skipping missing {container}")
                continue

            dsid, _, campaign = dsid_rtag_campaign(container)

            # debugging shortcuts / reducing workload
            if campaign_filter is not None and campaign not in campaign_filter:
                continue

            if dsid_filter is not None and dsid not in dsid_filter:
                continue

            weight_xs = sample_xs(campaign, dsid)
            lumi = integrated_luminosity(campaign)
            num_files = len(metadata["files_output"]) if max_files_per_sample is None else max_files_per_sample
            fileset[container] = {
                "files": dict((path, "reco") for path in metadata["files_output"][:num_files]),
                "metadata": {"dsid": dsid, "campaign": campaign, "category": category, "weight_xs": weight_xs, "lumi": lumi}
            }
            input_size_GB += sum(metadata["sizes_output_GB"][:num_files])

    print(f"[INFO] fileset has {len(fileset)} categories with {sum([len(f["files"]) for f in fileset.values()])} files total, size is {input_size_GB:.2f} GB")
    return fileset, input_size_GB


def preprocess_to_json(samples):
    """encode bytes"""
    serializable = []
    for s in samples:
        chunk = dataclasses.asdict(s)
        chunk["fileuuid"] = base64.b64encode(chunk["fileuuid"]).decode("ascii")
        serializable.append(chunk)

    return serializable


def json_to_preprocess(samples):
    """decode bytes"""
    for i in range(len(samples)):
        samples[i]["fileuuid"] = base64.b64decode(samples[i]["fileuuid"])
        samples[i] = coffea.processor.executor.WorkItem(**samples[i])

    return samples


##################################################
### custom pre-processing
##################################################

def custom_preprocess(fileset: dict, *, client, chunksize: int = 100_000, custom_func=None):
    """Dask-based pre-processing similar to coffea, can run user-provided function for more metadata"""
    files_to_preprocess = []
    for category_info in fileset.values():
        files_to_preprocess += [(k, v) for k, v in category_info["files"].items()]

    def extract_metadata(fname_and_treename: str, custom_func) -> dict:
        """read file and extract relevant information"""
        fname, treename = fname_and_treename
        meta = {"treename": treename}
        with uproot.open(fname) as f:
            meta["fileuuid"] = f.file.uuid.bytes
            meta["num_entries"] = f[treename].num_entries if treename in f else 0  # handle missing trees
            print("calling custom_func", custom_func, "on file", f)
            if custom_func:
                meta.update({"custom_meta": custom_func(f)})
        return {fname: meta}

    print(f"pre-processing {len(files_to_preprocess)} file(s)")
    tasks = client.map(functools.partial(extract_metadata, custom_func=custom_func), files_to_preprocess)
    futures = client.compute(tasks)

    with tqdm.notebook.tqdm(total=len(futures)) as pbar:
      for _ in dask.distributed.as_completed(futures):
        pbar.update(1)

    # turn into dict for easier use
    result_dict = {k: v for res in [f.result() for f in futures] for k, v in res.items()}

    # join back together per-file information with fileset-level information and turn into WorkItem list for coffea
    workitems = []
    for category_name, category_info in fileset.items():
        for fname, treename in category_info["files"].items():
            preprocess_meta = result_dict[fname]
            # split into chunks as done in coffea, taken from
            # https://github.com/scikit-hep/coffea/blob/f7e1249745484567d1e380865dc05fae83165084/src/coffea/dataset_tools/preprocess.py#L129-L140
            n_steps_target = max(round(preprocess_meta["num_entries"] / chunksize), 1)
            actual_step_size = math.ceil(preprocess_meta["num_entries"] / n_steps_target)
            chunks = np.array(
                [
                    [
                        i * actual_step_size,
                        min((i + 1) * actual_step_size, preprocess_meta["num_entries"]),
                    ]
                    for i in range(n_steps_target)
                ],
                dtype="int64",
            )
            for entry_start, entry_stop in chunks:
                if entry_stop - entry_start == 0:
                    continue
                workitems.append(
                    coffea.processor.executor.WorkItem(
                        dataset=category_name,
                        filename=fname,
                        treename=treename,
                        entrystart=int(entry_start),
                        entrystop=int(entry_stop),
                        fileuuid=preprocess_meta["fileuuid"],
                        usermeta=category_info["metadata"] | preprocess_meta["custom_meta"]
                    )
                )

    return workitems


##################################################
### custom processing
##################################################

def custom_process(workitems, processor_class, schema, client, preload: Optional[list] = None):
    """Dask-based processing similar to coffea, can return more metadata"""
    if preload is None:
        preload = []

    def run_analysis(wi: coffea.processor.executor.WorkItem):
        """workload to be distributed"""
        t0 = time.time()
        analysis_instance = processor_class()
        array_cache = {}
        f = uproot.open(wi.filename, array_cache=array_cache)
        events = coffea.nanoevents.NanoEventsFactory.from_root(
            f,
            treepath=wi.treename,
            mode="virtual",
            access_log=(access_log := []),
            preload=lambda b: b.name in preload,
            schemaclass=schema,
            entry_start=wi.entrystart,
            entry_stop=wi.entrystop,
        ).events()
        events.metadata.update(wi.usermeta)
        out = analysis_instance.process(events)
        bytesread = f.file.source.num_requested_bytes
        t1 = time.time()
        report = {
            "bytesread": bytesread,
            "entries": wi.entrystop - wi.entrystart,
            "processtime": t1 - t0,
            "chunks": 1,
            "columns": access_log,
            "chunk_info": {(wi.filename, wi.entrystart, wi.entrystop): (t0, t1, bytesread)},
        }
        return out, report

    def sum_output(a, b):
        """accumulation function"""
        return (
            {"hist": a[0]["hist"] + b[0]["hist"]},
            {
                "bytesread": a[1]["bytesread"] + b[1]["bytesread"],
                "entries": a[1]["entries"] + b[1]["entries"],
                "processtime": a[1]["processtime"] + b[1]["processtime"],
                "chunks": a[1]["chunks"] + b[1]["chunks"],
                "columns": list(set(a[1]["columns"]) | set(b[1]["columns"])),
                "chunk_info": a[1]["chunk_info"] | b[1]["chunk_info"],
            }
        )

    workitems_bag = dask.bag.from_sequence(workitems, partition_size=1)
    tasks = workitems_bag.map(run_analysis).to_delayed()
    futures = client.compute(tasks, rerun_exceptions_locally=True)
    workitems_bag = dask.bag.from_delayed(futures)
    res = client.compute(workitems_bag.fold(sum_output), rerun_exceptions_locally=True)

    with tqdm.notebook.tqdm(total=len(futures)) as pbar:
        for _ in dask.distributed.as_completed(futures):
            pbar.update(1)

    return res.result()
