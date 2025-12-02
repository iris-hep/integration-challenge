import collections
import concurrent.futures
import gzip
import itertools
import json
import os
import re
import subprocess
import urllib.request

import input_containers


def get_job_json(username, tag, fname):
    """get bigpanda information"""
    url = (
        f"https://bigpanda.cern.ch/tasks/?taskname=user.{username}*"
        f"{tag}*&limit=50000&display_limit=50000&days=250&json"
    )
    json_info = urllib.request.urlopen(url).read().decode()
    with open(fname, "w") as f:
        f.write(json_info)


def parse_job_json(fname):
    """extract metadata from bigpanda json"""
    with open(fname) as f:
        job_info = json.load(f)

    production_map = {}

    for job in job_info:
        if job["superstatus"] not in ["done", "finished"]:
            continue

        containernames = set([dataset["containername"] for dataset in job["datasets"]])
        if job["dsinfo"]["nfiles"] == 0:
            # handle jobs which timed out with zero files processed
            continue
        assert len(containernames) == 2  # one input, one output
        container_in = next(c for c in containernames if "out" not in c)
        container_out = next(c for c in containernames if "out" in c)

        production_map[container_in] = {
            "output": container_out,
            "jeditaskid": job["jeditaskid"],
            "nfiles_input": job["nfiles"],
            "size_input_GB": None,  # determine from rucio
            "nevts_input": job["neventsTot"],
            "nfiles_output": None,  # determine from rucio
            "size_output_GB": None,  # determine from rucio
            "nevts_output": None,  # not saved anywhere as metadata?
            "files_output": None,  # determine from rucio
        }

    return production_map


def rucio_container_metadata(name):
    """extract metadata from rucio list-files call"""
    cmd = f"rucio list-files {name}"
    output = subprocess.check_output(cmd, shell=True)
    out_tail = output[-100:].decode()

    try:
        nfiles = int(re.findall(r"Total files.*?(\d+)", out_tail)[0])
        size, unit = re.findall(r"([\d\.]+) (.B)", out_tail)[0]
        size = float(size) / 1000**({"TB": -1, "GB": 0, "MB": 1, "kB": 2}[unit])
        unit = "GB"
        nevts = re.findall(r"Total events.*?(\d+)", out_tail)
        if nevts:
            nevts = int(nevts[0])
        else:
            nevts = None  # unknown for output containers
    except:
        print(f"parsing failed for {name}:\n{out_tail}")

    return {"nfiles": nfiles, "size_GB": size, "nevts": nevts}


def rucio_file_paths(name, num_files_expected):
    """file paths from rucio list-file-replicas call"""
    cmd = f"rucio list-file-replicas --protocols root {name}"
    output = subprocess.check_output(cmd, shell=True)
    size_unit_rse_path = re.findall(r"(\d+\.\d+)\s(\wB).+?([\w-]+): (root:\/\/.*?)\s", output.decode())

    # select a single RSE for each file
    filenames = sorted(set([rp[-1].split("/")[-1] for rp in size_unit_rse_path]))
    unique_paths = []
    sizes_GB = []
    for filename in filenames:
        matches = [m for m in size_unit_rse_path if filename in m[-1]]
        # pick MWT2_UC_LOCALGROUPDISK match by default, otherwise first in the list
        match = next((m for m in matches if m[2] == "MWT2_UC_LOCALGROUPDISK"), matches[0])
        unique_paths.append(match[3])
        size_to_GB = lambda num, unit: float(num) * {"kB": 1e-6, "MB": 1e-3, "GB": 1}[unit]
        sizes_GB.append(size_to_GB(*match[:2]))

    assert len(unique_paths) == num_files_expected
    return unique_paths, sizes_GB


def process_one_category(category, container_list, production_map):
    """combine all metadata for a category"""
    print(f"starting {category}")
    metadata = {}
    for container in container_list:
        print(container)
        # get input container information
        info_input = rucio_container_metadata(container)
        metadata[container] = {
            "output": None,
            "jeditaskid": None,
            "nfiles_input": info_input["nfiles"],
            "size_input_GB": info_input["size_GB"],
            "nevts_input": info_input["nevts"],
            "nfiles_output": None,
            "size_output_GB": None,
            "nevts_output": None,
            "files_output": None,
        }

        if container in production_map:
            # job has run, combine bigpanda and rucio (information should match)
            assert info_input["nfiles"] == production_map[container]["nfiles_input"]
            assert info_input["nevts"] == production_map[container]["nevts_input"]

            # update task information
            metadata[container]["output"] = production_map[container]["output"]
            metadata[container]["jeditaskid"] = production_map[container]["jeditaskid"]

            # update output container information
            info_output = rucio_container_metadata(production_map[container]["output"])
            metadata[container]["nfiles_output"] = info_output["nfiles"]
            metadata[container]["size_output_GB"] = info_output["size_GB"]

            # add xrootd file paths
            paths, sizes = rucio_file_paths(production_map[container]["output"], info_output["nfiles"])
            metadata[container]["files_output"] = paths
            metadata[container]["sizes_output_GB"] = sizes
            assert abs(sum(sizes) - info_output["size_GB"]) < 0.01  # agree within 10 MB

    return {category: metadata}


def save_full_metadata(production_map, fname, max_workers=8):
    """combine all metadata into a file, multi-threaded"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        res = executor.map(
            process_one_category,
            input_containers.containers.keys(),
            input_containers.containers.values(),
            itertools.repeat(production_map)
        )

    metadata = {k: v for r in res for k, v in r.items()}

    # save compressed result
    json_enc = json.dumps(metadata, sort_keys=True, indent=4).encode("utf-8")
    with gzip.open(fname, "w") as f:
        f.write(json_enc)


if __name__ == "__main__":
    # run at UChicago, pick up the correct xcache
    os.environ["SITE_NAME"] = "AF_200"

    username = "alheld"
    production_tag = "IC-v1"

    fname_bigpanda = "production_status.json"
    get_job_json(username, production_tag, fname_bigpanda)
    production_map = parse_job_json(fname_bigpanda)

    fname_full = "file_metadata.json.gz"
    metadata = save_full_metadata(production_map, fname_full, max_workers=8)
