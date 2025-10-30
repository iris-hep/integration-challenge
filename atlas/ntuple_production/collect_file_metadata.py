import collections
import gzip
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
    rses_and_paths = re.findall(r"(\w+): (root:\/\/.*?)\s", output.decode())

    # select a single RSE for each file
    filenames = sorted(set([rp[1].split("/")[-1] for rp in rses_and_paths]))
    unique_paths = []
    for filename in filenames:
        fpaths = [rp for rp in rses_and_paths if filename in rp[1]]
        # pick NET2_LOCALGROUPDISK match by default, otherwise first in the list
        fpath = next((fp for fp in fpaths if fp[0] == "NET2_LOCALGROUPDISK"), fpaths[0])[1]
        unique_paths.append(fpath)

    assert len(unique_paths) == num_files_expected
    return unique_paths


def save_full_metadata(production_map, fname):
    """combine all metadata into a file"""
    metadata = collections.defaultdict(lambda: {})
    for category, container_list in input_containers.containers.items():
        print(category)
        metadata[category] = {}
        for container in container_list:
            print(container)
            # get input container information
            info_input = rucio_container_metadata(container)
            metadata[category][container] = {
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
                metadata[category][container]["output"] = production_map[container]["output"]
                metadata[category][container]["jeditaskid"] = production_map[container]["jeditaskid"]

                # update output container information
                info_output = rucio_container_metadata(production_map[container]["output"])
                metadata[category][container]["nfiles_output"] = info_output["nfiles"]
                metadata[category][container]["size_output_GB"] = info_output["size_GB"]

                # add xrootd file paths
                paths = rucio_file_paths(production_map[container]["output"], info_output["nfiles"])
                metadata[category][container]["files_output"] = paths

        # write after each category to retain partial results, save compressed
        json_enc = json.dumps(dict(metadata), sort_keys=True, indent=4).encode("utf-8")
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
    metadata = save_full_metadata(production_map, fname_full)
