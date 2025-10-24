import json
import subprocess
import re

import input_containers


if __name__ == "__main__":
    output_info = {}
    total_nfiles = 0
    total_size_TB = 0
    total_nevts = 0

    for category, container_list in input_containers.containers.items():
        for container in container_list:
            container = container.strip()

            cmd = f"rucio list-files {container}"
            output = subprocess.check_output(cmd, shell=True)

            nfiles, size, nevts = output[-100:].split(b"\n")[-4:-1]
            assert "Total files" in str(nfiles)
            assert "Total size" in str(size)
            assert "Total events" in str(nevts)

            nfiles = int(re.findall("\d+", str(nfiles))[0])
            size, unit = re.findall("([\d\.]+) (.B)", str(size))[0]
            size = round(float(size), 8)
            if unit == "GB":
                size /= 1000
                unit = "TB"
            if unit == "MB":
                size /= 1000**2
                unit = "TB"

            assert unit == "TB"

            nevts = int(re.findall("\d+", str(nevts))[0])
            print(f"\n{container}")
            print("  nfiles", nfiles)
            print("  size", size, unit)
            print("  nevts", nevts)
            total_nfiles += nfiles
            total_size_TB += size
            total_nevts += nevts
            output_info[container] = {"nfiles": nfiles, "size_TB": size, "nevts": nevts}

    print("\n=== total")
    print("  nfiles", total_nfiles)
    print("  size", total_size_TB, unit)
    print("  nevts", total_nevts)

    with open("container_metadata.json", "w") as f:
        f.write(json.dumps(output_info, sort_keys=True, indent=4))
