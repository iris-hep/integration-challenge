"""Cadi entry: https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2624&ancode=B2G-22-006&tp=an&line=B2G-22-006"""

from __future__ import annotations

import shutil
from pathlib import Path
import os
from typing import Iterable
from requests import Session
from os.path import isdir, expanduser
from getpass import getpass
import xml.etree.ElementTree as ET

from rich.console import Console
from rich.progress import Progress

console = Console()


class CERNSession(Session):
    LOGIN_URL = "https://login.cern.ch"
    CERT_DIRS = [
        "/etc/grid-security/certificates",
        "/cvmfs/grid.cern.ch/etc/grid-security/certificates",
    ]

    def __init__(
        self,
        key_file="~/.globus/userkey.pem",
        cert_file="~/.globus/usercert.pem",
        ca_cert_dir=True,
    ):
        super().__init__()

        if ca_cert_dir is True:
            for dir in filter(isdir, self.CERT_DIRS):
                ca_cert_dir = dir
                break
            else:
                ca_cert_dir = None

        c = self.get_adapter(self.LOGIN_URL).get_connection(self.LOGIN_URL)
        c.cert_file = expanduser(cert_file)
        c.key_file = expanduser(key_file)
        if ca_cert_dir:
            c.ca_cert_dir = expanduser(ca_cert_dir)
        c.key_password = lambda: getpass("Password (%s): " % key_file)
        assert set(c.pool.queue) == {None}

        self.headers["User-Agent"] = "curl-sso-certificate/0.6"

    def _get_form(self, resp):
        if resp.status_code == 200 and resp.url.startswith(self.LOGIN_URL):
            return ET.fromstring(resp.content).find("body/form")
        else:
            return None

    def get_redirect_target(self, resp):
        url = super().get_redirect_target(resp)
        if url is None:
            form = self._get_form(resp)
            if form:
                url = form.get("action")
        return url

    def rebuild_auth(self, preq, resp):
        super().rebuild_auth(preq, resp)
        form = self._get_form(resp)
        if form:
            preq.prepare_method("POST")
            preq.prepare_body(
                {el.get("name"): el.get("value") for el in form.findall("input")}, None
            )


# directory where to store files that should not be exposed publicly
# we put it explicitely in .gitignore
DONT_EXPOSE_CMS_INTERNAL = "DONT_EXPOSE_CMS_INTERNAL"  # NEVER CHANGE
base = Path(__file__).parent / DONT_EXPOSE_CMS_INTERNAL
protected_urls = ["cms-service-dqmdc.web.cern.ch"]


def _download(console, name: str, url: str, base_dst: str, to_disk: dict) -> dict:
    ext = "".join(Path(url).suffixes)
    fname = name + ext

    if any(p in url for p in protected_urls):
        session = CERNSession()
    else:
        session = Session()

    # destination to store the file
    dst = str(Path(base_dst) / fname)

    console.print(f"[cyan]Downloading {name} to {dst}")
    if url.startswith("http"):
        with session:
            with session.get(url, stream=True) as r:
                if not r.ok:
                    r.raise_for_status()
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    for chunk in r.iter_content():
                        f.write(chunk)
    elif url.startswith("/cvmfs/"):
        if not os.path.exists(url):
            raise FileNotFoundError(f"{url} does not exist")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(url, dst)
    else:
        raise ValueError(f"Cannot handle url {url}")
    # fill
    to_disk[name] = dst
    return to_disk


def download(it: Iterable, base_dst: str, to_disk: dict) -> None:
    with Progress(console=console) as progress:
        task = progress.add_task(
            "[cyan]Downloading ...", total=len(it), console=console
        )
        for item in it:
            name, url = item
            to_disk = _download(
                console=console, name=name, url=url, base_dst=base_dst, to_disk=to_disk
            )
            progress.advance(task)
    return to_disk


# mapping urls to local files
# Most corrections are found here:
# - https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun2LegacyAnalysis
# - https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG?ref_type=heads
# - https://cms-xpog.docs.cern.ch/commonJSONSFs/
_corrections_16 = {
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#2016
    "golden_json": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run2UL2016preVFP/
    "btagging_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz",
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run2UL2016postVFP/
    "btagging_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME?ref_type=heads
    "JEC_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jet_jerc.json.gz",
    "JEC_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jet_jerc.json.gz",
    "jetvetomaps_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jetvetomaps.json.gz",
    "jetvetomaps_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jetvetomaps.json.gz",
    "JME_ID_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/jmar.json.gz",
    "JME_ID_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/jmar.json.gz",
    "METPhi_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016preVFP_UL/met.json.gz",
    "METPhi_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2016postVFP_UL/met.json.gz",
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData#Centrally_produced_correctionlib
    "pileup_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2016preVFP_UL/puWeights.json.gz",
    "pileup_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2016postVFP_UL/puWeights.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO?ref_type=heads
    "muon_JPsi_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016preVFP_UL/muon_JPsi.json.gz",  # low-pt (pT < 30 GeV)
    "muon_JPsi_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016postVFP_UL/muon_JPsi.json.gz",  # low-pt (pT < 30 GeV)
    "muon_Z_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016preVFP_UL/muon_Z.json.gz",  # medium-pt (15 < pT < 200 GeV)
    "muon_Z_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json.gz",  # medium-pt (15 < pT < 200 GeV)
    "muon_HighPt_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016preVFP_UL/muon_HighPt.json.gz",  # high-pt (pT > 200 GeV)
    "muon_HighPt_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2016postVFP_UL/muon_HighPt.json.gz",  # high-pt (pT > 200 GeV)
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM?ref_type=heads
    "electron_preVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2016preVFP_UL/electron.json.gz",
    "electron_postVFP": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2016postVFP_UL/electron.json.gz",
}
_corrections_17 = {
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#2017
    "golden_json": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run2UL2017/
    "btagging": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2017_UL/btagging.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME?ref_type=heads
    "JEC": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jet_jerc.json.gz",
    "jetvetomaps": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jetvetomaps.json.gz",
    "JME_ID": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/jmar.json.gz",
    "METPhi": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2017_UL/met.json.gz",
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData#Centrally_produced_correctionlib
    "pileup": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2017_UL/puWeights.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO?ref_type=heads
    "muon_JPsi": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2017_UL/muon_JPsi.json.gz",  # low-pt (pT < 30 GeV)
    "muon_Z": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2017_UL/muon_Z.json.gz",  # medium-pt (15 < pT < 200 GeV)
    "muon_HighPt": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2017_UL/muon_HighPt.json.gz",  # high-pt (pT > 200 GeV)
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM?ref_type=heads
    "electron": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2017_UL/electron.json.gz",
}
_corrections_18 = {
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#2018
    "golden_json": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run2UL2018/
    "btagging": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2018_UL/btagging.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME?ref_type=heads
    "JEC": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jet_jerc.json.gz",
    "jetvetomaps": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jetvetomaps.json.gz",
    "JME_ID": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/jmar.json.gz",
    "METPhi": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2018_UL/met.json.gz",
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData#Centrally_produced_correctionlib
    "pileup": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/2018_UL/puWeights.json.gz",
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/MUO?ref_type=heads
    "muon_JPsi": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_JPsi.json.gz",  # low-pt (pT < 30 GeV)
    "muon_Z": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_Z.json.gz",  # medium-pt (15 < pT < 200 GeV)
    "muon_HighPt": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/2018_UL/muon_HighPt.json.gz",  # high-pt (pT > 200 GeV)
    # https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM?ref_type=heads
    "electron": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2018_UL/electron.json.gz",
}

_corrections = {
    16: _corrections_16,
    17: _corrections_17,
    18: _corrections_18,
}


def get_corrections(era: int) -> dict:
    """
    Get the correction files:

    ```python
    corrections = get_corrections(era)
    ```
    """
    assert era in (16, 17, 18)
    mapping = base / f"20{era}" / "corrections.json"

    if not os.path.exists(mapping):
        raise FileNotFoundError(
            f"{mapping} does not exist. Please run `python {str(Path(__file__))}` first."
        )

    with open(mapping, "r") as f:
        data = json.loads(f.read())

    return data


if __name__ == "__main__":
    import json

    for era in (16, 17, 18):
        to_disk = {}
        base_dst = base / f"20{era}"
        os.makedirs(base_dst, exist_ok=True)
        # trigger download and fill to_disk mapping
        to_disk = download(
            _corrections[era].items(), base_dst=base_dst, to_disk=to_disk
        )
        # save mapping
        with open(base_dst / "corrections.json", "w") as f:
            json.dump(to_disk, f, indent=2)
