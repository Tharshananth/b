"""
End-to-end check that one known patient record can be retrieved.

Run from anywhere (the script anchors itself to backend/):

    python scripts/test_patient_lookup.py                        # DB layer only
    python scripts/test_patient_lookup.py --database orchTest    # override config.yaml's job_db.database
    python scripts/test_patient_lookup.py --api                  # also ask the running chatbot

Layers tested:
  1. DB      - services.job_status.get_job_status() returns the expected document
  2. SCOPE   - the same patient under a different hospital returns nothing
  3. API     - POST /api/chat/ calls get_job_status and gets >0 jobs back
               (only with --api; uvicorn must be running)

Read-only. The --database override is in memory only; config.yaml is not touched.
The API layer uses whatever database the *running server* was started with.

NOTE: output includes real patientID / disorder values. Treat it as clinical data.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# The app loads config.yaml by relative path, so anchor to backend/ regardless
# of where this was launched from.
BACKEND = Path(__file__).resolve().parent.parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND / ".env")

OK, FAIL, INFO = "[OK]  ", "[FAIL]", "      "
failures = []

# Known record: orchTest.executionInfo _id 5fee660006fdf5b06571f358.
# Expected values are what Compass shows; update them if the record changes.
DEFAULT_HOSPITAL = "98"
DEFAULT_PATIENT = "0233"
DEFAULT_STUDY = "03"
EXPECTED = {
    "_id": "5fee660006fdf5b06571f358",
    "trialID": "6677",
    "executionID": "Data_entry_Visit_1",
    "disorder": "Mild_cognitive_impairment",
    "status.dicomToNifti": "Success",
    "status.fMRI_2": "Invalid Request",
    "status.sMRI_2": "Success",
    "status.dMRI_1": "Success",
    "status.inference.tumour_segmentation": "Success",
    "status.inference.functional_connectomes": "Started",
    "status.review_status": "REVIEWED",
}


def section(title):
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)


def check(cond, ok_msg, fail_msg):
    if cond:
        print(OK + " " + ok_msg)
    else:
        failures.append(fail_msg)
        print(FAIL + " " + fail_msg)
    return cond


def dig(doc, dotted):
    """doc['a']['b'] for 'a.b'; None if any step is missing."""
    for key in dotted.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    return doc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--hospital", default=DEFAULT_HOSPITAL)
    p.add_argument("--patient", default=DEFAULT_PATIENT)
    p.add_argument("--study", default=DEFAULT_STUDY)
    p.add_argument("--database", help="override job_db.database for the DB layers")
    p.add_argument("--api", action="store_true", help="also test the running chatbot")
    p.add_argument("--api-url", default="http://localhost:8000/api/chat/")
    p.add_argument("--show", action="store_true", help="print the full document")
    return p.parse_args()


args = parse_args()

import config
from services.job_status import get_job_status, TenantScope

cfg = config.get_config().job_db
configured_db = cfg.database
if args.database:
    cfg.database = args.database

# -- 1. DB ------------------------------------------------------------------
section("1. DB LOOKUP  (services.job_status.get_job_status)")
print(INFO + f" config.yaml database : {configured_db}")
print(INFO + f" testing against      : {cfg.database}.{cfg.collection}")
print(INFO + f" filter               : hospital={args.hospital} "
             f"patient={args.patient} study={args.study}")

try:
    docs = get_job_status(
        TenantScope(hospital=args.hospital),
        patient_id=args.patient,
        study_id=args.study,
    )
except Exception as e:
    docs = []
    check(False, "", f"lookup raised {type(e).__name__}: {e}")

found = check(len(docs) >= 1, f"{len(docs)} document(s) returned",
              f"no document for patient {args.patient} in {cfg.database}")

if found:
    doc = docs[0]
    using_defaults = (args.hospital, args.patient, args.study) == (
        DEFAULT_HOSPITAL, DEFAULT_PATIENT, DEFAULT_STUDY)
    if using_defaults:
        for field, want in EXPECTED.items():
            got = dig(doc, field)
            got = str(got) if field == "_id" else got
            check(got == want, f"{field} = {want!r}",
                  f"{field}: expected {want!r}, got {got!r}")
    else:
        print(INFO + " custom patient - skipping field comparison")
    if args.show:
        print(json.dumps(doc, default=str, indent=2))

# -- 2. SCOPE ---------------------------------------------------------------
section("2. TENANT SCOPE  (wrong hospital must see nothing)")
other = "__no_such_hospital__"
try:
    leaked = get_job_status(TenantScope(hospital=other),
                            patient_id=args.patient, study_id=args.study)
    check(len(leaked) == 0, f"hospital={other!r} sees 0 documents",
          f"hospital={other!r} saw {len(leaked)} document(s) - SCOPE LEAK")
except Exception as e:
    check(False, "", f"scope lookup raised {type(e).__name__}: {e}")

# -- 3. API -----------------------------------------------------------------
if args.api:
    section("3. CHAT API  (running server)")
    print(INFO + " the server uses the database it was STARTED with, not --database")
    body = json.dumps({
        "message": f"What is the processing status of patient {args.patient} "
                   f"in study {args.study}?",
        "session_id": "test_patient_lookup",
        "hospital": args.hospital,
        "patient_id": args.patient,
    }).encode()
    req = urllib.request.Request(args.api_url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
    except Exception as e:
        resp = None
        check(False, "", f"API call failed: {e}")

    if resp:
        tools = [t for t in resp.get("tools_used", [])
                 if t.get("name") == "get_job_status"]
        check(bool(tools), "chatbot called get_job_status",
              "chatbot did not call get_job_status")
        for t in tools:
            summary = t.get("summary", "")
            print(INFO + f" tool args={t.get('arguments')} summary={summary!r}")
            check(not summary.startswith("0 job"), "tool returned job data",
                  "tool returned 0 jobs - is the server on the right database?")
        print(INFO + " response: " + resp.get("response", "").replace("\n", " ")[:400])

# -- Summary ----------------------------------------------------------------
section("SUMMARY")
if failures:
    print(FAIL + f" {len(failures)} problem(s):")
    for f in failures:
        print(INFO + "  - " + f)
    sys.exit(1)
print(OK + " all checks passed")
