"""
Job database connectivity + schema probe.

Run from the backend/ directory. Any of these work:

    python scripts/check_job_db.py                       # prompts for user/password
    python scripts/check_job_db.py "mongodb+srv://..."   # full URI as an argument
    MONGODB_URI="mongodb+srv://..." python scripts/...   # from the environment

Verifies the credential works, confirms it is scoped and read-only, and dumps
the parts of the schema the chatbot needs. Read-only apart from one deliberate
write probe, which cleans up after itself.

NOTE: output includes real patientID / disorder values. Treat it as clinical data.
"""
import getpass
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Cluster host is not a secret; the credential is. Never hardcode the password
# here - this repo's .gitignore is broken, so anything written to it is tracked.
CLUSTER_HOST = "cluster-uat.aun3kcf.mongodb.net"

# The app loads config.yaml by relative path, so anchor to backend/ regardless
# of where this was launched from.
BACKEND = Path(__file__).resolve().parent.parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv()

OK, FAIL, WARN, INFO = "[OK]  ", "[FAIL]", "[WARN]", "      "
failures = []


def section(title):
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)


def fail(msg):
    failures.append(msg)
    print(FAIL + " " + msg)


def redact(uri):
    """mongodb+srv://user:pw@host/... -> mongodb+srv://user:***@host/..."""
    if "@" not in uri:
        return uri
    creds, host = uri.split("@", 1)
    if ":" in creds.rsplit("/", 1)[-1]:
        scheme_user, _ = creds.rsplit(":", 1)
        return scheme_user + ":***@" + host
    return creds + "@" + host


def typename(v):
    return type(v).__name__


# -- 1. Environment ---------------------------------------------------------
section("1. ENVIRONMENT")

def resolve_uri():
    """Command-line argument, then environment, then an interactive prompt."""
    if len(sys.argv) > 1 and sys.argv[1].startswith("mongodb"):
        print(OK + " using URI from command line")
        return sys.argv[1]

    from_env = os.getenv("MONGODB_URI")
    if from_env:
        print(OK + " using MONGODB_URI from environment")
        return from_env

    print(INFO + " no URI given - enter the Atlas credential")
    print(INFO + " cluster: " + CLUSTER_HOST)
    user = input("       username: ").strip()
    # getpass keeps the password off the screen and out of shell history.
    pw = getpass.getpass("       password: ")
    if not user or not pw:
        print(FAIL + " username and password are both required")
        sys.exit(1)
    # Atlas passwords routinely contain @ : / ? # characters, which break a URI
    # unless percent-encoded.
    return ("mongodb+srv://" + quote_plus(user) + ":" + quote_plus(pw)
            + "@" + CLUSTER_HOST + "/?retryWrites=false"
            + "&readPreference=secondaryPreferred")


uri = resolve_uri()
os.environ["MONGODB_URI"] = uri   # database.mongo reads it from here
print(INFO + " " + redact(uri))

import config

cfg = config.load_config("config.yaml").job_db
print(INFO + " target: " + cfg.database + "." + cfg.collection)
print(INFO + " tenant_field: " + repr(cfg.tenant_field))

from pymongo.errors import OperationFailure, PyMongoError, ServerSelectionTimeoutError
from database.mongo import get_mongo_client, get_job_db, close_mongo_client

# -- 2. Connectivity --------------------------------------------------------
section("2. CONNECTIVITY")

try:
    client = get_mongo_client()
    client.admin.command("ping")
    print(OK + " ping succeeded")
except ServerSelectionTimeoutError as e:
    fail("cannot reach cluster: " + str(e)[:200])
    print(INFO + " usually Atlas Network Access - is this machine's IP allowed?")
    sys.exit(1)
except PyMongoError as e:
    fail("connection failed: " + str(e)[:200])
    sys.exit(1)

try:
    build = client.admin.command("buildInfo")
    print(INFO + " MongoDB server " + str(build.get("version")))
except OperationFailure:
    print(INFO + " buildInfo not permitted (fine for a scoped user)")

# -- 3. Credential scope ----------------------------------------------------
section("3. CREDENTIAL SCOPE  (should see the job database and nothing else)")

try:
    dbs = client.list_database_names()
    print(INFO + " visible databases: " + str(dbs))
    extra = [d for d in dbs if d not in {cfg.database, "admin", "local", "config"}]
    if extra:
        print(WARN + " credential can see other databases: " + str(extra))
        print(INFO + " grant `read` on " + cfg.database + " ONLY, not readAnyDatabase")
    else:
        print(OK + " scoped to " + cfg.database)
except OperationFailure:
    print(OK + " listDatabases denied - credential is tightly scoped (good)")

# -- 4. Read-only check -----------------------------------------------------
section("4. READ-ONLY CHECK  (a write MUST be rejected)")

probe = "_chatbot_readonly_probe"
try:
    get_job_db()[probe].insert_one({"_probe": True})
    print(WARN + " WRITE SUCCEEDED - this credential is NOT read-only.")
    print(INFO + " the chatbot must never hold write access to trial data.")
    failures.append("credential has write access")
    try:
        get_job_db()[probe].drop()
        print(INFO + " probe collection dropped")
    except PyMongoError as e:
        print(WARN + " could not drop " + probe + ": " + str(e))
        print(INFO + " remove it manually")
except OperationFailure:
    print(OK + " write rejected - credential is read-only")
except PyMongoError as e:
    print(WARN + " inconclusive: " + str(e)[:150])

# -- 5. Collection ----------------------------------------------------------
section("5. COLLECTION")

coll = get_job_db()[cfg.collection]
try:
    names = get_job_db().list_collection_names()
    print(INFO + " collections: " + str(names))
    if cfg.collection not in names:
        fail(repr(cfg.collection) + " not found in " + cfg.database)
except OperationFailure as e:
    print(INFO + " listCollections denied: " + str(e)[:100])

try:
    total = coll.count_documents({})
    print(OK + " " + cfg.collection + ": " + str(total) + " documents")
except PyMongoError as e:
    fail("count failed: " + str(e))

# -- 6. Indexes -------------------------------------------------------------
section("6. INDEXES")

try:
    idx = list(coll.list_indexes())
    for i in idx:
        print(INFO + " " + i["name"] + ": " + str(dict(i["key"])))
    if len(idx) <= 1:
        print(WARN + " only the default _id_ index exists.")
        print(INFO + " every scoped query is a full collection scan. Needed:")
        print(INFO + "   db." + cfg.collection + ".createIndex({ "
              + cfg.tenant_field + ": 1, studyID: 1, patientID: 1, _id: -1 })")
except PyMongoError as e:
    print(WARN + " could not list indexes: " + str(e))

# -- 7. Schema --------------------------------------------------------------
section("7. SCHEMA  (top level)")

doc = coll.find_one()
if not doc:
    fail("collection is empty - nothing to inspect")
else:
    for k, v in doc.items():
        if isinstance(v, dict):
            print(INFO + " " + k.ljust(14) + " object(" + str(len(v)) + ")")
        elif isinstance(v, list):
            print(INFO + " " + k.ljust(14) + " array(" + str(len(v)) + ")")
        else:
            print(INFO + " " + k.ljust(14) + " " + typename(v).ljust(10) + " " + repr(v))

    # The two objects the diagnosis feature depends on.
    for field in ("status", "timespan"):
        section("7b. " + field.upper() + "  <- needed for the diagnosis feature")
        val = doc.get(field)
        if not isinstance(val, dict):
            print(WARN + " " + field + " is " + typename(val) + ", not an object")
            continue
        for k, v in val.items():
            shown = repr(v)
            if len(shown) > 70:
                shown = shown[:70] + "..."
            print(INFO + " " + k.ljust(22) + " " + typename(v).ljust(10) + " " + shown)

# -- 8. Field values --------------------------------------------------------
section("8. FIELD VALUES  (tenant + enum meanings)")

for field in (cfg.tenant_field, "provider", "env", "disorder", "studyID", "trialID"):
    try:
        vals = coll.distinct(field)
        preview = vals[:12]
        more = "  (+" + str(len(vals) - 12) + " more)" if len(vals) > 12 else ""
        print(INFO + " " + field.ljust(12) + str(len(vals)).rjust(4)
              + " distinct: " + str(preview) + more)
    except PyMongoError as e:
        print(WARN + " " + field + ": " + str(e)[:80])

# -- 9. Scoped query through the real service -------------------------------
section("9. SCOPED QUERY  (via services.job_status)")

from services.job_status import get_job_status, TenantScope, ScopeViolation

try:
    tenants = coll.distinct(cfg.tenant_field)
except PyMongoError:
    tenants = []

if not tenants:
    print(WARN + " no values for " + repr(cfg.tenant_field) + " - cannot test scoping")
else:
    t = tenants[0]
    print(INFO + " using " + cfg.tenant_field + "=" + repr(t))
    try:
        rows = get_job_status(TenantScope(hospital=t), limit=3)
        print(OK + " returned " + str(len(rows)) + " rows, all in scope")
        for r in rows:
            print(INFO + "   " + str(r.get("_id")))
    except ScopeViolation as e:
        fail("SCOPE VIOLATION: " + str(e))
    except PyMongoError as e:
        fail("query failed: " + str(e)[:150])

    if len(tenants) > 1:
        other = tenants[1]
        print("")
        print(INFO + " isolation check: scope=" + repr(t)
              + " must not return " + repr(other) + " data")
        try:
            rows = get_job_status(TenantScope(hospital=t), limit=50)
            leaked = [r for r in rows if r.get(cfg.tenant_field) != t]
            if leaked:
                fail(str(len(leaked)) + " rows leaked across tenants")
            else:
                print(OK + " no cross-tenant rows in " + str(len(rows)) + " results")
        except ScopeViolation as e:
            fail("SCOPE VIOLATION: " + str(e))
    else:
        print(WARN + " only one " + cfg.tenant_field + " value in UAT - "
              "isolation is untested until there are two")

# -- 10. Status vocabulary --------------------------------------------------
section("10. STATUS VOCABULARY  (every distinct value, per stage)")

from services.execution_summary import collect_status_vocabulary, summarize, classify

try:
    all_docs = list(coll.find({}, {"status": 1}))
    vocab = collect_status_vocabulary(all_docs)
    unknown = []
    for stage, values in vocab.items():
        print(INFO + " " + stage)
        for v in values:
            kind = classify(v)
            mark = "  <-- UNRECOGNISED" if kind == "unknown" else ""
            print(INFO + "     " + v.ljust(28) + kind + mark)
            if kind == "unknown":
                unknown.append(stage + "." + v)
    if unknown:
        print("")
        print(WARN + " " + str(len(unknown)) + " status value(s) the classifier "
              "does not recognise:")
        for u in unknown:
            print(INFO + "   " + u)
        print(INFO + " add these to classify() in services/execution_summary.py")
    else:
        print("")
        print(OK + " every status value maps to success / failed / running")
except PyMongoError as e:
    print(WARN + " vocabulary scan failed: " + str(e)[:120])

# -- 11. Summariser ---------------------------------------------------------
section("11. SUMMARISER  (what the LLM will actually receive)")

try:
    sample = coll.find_one()
    if sample:
        import json
        out = summarize(sample)
        print(json.dumps(out, indent=2, default=str))
except Exception as e:
    print(WARN + " summarise failed: " + str(e)[:200])

# -- Summary ----------------------------------------------------------------
section("SUMMARY")

close_mongo_client()

if failures:
    print(FAIL + " " + str(len(failures)) + " problem(s):")
    for f in failures:
        print(INFO + "  - " + f)
    sys.exit(1)

print(OK + " job database is reachable and correctly scoped")
