#!/usr/bin/env python3
"""Test the patched GID auto-selection against fake sysfs trees copied from spark-3 and spark-4.

Usage: python3 test_gid_auto.py /path/to/roce_oneshot.py
Take roce_oneshot.py from the serving image (b12x 1.5.0), e.g.
  docker run --rm --entrypoint cat glm53-b12x:ab86b70 \
    /usr/local/lib/python3.12/dist-packages/b12x/comm/roce/roce_oneshot.py > roce_oneshot.py
"""
import ast, logging, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
if len(sys.argv) != 2:
    sys.exit(__doc__)
ORIG = Path(sys.argv[1])

V4 = lambda ip: "0000:0000:0000:0000:0000:ffff:" + "%02x%02x:%02x%02x" % tuple(int(x) for x in ip.split("."))
LL = lambda s: "fe80:0000:0000:0000:" + s
ZERO = "0000:0000:0000:0000:0000:0000:0000:0000"

# (hca -> {index: (gid, type)}) as observed 2026-11-18
SPARK3 = {
    "rocep1s0f0": {0: (LL("4ebb:47ff:feea:aaab"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:feea:aaab"), "RoCE v2"),
                   2: (LL("3037:c686:cb11:edae"), "IB/RoCE v1"), 3: (LL("3037:c686:cb11:edae"), "RoCE v2"),
                   4: (V4("192.168.250.11"), "IB/RoCE v1"), 5: (V4("192.168.250.11"), "RoCE v2")},
    "rocep1s0f1": {0: (LL("4ebb:47ff:feea:aaac"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:feea:aaac"), "RoCE v2"),
                   2: (V4("192.168.100.11"), "IB/RoCE v1"), 3: (V4("192.168.100.11"), "RoCE v2")},
    "roceP2p1s0f0": {0: (LL("4ebb:47ff:feea:aaaf"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:feea:aaaf"), "RoCE v2"),
                     2: (LL("7a1d:977f:30e7:9b81"), "IB/RoCE v1"), 3: (LL("7a1d:977f:30e7:9b81"), "RoCE v2"),
                     4: (V4("192.168.251.11"), "IB/RoCE v1"), 5: (V4("192.168.251.11"), "RoCE v2")},
}
SPARK4 = {
    "rocep1s0f0": {0: (LL("4ebb:47ff:fee8:690c"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:fee8:690c"), "RoCE v2"),
                   2: (V4("192.168.250.12"), "IB/RoCE v1"), 4: (V4("192.168.250.12"), "RoCE v2")},
    "rocep1s0f1": {0: (LL("4ebb:47ff:fee8:690d"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:fee8:690d"), "RoCE v2"),
                   2: (V4("192.168.240.14"), "IB/RoCE v1"), 3: (V4("192.168.240.14"), "RoCE v2")},
    "roceP2p1s0f0": {0: (LL("4ebb:47ff:fee8:6910"), "IB/RoCE v1"), 1: (LL("4ebb:47ff:fee8:6910"), "RoCE v2"),
                     2: (V4("192.168.251.12"), "IB/RoCE v1"), 4: (V4("192.168.251.12"), "RoCE v2")},
}


def build(tree, root):
    for hca, gids in tree.items():
        port = root / hca / "ports" / "1"
        (port / "gids").mkdir(parents=True)
        (port / "gid_attrs" / "types").mkdir(parents=True)
        (port / "state").write_text("4: ACTIVE\n")
        for i in range(8):
            gid, kind = gids.get(i, (ZERO, None))
            (port / "gids" / str(i)).write_text(gid + "\n")
            if kind:  # empty slots: attrs unreadable on real sysfs -> omit file
                (port / "gid_attrs" / "types" / str(i)).write_text(kind + "\n")


def load_funcs(src):
    tree = ast.parse(src)
    want = {"_env_list", "_env_int", "_roce_v2_ipv4_gid_indices", "_auto_gid_index", "default_gid_index"}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("DEFAULT_GID_INDEX", "_GID_AUTO_LOGGED") for t in n.targets))]
    ns = {"os": os, "Path": Path, "Optional": Optional, "logger": logging.getLogger("t")}
    exec(compile(ast.Module(body=list[ast.stmt](nodes), type_ignores=[]), "patched", "exec"), ns)
    return ns


def main():
    logging.basicConfig(level=logging.INFO)
    work = Path(tempfile.mkdtemp())
    site = work / "site" / "b12x" / "comm" / "roce"
    site.mkdir(parents=True)
    shutil.copy(ORIG, site / "roce_oneshot.py")
    patch = HERE / "patch_roce_gid_auto.py"
    subprocess.run([sys.executable, str(patch), "--site-root", str(work / "site")], check=True)
    # idempotent second run
    subprocess.run([sys.executable, str(patch), "--site-root", str(work / "site")], check=True)
    src = (site / "roce_oneshot.py").read_text()
    compile(src, "roce_oneshot.py", "exec")
    ns = load_funcs(src)

    results = {}
    for label, tree in (("spark-3", SPARK3), ("spark-4", SPARK4)):
        root = work / label
        build(tree, root)
        ns["_GID_AUTO_LOGGED"] = False
        os.environ["NCCL_IB_HCA"] = "rocep1s0f0,roceP2p1s0f0"
        os.environ["NCCL_IB_GID_INDEX"] = "-1"
        os.environ.pop("B12X_ROCE_GID_INDEX", None)
        results[label] = ns["_auto_gid_index"](root)
    assert results == {"spark-3": 5, "spark-4": 4}, results

    # explicit index is respected; default_gid_index keeps explicit values
    os.environ["B12X_ROCE_GID_INDEX"] = "7"
    assert ns["default_gid_index"]() == 7
    os.environ.pop("B12X_ROCE_GID_INDEX")
    os.environ["NCCL_IB_GID_INDEX"] = "2"
    assert ns["default_gid_index"]() == 2
    # no common index -> None (caller falls back to 3)
    os.environ["NCCL_IB_HCA"] = "rocep1s0f0,rocep1s0f1"
    assert ns["_auto_gid_index"](work / "spark-4") is None
    # no explicit HCAs -> discovers active HCAs with IPv4 RoCE v2 GIDs
    os.environ.pop("NCCL_IB_HCA")
    assert ns["_auto_gid_index"](work / "spark-3") == 5  # roceP2p1s0f0 + rocep1s0f0 sorted -> common {5}? see note
    print("ALL TESTS PASSED", results)


if __name__ == "__main__":
    main()
