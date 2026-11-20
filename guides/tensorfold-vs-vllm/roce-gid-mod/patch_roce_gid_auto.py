#!/usr/bin/env python3
"""Make b12x's RoCE all-reduce pick its GID index the way NCCL does for -1.

b12x.comm.roce.default_gid_index() falls back to NCCL_IB_GID_INDEX when
B12X_ROCE_GID_INDEX is unset. eugr's cluster setup exports NCCL_IB_GID_INDEX=-1
(NCCL: "choose the GID yourself", with NCCL_IB_ADDR_FAMILY=AF_INET), and b12x
passed -1 straight to ibv_query_gid(), which fails with EINVAL on every rank:

    RoCEnante initialization failed: ... ibv_query_gid: Invalid argument

vLLM then silently keeps PyNCCL. A fixed index is no fix either: the IPv4
RoCE v2 GID sits at index 5 on spark-3's fabric HCAs and index 4 on spark-4's
(and spark-4's table has a hole at 3), so the index depends on the order the
node's addresses appeared.

This patch keeps an explicit non-negative B12X_ROCE_GID_INDEX/NCCL_IB_GID_INDEX
as before. A negative or unset value now selects, on each node, the lowest GID
index that holds an IPv4-mapped RoCE v2 GID on every selected HCA. The index is
local to the rank (b12x exchanges the GID itself and uses the index only as the
source index), so the two nodes may resolve to different indices. If no common
index exists it logs a warning and falls back to the original default (3).

Exact anchors fail clearly if roce_oneshot.py has drifted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "B12X-ROCE-GID-AUTO"
REL = "b12x/comm/roce/roce_oneshot.py"


class PatchError(RuntimeError):
    pass


ANCHOR = '''def default_gid_index() -> int:
    """``B12X_ROCE_GID_INDEX``, else NCCL's ``NCCL_IB_GID_INDEX``, else 3."""

    return _env_int(
        "B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX", default=DEFAULT_GID_INDEX
    )
'''

REPLACEMENT = f'''_GID_AUTO_LOGGED = False


def _roce_v2_ipv4_gid_indices(name: str, root: Path) -> set:
    """GID indices of HCA ``name`` (port 1) holding an IPv4-mapped RoCE v2 GID."""
    # {MARKER}
    port = root / name / "ports" / "1"
    found = set()
    try:
        entries = list((port / "gids").iterdir())
    except OSError:
        return found
    for entry in entries:
        try:
            index = int(entry.name)
            raw = entry.read_text().strip().replace(":", "").lower()
            kind = (port / "gid_attrs" / "types" / entry.name).read_text().strip()
        except (OSError, ValueError):
            continue  # empty slots raise EINVAL on their attrs
        if kind == "RoCE v2" and len(raw) == 32 and raw[:20] == "0" * 20 and raw[20:24] == "ffff":
            found.add(index)
    return found


def _auto_gid_index(root: Path = Path("/sys/class/infiniband")) -> Optional[int]:
    """Lowest index with an IPv4 RoCE v2 GID on every selected HCA (NCCL's -1, AF_INET)."""
    # {MARKER}
    global _GID_AUTO_LOGGED
    if os.getenv("NCCL_IB_ADDR_FAMILY", "AF_INET").upper() == "AF_INET6":
        logger.warning("[{MARKER}] NCCL_IB_ADDR_FAMILY=AF_INET6: no IPv6 auto-selection; "
                       "set B12X_ROCE_GID_INDEX explicitly")
        return None
    names = _env_list("B12X_ROCE_HCA", "NCCL_IB_HCA")[:2]
    if not names:
        candidates = []
        for dev in sorted(root.glob("*")):
            try:
                active = "ACTIVE" in (dev / "ports" / "1" / "state").read_text()
            except OSError:
                continue
            if active and _roce_v2_ipv4_gid_indices(dev.name, root):
                candidates.append(dev.name)
        names = tuple(candidates[:2])
    if not names:
        return None
    common = None
    for name in names:
        indices = _roce_v2_ipv4_gid_indices(name, root)
        common = indices if common is None else common & indices
    if not common:
        logger.warning("[{MARKER}] no common IPv4 RoCE v2 GID index on %s; falling back to %d",
                       ",".join(names), DEFAULT_GID_INDEX)
        return None
    index = min(common)
    if not _GID_AUTO_LOGGED:
        logger.info("[{MARKER}] gid_index=%d on %s (B12X_ROCE_GID_INDEX/NCCL_IB_GID_INDEX unset or negative)",
                    index, ",".join(names))
        _GID_AUTO_LOGGED = True
    return index


def default_gid_index() -> int:
    """``B12X_ROCE_GID_INDEX``, else NCCL's ``NCCL_IB_GID_INDEX``; negative or unset = auto.

    {MARKER}: auto picks the lowest index holding an IPv4 RoCE v2 GID on every
    selected HCA, as NCCL does for NCCL_IB_GID_INDEX=-1; else 3.
    """

    value = _env_int("B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX", default=-1)
    if value >= 0:
        return value
    auto = _auto_gid_index()
    return DEFAULT_GID_INDEX if auto is None else auto
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-root", required=True, help="site-packages containing b12x/")
    args = ap.parse_args()
    path = Path(args.site_root) / REL
    text = path.read_text()
    if MARKER in text:
        print(f"[{MARKER}] already applied to {path}")
        return 0
    count = text.count(ANCHOR)
    if count != 1:
        raise PatchError(
            f"ANCHOR FAILED for [default_gid_index] in {path}: expected exactly 1 occurrence, "
            f"found {count}. b12x has drifted -- re-derive the anchor.")
    path.write_text(text.replace(ANCHOR, REPLACEMENT, 1))
    print(f"[{MARKER}] patched {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PatchError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
