"""Headless renderer: vertexcomp-render project.vcp --first 1001 --last 1100 --write Write1"""
from __future__ import annotations

import argparse
import sys
import time

from vertexcomp import _vccore as vc
from vertexcomp import movie
from vertexcomp.project import Project, SchemaError, write_node_range


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vertexcomp-render", description="Render Write nodes of a Vertex Comp project.")
    ap.add_argument("project")
    ap.add_argument("--first", type=int, help="first frame (default: the Write node's range)")
    ap.add_argument("--last", type=int, help="last frame")
    ap.add_argument("--write", action="append", default=[], help="Write node to render (repeatable; default: all)")
    ap.add_argument("--threads", type=int, default=0, help="worker threads (0 = all cores)")
    ap.add_argument("--cache-gb", type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="open projects saved by a newer version")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    vc.set_threads(args.threads)
    movie.install(vc)
    proj = Project(int(args.cache_gb * (1 << 30)))
    try:
        proj.load(args.project, allow_newer=args.force)
    except SchemaError as e:
        print(f"error: {e} (use --force to try anyway)", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"error: cannot open {args.project}: {e}", file=sys.stderr)
        return 2

    writes = [n for n in proj.nodes() if n.type == "Write" and not n.disabled]
    if args.write:
        names = set(args.write)
        missing = names - {n.name for n in writes}
        if missing:
            print(f"error: no enabled Write node named {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        writes = [n for n in writes if n.name in names]
    if not writes:
        print("error: nothing to render", file=sys.stderr)
        return 2

    status = 0
    for w in writes:
        first = args.first if args.first is not None else int(w.get("first")[0])
        last = args.last if args.last is not None else int(w.get("last")[0])
        t0 = time.perf_counter()

        def progress(done, total, frame, name=w.name):
            if not args.quiet:
                print(f"\r{name}: {done}/{total} (frame {frame})", end="", flush=True)

        try:
            write_node_range(proj, w, first, last, progress=progress)
            if not args.quiet:
                print(f"\n{w.name}: {last - first + 1} frames in {time.perf_counter() - t0:.2f}s")
        except RuntimeError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
