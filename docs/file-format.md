# Project file format (.vcp)

UTF-8 JSON. Unknown keys are ignored. Example (abridged):

```json
{
 "schema": 2,
 "app_version": "0.1.0",
 "settings": {"first": 1, "last": 100, "fps": 24.0, "format_w": 1920, "format_h": 1080,
              "working_space": "Linear Rec.709"},
 "nodes": [
  {"name": "Read1", "type": "Read", "x": 0.0, "y": 0.0,
   "knobs": {"file": {"value": "plate/sphere.####.exr"}, "last": {"value": [24.0]}},
   "inputs": []},
  {"name": "Blur1", "type": "Blur", "x": 0.0, "y": 60.0, "label": "soften", "color": "#6e5560",
   "knobs": {"size": {"value": [3.0, 3.0],
                      "curves": [[[1.0, 0.0, 2, 0.0, 0.0], [10.0, 8.0, 2, 0.0, 0.0]], []],
                      "exprs": ["", "Blur1.size.x"]}},
   "inputs": ["Read1", null]},
  {"name": "Roto1", "type": "Roto", "knobs": {}, "inputs": [null],
   "roto": [{"name": "garbage", "closed": true, "blend": 0, "feather": 30.0, "opacity": 1.0,
             "keys": [[1.0, [[x, y, in_x, in_y, out_x, out_y, feather, smooth], ...]]],
             "transform": {"tx": {"value": 0.0, "keys": []}, "rot": {"value": null, "keys": [[1.0, 0.0], [24.0, 90.0]]}}}]}
 ],
 "backdrops": [{"title": "Keying", "x": -40, "y": 280, "w": 320, "h": 200, "color": "#4a4a44",
                "note": false, "group": true, "members": ["Read1", "Blur1"], "collapsed": false}],
 "viewer": {"slots": {"1": "Merge2", "2": "Read1"}, "active": 1, "frame": 12}
}
```

* `knobs` only lists values that differ from the defaults. String-like knobs store `value` as a string, numeric
  knobs as a list (one entry per component).
* `curves`: per component, a list of keys `[frame, value, interpolation, in_slope, out_slope]`;
  interpolation 0 constant, 1 linear, 2 smooth, 3 bezier (uses the slopes, value per frame).
* `exprs`: per component, empty when none.
* `inputs`: one entry per input slot, the source node's name or `null`. Merge has eight image slots followed by
  its mask input; files from schema 1 are migrated from the older two-image-plus-mask layout.
* Group backdrops store member node names and a collapsed state. Backdrops and sticky notes without `group: true`
  retain their previous behavior.
* Roto `blend`: 0 over, 1 add, 2 subtract, 3 max, 4 min, 5 multiply. Point lists of all keys of a shape have the
  same length.
* Relative paths: *File > Save As (relative paths)* stores Read/Write paths relative to the project file; on load
  relative paths are resolved against the project's directory.

## Versioning

`schema` is an integer. `Project.migrate()` upgrades older schemas step by step. A file with a newer schema is
refused with a message; the UI offers to open it anyway (unknown data is ignored) and the CLI accepts `--force`.

## Autosave and recovery

Autosaves go to the user config directory (`%APPDATA%\VertexComp\autosave` on Windows,
`~/.config/vertexcomp/autosave` on Linux) every N minutes while there are unsaved changes, keeping the configured
number of rotating backups. A lock file marks a running session; if it is still present at the next launch, the
app offers to recover the latest autosave.
