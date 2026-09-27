# Authoring and baking

## Schema and rig

Configure `PoseSearchSchema` for the consuming project's rig. `Root`,
`Pelvis`, `LeftFoot`, and `RightFoot` are required; additional named bones
(`Spine`, `Head`, `LeftHand`, `RightHand`) are stored for the declared
feature layout. Root forward and up axes must be finite, non-zero, and
non-parallel. The default trajectory times are six ascending values from
past through future (`-0.4, -0.2, 0.0, 0.2, 0.4, 0.6`) and must include
time zero.

Schema weights are query-time weights. Changing `PoseWeight` or
`TrajectoryWeight` does not require a rebake. Changes to feature layout,
sample rate, axes, skeleton extraction, or trajectory sampling do require a
new bake and a compatible database snapshot.

## Clip registry

`MotionMatchingEditorService` provides generic registry operations:

- `AddClips` and `AddClipFiles` register authored tags and loop flags.
- `ExportClipRegistry` writes deterministic, versioned JSON with content-
  relative paths, normalized tags, and loop values.
- `ImportClipRegistry` replaces the whole registry, groups entries by sorted
  tag set and loop value, and reports skipped entries.
- `ClipRegistryImportOptions` is the explicit extension point for a host path
  resolver or tag/loop inference. The generic editor has no folder, filename,
  or project-specific defaults: any naming convention must be supplied by
  the host through these callbacks.

`ClipRegistryImportOptions` has three optional callbacks:

- `InferTags`: `Func<string, string[]>` called with the resolved file path
  when a registry entry carries no tags. Return the host tags for that
  file, or null to leave the entry tagless (tagless entries are skipped).
- `InferLoop`: `Func<string, bool>` called with the resolved file path
  when a registry entry carries no loop value. Return the host loop flag
  for that file. Without it, a missing loop value defaults to `false`.
- `ResolvePath`: `Func<string, string, string>` called with the registry
  asset path and the registry file path. Return the resolved filesystem
  path. Without it, the importer tries the host Content folder first,
  then the registry file folder.

Registry JSON has this shape:

```json
{
  "version": 1,
  "clips": [
    {"path": "Animations/WalkLoop.flax", "tags": ["walk"], "loop": true}
  ]
}
```

The top-level `version` is an integer (currently `1`) and `clips` is the
entry array. Each entry has a `path` string (Content-relative preferred,
absolute accepted on import), a `tags` string array, and an optional
`loop` boolean. Paths may be Content-relative or absolute on import. Tags
are lowercased, trimmed, and de-duplicated by the database. A missing
loop value defaults to `false` unless `InferLoop` is explicitly supplied.
An entry without tags is skipped unless `InferTags` supplies them.
Import replaces the whole registry and groups entries by sorted tag set
plus loop value; the returned summary reports added groups and skipped
entries, and a bake is required afterwards.

## Bake and validate

Use `BakeDatabase`, or the combined `BakeValidateAndTest`, after the registry
and schema are configured. `BakeDatabase` samples the registered source
clips, bakes normalized pose and trajectory features, stores the schema
snapshot, and saves the database asset; the editor service reloads the
database after edits. Run `ValidatePersistence` and the editor-only search
self-test before shipping the asset:

```csharp
MotionMatchingEditorService.BakeDatabase(controller);
MotionMatchingEditorService.ValidatePersistence(controller);
MotionMatchingEditorService.BakeValidateAndTest(controller); // bake + validate + self-test
MotionMatchingEditorService.QueryTopCandidates(controller, 8); // inspect ranking, no play needed
```

`ValidatePersistence` checks the saved asset through
`MotionMatchingDatabase.ValidateSavedDatabase` and refreshes
`PersistenceValid`. `GenerateClipAudit` is an explicit diagnostic
command and writes `Cache/MMClipAudit.txt` only when requested:

```csharp
MotionMatchingEditorService.GenerateClipAudit(controller);
```

The default registry location is
`<ProjectContent>/MotionMatchingRegistry.json` (see
`DefaultRegistryPath`). Registry paths are authoring data and can be
resolved by `ClipRegistryImportOptions.ResolvePath` when source files
move between machines or Content layouts.
