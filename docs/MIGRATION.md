# Migration and asset identity

## Namespaces and the database TypeName

The public scripting namespace is `MotionMatching`. The serialized binary
database identity is a separate compatibility key and is intentionally frozen:

```text
Game.MotionMatchingDatabase
```

This legacy TypeName is the load key embedded in existing `.flax` database
headers. Keep it when moving the plugin between Flax projects. Do not rewrite
database JSON, scene text, or asset headers from `Game.MotionMatchingDatabase`
to `MotionMatching.MotionMatchingDatabase`; that would orphan existing baked
assets. A future intentional identity change requires a dedicated asset-header
migration and rebake plan.

The legacy key does not mean the plugin runtime depends on a game assembly. It
is only the binary asset factory key. The plugin's C# and native public types
are under the `MotionMatching` namespace.

History note: the `Game.` prefix is a leftover from when the database type
was first prototyped inside a host game assembly. The type now ships in this
plugin; only the serialized key keeps the old prefix for compatibility.

## Moving a host project

1. Add the plugin project reference and target module dependencies described in
   INTEGRATION.md.
2. Copy the database asset into the new project's Content tree or update the
   host's `DatabasePath` to its new Content-relative location.
3. Preserve the database's serialized TypeName and validate that the asset
   loads as `MotionMatchingDatabase`.
4. Recreate the host schema for the destination rig; do not assume a schema
   snapshot from another rig is compatible.
5. Export/import the clip registry when source paths change. Registry paths are
   authoring data and can be resolved by `ClipRegistryImportOptions`.
6. Rebake when schema contract, feature layout, sampling, rig extraction, or
   source animation metadata changes. A stale database is registry-only until
   it is rebaked; a structurally corrupt database must not be treated as valid.

## Legacy policy

The public migration guarantee is limited to preserving the database asset
identity and the documented runtime/editor APIs. Host scene components and
project-specific providers remain host-owned and must be migrated by the host.
The plugin does not silently infer old player types, bundled content folders, or
filename conventions during import.
