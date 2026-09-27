# Public plugin contract

This document is the contract for a project that consumes the plugin
without depending on any host-side integration code. The plugin ships
no locomotion policy, no host actor types, and no host content.

## Public types and ownership

| Type | Owner | Use |
| --- | --- | --- |
| `MotionMatchingController` | host facade, plugin runtime | Configure assets, call `Initialize`, `Update`, `ResetPlayback`, and `Shutdown`. |
| `IMotionMatchingInputProvider` | host | Convert host state into one `MotionMatchingFrameInput` per tick. |
| `MotionMatchingInputContext` | plugin -> provider | Time, frame id, actor world transform, database tags, and sample times. |
| `MotionMatchingFrameInput` | provider -> plugin | Trajectory, scope, and explicit caller filters for one tick. |
| `PoseSearchSchema` | host authoring | Rig names, axes, sample times, and live search weights. |
| `MotionMatchingPlayback` | plugin, borrowed by host | Read current playback/base-pose state; never destroy the handle. |
| `MotionMatchingEditorService` | host editor tooling | Registry, bake, validation, audit, tuning, and top-candidate operations. |

The provider and controller are separate responsibilities. The host
calls the provider; the controller does not discover or call host
code. The native boundary receives only plain scope/filter data and
trajectory points.

## Coordinates and tick order

- `ActorWorld` is the actor's world transform, not parent-local transform.
- `TrajectoryPoint.Position` and `Facing` are character-local samples. Their
  interpretation uses the schema root forward/up axes; do not assume a global
  world axis in a host integration.
- `SampleTimes` must contain the six strictly ascending times used by the
  schema, spanning negative and positive time around zero.
- A host tick updates its provider state, calls `BuildInput(context)`, then
  calls `controller.Update(deltaTime, actorWorld, frame)`.

## Scope and filter semantics

`AllowedTagSet` and `ExcludedTagSet` are normalized `+`-joined sets.

- A non-empty include scope uses **overlap**: a clip is eligible when it has at
  least one included tag.
- An exclusion set is a **veto**: a clip is rejected when it has any excluded
  tag. Include and exclude both apply.
- At the controller/policy boundary, an empty `AllowedTagSet` is an explicit
  **no-match** command. The current clip holds; no retry widens the scope.
- At the low-level `MotionMatchingSearch` API, an empty `allowedTags` array
  means **all tags**, which is the standalone search primitive's historical
  contract. The runtime controller deliberately converts an empty frame scope
  into no-match before calling search.

### LoopFilter values

`MotionLoopFilter` (`Source/MotionMatching/Search/MotionMatchingQuery.h`)
mirrors the `LoopFilter` integer on `MotionMatchingFrameInput`:

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `Any` | Loops and one-shots are both eligible. |
| 1 | `LoopOnly` | Only clips flagged loop are eligible. |
| 2 | `OneShotOnly` | Only clips flagged one-shot are eligible. |

Out-of-range integers are clamped to `Any` at the controller boundary.
Normal warmed queries use the caller's values verbatim after range
validation; the core does not infer them from velocity, grounded
state, gait, or intent.

### CandidateBand values

`BakedClipSpeedBand` database ids mirror the `CandidateBand` integer
on `MotionMatchingFrameInput`:

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `Any` | No gait-band restriction; turns and unknowns stay eligible everywhere. |
| 1 | `Walk` | Restrict one-shots to the walk band. |
| 2 | `Run` | Restrict one-shots to the run band. |
| 3 | `Sprint` | Restrict one-shots to the sprint band. |

The band is assigned per clip at bake time from asset names and stored
in the database. Loops are never band-cut: the band restricts only
one-shot candidates. Out-of-range integers are clamped to `Any`. The
input DTO carries this value as the explicit `CandidateBand` integer;
the provider owns the enum mapping.

## Neutral LoopOnly invariant

The runtime intentionally has two structural recovery paths:

1. After `ResetPlayback` or a detected teleport/discontinuity, the first real
   reseed query uses `LoopOnly`.
2. If a query produces no result while playback reports `IsAtEnd`, the runtime
   retries once with temporal exclusion removed and `LoopOnly`.

Both paths retain the provider's same tag scope (include and exclude sets) and candidate
band. They do not name or invent an idle/rest tag, read gameplay intent, or
widen the tag scope. Therefore a caller's `OneShotOnly` request is overridden
for this structural recovery query; the normal caller filter is restored on
the next ordinary query. This is an explicit neutral invariant, not a
configuration option, and preserves continuity without changing normal search
behavior.

## Outputs, cleanup, and files

`Playback`, `Trace`, and similar native handles are borrowed. `Shutdown` is
public and idempotent; it releases policy/playback/trace state and leaves host
configuration reusable. Call it on host disable/destroy. Export trace and
telemetry before shutdown when those diagnostics are enabled.

`FlushTelemetry` is an explicit session-end operation. Runtime ticks do not
write telemetry files. `GenerateClipAudit` is also explicit and editor/tool
driven; initialization and play do not scan the database to create an audit.
