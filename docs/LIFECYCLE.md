# Lifecycle guide

The controller lifecycle is deterministic and host-driven.

## Start

Configure `Schema`, `PlayerModel`, `CharacterBody`, `DatabasePath`, and any
session tuning before `Initialize`. `Initialize` releases a stale binding and
loads the configured database. The first `Update` creates the native policy if
the database is baked and the body has a compatible `SkinnedModel`.

The host should verify `Database.IsBaked`, `PersistenceValid`, and `IsReady`
before treating playback output as available. A missing database, invalid bake,
missing body/model, or unresolved required skeleton node leaves the controller
disabled and produces an actionable log.

## Tick

The host owns the tick order: update provider state, build the frame, call
`Update`, read observations/base pose, then apply host-owned pose passes. The
controller samples the base/search pose, builds trajectory data, runs the
continuation fast path or a full search, and advances playback.

## Reset and discontinuity

For teleport, respawn, or another intentional discontinuity, call both:

```csharp
provider.Reset();
controller.ResetPlayback();
```

`ResetPlayback` clears trajectory/chooser history while preserving the current
playback clip and pose continuity. The next real query is a zero-velocity
reseed and is forced to `LoopOnly`; see the neutral invariant in
PLUGIN_CONTRACT.md. It does not require a transform argument. The runtime
also detects a large unannounced actor displacement and applies the same
structural reset.

## Rebind

Changing the database reference, database path, character body/model, or any
schema field that defines the binding causes the old native session to be
released before the next native read. Query weights are live tuning and do not
by themselves change binding identity. After a rebind, reacquire borrowed
`Playback` and `Trace` handles.

## Stop

At host disable/destroy, export any diagnostics, call `Shutdown`, and discard
borrowed handles. `Shutdown` may be called repeatedly and the configured
controller can be initialized/updated again later. `Dispose` is an equivalent
host convenience.
