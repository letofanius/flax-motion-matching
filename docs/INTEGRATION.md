# Integration guide

This plugin is consumed by a Flax 1.12 project as a normal project
reference. The host owns the actor, the input provider, movement
authority, the camera, and any pose post-processing. The plugin owns
database loading, pose search, playback, and the neutral input/output
contract.

## Project and module setup

1. Copy or clone this repository to
   `<HostProject>/Plugins/MotionMatching`, then reference
   `Plugins/MotionMatching/MotionMatching.flaxproj` from the host
   project's `.flaxproj` next to the engine reference:

   ```json
   "References": [
       { "Name": "$(EnginePath)/Flax.flaxproj" },
       { "Name": "$(ProjectPath)/Plugins/MotionMatching/MotionMatching.flaxproj" }
   ]
   ```

2. Add `MotionMatching` to the game target modules
   (`MotionMatchingTarget` is the plugin's own game target; the host
   game target adds the `MotionMatching` module):

   ```csharp
   Modules.Add("MotionMatching");
   ```

3. Add `MotionMatchingEditor` to the editor target modules when editor
   bake, registry, tuning, or contract-test tooling is needed
   (`MotionMatchingEditorTarget` is the plugin's own editor target):

   ```csharp
   Modules.Add("MotionMatching");
   Modules.Add("MotionMatchingEditor");
   ```

4. Implement `IMotionMatchingInputProvider` in the host. The plugin
   ships no locomotion module and no host content; gait, scope, and
   intent decisions live in the host provider.

The plugin project itself references only the Flax engine
(`$(EnginePath)/Flax.flaxproj`, `MinEngineVersion` 1.12.6912). The
runtime module does not reference a game assembly, an actor type, a
player coordinator, or host content.

## Schema authoring pointer

Author one `PoseSearchSchema` per consuming rig before baking. The
skeleton profile ships empty: set `Skeleton.Root`, `Skeleton.Pelvis`,
`Skeleton.LeftFoot`, and `Skeleton.RightFoot` to the host bone names
(scene to `MotionMatchingController` to `Schema` to `Skeleton`).
`Spine`, `Head`, `LeftHand`, and `RightHand` are optional and stored
for future channels. Root forward/up axes must be finite, non-zero,
and non-parallel. Keep the six default `TrajectorySampleTimes`
(`-0.4, -0.2, 0.0, 0.2, 0.4, 0.6`) unless the host predictor is
retuned to match; `PoseWeight` and `TrajectoryWeight` are live query
weights and do not require a rebake. Full authoring detail is in
AUTHORING.md; the coordinate contract is in PLUGIN_CONTRACT.md.

## Provider implementation

The provider converts host state into one `MotionMatchingFrameInput`
per tick. `MotionMatchingInputContext` carries only what the plugin
side provides: `ActorWorld`, `DeltaTime`, `FrameId`,
`DatabaseTags` (baked tag inventory, null skips validation), and
`SampleTimes` (strictly ascending offsets). The controller has no
provider slot and never calls the provider itself.

```csharp
public MotionMatchingFrameInput BuildInput(MotionMatchingInputContext ctx)
{
    // 1. Advance the host trajectory predictor from actor state.
    //    Points are character-local and aligned 1:1 with ctx.SampleTimes.
    // 2. Derive scope: '+'-joined AllowedTagSet / ExcludedTagSet.
    //    Empty AllowedTagSet is an explicit no-match command.
    // 3. Derive the caller filter: LoopFilter (0 Any, 1 LoopOnly,
    //    2 OneShotOnly), CandidateBand (0 Any, 1 Walk, 2 Run, 3 Sprint),
    //    QueryYawRate (rad/s, >= 0), CallerPlanarSpeed (units/s, >= 0).
    return new MotionMatchingFrameInput
    {
        Points = predictedPoints,
        AllowedTagSet = includeScope,
        ExcludedTagSet = excludeScope,
        LoopFilter = 0,
        CandidateBand = 0,
        QueryYawRate = 0.0f,
        CallerPlanarSpeed = 0.0f,
    };
}
```

`Reset()` snaps scope hysteresis and reseeds the predictor lazily on
the next `BuildInput`. It takes no transform argument. The host calls
it alongside `controller.ResetPlayback()` on teleport or respawn.

## Controller init, tick, reset, shutdown

`MotionMatchingController` is a plain C# object; it is not a scene
script. Create and retain one controller in the host facade, then
configure it before `Initialize`:

```csharp
var controller = new MotionMatchingController
{
    Schema = hostSchema,               // PoseSearchSchema for the host rig
    PlayerModel = hostModel,           // SkinnedModel
    CharacterBody = hostAnimatedModel, // AnimatedModel
    DatabasePath = "MotionMatching/MyDatabase.flax",
};
controller.Initialize();
```

`Initialize` releases any stale binding and loads the configured
database. Verify `Database.IsBaked`, `PersistenceValid`, and `IsReady`
before treating playback output as available. A missing database, an
invalid bake, a missing body or model, or an unresolved required
skeleton node leaves the controller disabled and produces an
actionable log.

On every host tick:

```csharp
MotionMatchingInputContext context = BuildContext();
MotionMatchingFrameInput frame = provider.BuildInput(context);
controller.Update(deltaTime, actorWorldTransform, frame);
```

The provider owns gameplay interpretation and produces trajectory
points, include/exclude tag sets, the loop filter, the candidate
speed band, and query yaw rate. The controller validates filter enum
ranges, loads the database on first use when necessary, and forwards
the frame to the native policy.

The base search pose is available through `TryGetBasePose` or the
reusable `TryGetBasePoseInto` overload. The host may apply its own
layers or IK and writes the final pose. Playback is a borrowed
handle: do not destroy it, and reacquire it after reset, rebind, or
shutdown. `Shutdown` is public and idempotent; call it on host
disable or destroy and export diagnostics first when enabled.
`Dispose` is an equivalent host convenience. See LIFECYCLE.md for the
full start, tick, reset, rebind, and stop sequence.

## AnimGraph warning

The body must not run an AnimGraph that overwrites manually written
poses. The plugin writes the base pose through the controller; the
host owns all layering and IK after that write. Disable or bypass
any competing graph output on the `CharacterBody` (`AnimatedModel`),
or the graph will fight the motion-matching pose every frame.
Character movement and root motion remain host responsibilities.

## Database path

`DatabasePath` may be a host Content-relative path or an absolute
path. The default is `MotionMatching/MotionMatchingDatabase.flax`.
The host database asset must be baked with a schema compatible with
the configured rig; the database stores the bake-time schema
snapshot. `DatabaseFile` resolves the configured path against the
host Content folder when it is not rooted.
