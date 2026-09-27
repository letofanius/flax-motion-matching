using FlaxEngine;

namespace MotionMatching;

// Motion-matching input contract: host facade <-> plugin runtime boundary.
//
// Purpose: declare the per-tick data the host builds and the plugin consumes.
// Ownership: this file is plugin runtime owned; the host facade implements
// IMotionMatchingInputProvider and calls BuildInput itself, then passes the
// resulting frame to MotionMatchingController.Update.
// Tick/lifecycle role: BuildInput runs once per tick on the host side; the
// controller holds no provider slot and never calls BuildInput or Reset
// itself. Reset runs on teleport/respawn alongside controller.ResetPlayback.
// This file must never reference game/player types or sample-project paths:
// it names only plugin types (controller, frame, context, trajectory point).
//
// Managed data-transfer objects only: MotionMatchingInputContext never
// crosses the native boundary (C# -> C#), so managed arrays are fine here.
// Only the plain String/POD fields copied into MotionMatchingPlayerSnapshot /
// MotionMatchingPolicyTuning cross into C++.

/// <summary>
/// The single call the motion-matching side needs from host-side code,
/// once per tick. The provider owns the player snapshot (pushed host-side
/// via its own Update method), all hysteresis/latch state, and the
/// trajectory predictor. ctx carries only what the motion-matching side
/// provides: time, world transform, schema inventory and database inventory.
/// BuildInput assembles the frame in one call: snapshot, then trajectory,
/// then scope reads the fresh points, then the frame.
/// </summary>
public interface IMotionMatchingInputProvider
{
    MotionMatchingFrameInput BuildInput(MotionMatchingInputContext ctx);

    /// <summary>
    /// Snap scope hysteresis and reseed the predictor. The host calls this
    /// alongside controller.ResetPlayback (teleport/respawn). Must not require a transform:
    /// predictor reseed is lazy on the next BuildInput.
    /// </summary>
    void Reset();
}

/// <summary>
/// Motion-matching-side input to the provider. No raw movement here on purpose:
/// movement arrives host-side through the provider's own Update method,
/// so there is exactly one writer and no push-then-tick staleness.
/// </summary>
public struct MotionMatchingInputContext
{
    // Actor WORLD transform (Flax Actor.Transform is world space; parent-local
    // would be Actor.LocalTransform). The predictor and the query run in this
    // frame; predicted Points are actor-local offsets. Verified against
    // Engine/Level/Actor.h (Transform = world, LocalTransform = parent-local).
    public Transform ActorWorld;
    public float DeltaTime;
    public int FrameId;
    // Baked tag inventory for validate-then-drop (null = skip validation).
    public string[] DatabaseTags;
    // Trajectory sample offsets, seconds, strictly ascending.
    public float[] SampleTimes;
}

/// <summary>
/// Everything motion matching needs for one tick, produced in one call.
/// Contents: trajectory + scope + caller filter only. All gameplay decisions
/// (gait roles, speed ceilings, yaw/direction thresholds, driver flags,
/// overlay angles/gates, velocities, grounded/sprint) live host-side in the
/// provider/selection policy and never cross into the core. Includes use tag
/// overlap; exclusions veto any overlap. Empty AllowedTagSet = explicit no-match.
/// Normal queries use the caller filter. Structural reseed/end-recovery
/// searches override LoopFilter to LoopOnly within the same tag scope and
/// exclusions, even for OneShotOnly callers (docs/PLUGIN_CONTRACT.md).
/// </summary>
public struct MotionMatchingFrameInput
{
    // 1. Predicted trajectory, character-local, aligned 1:1 with the
    // sample times from the context. Empty/null = prediction failed;
    // the core skips the query, same as a failed trajectory build.
    public TrajectoryPoint[] Points;

    // 2. Scope: '+'-joined include/exclude tag sets decided provider-side.
    // Empty AllowedTagSet = explicit no-match (hold current, never fall
    // back to any pool). Every query is provider-scoped; there is no
    // fallback chooser inside the core.
    public string AllowedTagSet;
    public string ExcludedTagSet;

    // 3. Caller filter (host selection-policy output):
    // LoopFilter mirrors MotionLoopFilter (0 Any, 1 LoopOnly, 2 OneShotOnly);
    // CandidateBand mirrors BakedClipSpeedBand DB ids (0 Any, 1 Walk, 2 Run,
    // 3 Sprint); QueryYawRate (rad/s, >= 0) scales the straightness prior.
    // CallerPlanarSpeed (u/s, >= 0) is the host-side character planar speed
    // for play-rate only (clip time bends to world speed, clamped 0.85-1.15).
    // The core never infers these from movement.
    public int LoopFilter;
    public int CandidateBand;
    public float QueryYawRate;
    public float CallerPlanarSpeed;
}

// History note: earlier additive-only filter/tuning DTOs were never read by
// Tick and are deleted. The live host-side request object carries scope into
// the core through MotionMatchingFrameInput above (direct path); intent/loop/
// band mirrors stay host-side. No native struct field is added
// (MotionMatchingTraceEntry stays 61/64 for the Flax 1.12 bindings cap).
