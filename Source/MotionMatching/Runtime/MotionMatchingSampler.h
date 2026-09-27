// MotionMatchingSampler.h: pose sampler and shared sample-domain math (MotionMatchingSampler, MotionSamplingStatus).
// Ownership: plugin runtime, no game/host references.
// Key invariants: SampleClipPose/SampleClipLocalPose/GetNodeReferenceLocal/GetSkeletonHierarchy/StripRootLocal/BlendLocalPoses/ComposeModelPose share one channel-evaluation copy with the baker (game builds never link editor baker code); GetFrameDt/ComputeSampleCount/ComputeSampleTime/ClassifyClipTime/WrapLoopTime/TimeToSampleIndex/GetCentralDiffIndices/IsEndpointSample/UnwrapYawDelta/ComputeYawRate/YawFromForwardXZ/IsTrajectoryTargetValid/IsNonLoopFutureOutside are the single sample-domain definition (loop wrap to [0, length), interior-central / edge-one-sided stencils over the true dt, snapshot-copy outputs, row-vector childLocal * parentWorld).
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Scripting/ScriptingType.h"

class Animation;
class SkinnedModel;

/// <summary>
/// Shared sampling contract.
/// Single definition of the sample-domain semantics used by the editor baker
/// (MotionMatchingBaker.cpp) and the live query (MotionMatchingQuery.cpp).
/// Both sides must call MotionMatchingSampler helpers below; duplicating the
/// formulas locally (wrap math, index rounding, yaw unwrap, dt guards) is
/// forbidden — drift between two copies is exactly what broke seam/endpoint
/// parity before.
///
/// Contract fields:
/// - sample rate: schema SampleRate; frame dt = 1/rate (GetFrameDt).
/// - sample time: local index i gives time = min(i/rate, clipLength), count =
///   ceil(length*rate), duration = true clip length (ComputeSampleCount /
///   ComputeSampleTime). count/rate is at least length; the last interval may be short.
/// - interpolation: pose channels = Flax channel Evaluate at the exact sample
///   frame (no added resampling on top); root-at-time = nearest baked sample
///   + cycle transform (bake-side GetRootAtTimeWithStatus).
/// - finite-difference dt: interior central over two frames, non-loop edges
///   one-sided over one frame. The dt is always the TRUE
///   time span of the stencil actually used, never a hardcoded 2*dt against
///   a clamped twin.
/// - endpoint mode: non-loop timestamps outside [0, length] return
///   OutsideNonLoop (position clamps to the endpoint for playback continuity
///   but the STATUS tells feature/search code it is outside); degenerate
///   input returns Invalid. Silent clamp-then-derive is forbidden.
/// - loop mode: time wraps to [0, length) with a cycle index; root displacement
///   across the seam applies CycleRootTransform^cycle; each side of a ±dt
///   stencil resolves its own cycle so seam neighbors come from the adjacent
///   cycle, not from a duplicated endpoint.
///
/// Timescale note (bake vs query): pose velocity is central at bake
/// (±1 bake sample, non-causal) but backward at query (causal live frame —
/// central would need the future), so live velocity lags by half a sample
/// interval by construction; yaw-rate stencil is ±1 bake sample at bake vs
/// ±1 trajectory interval (schema times) at query. Both sides divide
/// unwrapped yaw by their OWN true dt through ComputeYawRate, so rad/s means
/// the same thing everywhere. The lag is documented, not numerically
/// compensated (within trajectory tolerance).
///
/// Non-loop tail validity: baked future targets past the clip end keep
/// a clamped endpoint pose for playback, but search must treat them as
/// invalid/penalized via IsNonLoopFutureOutside (runtime horizon check from
/// clip lengths + sample times, no rebake — same pattern as the existing
/// LoopTailExclusion). Search integration lives in the search layer; the sampler
/// only provides the helper and stops fabricating zero futures.
/// </summary>
enum class MotionSamplingStatus : int32
{
    Valid = 0,
    OutsideNonLoop = 1,
    Invalid = 2,
};

/// <summary>
/// Runtime clip pose sampler for motion-matching playback.
/// Duplicates the baker's channel-evaluation math (kept separate so the game
/// build never depends on editor-only baker code); outputs actor-space node
/// transforms ready for AnimatedModel.SetCurrentPose(worldSpace: false).
/// </summary>
API_CLASS(Static, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingSampler
{
    DECLARE_SCRIPTING_TYPE_NO_SPAWN(MotionMatchingSampler);

public:
    /// <summary>
    /// Samples one clip at the given time and writes per-node actor-space
    /// transforms. Time is clamped into the clip range by the caller.
    /// </summary>
    API_FUNCTION()
    static bool SampleClipPose(
        Animation* clip,
        SkinnedModel* model,
        float time,
        API_PARAM(Out) Array<Matrix>& outPose,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Returns the bind (reference) local matrix of one skeleton node, for
    /// root-motion stripping that preserves skeleton calibration. Playback
    /// restores this reference instead of identity so imported root rotation /
    /// scale and root height survive while animated travel/yaw are removed.
    /// </summary>
    API_FUNCTION()
    static bool GetNodeReferenceLocal(
        SkinnedModel* model,
        int32 nodeIndex,
        API_PARAM(Out) Matrix& outLocal,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Samples one clip at the given time and writes per-node BONE-LOCAL
    /// transforms (no hierarchy composition). This is the hierarchy-correct
    /// source for stripping and blending: root motion is removed and poses
    /// are blended in local space BEFORE forward kinematics, so descendants
    /// stay consistent and bone lengths are preserved. Compose with
    /// ComposeModelPose for the final actor-space pose expected by
    /// AnimatedModel.SetCurrentPose(worldSpace: false).
    /// Snapshot contract: the output array is a FRESH snapshot copy every call —
    /// never an alias into baker/clip storage — so playback-side IK/layer may
    /// mutate it in place while the saved search snapshot stays intact.
    /// Row-vector convention: ComposeModelPose builds
    /// childLocal * parentWorld (parent on the RIGHT).
    /// NOTE: channel evaluation below duplicates the baker's
    /// math by design (game build must not depend on editor-only baker code).
    /// The two copies must stay in sync; any change here requires the same
    /// change in MotionMatchingBaker.cpp plus a golden-vector comparison.
    /// </summary>
    API_FUNCTION()
    static bool SampleClipLocalPose(
        Animation* clip,
        SkinnedModel* model,
        float time,
        API_PARAM(Out) Array<Matrix>& outLocals,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Returns the skeleton hierarchy (parent index per node, -1 for roots)
    /// plus the bind-pose local matrix per node. Playback caches this once at
    /// Setup so per-frame work needs no skeleton walks.
    /// </summary>
    API_FUNCTION()
    static bool GetSkeletonHierarchy(
        SkinnedModel* model,
        API_PARAM(Out) Array<int32>& outParents,
        API_PARAM(Out) Array<Matrix>& outReferenceLocals,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Copies locals to outStripped with the root bone pinned to its bind
    /// reference (keeps the sampled root scale, restores reference rotation
    /// and translation). Must run in LOCAL space before ComposeModelPose:
    /// editing only the root of a composed model-space pose leaves every
    /// descendant carrying the old root travel/yaw.
    /// </summary>
    API_FUNCTION()
    static bool StripRootLocal(
        const Array<Matrix>& locals,
        int32 rootIndex,
        const Matrix& referenceRootLocal,
        API_PARAM(Out) Array<Matrix>& outStripped,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Blends two local-space poses bone by bone (lerp translation/scale,
    /// slerp rotation) then returns the blended locals. Blending must happen
    /// in local space: lerping composed model-space matrices independently
    /// breaks parent/child consistency and visibly stretches bones.
    /// </summary>
    API_FUNCTION()
    static bool BlendLocalPoses(
        const Array<Matrix>& from,
        const Array<Matrix>& to,
        float t,
        API_PARAM(Out) Array<Matrix>& outBlended,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Forward kinematics: composes bone-local matrices into actor-space
    /// (model-space) matrices parent-first, the layout SetCurrentPose expects.
    /// </summary>
    API_FUNCTION()
    static bool ComposeModelPose(
        SkinnedModel* model,
        const Array<Matrix>& locals,
        API_PARAM(Out) Array<Matrix>& outModelPose,
        API_PARAM(Out) String& error);

    // ---- Shared sampling contract ----
    // Pure helpers, no engine state. The baker and the live query must call
    // these instead of copying literals or mirror logic. Deliberately NOT
    // API_FUNCTION (C# scripting does not need them; C++-only keeps Matrix /
    // reference signatures free of binding constraints).
    static float GetFrameDt(int32 sampleRate);
    static int32 ComputeSampleCount(float clipLength, int32 sampleRate);
    static float ComputeSampleTime(int32 localIndex, float clipLength, int32 sampleRate);
    static MotionSamplingStatus ClassifyClipTime(float time, float clipLength, bool loop, float& outLocalTime, int32& outCycleIndex);
    static float WrapLoopTime(float time, float clipLength, int32& outCycleIndex);
    static MotionSamplingStatus TimeToSampleIndex(float time, float clipLength, int32 sampleCount, int32 sampleRate, bool loop, int32& outLocalIndex);
    static void GetCentralDiffIndices(int32 localIndex, int32 sampleCount, bool loop, int32& outPrevLocal, int32& outNextLocal);
    static bool IsEndpointSample(int32 localIndex, int32 sampleCount, bool loop);
    static float UnwrapYawDelta(float deltaYaw);
    static float ComputeYawRate(float yawPast, float yawFuture, float dtTotal);
    static float YawFromForwardXZ(float x, float z);
    static bool IsTrajectoryTargetValid(float targetTime, float clipLength, bool loop);
    static bool IsNonLoopFutureOutside(float sampleTime, float horizon, float clipLength, bool loop);
};
