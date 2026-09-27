// MotionMatchingPlayback.h: manual clip playback with inertial blending (MotionMatchingPlayback: Setup/Play/Stop/Tick, TryDecompose/CalcInertialFloat/BuildInertialOffsets/TwistAbout).
// Ownership: plugin runtime, no game/host references.
// Key invariants: Playback is the only place that applies time (PlayRate clamped to [0.85, 1.15]); strip/blend run in bone-local space before ComposeModelPose forward kinematics; the post-inertial snapshot (GetSearchLocals/GetSearchLocal) is a deep copy the provider post-passes; Tick asserts the sample-search-save order via SampleFrameId/SearchFrameId/SaveSearchPoseFrameId; turn-in-place yaw comes from the ConjugateQ(prev)*cur delta quaternion once per tick.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Quaternion.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Scripting/ScriptingObject.h"
#include "../Schema/MotionMatchingTypes.h"

class AnimatedModel;
class SkinnedModel;
class Animation;
class MotionMatchingDatabase;


/// No pose-overlay seam: the provider applies its own post passes onto the
/// GetSearchLocals snapshot, then composes and writes the final pose itself.
/// Playback never calls out.



/// <summary>
/// Manual clip playback. Advances the winning clip at the
/// baked sample rate, wraps loop clips, clamps one-shots, and blends between
/// clips with quintic inertialization (UE AnimNode_Inertialization port, see
/// CalcInertialFloat): offset = oldPose - newPose decayed per bone with zero
/// end value/velocity/acceleration, so phase-mismatched switches settle
/// without the foot-skate of linear crossfade. Root motion stays Disabled:
/// CharacterController owns movement. Playback advances time and holds
/// the inertial search pose only — it never overlays, solves IK, composes
/// or writes the final pose (the provider owns all post passes).
/// Requires the body to have NO AnimGraph assigned — the graph would
/// overwrite the game-written pose every frame.
///
/// Hierarchy correctness: stripping and blending MUST happen in bone-LOCAL
/// space before forward kinematics. This class therefore samples locals,
/// strips the root local to its bind reference, blends locals (native TRS
/// blend) and snapshots the result as the search pose. No overlay, no
/// IK, no compose, no pose write here — the provider owns all post passes.
/// </summary>
API_CLASS(Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingPlayback : public ScriptingObject
{
    DECLARE_SCRIPTING_TYPE_WITH_CONSTRUCTOR_IMPL(MotionMatchingPlayback, ScriptingObject);

public:
    // Play-rate ownership: Playback is the ONLY place that
    // applies time. Rate defaults to 1.0, clamped to [0.85, 1.15].
    API_PROPERTY() float GetPlayRate() const;
    API_PROPERTY() void SetPlayRate(float value);
    API_PROPERTY() float GetPlayRateBefore() const;
    API_PROPERTY() float GetPlayRateAfter() const;
    API_FIELD() float LastEffectiveDt = 0.0f;
    API_FIELD() int32 ClipCycle = 0;
    // Inertialization blend time (seconds, replaces linear-crossfade
    // BlendDuration — single path, no parallel blend modes).
    API_FIELD() float InertializationDuration = 0.25f;
    // True while a decay offset is active (telemetry only).
    API_PROPERTY() bool GetInertialActive() const;

    // Turn-in-place visual yaw: per-tick root yaw delta in radians,
    // measured AFTER sample-time update and BEFORE inertial/strip/pose-snapshot.
    // C# binds as read-only property RootYawDelta (same pattern as PlayRate).
    API_PROPERTY() float GetRootYawDelta() const { return _rootYawDelta; }


    // No foot solving here: the provider owns all foot IK; baked contact
    // channels (bake + query features) are untouched by that separation.



    API_FIELD() int32 CurrentClip = -1;
    API_FIELD() float CurrentTime = 0.0f;
    API_FIELD() bool IsAtEnd = false;
    API_PROPERTY() float GetCurrentClipLength() const;

    API_FIELD() int64 SampleFrameId = -1;
    API_FIELD() int64 SearchFrameId = -1;
    API_FIELD() int64 SaveSearchPoseFrameId = -1;

public:
    API_FUNCTION() void Setup(
        MotionMatchingDatabase* database,
        SkinnedModel* fallbackModel,
        const SkeletonProfile& skeleton,
        const Float3& controllerUpAxis);
    API_FUNCTION() void Stop();
    API_FUNCTION() void Play(int32 clipIndex, float startTime, bool forceSeek);
    API_FUNCTION() void ResetOnTeleport();
    API_FUNCTION() void NotifySearchExecuted(int64 tickFrameId);
    API_FUNCTION() Array<Matrix> GetSearchLocals() const;
    /// <summary>
    /// Zero-alloc pose read: bone count for the copy API below. Returns 0
    /// when no search pose is available. Bound: model bone count (91 on the
    /// reference rig, reuse buffers cap 512).
    /// </summary>
    API_FUNCTION() int32 GetSearchLocalsCount() const;
    /// <summary>
    /// Zero-alloc pose read: copies one bone local into the caller struct
    /// (no managed array allocation, unlike GetSearchLocals). Returns false
    /// when the index is out of range (outLocal set to identity). The caller
    /// reuses its own buffers and resizes only when GetSearchLocalsCount
    /// changes. Same snapshot data as GetSearchLocals (post-inertial search
    /// pose); the native history is never aliased with game-owned buffers.
    /// </summary>
    API_FUNCTION() bool GetSearchLocal(int32 index, API_PARAM(Out) Matrix& outLocal) const;
    /// <summary>
    /// Internal fast path (no bindings): const reference to the search
    /// snapshot for the policy Tick sampler. Avoids the Array-by-value copy
    /// in GetSearchLocals (which stays for diagnostics/tools). Not exposed
    /// to scripting (no API_FUNCTION) so it needs no binding regen.
    /// </summary>
    const Array<Matrix>& GetSearchLocalsRef() const { return _searchLocals; }
    API_FUNCTION() void Tick(float deltaTime, AnimatedModel* body, bool grounded);

#if USE_EDITOR
    API_FUNCTION() static bool RunPlaybackSelfTest(MotionMatchingDatabase* database, AnimatedModel* body, API_PARAM(Out) String& log);
#endif

private:
    MotionMatchingDatabase* _database = nullptr;
    SkinnedModel* _fallbackModel = nullptr;
    SkinnedModel* _hierarchyModel = nullptr;
    int32 _rootNodeIndex = -1;
    Array<Animation*> _clips;
    Array<float> _lengths;
    Array<bool> _loops;

    Array<int32> _parents;
    Array<Matrix> _referenceLocals;
    Matrix _referenceRootLocal = Matrix::Identity;
    bool _hasReferenceRoot = false;

    float _playRate = 1.0f;
    float _playRateBefore = 1.0f;
    // Turn-in-place root yaw delta state (radians, once per tick).
    float _rootYawDelta = 0.0f;
    Quaternion _prevRootSampleRotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    bool _hasPrevRootSample = false;
    Float3 _rootUpAxis = Float3::Zero;
    bool _hasRootUpAxis = false;
    // Lazy Play->first-tick baseline (no sampling in Play): Play stores its
    // target time here; the next Tick samples the baseline pose (same clip)
    // and reports the full Play->tick interval without skipping the frame.
    float _rootYawPlayStartTime = 0.0f;
    bool _hasRootYawPlayStart = false;

    Array<Matrix> _searchLocals;
    Array<Matrix> _prevSearchLocals;

    int64 _tickFrameId = 0;
    bool _orderWarned = false;

    bool _inertialPending = false;
    bool _inertialActive = false;
    float _inertialT = 0.0f;
    float _inertialDur = 0.0f;
    Array<Float3> _inertPosDir;
    Array<float> _inertPosMag;
    Array<float> _inertPosVel;
    Array<Float3> _inertRotAxis;
    Array<float> _inertRotAngle;
    Array<float> _inertRotVel;
    Array<Float3> _inertSclDir;
    Array<float> _inertSclMag;
    Array<float> _inertSclVel;
    bool _warned = false;
    // Scratch storage for BuildInertialOffsets (the extra new-clip sample
    // must outlive the call scope while offsets are computed).
    Array<Matrix> _prevSampleScratch;

private:

    static float ClampF(float value, float minValue, float maxValue);
    static float MaxF(float a, float b);
    static float MinF(float a, float b);
    static float AbsF(float value);
    static bool IsFiniteF(float value);

    void ArmInertial();
    void ClearInertial();
    void AssertTickOrder();
    void ApplyInertialization(Array<Matrix>& baseLocals, SkinnedModel* model, float dt);
    void BuildInertialOffsets(Array<Matrix>& newBase, SkinnedModel* model, float dt);
    static float CalcInertialFloat(float x0, float v0, float t, float t1);
    static Quaternion NormalizeQ(const Quaternion& value);
    static Quaternion ConjugateQ(const Quaternion& value);
    static void QuatToAxisAngle(const Quaternion& value, Float3& axis, float& angle);
    static Quaternion AxisAngleToQuat(const Float3& axis, float angle);
    static float TwistAbout(const Quaternion& value, const Float3& axis);
    static bool TryDecompose(const Matrix& value, Float3& scl, Quaternion& rot, Float3& pos);
    void EnsureHierarchy(SkinnedModel* model);
    void WarnOnce(const String& message);
    void ResetRootYawState();
    static bool IsValidUpAxis(const Float3& value);
    bool TryGetRootSampleRotation(const Array<Matrix>& locals, Quaternion& outRotation) const;
    float ClampRootYawDelta(float delta, float deltaTime) const;
    void ApplyRootYawSample(bool currentValid, const Quaternion& currentRotation, float deltaTime, bool wasAtEnd, bool wrapped, bool hasBaseline, const Quaternion& baselineRotation);
};
