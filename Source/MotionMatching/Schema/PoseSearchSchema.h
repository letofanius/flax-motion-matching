// PoseSearchSchema.h: pose-search bake contract (PoseSearchSchema: layout, weights, axes, validation, contacts).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the baker stores a snapshot in every database so queries never depend on scene settings; GetPoseFeatureSize/GetTrajectoryFeatureSize/DescribePoseChannels/DescribeTrajectoryChannels are the single layout truth (fixed 7-bone pose order, 6-point trajectory); NormalizationRevision + GetFeatureLayoutHash gate stale bakes to registry-only; contacts resolve through TryGetNormalizedUpAxis/ComputeContactHeight/IsPlantedContact so height never hardcodes .Y.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Types/String.h"
#include "FeatureChannel.h"
#include "MotionMatchingTypes.h"

class ReadStream;
class WriteStream;

/// <summary>
/// Bake contract shared by the baker and the runtime query builder.
/// A snapshot of the schema used for baking is stored inside MotionMatchingDatabase,
/// so a database never depends on scene-side settings to be interpreted.
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API PoseSearchSchema
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(PoseSearchSchema);

    /// <summary>Samples per second baked for every clip.</summary>
    API_FIELD(Attributes="Limit(1, 240)") int32 SampleRate = 30;

    /// <summary>Root-node local axis that points forward.</summary>
    API_FIELD() Float3 RootForwardAxis = Float3(0, -1, 0);

    /// <summary>Root-node local axis that points up.</summary>
    API_FIELD() Float3 RootUpAxis = Float3(0, 0, -1);

    /// <summary>
    /// Relative importance of the pose block in the search cost.
    /// Design note: applied at query/cost time only; it is never baked into
    /// features, so tuning it does not require a re-bake.
    /// </summary>
    API_FIELD(Attributes="Limit(0, 100)") float PoseWeight = 1.0f;

    /// <summary>
    /// Relative importance of the trajectory block in the search cost.
    /// Design note: applied at query/cost time only; it is never baked into
    /// features, so tuning it does not require a re-bake.
    /// </summary>
    API_FIELD(Attributes="Limit(0, 100)") float TrajectoryWeight = 1.0f;

    /// <summary>
    /// Skeleton node names. Empty by default — the host MUST configure them
    /// for its own rig (scene → MotionMatchingController → Schema → Skeleton).
    /// Only Root, Pelvis, LeftFoot and RightFoot are required and extracted;
    /// the remaining names are stored for future channels.
    /// </summary>
    API_FIELD() SkeletonProfile Skeleton;

    /// <summary>
    /// Trajectory sampling times in seconds relative to the current pose.
    /// Exactly TrajectoryPointCount values, strictly ascending, spanning past and
    /// future around time zero (for example -0.4 .. 0.6).
    /// </summary>
    API_FIELD() Array<float> TrajectorySampleTimes = { -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f };

    // Feature layout (v3 upper body). Pose bones are extracted in this
    // fixed order: [0] pelvis, [1] left foot, [2] right foot, [3] spine,
    // [4] head, [5] left hand, [6] right hand. After the velocity block come
    // PoseExtraCount foot-contact flags (left, right; 1 = planted).
    static constexpr int32 PoseBoneCount = 7;
    static constexpr int32 PoseFloatsPerBone = 6;           // position xyz + velocity xyz
    static constexpr int32 PoseExtraCount = 2;              // L/R foot contact flags
    static constexpr int32 TrajectoryPointCount = 6;
    static constexpr int32 TrajectoryFloatsPerPoint = 5;    // planar position xy + planar facing xy + yaw rate
    static constexpr float AxisTolerance = 0.001f;       // tolerance for axis validation
    // Foot-contact gate shared by the baker and the live query: a foot counts
    // as planted while slow and low in the root frame.
    static constexpr float ContactMaxSpeed = 300.0f;
    static constexpr float ContactMaxHeight = 12.0f;
    // Normalization revision: bumped when the bake-time statistics change shape.
    // Stored per-database (NOT in the schema snapshot); databases baked with an
    // older revision load registry-only with a "rebake required" message instead
    // of silently mismatching. Rev 1 = pooled grouped (mean/std over samples x
    // grouped dims). Rev 2 = scalar per-dim mean/population-variance with baked
    // z-scored features + stored means. Rev 3 = NO means: raw baked/query values
    // with one mean-row-norm deviation per group floored at `dev > 0.1 ? dev :
    // 1.0`; the search applies sqrt(sum-normalized weight)/deviation once at cost
    // time.
    static constexpr int32 NormalizationRevision = 3;
    // Normalization groups: channels that share a class, cardinality and semantic
    // share one group so left/right statistics cannot skew turns; solo bones and
    // every trajectory dimension keep their own group.
    // Pose layout: [0] pelvis, [1] L foot, [2] R foot, [3] spine, [4] head,
    // [5] L hand, [6] R hand; positions 0..20, velocities 21..41, contacts 42..43.
    static int32 GetPoseNormalizationGroup(int32 dim)
    {
        if (dim < 0)
            return -1;
        if (dim < 21)
        {
            const int32 bone = dim / 3;
            if (bone == 0)
                return 0;
            if (bone == 1 || bone == 2)
                return 1;
            if (bone == 3)
                return 2;
            if (bone == 4)
                return 3;
            return 4;
        }
        if (dim < 42)
        {
            const int32 bone = (dim - 21) / 3;
            if (bone == 0)
                return 5;
            if (bone == 1 || bone == 2)
                return 6;
            if (bone == 3)
                return 7;
            if (bone == 4)
                return 8;
            return 9;
        }
        return 10;
    }

    static int32 GetTrajectoryNormalizationGroup(int32 dim)
    {
        return dim;
    }

    /// <summary>
    /// Contact metadata: block-local pose offset of a planted flag.
    /// side 0 = left, 1 = right; returns -1 for any other side. All bake,
    /// query and search code must resolve contacts through this; deriving
    /// them from poseFeatureSize-2/-1 is forbidden (layout coupling).
    /// </summary>
    static int32 GetContactFeatureOffset(int32 side)
    {
        if (side < 0 || side >= PoseExtraCount)
            return -1;
        return GetPoseFeatureSize() - PoseExtraCount + side;
    }

    /// <summary>
    /// Layout hash (FNV-1a over block/offset/dim/kind/group/symmetry/
    /// semantic/contact-role of every channel). Stored per-database next to
    /// the normalization revision; a mismatch drops baked arrays to
    /// registry-only with REBAKE REQUIRED. Never 0 (0 = unknown: files without the tail int).
    /// </summary>
    static uint32 GetFeatureLayoutHash();

    /// <summary>
    /// Channel metadata, single source of truth for layout. Pose block:
    /// 14 vector channels (7 positions + 7 velocities, dim 3) + 2 scalar
    /// contact channels. Trajectory block: 30 scalar channels (identity
    /// groups). Offsets are block-local. Kind/cardinality are documentation
    /// metadata: the rev-3 deviation uses one mean-row-norm formula for every
    /// group, so no scalar/vector dispatch exists anymore.
    /// </summary>
    static void DescribePoseChannels(Array<FeatureChannel>& outChannels);
    static void DescribeTrajectoryChannels(Array<FeatureChannel>& outChannels);

    // NOTE: the bake-time filename->band mapping used to live here
    // (ClipSpeedBand + BandFromClipName). It moved to Baker/
    // MotionMatchingBakeHeuristics.h: the runtime schema must not know
    // filename conventions. Stored bands stay plain ints in the database.

    /// <summary>
    /// Layout validation: every channel inside its block bounds, no two
    /// channels overlapping, contact channels exactly at
    /// GetContactFeatureOffset. Called by TryValidate.
    /// </summary>
    static bool ValidateLayout(String& error);

    static constexpr int32 GetPosePositionBlockSize()
    {
        return PoseBoneCount * 3;
    }

    static constexpr int32 GetPoseFeatureSize()
    {
        return PoseBoneCount * PoseFloatsPerBone + PoseExtraCount;
    }

    static constexpr int32 GetTrajectoryFeatureSize()
    {
        return TrajectoryPointCount * TrajectoryFloatsPerPoint;
    }

    /// <summary>
    /// Checks sample rate, axes, weights, required bone names and trajectory times.
    /// </summary>
    bool TryValidate(String& error) const;

    /// <summary>
    /// Returns the normalized right axis (forward x up) or false when axes are degenerate.
    /// </summary>
    bool TryGetRightAxis(Float3& right) const;

    /// <summary>
    /// Shared contact helper: normalized root-up axis, or false when the
    /// schema up axis is zero/non-finite. Baker, live query and playback all
    /// resolve the axis through this (baker normalizes on snapshot; callers
    /// with a live schema normalize here) so contact height never hardcodes .Y.
    /// </summary>
    bool TryGetNormalizedUpAxis(Float3& up) const;

    /// <summary>
    /// Shared contact helper: height = dot(footPosition - rootPosition,
    /// normalizedUpAxis). footPosition and rootPosition must be in the SAME
    /// space (baker/query: root-relative with root at origin; playback:
    /// model/body space from the composed pose).
    /// </summary>
    static float ComputeContactHeight(
        const Float3& footPosition,
        const Float3& rootPosition,
        const Float3& normalizedUpAxis);

    /// <summary>
    /// Shared contact gate: planted while slow and low. Speed is passed
    /// squared to avoid a sqrt on the hot path. Bake/query pass the
    /// root-relative pose velocity; playback passes the world foot velocity
    /// with its own runtime-owned threshold (same default, independent knob).
    /// </summary>
    static bool IsPlantedContact(float speedSquared, float maxSpeed, float height, float maxHeight);

    bool operator==(const PoseSearchSchema& other) const;
    bool operator!=(const PoseSearchSchema& other) const
    {
        return !(*this == other);
    }

    void Serialize(WriteStream& stream) const;
    void Deserialize(ReadStream& stream);
};
