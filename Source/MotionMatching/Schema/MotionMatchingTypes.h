// MotionMatchingTypes.h: skeleton profile configuration and resolved node indices (SkeletonProfile, NodeIndices, MotionMatchingSkeleton).
// Ownership: plugin runtime, no game/host references.
// Key invariants: SkeletonProfile ships empty and the host must configure every bone name for its rig; TryResolve fails fast naming the scene configuration path when a required node (Root, Pelvis, LeftFoot, RightFoot) is missing and never falls back to built-in bone names; GetPoseBoneIndices fills the fixed 7-bone feature order with -1 for missing optional upper-body nodes (IsValid/HasOptionalNodes report coverage).
#pragma once

#include "Engine/Core/Types/String.h"
#include "Engine/Content/Assets/SkinnedModel.h"

/// <summary>
/// Resolved node indices used by motion-matching feature extraction.
/// </summary>
API_STRUCT(Namespace="MotionMatching") struct MOTIONMATCHING_API NodeIndices
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(NodeIndices);

    API_FIELD() int32 Root = -1;
    API_FIELD() int32 Pelvis = -1;
    API_FIELD() int32 Spine = -1;
    API_FIELD() int32 Head = -1;
    API_FIELD() int32 LeftFoot = -1;
    API_FIELD() int32 RightFoot = -1;
    API_FIELD() int32 LeftHand = -1;
    API_FIELD() int32 RightHand = -1;

    /// <summary>
    /// True when every node used by pose feature extraction was resolved
    /// (Root, Pelvis, LeftFoot, RightFoot). Spine/Head/Hands are optional.
    /// </summary>
    bool IsValid() const
    {
        return Root >= 0 &&
               Pelvis >= 0 &&
               LeftFoot >= 0 &&
               RightFoot >= 0;
    }

    /// <summary>
    /// True when the optional upper-body nodes were also resolved.
    /// </summary>
    bool HasOptionalNodes() const
    {
        return Spine >= 0 &&
               Head >= 0 &&
               LeftHand >= 0 &&
               RightHand >= 0;
    }

    /// <summary>
    /// Fills the pose bone indices in the fixed feature order:
    /// [0] pelvis, [1] left foot, [2] right foot, [3] spine, [4] head,
    /// [5] left hand, [6] right hand. Entries may be -1 when the optional
    /// upper-body nodes are missing; baker and query bake/sample zeros for
    /// those channels instead of failing.
    /// </summary>
    void GetPoseBoneIndices(int32* indices) const
    {
        indices[0] = Pelvis;
        indices[1] = LeftFoot;
        indices[2] = RightFoot;
        indices[3] = Spine;
        indices[4] = Head;
        indices[5] = LeftHand;
        indices[6] = RightHand;
    }
};

/// <summary>
/// Names of the skeleton nodes used by the motion-matching baker.
/// Empty by default: the host MUST configure every name for its own rig
/// (scene → MotionMatchingController → Schema → Skeleton). Root, Pelvis,
/// LeftFoot and RightFoot are required; the rest are stored for future channels.
/// There is deliberately no mannequin preset here: a flat shared string
/// array would lose the per-role semantics (Root strips travel, Pelvis/Feet
/// feed feature extraction, Spine/Head/Hands are optional), so the struct
/// keeps one named field per role and ships empty.
/// </summary>
API_STRUCT(Namespace="MotionMatching") struct MOTIONMATCHING_API SkeletonProfile
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(SkeletonProfile);

    API_FIELD() String Root;
    API_FIELD() String Pelvis;
    API_FIELD() String Spine;
    API_FIELD() String Head;
    API_FIELD() String LeftFoot;
    API_FIELD() String RightFoot;
    API_FIELD() String LeftHand;
    API_FIELD() String RightHand;

};

/// <summary>
/// Native skeleton-profile helpers exposed to C#.
/// </summary>
API_CLASS(Static, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingSkeleton
{
    DECLARE_SCRIPTING_TYPE_NO_SPAWN(MotionMatchingSkeleton);

public:
    /// <summary>
    /// Returns an empty profile. The host MUST fill every bone name for its
    /// own rig (scene → MotionMatchingController → Schema → Skeleton);
    /// there is no built-in mannequin preset.
    /// </summary>
    API_FUNCTION() static SkeletonProfile CreateDefault();

    /// <summary>
    /// Resolves the configured names against a model skeleton.
    /// Fails when a required name is empty (profile not configured for this
    /// rig) or when a required node (Root, Pelvis, LeftFoot, RightFoot) is
    /// missing from the model. The error always tells the host where to
    /// configure the profile (scene → MotionMatchingController → Schema →
    /// Skeleton); it never falls back to built-in bone names.
    /// </summary>
    API_FUNCTION() static bool TryResolve(
        const SkeletonProfile& profile,
        SkinnedModel* model,
        API_PARAM(Out) NodeIndices& result,
        API_PARAM(Out) String& error);
};
