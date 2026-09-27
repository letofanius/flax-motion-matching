// MotionMatchingBaker.h: editor-side database baker interface (MotionMatchingBaker::Bake).
// Ownership: plugin runtime, no game/host references.
// Key invariants: Bake samples every source clip into pose/trajectory features, stores the PoseSearchSchema snapshot, then saves the asset; schema weights are stored for the runtime cost function, never baked into features.
#pragma once

#if USE_EDITOR

#include "Engine/Content/Assets/SkinnedModel.h"
#include "../Schema/PoseSearchSchema.h"

class MotionMatchingDatabase;

/// <summary>
/// Editor-side native motion-matching database baker.
/// </summary>
API_CLASS(Static, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingBaker
{
    DECLARE_SCRIPTING_TYPE_NO_SPAWN(MotionMatchingBaker);

public:
    /// <summary>
    /// Samples the database source clips, bakes normalized pose and trajectory
    /// features, stores the schema snapshot, then saves the database asset.
    /// Feature sizes are defined by PoseSearchSchema (pose: 7 bones x pos+vel
    /// + 2 foot contacts, trajectory: 6 points x planar pos+dir+yawrate).
    /// Schema weights are stored but not applied.
    /// </summary>
    API_FUNCTION()
    static bool Bake(
        SkinnedModel* model,
        const PoseSearchSchema& schema,
        MotionMatchingDatabase* database,
        API_PARAM(Out) int32& bakedClipCount,
        API_PARAM(Out) int32& bakedSampleCount);
};

#endif
