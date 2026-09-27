// MotionMatchingQuery.cpp: raw query-feature builders (SamplePoseFeatures/SampleSearchPose/BuildRawTrajectory, SampleLivePose/SampleLivePoseWithSchema/BuildLiveTrajectory).
// Ownership: plugin runtime, no game/host references.
// Key invariants: one shared feature core mirrors the baker exactly (world * rootInverse, root-relative translation, backward live velocity, schema-axis contacts via metadata offsets); the live path re-anchors trajectory points/facing to the zero-time sample; bake and query divide unwrapped yaw by their own true dt through MotionMatchingSampler::ComputeYawRate so rad/s matches on both sides.
#include "MotionMatchingQuery.h"

#include "../Runtime/MotionMatchingSampler.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Level/Actors/AnimatedModel.h"

#include <cmath>

namespace
{
    constexpr float MinimumDeltaTime = 0.000001f;
    constexpr float MinimumDirectionLengthSquared = 0.000001f;

    bool IsFinite(const Float3& value)
    {
        return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
    }
}

bool MotionMatchingQuery::SamplePoseFeatures(
    AnimatedModel* model,
    const NodeIndices& nodes,
    float deltaTime,
    Float3* prevPositions,
    float* outPose,
    String& error)
{
    // Default-axis entry: default schema axis (up = (0,0,-1)). New callers use the
    // schema-aware overload so bake/live share one axis semantic.
    const PoseSearchSchema defaultSchema;
    return SamplePoseFeatures(model, nodes, deltaTime, prevPositions, defaultSchema, outPose, error);
}

// Shared feature core: bone WORLD matrices supplied by the
// caller — either read live from the animated model or composed from the
// playback's pre-layer/pre-IK search snapshot. One definition, no copies:
// mirror the baker exactly (world * rootInverse, translation = root-relative
// position in the root frame; backward velocity; schema-axis contacts via
// metadata offsets). Missing optional upper-body nodes sample zeros (same
// as the baker).
static bool SampleFeaturesFromWorlds(
    const Matrix* nodeWorlds,
    int32 nodeWorldCount,
    const NodeIndices& nodes,
    float deltaTime,
    Float3* prevPositions,
    const PoseSearchSchema& schema,
    float* outPose,
    String& error)
{
    error = String::Empty;
    if (nodeWorlds == nullptr || outPose == nullptr || prevPositions == nullptr)
    {
        error = TEXT("Motion matching query failed: null world matrices or output buffer.");
        return false;
    }
    if (!nodes.IsValid())
    {
        error = TEXT("Motion matching query failed: skeleton node indices are not resolved.");
        return false;
    }
    if (nodes.Root < 0 || nodes.Root >= nodeWorldCount)
    {
        error = TEXT("Motion matching query failed: root node is outside the world-matrix range.");
        return false;
    }

    int32 boneIndices[PoseSearchSchema::PoseBoneCount];
    nodes.GetPoseBoneIndices(boneIndices);

    const Matrix rootInverse = Matrix::Invert(nodeWorlds[nodes.Root]);

    // Mirror the baker: bone transform composed with the root inverse, then the
    // translation is the root-relative position in the root node's local frame.
    // Missing optional upper-body nodes sample zeros (same as the baker).
    // Root-relative pose velocity (feature) — never the world foot velocity used
    // by foot solving. Named distinctly; thresholds never shared.
    Float3 poseVelocityRootRelative[PoseSearchSchema::PoseBoneCount];
    for (int32 boneIndex = 0; boneIndex < PoseSearchSchema::PoseBoneCount; boneIndex++)
    {
        Float3 position = Float3::Zero;
        const int32 nodeIndex = boneIndices[boneIndex];
        if (nodeIndex >= 0 && nodeIndex < nodeWorldCount)
        {
            Matrix relative;
            Matrix::Multiply(nodeWorlds[nodeIndex], rootInverse, relative);
            position = relative.GetTranslation();
        }

        outPose[boneIndex * 3] = position.X;
        outPose[boneIndex * 3 + 1] = position.Y;
        outPose[boneIndex * 3 + 2] = position.Z;

        // Backward difference against the previous frame. The baker uses central
        // differences, so live velocity lags by half a sample interval; this is
        // standard practice and well within the trajectory weight's tolerance.
        // Timescale note: bake stencil = central ±1 bake
        // sample (2/rate s, one-sided at non-loop edges); live stencil =
        // backward over the live frame dt (causal — central would need the
        // future). Both are root-relative m/s under one velocity definition;
        // the half-interval lag is documented, not compensated, and query
        // yaw-rate below shares MotionMatchingSampler unwrap/rate helpers with
        // the baker so rad/s means the same thing on both sides.
        Float3 velocity = Float3::Zero;
        if (deltaTime > MinimumDeltaTime)
            velocity = (position - prevPositions[boneIndex]) / deltaTime;
        poseVelocityRootRelative[boneIndex] = velocity;
        const int32 velocityOffset = PoseSearchSchema::GetPosePositionBlockSize() + boneIndex * 3;
        outPose[velocityOffset] = velocity.X;
        outPose[velocityOffset + 1] = velocity.Y;
        outPose[velocityOffset + 2] = velocity.Z;

        prevPositions[boneIndex] = position;
    }

    // Foot-contact flags: same gate as the baker (feet are required,
    // so indices 1/2 are always valid here). Height via the shared helper
    // with the schema up axis — never hardcoded .Y. Positions above are
    // root-relative (root at origin), so relativeFoot = foot - root(origin).
    {
        Float3 normalizedUp;
        if (!schema.TryGetNormalizedUpAxis(normalizedUp))
        {
            error = TEXT("Motion matching query failed: root up axis is degenerate.");
            return false;
        }
        const Float3 rootOrigin = Float3::Zero;
        const Float3 footL(outPose[3], outPose[4], outPose[5]);
        const Float3 footR(outPose[6], outPose[7], outPose[8]);
        const float heightL = PoseSearchSchema::ComputeContactHeight(footL, rootOrigin, normalizedUp);
        const float heightR = PoseSearchSchema::ComputeContactHeight(footR, rootOrigin, normalizedUp);
        // Contact offsets resolve via channel metadata, never size arithmetic.
        const int32 contactL = PoseSearchSchema::GetContactFeatureOffset(0);
        const int32 contactR = PoseSearchSchema::GetContactFeatureOffset(1);
        if (contactL < 0 || contactR < 0 ||
            contactL >= PoseSearchSchema::GetPoseFeatureSize() ||
            contactR >= PoseSearchSchema::GetPoseFeatureSize())
        {
            error = TEXT("Motion matching query failed: contact feature offsets are invalid.");
            return false;
        }
        outPose[contactL] = PoseSearchSchema::IsPlantedContact(
            poseVelocityRootRelative[1].LengthSquared(), PoseSearchSchema::ContactMaxSpeed,
            heightL, PoseSearchSchema::ContactMaxHeight) ? 1.0f : 0.0f;
        outPose[contactR] = PoseSearchSchema::IsPlantedContact(
            poseVelocityRootRelative[2].LengthSquared(), PoseSearchSchema::ContactMaxSpeed,
            heightR, PoseSearchSchema::ContactMaxHeight) ? 1.0f : 0.0f;
    }

    return true;
}

bool MotionMatchingQuery::SamplePoseFeatures(
    AnimatedModel* model,
    const NodeIndices& nodes,
    float deltaTime,
    Float3* prevPositions,
    const PoseSearchSchema& schema,
    float* outPose,
    String& error)
{
    error = String::Empty;
    if (model == nullptr || outPose == nullptr || prevPositions == nullptr)
    {
        error = TEXT("Motion matching query failed: null model or output buffer.");
        return false;
    }
    if (!nodes.IsValid())
    {
        error = TEXT("Motion matching query failed: skeleton node indices are not resolved.");
        return false;
    }

    // Gather one world matrix per used skeleton node, then run the shared
    // core so live reads and search snapshots cannot drift apart.
    int32 boneIndices[PoseSearchSchema::PoseBoneCount];
    nodes.GetPoseBoneIndices(boneIndices);
    int32 maxNode = nodes.Root;
    for (int32 boneIndex = 0; boneIndex < PoseSearchSchema::PoseBoneCount; boneIndex++)
    {
        if (boneIndices[boneIndex] > maxNode)
            maxNode = boneIndices[boneIndex];
    }
    Array<Matrix> worlds;
    worlds.Resize(maxNode + 1);
    for (int32 nodeIndex = 0; nodeIndex <= maxNode; nodeIndex++)
        model->GetNodeTransformation(nodeIndex, worlds[nodeIndex], true);

    return SampleFeaturesFromWorlds(
        worlds.Get(), worlds.Count(), nodes, deltaTime, prevPositions, schema, outPose, error);
}

bool MotionMatchingQuery::SampleSearchPose(
    AnimatedModel* animatedModel,
    const Array<Matrix>& searchLocals,
    const NodeIndices& nodes,
    float deltaTime,
    const Array<float>& prevPositions,
    const PoseSearchSchema& schema,
    Array<float>& outPose,
    Array<float>& outPrevPositions,
    Matrix& outRootWorld,
    String& error)
{
    outPose.Clear();
    outPrevPositions.Clear();
    outRootWorld = Matrix::Identity;
    error = String::Empty;

    if (animatedModel == nullptr)
    {
        error = TEXT("Motion matching query failed: null animated model.");
        return false;
    }
    if (searchLocals.Count() == 0)
    {
        error = TEXT("Motion matching query failed: empty search snapshot.");
        return false;
    }
    if (prevPositions.Count() != PoseSearchSchema::GetPosePositionBlockSize())
    {
        error = TEXT("Motion matching query failed: prevPositions has wrong size.");
        return false;
    }

    SkinnedModel* skinned = animatedModel->SkinnedModel.Get();
    if (skinned == nullptr)
    {
        error = TEXT("Motion matching query failed: skinned model asset is not loaded.");
        return false;
    }
    Array<Matrix> modelPose;
    if (!MotionMatchingSampler::ComposeModelPose(skinned, searchLocals, modelPose, error))
        return false;

    Float3 prev[PoseSearchSchema::PoseBoneCount];
    for (int32 i = 0; i < PoseSearchSchema::PoseBoneCount; i++)
        prev[i] = Float3(prevPositions[i * 3], prevPositions[i * 3 + 1], prevPositions[i * 3 + 2]);

    float pose[PoseSearchSchema::GetPoseFeatureSize()];
    if (!SampleFeaturesFromWorlds(
            modelPose.Get(), modelPose.Count(), nodes, deltaTime, prev, schema, pose, error))
        return false;

    if (nodes.Root < 0 || nodes.Root >= modelPose.Count())
    {
        error = TEXT("Motion matching query failed: root node is outside the composed pose.");
        return false;
    }
    outRootWorld = modelPose[nodes.Root];

    outPose.Resize(PoseSearchSchema::GetPoseFeatureSize());
    outPrevPositions.Resize(PoseSearchSchema::GetPosePositionBlockSize());
    for (int32 i = 0; i < PoseSearchSchema::GetPoseFeatureSize(); i++)
        outPose[i] = pose[i];
    for (int32 i = 0; i < PoseSearchSchema::PoseBoneCount; i++)
    {
        outPrevPositions[i * 3] = prev[i].X;
        outPrevPositions[i * 3 + 1] = prev[i].Y;
        outPrevPositions[i * 3 + 2] = prev[i].Z;
    }
    return true;
}

bool MotionMatchingQuery::BuildRawTrajectory(
    const TrajectoryPoint* points,
    const Matrix& actorWorld,
    const Matrix& rootWorld,
    const PoseSearchSchema& schema,
    float* outTrajectory,
    String& error)
{
    error = String::Empty;
    if (points == nullptr || outTrajectory == nullptr)
    {
        error = TEXT("Motion matching query failed: null trajectory points or output buffer.");
        return false;
    }

    Float3 localRight;
    if (!schema.TryGetRightAxis(localRight))
    {
        error = TEXT("Motion matching query failed: root forward and up axes cannot be parallel.");
        return false;
    }
    const Float3& rootForwardAxis = schema.RootForwardAxis;

    // Points live in actor space; the baked features live in the root node's
    // local frame, so map actor -> world -> root frame (row-vector convention,
    // same composition order the baker uses for bone transforms).
    const Matrix rootInverse = Matrix::Invert(rootWorld);
    Matrix actorToRoot;
    Matrix::Multiply(actorWorld, rootInverse, actorToRoot);

    for (int32 pointIndex = 0; pointIndex < PoseSearchSchema::TrajectoryPointCount; pointIndex++)
    {
        const TrajectoryPoint& point = points[pointIndex];
        const Float3 relativePosition = Float3::Transform(point.Position, actorToRoot);
        Float3 relativeFacing;
        Float3::TransformNormal(point.Facing, actorToRoot, relativeFacing);

        if (!IsFinite(relativePosition) || !IsFinite(relativeFacing))
        {
            error = String::Format(TEXT("Motion matching query failed: non-finite trajectory at point {}."), pointIndex);
            return false;
        }

        // Same planar projection as the baker: dot with the root-local right
        // and forward axes, direction normalized with a (0, 1) fallback.
        const Float2 planarPosition(
            Float3::Dot(relativePosition, localRight),
            Float3::Dot(relativePosition, rootForwardAxis));
        Float2 planarDirection(
            Float3::Dot(relativeFacing, localRight),
            Float3::Dot(relativeFacing, rootForwardAxis));
        if (planarDirection.LengthSquared() > MinimumDirectionLengthSquared)
            planarDirection.Normalize();
        else
            planarDirection = Float2(0.0f, 1.0f);

        const int32 offset = pointIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
        outTrajectory[offset] = planarPosition.X;
        outTrajectory[offset + 1] = planarPosition.Y;
        outTrajectory[offset + 2] = planarDirection.X;
        outTrajectory[offset + 3] = planarDirection.Y;
    }

    // Rebase to the zero-time point so the query frame matches the baker's:
    // the baker maps currentRoot -> itself, so its zero-time position is
    // exactly (0, 0) with facing (0, 1). The live path maps actor-space points
    // via actorWorld * rootInverse, so actor->root offsets (mesh mounts,
    // stripped-root placement, actor origin at the feet) would otherwise bias
    // every point by a constant. Subtract the zero-time position and rotate so
    // zero-time facing is forward; relative shape is preserved.
    int32 zeroIndex = -1;
    for (int32 i = 0; i < PoseSearchSchema::TrajectoryPointCount; i++)
    {
        if (i < schema.TrajectorySampleTimes.Count() &&
            std::fabs(schema.TrajectorySampleTimes[i]) <= 0.0001f)
        {
            zeroIndex = i;
            break;
        }
    }
    if (zeroIndex < 0)
        zeroIndex = PoseSearchSchema::TrajectoryPointCount / 2; // schema validated to contain zero; fallback
    const int32 zeroOffset = zeroIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
    const float zeroX = outTrajectory[zeroOffset];
    const float zeroY = outTrajectory[zeroOffset + 1];
    float zeroDirX = outTrajectory[zeroOffset + 2];
    float zeroDirY = outTrajectory[zeroOffset + 3];
    const float zeroDirLenSq = zeroDirX * zeroDirX + zeroDirY * zeroDirY;
    float cosA = 1.0f;
    float sinA = 0.0f;
    if (zeroDirLenSq > MinimumDirectionLengthSquared)
    {
        const float invLen = 1.0f / std::sqrt(zeroDirLenSq);
        zeroDirX *= invLen;
        zeroDirY *= invLen;
        // Rotation taking zeroDir -> (0, 1): cosA = dot(zeroDir, forward),
        // sinA = cross-equivalent (zeroDir.x * 1 - zeroDir.y * 0) = zeroDir.x.
        // Apply R(-angle): x' = x * cosA - y * sinA? Verified: with zeroDir as
        // reference, R maps zeroDir to (0,1) exactly.
        cosA = zeroDirY;
        sinA = zeroDirX;
    }
    for (int32 pointIndex = 0; pointIndex < PoseSearchSchema::TrajectoryPointCount; pointIndex++)
    {
        const int32 offset = pointIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
        const float relX = outTrajectory[offset] - zeroX;
        const float relY = outTrajectory[offset + 1] - zeroY;
        const float dirX = outTrajectory[offset + 2];
        const float dirY = outTrajectory[offset + 3];
        outTrajectory[offset] = relX * cosA - relY * sinA;
        outTrajectory[offset + 1] = relX * sinA + relY * cosA;
        outTrajectory[offset + 2] = dirX * cosA - dirY * sinA;
        outTrajectory[offset + 3] = dirX * sinA + dirY * cosA;
    }
    // Renormalize directions after rotation (rotation preserves length, but the
    // (0,1) fallback path and float error can drift); enforce zero-time exactly.
    for (int32 pointIndex = 0; pointIndex < PoseSearchSchema::TrajectoryPointCount; pointIndex++)
    {
        const int32 offset = pointIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
        float dx = outTrajectory[offset + 2];
        float dy = outTrajectory[offset + 3];
        const float lenSq = dx * dx + dy * dy;
        if (lenSq > MinimumDirectionLengthSquared)
        {
            const float invLen = 1.0f / std::sqrt(lenSq);
            outTrajectory[offset + 2] = dx * invLen;
            outTrajectory[offset + 3] = dy * invLen;
        }
        else
        {
            outTrajectory[offset + 2] = 0.0f;
            outTrajectory[offset + 3] = 1.0f;
        }
    }
    outTrajectory[zeroOffset] = 0.0f;
    outTrajectory[zeroOffset + 1] = 0.0f;
    outTrajectory[zeroOffset + 2] = 0.0f;
    outTrajectory[zeroOffset + 3] = 1.0f;

    // Yaw-rate channel: explicit turn signal from the predicted facings,
    // central difference of facing yaw over the schema sample times. Same
    // physical quantity (rad/s, unwrap [-pi,pi]) the baker derives from root
    // motion, through the SAME MotionMatchingSampler::ComputeYawRate helper.
    // Stencil differs by construction: bake = ±1 bake sample, query = ±1
    // trajectory interval (schema times, non-uniform allowed). No timescale
    // conversion is needed — both sides divide unwrapped yaw by their own true
    // dt (bake: 2*frameDt or one-sided 1*frameDt at non-loop edges; query: the
    // actual schema-time span below).
    {
        float yaws[PoseSearchSchema::TrajectoryPointCount];
        for (int32 pointIndex = 0; pointIndex < PoseSearchSchema::TrajectoryPointCount; pointIndex++)
        {
            const int32 offset = pointIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
            yaws[pointIndex] = Math::Atan2(outTrajectory[offset + 2], outTrajectory[offset + 3]);
        }
        for (int32 pointIndex = 0; pointIndex < PoseSearchSchema::TrajectoryPointCount; pointIndex++)
        {
            const int32 prevIndex = Math::Max(pointIndex - 1, 0);
            const int32 nextIndex = Math::Min(pointIndex + 1, PoseSearchSchema::TrajectoryPointCount - 1);
            const float dt = schema.TrajectorySampleTimes[nextIndex] - schema.TrajectorySampleTimes[prevIndex];
            const int32 offset = pointIndex * PoseSearchSchema::TrajectoryFloatsPerPoint;
            outTrajectory[offset + 4] = MotionMatchingSampler::ComputeYawRate(yaws[prevIndex], yaws[nextIndex], dt);
        }
    }

    return true;
}

bool MotionMatchingQuery::SampleLivePose(
    AnimatedModel* animatedModel,
    const NodeIndices& nodes,
    float deltaTime,
    const Array<float>& prevPositions,
    Array<float>& outPose,
    Array<float>& outPrevPositions,
    Matrix& outRootWorld,
    String& error)
{
    const PoseSearchSchema defaultSchema;
    return SampleLivePoseWithSchema(
        animatedModel, nodes, deltaTime, prevPositions, defaultSchema,
        outPose, outPrevPositions, outRootWorld, error);
}

bool MotionMatchingQuery::SampleLivePoseWithSchema(
    AnimatedModel* animatedModel,
    const NodeIndices& nodes,
    float deltaTime,
    const Array<float>& prevPositions,
    const PoseSearchSchema& schema,
    Array<float>& outPose,
    Array<float>& outPrevPositions,
    Matrix& outRootWorld,
    String& error)
{
    outPose.Clear();
    outPrevPositions.Clear();
    outRootWorld = Matrix::Identity;
    error = String::Empty;

    if (prevPositions.Count() != PoseSearchSchema::GetPosePositionBlockSize())
    {
        error = String::Format(
            TEXT("Motion matching query failed: prevPositions has {} floats, expected {}."),
            prevPositions.Count(),
            PoseSearchSchema::GetPosePositionBlockSize());
        return false;
    }

    Float3 prev[PoseSearchSchema::PoseBoneCount];
    for (int32 i = 0; i < PoseSearchSchema::PoseBoneCount; i++)
        prev[i] = Float3(prevPositions[i * 3], prevPositions[i * 3 + 1], prevPositions[i * 3 + 2]);

    float pose[PoseSearchSchema::GetPoseFeatureSize()];
    if (!SamplePoseFeatures(animatedModel, nodes, deltaTime, prev, schema, pose, error))
        return false;

    // Root world is re-read here (one extra node lookup at 10 Hz) so the C#
    // policy never touches bone matrices itself.
    if (animatedModel)
        animatedModel->GetNodeTransformation(nodes.Root, outRootWorld, true);

    outPose.Resize(PoseSearchSchema::GetPoseFeatureSize());
    outPrevPositions.Resize(PoseSearchSchema::GetPosePositionBlockSize());
    for (int32 i = 0; i < PoseSearchSchema::GetPoseFeatureSize(); i++)
        outPose[i] = pose[i];
    for (int32 i = 0; i < PoseSearchSchema::PoseBoneCount; i++)
    {
        outPrevPositions[i * 3] = prev[i].X;
        outPrevPositions[i * 3 + 1] = prev[i].Y;
        outPrevPositions[i * 3 + 2] = prev[i].Z;
    }
    return true;
}

bool MotionMatchingQuery::BuildLiveTrajectory(
    const Array<float>& points,
    const Matrix& actorWorld,
    const Matrix& rootWorld,
    const PoseSearchSchema& schema,
    Array<float>& outTrajectory,
    String& error)
{
    outTrajectory.Clear();
    error = String::Empty;

    const int32 expected = PoseSearchSchema::TrajectoryPointCount * 6;
    if (points.Count() != expected)
    {
        error = String::Format(
            TEXT("Motion matching query failed: trajectory packs {} floats, expected {} (6 points x pos xyz + facing xyz)."),
            points.Count(),
            expected);
        return false;
    }

    TrajectoryPoint unpacked[PoseSearchSchema::TrajectoryPointCount];
    for (int32 i = 0; i < PoseSearchSchema::TrajectoryPointCount; i++)
    {
        unpacked[i].Position = Float3(points[i * 6], points[i * 6 + 1], points[i * 6 + 2]);
        unpacked[i].Facing = Float3(points[i * 6 + 3], points[i * 6 + 4], points[i * 6 + 5]);
    }

    float trajectory[PoseSearchSchema::GetTrajectoryFeatureSize()];
    if (!BuildRawTrajectory(unpacked, actorWorld, rootWorld, schema, trajectory, error))
        return false;

    outTrajectory.Resize(PoseSearchSchema::GetTrajectoryFeatureSize());
    for (int32 i = 0; i < PoseSearchSchema::GetTrajectoryFeatureSize(); i++)
        outTrajectory[i] = trajectory[i];
    return true;
}
