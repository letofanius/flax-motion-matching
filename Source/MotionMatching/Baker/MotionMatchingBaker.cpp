// MotionMatchingBaker.cpp: editor bake pipeline (MotionMatchingBaker::Bake, ClipBakeInfo/SampleBakeInfo, ComputeGroupDeviations, GetRootAtTime).
// Ownership: plugin runtime, no game/host references.
// Key invariants: sample count/time come only from the shared MotionMatchingSampler helpers (loop wrap to [0, length), interior-central / edge-one-sided stencils over the true dt); features stay raw with one mean-row-norm deviation per schema group; snapshot + weights persist with the data so tuning never needs a rebake; a retry loop guards the async save/reload persistence check.
#include "MotionMatchingBaker.h"

#if USE_EDITOR

#include "../Database/MotionMatchingDatabase.h"
#include "../Runtime/MotionMatchingSampler.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Platform/StringUtils.h"

#include <cmath>

namespace
{
    // Feature layout comes from the schema so the baker, the database validator and
    // the (future) runtime query builder share one definition.
    constexpr int32 PoseBoneCount = PoseSearchSchema::PoseBoneCount;
    constexpr int32 PositionFeatureSize = PoseSearchSchema::GetPosePositionBlockSize();
    constexpr int32 PoseFeatureSize = PoseSearchSchema::GetPoseFeatureSize();
    constexpr int32 TrajectoryPointCount = PoseSearchSchema::TrajectoryPointCount;
    constexpr int32 TrajectoryPointSize = PoseSearchSchema::TrajectoryFloatsPerPoint;
    constexpr int32 TrajectoryFeatureSize = PoseSearchSchema::GetTrajectoryFeatureSize();
    constexpr float MinimumDeltaTime = 0.000001f;
    // Foot-contact gate lives in PoseSearchSchema (shared with the live query).
    constexpr float ContactMaxSpeed = PoseSearchSchema::ContactMaxSpeed;
    constexpr float ContactMaxHeight = PoseSearchSchema::ContactMaxHeight;

    struct ClipBakeInfo
    {
        Animation* Clip;
        int32 SourceIndex;
        int32 SampleStart;
        int32 SampleCount;
        float Length;
        bool Loop;
        Matrix StartRootTransform;
        Matrix StartRootInverse;
        Matrix CycleRootTransform;
    };

    struct SampleBakeInfo
    {
        int32 ClipIndex;
        float Time;
        Matrix RootTransform;
        Float3 Positions[PoseBoneCount];
    };

    bool IsFinite(const Float3& value)
    {
        return std::isfinite(value.X) &&
               std::isfinite(value.Y) &&
               std::isfinite(value.Z);
    }

    void IncludeRange(float* minimums, float* maximums, int32 index, float value)
    {
        minimums[index] = Math::Min(minimums[index], value);
        maximums[index] = Math::Max(maximums[index], value);
    }

    void IncludeVectorRange(float* minimums, float* maximums, int32 offset, const Float3& value)
    {
        IncludeRange(minimums, maximums, offset, value.X);
        IncludeRange(minimums, maximums, offset + 1, value.Y);
        IncludeRange(minimums, maximums, offset + 2, value.Z);
    }

    bool SampleAnimationModelPose(
        Animation* animation,
        SkinnedModel* model,
        float time,
        Array<Matrix>& nodeTransforms)
    {
        nodeTransforms.Clear();

        if (animation == nullptr || animation->WaitForLoaded() ||
            model == nullptr || model->WaitForLoaded())
        {
            return false;
        }

        const float framesPerSecond = static_cast<float>(animation->Data.FramesPerSecond);
        if (framesPerSecond <= 0.0f)
            return false;

        const auto& nodes = model->GetNodes();
        const auto& channels = animation->Data.Channels;
        const int32 nodeCount = nodes.Count();
        const float sampleFrame = time * framesPerSecond;

        nodeTransforms.Resize(nodeCount);

        for (int32 nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
        {
            const auto& node = nodes[nodeIndex];
            Transform local = node.LocalTransform;

            for (int32 channelIndex = 0; channelIndex < channels.Count(); channelIndex++)
            {
                const auto& channel = channels[channelIndex];
                if (channel.NodeName != node.Name)
                    continue;

                // Keep bind-pose values for animation sub-tracks removed by
                // import or keyframe optimization.
                if (channel.Position.GetKeyframes().HasItems())
                    channel.Position.Evaluate(local.Translation, sampleFrame, false);
                if (channel.Rotation.GetKeyframes().HasItems())
                    channel.Rotation.Evaluate(local.Orientation, sampleFrame, false);
                if (channel.Scale.GetKeyframes().HasItems())
                    channel.Scale.Evaluate(local.Scale, sampleFrame, false);
                break;
            }

            const Matrix localMatrix = local.GetWorld();
            if (node.ParentIndex >= 0 && node.ParentIndex < nodeIndex)
            {
                Matrix::Multiply(
                    localMatrix,
                    nodeTransforms[node.ParentIndex],
                    nodeTransforms[nodeIndex]);
            }
            else
            {
                nodeTransforms[nodeIndex] = localMatrix;
            }
        }

        return true;
    }

    Float3 GetRelativePosition(const Matrix& transform, const Matrix& referenceInverse)
    {
        return Float3::Transform(transform.GetTranslation(), referenceInverse);
    }

    Float3 GetModelDirection(const Matrix& transform, const Float3& localDirection)
    {
        Float3 result;
        Float3::TransformNormal(localDirection, transform, result);
        if (result.LengthSquared() > MinimumDeltaTime)
            result.Normalize();
        return result;
    }

    Matrix MultiplyMatrices(const Matrix& left, const Matrix& right)
    {
        Matrix result;
        Matrix::Multiply(left, right, result);
        return result;
    }

    Matrix MatrixPower(Matrix value, int32 exponent)
    {
        if (exponent < 0)
        {
            value = Matrix::Invert(value);
            exponent = -exponent;
        }

        Matrix result = Matrix::Identity;
        while (exponent > 0)
        {
            if ((exponent & 1) != 0)
                result = MultiplyMatrices(result, value);
            exponent >>= 1;
            if (exponent > 0)
                value = MultiplyMatrices(value, value);
        }
        return result;
    }

    // Loop wrap lives in MotionMatchingSampler::WrapLoopTime (single definition
    // shared with the live query side); the baker only adds the cycle-transform
    // application on top. Each stencil side resolves its own cycle independently,
    // so seam neighbors come from the adjacent cycle with the matching
    // CycleRootTransform power — never from a duplicated endpoint.
    Matrix GetRootAtTimeWithStatus(
        const ClipBakeInfo& clip,
        const Array<SampleBakeInfo>& samples,
        float time,
        int32 sampleRate,
        MotionSamplingStatus& outStatus)
    {
        int32 localIndex = -1;
        outStatus = MotionMatchingSampler::TimeToSampleIndex(
            time, clip.Length, clip.SampleCount, sampleRate, clip.Loop, localIndex);
        if (outStatus == MotionSamplingStatus::Invalid || localIndex < 0)
            return clip.StartRootTransform;
        if (!clip.Loop)
            return samples[clip.SampleStart + localIndex].RootTransform;

        int32 cycleIndex = 0;
        MotionMatchingSampler::WrapLoopTime(time, clip.Length, cycleIndex);
        const Matrix rootDelta = MultiplyMatrices(
            samples[clip.SampleStart + localIndex].RootTransform,
            clip.StartRootInverse);
        const Matrix accumulatedCycle = MatrixPower(clip.CycleRootTransform, cycleIndex);
        return MultiplyMatrices(
            MultiplyMatrices(rootDelta, accumulatedCycle),
            clip.StartRootTransform);
    }

    Matrix GetRootAtTime(
        const ClipBakeInfo& clip,
        const Array<SampleBakeInfo>& samples,
        float time,
        int32 sampleRate)
    {
        // Clamped-for-playback wrapper: keeps the endpoint pose for display.
        // Feature code (velocity/trajectory/yaw) must use
        // GetRootAtTimeWithStatus + MotionMatchingSampler validity helpers so
        // a clamped non-loop tail never reads as a stopped future (endpoint-validity rule).
        MotionSamplingStatus status = MotionSamplingStatus::Invalid;
        return GetRootAtTimeWithStatus(clip, samples, time, sampleRate, status);
    }

    // Per-group deviation (NormalizationRevision 3). Baked features stay
    // RAW; only a per-dimension deviation is computed here and stored. One
    // formula for every normalization group regardless of cardinality:
    // - mu[d] = column mean (centroid);
    // - center every column;
    // - deviation = MEAN OVER SAMPLES OF THE CENTERED ROW NORM (mean absolute
    //   deviation for a cardinality-1 scalar group — there is NO separate
    //   scalar path);
    // - floor contract: dev > 0.1f ? dev : 1.0f (NOT max(dev, 0.1)); a true
    //   0.05 deviation divides by 1.0.
    // Symmetry safety: only same-semantic, same-cardinality channels ever
    // share a group (see Describe*Channels) — feet positions L/R, feet
    // velocities L/R and contacts L/R share; trajectory scalars stay
    // per-dimension (identity). Weights are applied once at query/cost time
    // (sqrt of sum-normalized weight, divided by this deviation), never
    // baked, so retunes never need a rebake. Returns the count of dimensions
    // whose PRE-FLOOR deviation sat at/below the unit floor (log-only).
    int32 ComputeGroupDeviations(
        const Array<float>& features,
        int32 sampleCount,
        int32 featureCount,
        int32 (*groupOf)(int32),
        Array<float>& deviations)
    {
        constexpr double DeviationFloor = 0.1;
        deviations.Resize(featureCount);
        int32 constantFeatureCount = 0;

        // Collect distinct groups in first-seen order (stable logs).
        Array<int32> groups;
        for (int32 dim = 0; dim < featureCount; dim++)
        {
            const int32 g = groupOf(dim);
            if (!groups.Contains(g))
                groups.Add(g);
        }

        Array<int32> groupDims;
        Array<double> centroid;
        for (int32 gi = 0; gi < groups.Count(); gi++)
        {
            const int32 group = groups[gi];
            groupDims.Clear();
            for (int32 dim = 0; dim < featureCount; dim++)
            {
                if (groupOf(dim) == group)
                    groupDims.Add(dim);
            }
            if (groupDims.IsEmpty() || sampleCount <= 0)
                continue;

            const int32 dimCount = groupDims.Count();
            centroid.Resize(dimCount);
            for (int32 i = 0; i < dimCount; i++)
            {
                const int32 dim = groupDims[i];
                double sum = 0.0;
                for (int32 sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
                    sum += static_cast<double>(features[sampleIndex * featureCount + dim]);
                centroid[i] = sum / static_cast<double>(sampleCount);
                if (!std::isfinite(centroid[i]))
                    centroid[i] = 0.0;
            }
            double normSum = 0.0;
            for (int32 sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
            {
                double squaredNorm = 0.0;
                for (int32 i = 0; i < dimCount; i++)
                {
                    const int32 dim = groupDims[i];
                    const double centered = static_cast<double>(
                        features[sampleIndex * featureCount + dim]) - centroid[i];
                    squaredNorm += centered * centered;
                }
                normSum += std::sqrt(squaredNorm > 0.0 ? squaredNorm : 0.0);
            }
            const double preFloorDeviation = normSum / static_cast<double>(sampleCount);
            double deviation = preFloorDeviation;
            // !(dev > floor) also catches NaN/non-finite: never divide by
            // zero and never propagate a bad stat into the database.
            if (!(deviation > DeviationFloor))
                deviation = 1.0;
            if (!(preFloorDeviation > DeviationFloor))
                constantFeatureCount += dimCount;
            for (int32 i = 0; i < dimCount; i++)
                deviations[groupDims[i]] = static_cast<float>(deviation);
        }

        return constantFeatureCount;
    }
}

bool MotionMatchingBaker::Bake(
    SkinnedModel* model,
    const PoseSearchSchema& schema,
    MotionMatchingDatabase* database,
    int32& bakedClipCount,
    int32& bakedSampleCount)
{
    bakedClipCount = 0;
    bakedSampleCount = 0;

    if (database == nullptr || database->WaitForLoaded())
    {
        LOG(Error, "Motion matching bake failed: database is not loaded.");
        return false;
    }

    Array<Animation*> animations;
    Array<Array<String>> tags;
    Array<int32> loopingClips;
    database->GetSourceData(animations, tags, loopingClips);

    Array<int32> sourceClipIndices;
    Array<float> clipLengths;
    Array<int32> clipSampleCounts;
    Array<int32> sampleClipIndices;
    Array<float> sampleTimes;
    Array<Float3> rootPositions;
    Array<Float3> rootVelocities;
    Array<Float3> rootForwards;
    Array<float> poseFeatures;
    Array<float> trajectoryFeatures;
    Array<float> poseFeatureDeviations;
    Array<float> trajectoryFeatureDeviations;

    if (model == nullptr || model->WaitForLoaded())
    {
        LOG(Error, "Motion matching bake failed: player model is not loaded.");
        return false;
    }
    String schemaError;
    if (!schema.TryValidate(schemaError))
    {
        LOG(Error, "Motion matching bake failed: invalid schema. {}", schemaError);
        return false;
    }
    if (animations.Count() != tags.Count() ||
        animations.Count() != loopingClips.Count())
    {
        LOG(Error, "Motion matching bake failed: animation, tag, and loop-flag counts differ.");
        return false;
    }

    // Snapshot the schema with normalized axes so the stored contract matches what was baked.
    PoseSearchSchema bakedSchema = schema;
    bakedSchema.RootForwardAxis.Normalize();
    bakedSchema.RootUpAxis.Normalize();

    const SkeletonProfile& skeleton = bakedSchema.Skeleton;
    const int32 sampleRate = bakedSchema.SampleRate;
    const Array<float>& schemaTrajectorySampleTimes = bakedSchema.TrajectorySampleTimes;
    const Float3 rootForwardAxis = bakedSchema.RootForwardAxis;
    // Normalized root-up axis from the schema snapshot (manifest records
    // the raw axis; bake uses the normalized copy). All contact heights go
    // through PoseSearchSchema::ComputeContactHeight with this axis.
    const Float3 rootUpAxis = bakedSchema.RootUpAxis;
    Float3 localRight;
    if (!bakedSchema.TryGetRightAxis(localRight))
    {
        LOG(Error, "Motion matching bake failed: root forward and up axes cannot be parallel.");
        return false;
    }

    NodeIndices nodeIndices;
    String skeletonError;
    if (!MotionMatchingSkeleton::TryResolve(skeleton, model, nodeIndices, skeletonError))
    {
        LOG(Error, "Motion matching bake failed: {}", skeletonError);
        return false;
    }

    const int32 rootIndex = nodeIndices.Root;
    int32 poseBoneIndices[PoseBoneCount];
    nodeIndices.GetPoseBoneIndices(poseBoneIndices);

    LOG(
        Info,
        "Bake schema: sampleRate={}, forward=({:.2f}, {:.2f}, {:.2f}), up=({:.2f}, {:.2f}, {:.2f}), "
        "root='{}', poseBones=[{}, {}, {}].",
        sampleRate,
        rootForwardAxis.X, rootForwardAxis.Y, rootForwardAxis.Z,
        bakedSchema.RootUpAxis.X, bakedSchema.RootUpAxis.Y, bakedSchema.RootUpAxis.Z,
        skeleton.Root,
        skeleton.Pelvis, skeleton.LeftFoot, skeleton.RightFoot);
    if (!nodeIndices.HasOptionalNodes())
    {
        LOG(
            Info,
            "Optional skeleton nodes (spine/head/hands) are not all present in the model; "
            "they are stored in the schema but not used by pose features.");
    }
    // Design note: weights are stored for the runtime cost function only. They are
    // intentionally NOT multiplied into the baked features, so changing them later
    // does not require a re-bake.
    LOG(
        Info,
        "Schema weights stored (not applied at bake time): pose={:.2f}, trajectory={:.2f}.",
        bakedSchema.PoseWeight,
        bakedSchema.TrajectoryWeight);

    Array<ClipBakeInfo> clips;
    int32 totalSampleCount = 0;
    for (int32 sourceIndex = 0; sourceIndex < animations.Count(); sourceIndex++)
    {
        Animation* animation = animations[sourceIndex];
        if (animation == nullptr || animation->WaitForLoaded())
        {
            LOG(Warning, "Skipping null or unloaded animation at source index {}.", sourceIndex);
            continue;
        }

        // Sampling contract: count = ceil(length*rate), duration = true clip
        // length, time = min(index/rate, length). count/rate >= length; the
        // last interval may be short. Same helpers as the live query side.
        const float length = animation->GetLength();
        const int32 count = MotionMatchingSampler::ComputeSampleCount(length, sampleRate);
        if (length <= 0.0f || count <= 0)
        {
            LOG(Warning, "Skipping zero-length animation at source index {}.", sourceIndex);
            continue;
        }

        ClipBakeInfo info;
        info.Clip = animation;
        info.SourceIndex = sourceIndex;
        info.SampleStart = totalSampleCount;
        info.SampleCount = count;
        info.Length = length;
        info.Loop = loopingClips[sourceIndex] != 0;
        clips.Add(info);
        totalSampleCount += count;

        sourceClipIndices.Add(sourceIndex);
        clipLengths.Add(length);
        clipSampleCounts.Add(count);
        LOG(
            Info,
            "Added clip '{}' for tags '{}' with length {}, sample count {}, loop={}.",
            StringUtils::GetFileNameWithoutExtension(animation->GetPath()),
            MotionMatchingDatabase::JoinTags(tags[sourceIndex]),
            length,
            count,
            info.Loop);
    }

    if (clips.IsEmpty() || totalSampleCount <= 0)
    {
        LOG(Error, "Motion matching bake failed: no valid animation samples.");
        return false;
    }

    LOG(Info, "Motion clips counted: {}", clips.Count());
    LOG(Info, "Motion samples generated: {}", totalSampleCount);
    LOG(
        Info,
        "Sampling contract T1: count=ceil(length*rate), time=min(index/rate,length), "
        "duration=true clip length; loop wraps [0,length), non-loop endpoints one-sided "
        "with Valid/OutsideNonLoop/Invalid status (PLAN Phase 2 §3, §18.6).");

    Array<SampleBakeInfo> samples;
    samples.Resize(totalSampleCount);
    sampleClipIndices.Resize(totalSampleCount);
    sampleTimes.Resize(totalSampleCount);
    rootPositions.Resize(totalSampleCount);
    rootVelocities.Resize(totalSampleCount);
    rootForwards.Resize(totalSampleCount);
    poseFeatures.Resize(totalSampleCount * PoseFeatureSize);
    trajectoryFeatures.Resize(totalSampleCount * TrajectoryFeatureSize);

    float featureMinimums[PoseFeatureSize];
    float featureMaximums[PoseFeatureSize];
    float rootPositionMinimums[3] = { MAX_float, MAX_float, MAX_float };
    float rootPositionMaximums[3] = { -MAX_float, -MAX_float, -MAX_float };
    float rootVelocityMinimums[3] = { MAX_float, MAX_float, MAX_float };
    float rootVelocityMaximums[3] = { -MAX_float, -MAX_float, -MAX_float };
    float trajectoryMinimums[TrajectoryFeatureSize];
    float trajectoryMaximums[TrajectoryFeatureSize];
    for (int32 i = 0; i < PoseFeatureSize; i++)
    {
        featureMinimums[i] = MAX_float;
        featureMaximums[i] = -MAX_float;
    }
    for (int32 i = 0; i < TrajectoryFeatureSize; i++)
    {
        trajectoryMinimums[i] = MAX_float;
        trajectoryMaximums[i] = -MAX_float;
    }

    Array<Matrix> pose;
    for (int32 clipIndex = 0; clipIndex < clips.Count(); clipIndex++)
    {
        ClipBakeInfo& clip = clips[clipIndex];
        for (int32 localSampleIndex = 0; localSampleIndex < clip.SampleCount; localSampleIndex++)
        {
            const int32 sampleIndex = clip.SampleStart + localSampleIndex;
            const float time = MotionMatchingSampler::ComputeSampleTime(
                localSampleIndex, clip.Length, sampleRate);

            if (!SampleAnimationModelPose(clip.Clip, model, time, pose))
            {
                LOG(
                    Error,
                    "Motion matching bake failed while sampling clip {} at {:.4f}s.",
                    clipIndex,
                    time);
                return false;
            }

            const int32 poseCount = pose.Count();
            if (rootIndex < 0 || rootIndex >= poseCount)
            {
                LOG(Error, "Motion matching bake failed: root node index is outside pose array.");
                return false;
            }

            SampleBakeInfo& sample = samples[sampleIndex];
            sample.ClipIndex = clipIndex;
            sample.Time = time;
            sample.RootTransform = pose[rootIndex];
            const Matrix rootInverse = Matrix::Invert(sample.RootTransform);

            Matrix relative;
            for (int32 boneIndex = 0; boneIndex < PoseBoneCount; boneIndex++)
            {
                // Missing optional upper-body nodes bake zeros (constant
                // features normalize to harmless no-ops) instead of failing.
                const int32 nodeIndex = poseBoneIndices[boneIndex];
                if (nodeIndex < 0 || nodeIndex >= poseCount)
                {
                    if (boneIndex <= 2)
                    {
                        LOG(Error, "Motion matching bake failed: required bone {} is outside pose array.", boneIndex);
                        return false;
                    }
                    sample.Positions[boneIndex] = Float3::Zero;
                    continue;
                }
                Matrix::Multiply(pose[nodeIndex], rootInverse, relative);
                sample.Positions[boneIndex] = relative.GetTranslation();
                if (!IsFinite(sample.Positions[boneIndex]))
                {
                    LOG(Error, "Motion matching bake failed: non-finite position at sample {}.", sampleIndex);
                    return false;
                }
            }

            sampleClipIndices[sampleIndex] = clipIndex;
            sampleTimes[sampleIndex] = time;
            rootPositions[sampleIndex] = sample.RootTransform.GetTranslation();
            rootForwards[sampleIndex] = GetModelDirection(sample.RootTransform, rootForwardAxis);
            if (!IsFinite(rootPositions[sampleIndex]) ||
                !IsFinite(rootForwards[sampleIndex]) ||
                rootForwards[sampleIndex].LengthSquared() <= MinimumDeltaTime)
            {
                LOG(Error, "Motion matching bake failed: invalid root transform at sample {}.", sampleIndex);
                return false;
            }

            const int32 featureOffset = sampleIndex * PoseFeatureSize;
            for (int32 boneIndex = 0; boneIndex < PoseBoneCount; boneIndex++)
            {
                const int32 boneOffset = featureOffset + boneIndex * 3;
                const Float3& position = sample.Positions[boneIndex];
                poseFeatures[boneOffset] = position.X;
                poseFeatures[boneOffset + 1] = position.Y;
                poseFeatures[boneOffset + 2] = position.Z;
                IncludeVectorRange(featureMinimums, featureMaximums, boneIndex * 3, position);
            }

            IncludeVectorRange(rootPositionMinimums, rootPositionMaximums, 0, rootPositions[sampleIndex]);
        }

        clip.StartRootTransform = samples[clip.SampleStart].RootTransform;
        clip.StartRootInverse = Matrix::Invert(clip.StartRootTransform);
        if (!SampleAnimationModelPose(clip.Clip, model, clip.Length, pose) ||
            rootIndex >= pose.Count())
        {
            LOG(
                Error,
                "Motion matching bake failed while sampling clip {} endpoint at {:.4f}s.",
                clipIndex,
                clip.Length);
            return false;
        }
        clip.CycleRootTransform = MultiplyMatrices(pose[rootIndex], clip.StartRootInverse);
    }

    LOG(Info, "Baking pose positions: {}/{}", totalSampleCount, totalSampleCount);

    float maximumSpeeds[PoseBoneCount] = { 0.0f };
    int32 contactSamples = 0;
    for (int32 sampleIndex = 0; sampleIndex < totalSampleCount; sampleIndex++)
    {
        const SampleBakeInfo& current = samples[sampleIndex];
        const ClipBakeInfo& clip = clips[current.ClipIndex];
        const int32 localIndex = sampleIndex - clip.SampleStart;
        // Endpoint contract (interior central, edges one-sided). Indices come from
        // the shared helper — no ad-hoc prev/next literals here; the live query
        // documents the same stencil.
        int32 previousLocalIndex = 0;
        int32 nextLocalIndex = 0;
        MotionMatchingSampler::GetCentralDiffIndices(
            localIndex, clip.SampleCount, clip.Loop, previousLocalIndex, nextLocalIndex);
        const int32 previousIndex = clip.SampleStart + previousLocalIndex;
        const int32 nextIndex = clip.SampleStart + nextLocalIndex;
        const SampleBakeInfo& previous = samples[previousIndex];
        const SampleBakeInfo& next = samples[nextIndex];
        const float frameDeltaTime = MotionMatchingSampler::GetFrameDt(sampleRate);
        // Central dt spans two frames; at non-loop edges the clamped neighbor
        // collapses (next.Time - previous.Time) to a single frame, which IS
        // the one-sided stencil — never a fabricated zero future.
        const float deltaTime = clip.Loop
            ? frameDeltaTime * 2.0f
            : next.Time - previous.Time;

        // Root-relative pose velocity (bake/query feature). Positions are
        // already root-relative, so this is the feature-space velocity — NOT
        // the world-space foot velocity used by foot-lock. Named distinctly
        // so the two semantics can never share a threshold blindly.
        Float3 poseVelocityRootRelative[PoseBoneCount] = { Float3::Zero, Float3::Zero, Float3::Zero };
        Float3 rootVelocity = Float3::Zero;
        if (deltaTime > MinimumDeltaTime)
        {
            for (int32 boneIndex = 0; boneIndex < PoseBoneCount; boneIndex++)
                poseVelocityRootRelative[boneIndex] = (next.Positions[boneIndex] - previous.Positions[boneIndex]) / deltaTime;
            if (clip.Loop)
            {
                // Each stencil side wraps in its own cycle with the matching
                // CycleRootTransform power, so seam neighbors are cycle-correct
                // (no endpoint duplication).
                const Matrix previousRoot = GetRootAtTime(
                    clip,
                    samples,
                    current.Time - frameDeltaTime,
                    sampleRate);
                const Matrix nextRoot = GetRootAtTime(
                    clip,
                    samples,
                    current.Time + frameDeltaTime,
                    sampleRate);
                rootVelocity =
                    (nextRoot.GetTranslation() - previousRoot.GetTranslation()) /
                    deltaTime;
            }
            else
            {
                rootVelocity =
                    (next.RootTransform.GetTranslation() - previous.RootTransform.GetTranslation()) /
                    deltaTime;
            }
        }

        bool velocitiesFinite = IsFinite(rootVelocity);
        for (int32 boneIndex = 0; boneIndex < PoseBoneCount; boneIndex++)
            velocitiesFinite &= IsFinite(poseVelocityRootRelative[boneIndex]);
        if (!velocitiesFinite)
        {
            LOG(Error, "Motion matching bake failed: non-finite velocity at sample {}.", sampleIndex);
            return false;
        }

        rootVelocities[sampleIndex] = rootVelocity;
        IncludeVectorRange(rootVelocityMinimums, rootVelocityMaximums, 0, rootVelocity);

        const int32 featureOffset = sampleIndex * PoseFeatureSize + PositionFeatureSize;
        for (int32 boneIndex = 0; boneIndex < PoseBoneCount; boneIndex++)
        {
            const int32 boneOffset = featureOffset + boneIndex * 3;
            const Float3& velocity = poseVelocityRootRelative[boneIndex];
            poseFeatures[boneOffset] = velocity.X;
            poseFeatures[boneOffset + 1] = velocity.Y;
            poseFeatures[boneOffset + 2] = velocity.Z;
            IncludeVectorRange(
                featureMinimums,
                featureMaximums,
                PositionFeatureSize + boneIndex * 3,
                velocity);
            maximumSpeeds[boneIndex] = Math::Max(
                maximumSpeeds[boneIndex],
                Math::Sqrt(velocity.LengthSquared()));
        }

        // Foot-contact flags: planted = slow and low in the root frame.
        // Feet are required bones, so indices 1/2 are always valid here.
        // Height = dot(relativeFoot, normalizedRootUpAxis) via the shared
        // helper — never hardcoded .Y. Positions are root-relative (root at
        // origin), so relativeFoot = foot - root(role) is explicit below.
        // NOTE: featureOffset already points at the velocity block (+21), so
        // rebase from the sample start here. Contact offsets resolve via channel
        // metadata, never size arithmetic.
        const int32 contactOffsetL = sampleIndex * PoseFeatureSize +
            PoseSearchSchema::GetContactFeatureOffset(0);
        const int32 contactOffsetR = sampleIndex * PoseFeatureSize +
            PoseSearchSchema::GetContactFeatureOffset(1);
        const Float3 rootOrigin = Float3::Zero;
        const float contactHeightL = PoseSearchSchema::ComputeContactHeight(
            current.Positions[1], rootOrigin, rootUpAxis);
        const float contactHeightR = PoseSearchSchema::ComputeContactHeight(
            current.Positions[2], rootOrigin, rootUpAxis);
        const float contactL = PoseSearchSchema::IsPlantedContact(
            poseVelocityRootRelative[1].LengthSquared(), ContactMaxSpeed,
            contactHeightL, ContactMaxHeight) ? 1.0f : 0.0f;
        const float contactR = PoseSearchSchema::IsPlantedContact(
            poseVelocityRootRelative[2].LengthSquared(), ContactMaxSpeed,
            contactHeightR, ContactMaxHeight) ? 1.0f : 0.0f;
        poseFeatures[contactOffsetL] = contactL;
        poseFeatures[contactOffsetR] = contactR;
        IncludeRange(featureMinimums, featureMaximums, PoseSearchSchema::GetContactFeatureOffset(0), contactL);
        IncludeRange(featureMinimums, featureMaximums, PoseSearchSchema::GetContactFeatureOffset(1), contactR);
        if (contactL > 0.5f)
            contactSamples++;
        if (contactR > 0.5f)
            contactSamples++;
    }

    LOG(Info, "Baking pose velocities: {}/{}", totalSampleCount, totalSampleCount);

    // Tail-validity telemetry: non-loop future targets past the clip
    // end keep a clamped endpoint pose for playback, but search must treat them
    // as invalid/penalized via MotionMatchingSampler::IsNonLoopFutureOutside
    // (search-side integration, runtime horizon check, no rebake).
    int32 tailClampedTargets = 0;

    for (int32 sampleIndex = 0; sampleIndex < totalSampleCount; sampleIndex++)
    {
        const SampleBakeInfo& current = samples[sampleIndex];
        const ClipBakeInfo& clip = clips[current.ClipIndex];
        const Matrix currentRootInverse = Matrix::Invert(current.RootTransform);
        const int32 trajectoryOffset = sampleIndex * TrajectoryFeatureSize;
        const float frameDt = MotionMatchingSampler::GetFrameDt(sampleRate);

        for (int32 pointIndex = 0; pointIndex < TrajectoryPointCount; pointIndex++)
        {
            const float targetTime = current.Time + schemaTrajectorySampleTimes[pointIndex];
            if (!MotionMatchingSampler::IsTrajectoryTargetValid(targetTime, clip.Length, clip.Loop))
                tailClampedTargets++;
            const Matrix targetRoot = GetRootAtTime(
                clip,
                samples,
                targetTime,
                sampleRate);

            const Float3 relativePosition = GetRelativePosition(
                targetRoot,
                currentRootInverse);
            const Float3 targetForward = GetModelDirection(targetRoot, rootForwardAxis);
            Float3 relativeForward;
            Float3::TransformNormal(targetForward, currentRootInverse, relativeForward);

            if (!IsFinite(relativePosition) || !IsFinite(relativeForward))
            {
                LOG(Error, "Motion matching bake failed: invalid trajectory at sample {}.", sampleIndex);
                return false;
            }

            Float2 planarPosition(
                Float3::Dot(relativePosition, localRight),
                Float3::Dot(relativePosition, rootForwardAxis));
            Float2 planarDirection(
                Float3::Dot(relativeForward, localRight),
                Float3::Dot(relativeForward, rootForwardAxis));
            if (planarDirection.LengthSquared() > MinimumDeltaTime)
                planarDirection.Normalize();
            else
                planarDirection = Float2(0.0f, 1.0f);

            // Yaw-rate channel: explicit turn signal from baked root
            // motion — the same physical quantity (rad/s, unwrap [-pi,pi]) the
            // live query derives from predicted facings, through the SAME
            // MotionMatchingSampler unwrap/rate helpers. Stencil differs by
            // construction and is documented, not hidden: bake = central ±1
            // bake sample (2*frameDt); query = central ±1 trajectory interval
            // (schema times). Non-loop edges use one-sided differences over a
            // single true frame instead of halving the rate against a clamped
            // twin, so a moving tail never bakes a fake slowdown.
            const float pastTime = targetTime - frameDt;
            const float futureTime = targetTime + frameDt;
            const bool pastValid = MotionMatchingSampler::IsTrajectoryTargetValid(pastTime, clip.Length, clip.Loop);
            const bool futureValid = MotionMatchingSampler::IsTrajectoryTargetValid(futureTime, clip.Length, clip.Loop);
            const Float3 pastForward = GetModelDirection(
                GetRootAtTime(clip, samples, pastTime, sampleRate), rootForwardAxis);
            const Float3 futureForward = GetModelDirection(
                GetRootAtTime(clip, samples, futureTime, sampleRate), rootForwardAxis);
            float yawRate = 0.0f;
            if (IsFinite(pastForward) && IsFinite(futureForward) && IsFinite(targetForward))
            {
                if (pastValid && futureValid)
                {
                    yawRate = MotionMatchingSampler::ComputeYawRate(
                        MotionMatchingSampler::YawFromForwardXZ(pastForward.X, pastForward.Z),
                        MotionMatchingSampler::YawFromForwardXZ(futureForward.X, futureForward.Z),
                        2.0f * frameDt);
                }
                else if (pastValid)
                {
                    yawRate = MotionMatchingSampler::ComputeYawRate(
                        MotionMatchingSampler::YawFromForwardXZ(pastForward.X, pastForward.Z),
                        MotionMatchingSampler::YawFromForwardXZ(targetForward.X, targetForward.Z),
                        frameDt);
                }
                else if (futureValid)
                {
                    yawRate = MotionMatchingSampler::ComputeYawRate(
                        MotionMatchingSampler::YawFromForwardXZ(targetForward.X, targetForward.Z),
                        MotionMatchingSampler::YawFromForwardXZ(futureForward.X, futureForward.Z),
                        frameDt);
                }
                // Neither side valid (clip shorter than one frame): leave 0;
                // pose/root velocity is likewise degenerate there, consistently.
            }

            const int32 pointOffset = trajectoryOffset + pointIndex * TrajectoryPointSize;
            trajectoryFeatures[pointOffset] = planarPosition.X;
            trajectoryFeatures[pointOffset + 1] = planarPosition.Y;
            trajectoryFeatures[pointOffset + 2] = planarDirection.X;
            trajectoryFeatures[pointOffset + 3] = planarDirection.Y;
            trajectoryFeatures[pointOffset + 4] = yawRate;
            for (int32 component = 0; component < TrajectoryPointSize; component++)
            {
                IncludeRange(
                    trajectoryMinimums,
                    trajectoryMaximums,
                    pointIndex * TrajectoryPointSize + component,
                    trajectoryFeatures[pointOffset + component]);
            }
        }
    }

    LOG(Info, "Baking trajectory features: {}/{}", totalSampleCount, totalSampleCount);
    LOG(
        Info,
        "Non-loop trajectory targets clamped for playback (validity for search via "
        "MotionMatchingSampler::IsNonLoopFutureOutside, S1): {}.",
        tailClampedTargets);
    LOG(
        Info,
        "Pose feature baking finished: {}/{}, dimensions={}.",
        totalSampleCount,
        totalSampleCount,
        PoseFeatureSize);
    LOG(
        Info,
        "pos_pelvis_x=[{:.2f}, {:.2f}], pos_pelvis_y=[{:.2f}, {:.2f}], pos_pelvis_z=[{:.2f}, {:.2f}]",
        featureMinimums[0], featureMaximums[0],
        featureMinimums[1], featureMaximums[1],
        featureMinimums[2], featureMaximums[2]);
    LOG(
        Info,
        "pos_leftFoot_x=[{:.2f}, {:.2f}], pos_leftFoot_y=[{:.2f}, {:.2f}], pos_leftFoot_z=[{:.2f}, {:.2f}]",
        featureMinimums[3], featureMaximums[3],
        featureMinimums[4], featureMaximums[4],
        featureMinimums[5], featureMaximums[5]);
    LOG(
        Info,
        "pos_rightFoot_x=[{:.2f}, {:.2f}], pos_rightFoot_y=[{:.2f}, {:.2f}], pos_rightFoot_z=[{:.2f}, {:.2f}]",
        featureMinimums[6], featureMaximums[6],
        featureMinimums[7], featureMaximums[7],
        featureMinimums[8], featureMaximums[8]);
    LOG(
        Info,
        "vel_pelvis_x=[{:.2f}, {:.2f}], vel_pelvis_y=[{:.2f}, {:.2f}], vel_pelvis_z=[{:.2f}, {:.2f}]",
        featureMinimums[9], featureMaximums[9],
        featureMinimums[10], featureMaximums[10],
        featureMinimums[11], featureMaximums[11]);
    LOG(
        Info,
        "vel_leftFoot_x=[{:.2f}, {:.2f}], vel_leftFoot_y=[{:.2f}, {:.2f}], vel_leftFoot_z=[{:.2f}, {:.2f}]",
        featureMinimums[12], featureMaximums[12],
        featureMinimums[13], featureMaximums[13],
        featureMinimums[14], featureMaximums[14]);
    LOG(
        Info,
        "vel_rightFoot_x=[{:.2f}, {:.2f}], vel_rightFoot_y=[{:.2f}, {:.2f}], vel_rightFoot_z=[{:.2f}, {:.2f}]",
        featureMinimums[15], featureMaximums[15],
        featureMinimums[16], featureMaximums[16],
        featureMinimums[17], featureMaximums[17]);
    LOG(
        Info,
        "pos_spine=[{:.2f}, {:.2f}][{:.2f}, {:.2f}][{:.2f}, {:.2f}], pos_head=[{:.2f}, {:.2f}][{:.2f}, {:.2f}][{:.2f}, {:.2f}]",
        featureMinimums[9], featureMaximums[9],
        featureMinimums[10], featureMaximums[10],
        featureMinimums[11], featureMaximums[11],
        featureMinimums[12], featureMaximums[12],
        featureMinimums[13], featureMaximums[13],
        featureMinimums[14], featureMaximums[14]);
    LOG(
        Info,
        "pos_handL=[{:.2f}, {:.2f}][{:.2f}, {:.2f}][{:.2f}, {:.2f}], pos_handR=[{:.2f}, {:.2f}][{:.2f}, {:.2f}][{:.2f}, {:.2f}]",
        featureMinimums[15], featureMaximums[15],
        featureMinimums[16], featureMaximums[16],
        featureMinimums[17], featureMaximums[17],
        featureMinimums[18], featureMaximums[18],
        featureMinimums[19], featureMaximums[19],
        featureMinimums[20], featureMaximums[20]);
    LOG(
        Info,
        "vel_spine_max={:.2f}, vel_head_max={:.2f}, vel_handL_max={:.2f}, vel_handR_max={:.2f} units/s.",
        maximumSpeeds[3],
        maximumSpeeds[4],
        maximumSpeeds[5],
        maximumSpeeds[6]);
    LOG(
        Info,
        "contact_L=[{:.2f}, {:.2f}], contact_R=[{:.2f}, {:.2f}], planted fraction={:.3f}.",
        featureMinimums[PoseSearchSchema::GetContactFeatureOffset(0)], featureMaximums[PoseSearchSchema::GetContactFeatureOffset(0)],
        featureMinimums[PoseSearchSchema::GetContactFeatureOffset(1)], featureMaximums[PoseSearchSchema::GetContactFeatureOffset(1)],
        static_cast<float>(contactSamples) / static_cast<float>(totalSampleCount * 2));
    LOG(
        Info,
        "root_pos_x=[{:.2f}, {:.2f}], root_pos_y=[{:.2f}, {:.2f}], root_pos_z=[{:.2f}, {:.2f}]",
        rootPositionMinimums[0], rootPositionMaximums[0],
        rootPositionMinimums[1], rootPositionMaximums[1],
        rootPositionMinimums[2], rootPositionMaximums[2]);
    LOG(
        Info,
        "root_vel_x=[{:.2f}, {:.2f}], root_vel_y=[{:.2f}, {:.2f}], root_vel_z=[{:.2f}, {:.2f}]",
        rootVelocityMinimums[0], rootVelocityMaximums[0],
        rootVelocityMinimums[1], rootVelocityMaximums[1],
        rootVelocityMinimums[2], rootVelocityMaximums[2]);
    LOG(
        Info,
        "max_speed_pelvis={:.2f}, max_speed_leftFoot={:.2f}, max_speed_rightFoot={:.2f} units/s.",
        maximumSpeeds[0],
        maximumSpeeds[1],
        maximumSpeeds[2]);

    for (int32 pointIndex = 0; pointIndex < TrajectoryPointCount; pointIndex++)
    {
        const int32 offset = pointIndex * TrajectoryPointSize;
        LOG(
            Info,
            "trajectory_{:.2f}s_pos_right=[{:.2f}, {:.2f}], trajectory_{:.2f}s_pos_forward=[{:.2f}, {:.2f}], "
            "trajectory_{:.2f}s_dir_right=[{:.2f}, {:.2f}], trajectory_{:.2f}s_dir_forward=[{:.2f}, {:.2f}], "
            "trajectory_{:.2f}s_yawrate=[{:.2f}, {:.2f}]",
            schemaTrajectorySampleTimes[pointIndex], trajectoryMinimums[offset], trajectoryMaximums[offset],
            schemaTrajectorySampleTimes[pointIndex], trajectoryMinimums[offset + 1], trajectoryMaximums[offset + 1],
            schemaTrajectorySampleTimes[pointIndex], trajectoryMinimums[offset + 2], trajectoryMaximums[offset + 2],
            schemaTrajectorySampleTimes[pointIndex], trajectoryMinimums[offset + 3], trajectoryMaximums[offset + 3],
            schemaTrajectorySampleTimes[pointIndex], trajectoryMinimums[offset + 4], trajectoryMaximums[offset + 4]);
    }

    // Revision 3: one mean-row-norm deviation per normalization group; baked
    // features stay raw. Groups come from the schema group function (single
    // source of truth), never from hardcoded dim ranges.
    const int32 constantPoseFeatures = ComputeGroupDeviations(
        poseFeatures,
        totalSampleCount,
        PoseFeatureSize,
        &PoseSearchSchema::GetPoseNormalizationGroup,
        poseFeatureDeviations);
    const int32 constantTrajectoryFeatures = ComputeGroupDeviations(
        trajectoryFeatures,
        totalSampleCount,
        TrajectoryFeatureSize,
        &PoseSearchSchema::GetTrajectoryNormalizationGroup,
        trajectoryFeatureDeviations);
    LOG(
        Info,
        "Feature deviations computed (N1b UE-style rev {}): raw features kept, pose dimensions={} ({} floored), trajectory dimensions={} ({} floored).",
        PoseSearchSchema::NormalizationRevision,
        PoseFeatureSize,
        constantPoseFeatures,
        TrajectoryFeatureSize,
        constantTrajectoryFeatures);
    // Grouped-stats audit: one line per pose group proves L/R limbs share
    // the group deviation by construction (symmetric costs for mirrored
    // queries). All groups use the same mean-row-norm formula, so scalar
    // (contacts) and vector limbs are logged identically.
    for (int32 group = 0; group <= 10; group++)
    {
        int32 firstDim = -1;
        int32 dimCount = 0;
        for (int32 dim = 0; dim < PoseFeatureSize; dim++)
        {
            if (PoseSearchSchema::GetPoseNormalizationGroup(dim) == group)
            {
                if (firstDim < 0)
                    firstDim = dim;
                dimCount++;
            }
        }
        if (firstDim >= 0)
        {
            LOG(
                Info,
                "pose norm group {}: dims={} deviation={:.4f}.",
                group,
                dimCount,
                poseFeatureDeviations[firstDim]);
        }
    }

    database->SetBakedData(
        bakedSchema,
        sourceClipIndices,
        clipLengths,
        clipSampleCounts,
        sampleClipIndices,
        sampleTimes,
        rootPositions,
        rootVelocities,
        rootForwards,
        poseFeatures,
        trajectoryFeatures,
        poseFeatureDeviations,
        trajectoryFeatureDeviations);
    if (!database->ValidateData())
    {
        LOG(Error, "Motion matching bake failed: generated database did not pass validation.");
        return false;
    }
    const int32 expectedClipCount = database->GetClipCount();
    const int32 expectedSampleCount = database->GetSampleCount();
    if (database->Save())
    {
        LOG(Error, "Motion matching bake failed while saving database '{}'.", database->GetPath());
        return false;
    }

    // Persistence check with retries: SaveAsset triggers an async engine
    // reimport ("Creating package ... file is in use") and an immediate
    // Reload can read pre-save bytes (observed: revision tail missing on
    // first read). Retry until the reloaded database matches, revision and
    // layout hash included — a stale read fails the checks, not silently.
    bool persisted = false;
    for (int32 attempt = 0; attempt < 6 && !persisted; attempt++)
    {
        if (attempt > 0)
            LOG(Info, "Motion matching persistence check retry {}/5.", attempt);
        database->Reload();
        if (database->WaitForLoaded(10000) ||
            !database->ValidateData() ||
            database->GetClipCount() != expectedClipCount ||
            database->GetSampleCount() != expectedSampleCount ||
            database->GetSchema() != bakedSchema ||
            database->GetNormalizationRevision() != PoseSearchSchema::NormalizationRevision ||
            database->GetFeatureLayoutHash() != static_cast<int32>(PoseSearchSchema::GetFeatureLayoutHash()))
            continue;
        persisted = true;
    }
    if (!persisted)
    {
        LOG(Error, "Motion matching bake failed persistence reload check for '{}'.", database->GetPath());
        return false;
    }

    LOG(
        Info,
        "Motion matching database saved: clips={}, samples={}, path='{}'.",
        database->GetClipCount(),
        database->GetSampleCount(),
        database->GetPath());
    bakedClipCount = expectedClipCount;
    bakedSampleCount = expectedSampleCount;
    return true;
}

#endif
