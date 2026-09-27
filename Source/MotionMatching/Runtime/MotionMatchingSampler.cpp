// MotionMatchingSampler.cpp: sampler implementation (SampleClipPose/SampleClipLocalPose/ComposeModelPose plus the ClassifyClipTime/WrapLoopTime/TimeToSampleIndex/GetCentralDiffIndices/IsNonLoopFutureOutside sample-domain helpers).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the channel-evaluation block intentionally duplicates the baker math (game builds never link editor baker code) with row-vector childLocal * parentWorld composition; outputs are snapshot copies, never aliases; non-loop timestamps outside [0, length] report OutsideNonLoop while clamping position for playback continuity.
#include "MotionMatchingSampler.h"

#include "Engine/Content/Assets/Animation.h"
#include "Engine/Content/Assets/SkinnedModel.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Transform.h"

#include <cmath>

namespace
{
    bool IsFinite(const Float3& value)
    {
        return std::isfinite(value.X) &&
                std::isfinite(value.Y) &&
                std::isfinite(value.Z);
    }

    bool IsFiniteQuat(const Quaternion& value)
    {
        return std::isfinite(value.X) &&
                std::isfinite(value.Y) &&
                std::isfinite(value.Z) &&
                std::isfinite(value.W);
    }

    // Shared channel evaluation: animation channels override bind-pose local
    // transforms, missing sub-tracks keep bind values. Identical to the baker.
    bool SampleLocalsInternal(
        Animation* clip,
        const Array<SkeletonNode>& nodes,
        float time,
        Array<Matrix>& outLocals,
        String& error)
    {
        const auto& channels = clip->Data.Channels;
        const int32 nodeCount = nodes.Count();
        const float framesPerSecond = static_cast<float>(clip->Data.FramesPerSecond);
        const float sampleFrame = time * framesPerSecond;

        outLocals.Resize(nodeCount);
        for (int32 nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
        {
            const auto& node = nodes[nodeIndex];
            Transform local = node.LocalTransform;

            for (int32 channelIndex = 0; channelIndex < channels.Count(); channelIndex++)
            {
                const auto& channel = channels[channelIndex];
                if (channel.NodeName != node.Name)
                    continue;
                if (channel.Position.GetKeyframes().HasItems())
                    channel.Position.Evaluate(local.Translation, sampleFrame, false);
                if (channel.Rotation.GetKeyframes().HasItems())
                    channel.Rotation.Evaluate(local.Orientation, sampleFrame, false);
                if (channel.Scale.GetKeyframes().HasItems())
                    channel.Scale.Evaluate(local.Scale, sampleFrame, false);
                break;
            }

            local.GetWorld(outLocals[nodeIndex]);
        }

        if (nodeCount > 0 && !IsFinite(outLocals[0].GetTranslation()))
        {
            error = TEXT("Motion matching playback failed: sampled pose is non-finite.");
            return false;
        }
        return true;
    }
}

bool MotionMatchingSampler::SampleClipPose(
    Animation* clip,
    SkinnedModel* model,
    float time,
    Array<Matrix>& outPose,
    String& error)
{
    outPose.Clear();
    error = String::Empty;

    if (clip == nullptr || clip->WaitForLoaded() ||
        model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("Motion matching playback failed: clip or model is not loaded.");
        return false;
    }

    const float framesPerSecond = static_cast<float>(clip->Data.FramesPerSecond);
    if (framesPerSecond <= 0.0f)
    {
        error = TEXT("Motion matching playback failed: clip has no frame rate.");
        return false;
    }

    // Same evaluation the baker uses: animation channels override bind-pose
    // local transforms, missing sub-tracks keep bind values, hierarchy is
    // composed parent-first (model space == actor space for a model
    // authored at the origin).
    // Intentional duplication note: this block DUPLICATES the baker's
    // channel math (game build cannot link editor-only baker code). Keep the
    // two in sync; drift is caught by golden-vector comparison, not by eye.
    // Row-vector: world = childLocal * parentWorld (parent on the RIGHT —
    // Matrix::Multiply(locals[nodeIndex], parentPose)). Reversing the operand
    // order compiles fine and silently produces garbage world poses.
    const auto& nodes = model->GetNodes();
    const int32 nodeCount = nodes.Count();

    Array<Matrix> locals;
    if (!SampleLocalsInternal(clip, nodes, time, locals, error))
        return false;

    outPose.Resize(nodeCount);
    for (int32 nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
    {
        const int32 parentIndex = nodes[nodeIndex].ParentIndex;
        if (parentIndex >= 0 && parentIndex < nodeCount)
            Matrix::Multiply(locals[nodeIndex], outPose[parentIndex], outPose[nodeIndex]);
        else
            outPose[nodeIndex] = locals[nodeIndex];
    }

    return true;
}

bool MotionMatchingSampler::GetNodeReferenceLocal(
    SkinnedModel* model,
    int32 nodeIndex,
    Matrix& outLocal,
    String& error)
{
    outLocal = Matrix::Identity;
    error = String::Empty;

    if (model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("Motion matching playback failed: model is not loaded.");
        return false;
    }

    const auto& nodes = model->GetNodes();
    if (nodeIndex < 0 || nodeIndex >= nodes.Count())
    {
        error = TEXT("Motion matching playback failed: root node index is out of range.");
        return false;
    }

    outLocal = nodes[nodeIndex].LocalTransform.GetWorld();
    return true;
}

bool MotionMatchingSampler::SampleClipLocalPose(
    Animation* clip,
    SkinnedModel* model,
    float time,
    Array<Matrix>& outLocals,
    String& error)
{
    outLocals.Clear();
    error = String::Empty;

    if (clip == nullptr || clip->WaitForLoaded() ||
        model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("Motion matching playback failed: clip or model is not loaded.");
        return false;
    }

    const float framesPerSecond = static_cast<float>(clip->Data.FramesPerSecond);
    if (framesPerSecond <= 0.0f)
    {
        error = TEXT("Motion matching playback failed: clip has no frame rate.");
        return false;
    }

    return SampleLocalsInternal(clip, model->GetNodes(), time, outLocals, error);
}

bool MotionMatchingSampler::GetSkeletonHierarchy(
    SkinnedModel* model,
    Array<int32>& outParents,
    Array<Matrix>& outReferenceLocals,
    String& error)
{
    outParents.Clear();
    outReferenceLocals.Clear();
    error = String::Empty;

    if (model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("Motion matching playback failed: model is not loaded.");
        return false;
    }

    const auto& nodes = model->GetNodes();
    const int32 nodeCount = nodes.Count();
    if (nodeCount <= 0)
    {
        error = TEXT("Motion matching playback failed: model has no skeleton nodes.");
        return false;
    }

    outParents.Resize(nodeCount);
    outReferenceLocals.Resize(nodeCount);
    for (int32 i = 0; i < nodeCount; i++)
    {
        outParents[i] = nodes[i].ParentIndex;
        nodes[i].LocalTransform.GetWorld(outReferenceLocals[i]);
    }
    return true;
}

bool MotionMatchingSampler::StripRootLocal(
    const Array<Matrix>& locals,
    int32 rootIndex,
    const Matrix& referenceRootLocal,
    Array<Matrix>& outStripped,
    String& error)
{
    error = String::Empty;
    const int32 nodeCount = locals.Count();
    if (nodeCount <= 0 || rootIndex < 0 || rootIndex >= nodeCount)
    {
        error = TEXT("Motion matching playback failed: invalid locals or root index for stripping.");
        return false;
    }

    outStripped.Resize(nodeCount);
    for (int32 i = 0; i < nodeCount; i++)
        outStripped[i] = locals[i];

    // Keep the sampled root scale (usually ~1, carries any animated scale),
    // restore the bind reference rotation + translation so imported
    // calibration (root height / basis rotation) survives while animated
    // travel and yaw are removed.
    Float3 sampledScale;
    Quaternion sampledRotation;
    Float3 sampledTranslation;
    locals[rootIndex].Decompose(sampledScale, sampledRotation, sampledTranslation);
    Float3 referenceScale;
    Quaternion referenceRotation;
    Float3 referenceTranslation;
    referenceRootLocal.Decompose(referenceScale, referenceRotation, referenceTranslation);
    if (!IsFinite(sampledScale) || !IsFinite(referenceTranslation) || !IsFiniteQuat(referenceRotation))
    {
        error = TEXT("Motion matching playback failed: non-finite root transform for stripping.");
        return false;
    }
    if (!IsFiniteQuat(sampledRotation))
        sampledRotation = Quaternion::Identity;

    Transform stripped;
    stripped.Translation = referenceTranslation;
    stripped.Orientation = referenceRotation;
    stripped.Scale = sampledScale;
    stripped.GetWorld(outStripped[rootIndex]);
    return true;
}

bool MotionMatchingSampler::BlendLocalPoses(
    const Array<Matrix>& from,
    const Array<Matrix>& to,
    float t,
    Array<Matrix>& outBlended,
    String& error)
{
    error = String::Empty;
    const int32 nodeCount = to.Count();
    if (nodeCount <= 0 || from.Count() != nodeCount)
    {
        error = TEXT("Motion matching playback failed: blend poses have mismatched sizes.");
        return false;
    }

    const float clampedT = Math::Clamp(t, 0.0f, 1.0f);
    const float smoothT = clampedT * clampedT * (3.0f - 2.0f * clampedT);
    outBlended.Resize(nodeCount);
    for (int32 i = 0; i < nodeCount; i++)
    {
        Float3 scaleA, posA, scaleB, posB;
        Quaternion rotA, rotB;
        from[i].Decompose(scaleA, rotA, posA);
        to[i].Decompose(scaleB, rotB, posB);
        if (!IsFiniteQuat(rotA))
            rotA = Quaternion::Identity;
        if (!IsFiniteQuat(rotB))
            rotB = Quaternion::Identity;

        Transform blended;
        blended.Translation = Float3::Lerp(posA, posB, smoothT);
        blended.Scale = Float3::Lerp(scaleA, scaleB, smoothT);
        Quaternion::Slerp(rotA, rotB, smoothT, blended.Orientation);
        blended.GetWorld(outBlended[i]);
    }
    return true;
}

bool MotionMatchingSampler::ComposeModelPose(
    SkinnedModel* model,
    const Array<Matrix>& locals,
    Array<Matrix>& outModelPose,
    String& error)
{
    outModelPose.Clear();
    error = String::Empty;

    if (model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("Motion matching playback failed: model is not loaded.");
        return false;
    }

    const auto& nodes = model->GetNodes();
    const int32 nodeCount = nodes.Count();
    if (locals.Count() != nodeCount || nodeCount <= 0)
    {
        error = TEXT("Motion matching playback failed: locals count does not match skeleton.");
        return false;
    }

    // Snapshot-copy guarantee: outModelPose is resized and filled
    // element-wise — never an alias into locals/model storage — so callers
    // may mutate locals (IK/layer writeback) without corrupting saved poses.
    // Row-vector FK: childLocal * parentWorld (see note above).
    outModelPose.Resize(nodeCount);
    for (int32 i = 0; i < nodeCount; i++)
    {
        const int32 parentIndex = nodes[i].ParentIndex;
        if (parentIndex >= 0 && parentIndex < nodeCount)
            Matrix::Multiply(locals[i], outModelPose[parentIndex], outModelPose[i]);
        else
            outModelPose[i] = locals[i];
    }

    if (!IsFinite(outModelPose[0].GetTranslation()))
    {
        error = TEXT("Motion matching playback failed: composed pose is non-finite.");
        return false;
    }
    return true;
}

// ---- Shared sampling contract ----
// Single definition of sample-domain math for the editor baker and the live
// query. Bake-time cost is irrelevant here; correctness and one-definition
// are what matter.

float MotionMatchingSampler::GetFrameDt(int32 sampleRate)
{
    return sampleRate > 0 ? 1.0f / static_cast<float>(sampleRate) : 0.0f;
}

int32 MotionMatchingSampler::ComputeSampleCount(float clipLength, int32 sampleRate)
{
    if (!std::isfinite(clipLength) || !(clipLength > 0.0f) || sampleRate <= 0)
        return 0;
    return Math::CeilToInt(clipLength * static_cast<float>(sampleRate));
}

float MotionMatchingSampler::ComputeSampleTime(int32 localIndex, float clipLength, int32 sampleRate)
{
    if (localIndex < 0)
        localIndex = 0;
    return Math::Min(GetFrameDt(sampleRate) * static_cast<float>(localIndex), clipLength);
}

float MotionMatchingSampler::WrapLoopTime(float time, float clipLength, int32& outCycleIndex)
{
    outCycleIndex = 0;
    if (!std::isfinite(time) || !std::isfinite(clipLength) || !(clipLength > 0.0f))
        return 0.0f;
    // Cycle-aware wrap into [0, length): no duplicated endpoint, negative
    // times land in the previous cycle (cycle < 0) with the matching
    // CycleRootTransform power applied by the caller.
    const int32 cycleIndex = static_cast<int32>(std::floor(time / clipLength));
    float localTime = time - static_cast<float>(cycleIndex) * clipLength;
    if (localTime < 0.0f)
        localTime += clipLength;
    else if (localTime >= clipLength)
        localTime = 0.0f;
    outCycleIndex = cycleIndex;
    return localTime;
}

MotionSamplingStatus MotionMatchingSampler::ClassifyClipTime(
    float time,
    float clipLength,
    bool loop,
    float& outLocalTime,
    int32& outCycleIndex)
{
    outLocalTime = 0.0f;
    outCycleIndex = 0;
    if (!std::isfinite(time) || !std::isfinite(clipLength) || !(clipLength > 0.0f))
        return MotionSamplingStatus::Invalid;
    if (loop)
    {
        outLocalTime = WrapLoopTime(time, clipLength, outCycleIndex);
        return MotionSamplingStatus::Valid;
    }
    if (time < 0.0f)
    {
        outLocalTime = 0.0f;
        return MotionSamplingStatus::OutsideNonLoop;
    }
    if (time > clipLength)
    {
        outLocalTime = clipLength;
        return MotionSamplingStatus::OutsideNonLoop;
    }
    outLocalTime = time;
    return MotionSamplingStatus::Valid;
}

MotionSamplingStatus MotionMatchingSampler::TimeToSampleIndex(
    float time,
    float clipLength,
    int32 sampleCount,
    int32 sampleRate,
    bool loop,
    int32& outLocalIndex)
{
    outLocalIndex = -1;
    if (sampleCount <= 0 || sampleRate <= 0)
        return MotionSamplingStatus::Invalid;
    float localTime = 0.0f;
    int32 cycleIndex = 0;
    const MotionSamplingStatus status = ClassifyClipTime(time, clipLength, loop, localTime, cycleIndex);
    if (status == MotionSamplingStatus::Invalid)
        return status;
    // Nearest baked sample of the (possibly wrapped) local time; Round can
    // overshoot to sampleCount at the seam, hence the clamp (never a duplicate
    // endpoint: localTime < length for loops, <= length for non-loops).
    outLocalIndex = Math::Clamp(Math::RoundToInt(localTime * static_cast<float>(sampleRate)), 0, sampleCount - 1);
    return status;
}

void MotionMatchingSampler::GetCentralDiffIndices(
    int32 localIndex,
    int32 sampleCount,
    bool loop,
    int32& outPrevLocal,
    int32& outNextLocal)
{
    if (sampleCount <= 1)
    {
        outPrevLocal = 0;
        outNextLocal = 0;
        return;
    }
    if (loop)
    {
        int32 prev = (localIndex - 1) % sampleCount;
        int32 next = (localIndex + 1) % sampleCount;
        if (prev < 0)
            prev += sampleCount;
        if (next < 0)
            next += sampleCount;
        outPrevLocal = prev;
        outNextLocal = next;
        return;
    }
    // Interior central, edges one-sided via the clamped
    // neighbor (dt collapses to one frame at the call site — never a
    // fabricated zero future).
    outPrevLocal = localIndex - 1 < 0 ? 0 : localIndex - 1;
    outNextLocal = localIndex + 1 > sampleCount - 1 ? sampleCount - 1 : localIndex + 1;
}

bool MotionMatchingSampler::IsEndpointSample(int32 localIndex, int32 sampleCount, bool loop)
{
    return !loop && (localIndex <= 0 || localIndex >= sampleCount - 1);
}

float MotionMatchingSampler::UnwrapYawDelta(float deltaYaw)
{
    float delta = deltaYaw;
    while (delta > PI)
        delta -= TWO_PI;
    while (delta < -PI)
        delta += TWO_PI;
    return delta;
}

float MotionMatchingSampler::ComputeYawRate(float yawPast, float yawFuture, float dtTotal)
{
    constexpr float MinimumDt = 0.000001f;
    if (!std::isfinite(yawPast) || !std::isfinite(yawFuture) ||
        !std::isfinite(dtTotal) || !(dtTotal > MinimumDt))
        return 0.0f;
    return UnwrapYawDelta(yawFuture - yawPast) / dtTotal;
}

float MotionMatchingSampler::YawFromForwardXZ(float x, float z)
{
    // Bake convention: yaw of the root forward around the up axis.
    return Math::Atan2(x, z);
}

bool MotionMatchingSampler::IsTrajectoryTargetValid(float targetTime, float clipLength, bool loop)
{
    if (!std::isfinite(targetTime) || !std::isfinite(clipLength) || !(clipLength > 0.0f))
        return false;
    if (loop)
        return true;
    return targetTime >= 0.0f && targetTime <= clipLength;
}

bool MotionMatchingSampler::IsNonLoopFutureOutside(float sampleTime, float horizon, float clipLength, bool loop)
{
    if (loop)
        return false;
    if (!std::isfinite(sampleTime) || !std::isfinite(horizon) ||
        !std::isfinite(clipLength) || !(clipLength > 0.0f))
        return true;
    return sampleTime + horizon > clipLength;
}
