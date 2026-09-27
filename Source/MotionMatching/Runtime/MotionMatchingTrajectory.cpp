// MotionMatchingTrajectory.cpp: trajectory prediction (MotionMatchingTrajectory::CalculateTrajectory/UpdateTrajectory/AddHistory/SampleHistory, BendVelocityTowardsDesired, RotateY).
// Ownership: plugin runtime, no game/host references.
// Key invariants: past points replay history exactly, the zero-time point stays exact, and only the future spring rollout bends toward intent; the steady-yaw curve rotates the spring target by yawRate * t at half angle so constant turns predict arcs; planar only, Y untouched.
#include "MotionMatchingTrajectory.h"
#include "SpringDamper.h"
#include "Engine/Core/Math/Math.h"

namespace
{
    const float DefaultOffsets[6] = { -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f };

    // BendVelocityTowardsDesired (fixed, NOT a live knob): magnitude-preserving lerp
    // of the current planar velocity toward the desired-velocity direction BEFORE
    // spring prediction, so input flips register instantly instead of waiting for
    // the spring. Stateless function of the snapshot inputs, so replay-safe (the
    // recorded trajectory points already contain its effect). Blend factor 0.4
    // (measured: 0.6 latencies 0.95/1.32s, no better than 0.4's 0.92/never and
    // baseline 0.7-1.4): steady state current~=desired so bend is inert there and
    // only transients rotate.
    constexpr float BendVelocityFactor = 0.4f;
    constexpr float BendMinSpeedSq = 100.0f; // 10 u/s: moving only, rest noise untouched

    Float3 BendVelocityTowardsDesired(const Float3& currentVelocity, const Float3& desiredVelocity)
    {
        const float curSq = currentVelocity.X * currentVelocity.X + currentVelocity.Z * currentVelocity.Z;
        const float desSq = desiredVelocity.X * desiredVelocity.X + desiredVelocity.Z * desiredVelocity.Z;
        if (curSq <= BendMinSpeedSq || desSq <= BendMinSpeedSq)
            return currentVelocity;
        const float curSpeed = Math::Sqrt(curSq);
        const float desSpeed = Math::Sqrt(desSq);
        if (curSpeed <= 0.00001f || desSpeed <= 0.00001f)
            return currentVelocity;
        const float curDirX = currentVelocity.X / curSpeed;
        const float curDirZ = currentVelocity.Z / curSpeed;
        const float desDirX = desiredVelocity.X / desSpeed;
        const float desDirZ = desiredVelocity.Z / desSpeed;
        const float bx = curDirX + (desDirX - curDirX) * BendVelocityFactor;
        const float bz = curDirZ + (desDirZ - curDirZ) * BendVelocityFactor;
        const float blenSq = bx * bx + bz * bz;
        if (blenSq <= 1e-10f)
            return currentVelocity;
        const float invLen = 1.0f / Math::Sqrt(blenSq);
        return Float3(bx * invLen * curSpeed, 0.0f, bz * invLen * curSpeed);
    }
    // Steady-yaw curve: rotate a planar velocity by a yaw delta so a constant
    // turn predicts an arc, not a straight line. Left-handed Y-up Flax
    // convention: yaw θ maps Forward(0,0,1)->(sinθ,0,cosθ), hence
    // x' = x*cosΔ + z*sinΔ, z' = -x*sinΔ + z*cosΔ. Magnitude-preserving.
    Float3 RotateY(const Float3& v, float angleRad)
    {
        const float c = Math::Cos(angleRad);
        const float s = Math::Sin(angleRad);
        return Float3(v.X * c + v.Z * s, v.Y, -v.X * s + v.Z * c);
    }
}

void MotionMatchingTrajectory::AddHistory(const Float3& position, const Float3& facing, float time)
{
    if (_historyCount < HistoryCapacity)
    {
        const int32 writeIndex = (_historyHead + _historyCount) % HistoryCapacity;
        _historyBuffer[writeIndex] = { position, facing, time };
        _historyCount++;
    }
    else
    {
        _historyBuffer[_historyHead] = { position, facing, time };
        _historyHead = (_historyHead + 1) % HistoryCapacity;
    }
}

const MotionMatchingTrajectory::HistorySample& MotionMatchingTrajectory::GetHistory(int32 index) const
{
    return _historyBuffer[(_historyHead + index) % HistoryCapacity];
}

void MotionMatchingTrajectory::SampleHistory(float targetTime, Float3& outPosition, Float3& outFacing) const
{
    if (_historyCount == 0)
    {
        outPosition = Float3::Zero;
        outFacing = Float3::Forward;
        return;
    }

    const HistorySample& first = GetHistory(0);
    if (targetTime <= first.Time)
    {
        outPosition = first.Position;
        outFacing = first.Facing;
        return;
    }

    const HistorySample& last = GetHistory(_historyCount - 1);
    if (targetTime >= last.Time)
    {
        outPosition = last.Position;
        outFacing = last.Facing;
        return;
    }

    for (int32 i = 0; i < _historyCount - 1; i++)
    {
        const HistorySample& a = GetHistory(i);
        const HistorySample& b = GetHistory(i + 1);

        if (targetTime >= a.Time && targetTime <= b.Time)
        {
            float duration = b.Time - a.Time;
            if (duration <= 0.00001f)
            {
                outPosition = a.Position;
                outFacing = a.Facing;
                return;
            }

            float alpha = (targetTime - a.Time) / duration;
            outPosition = Float3::Lerp(a.Position, b.Position, alpha);
            // Lerp + normalize is fine except for ~180deg flips where it collapses to zero.
            const Float3 lerped = Float3::Lerp(a.Facing, b.Facing, alpha);
            if (lerped.LengthSquared() > 1e-6f)
            {
                const float invLen = 1.0f / Math::Sqrt(lerped.LengthSquared());
                outFacing = Float3(lerped.X * invLen, lerped.Y * invLen, lerped.Z * invLen);
            }
            else
            {
                outFacing = a.Facing;
            }
            return;
        }
    }

    outPosition = last.Position;
    outFacing = last.Facing;
}

const Array<TrajectoryPoint>& MotionMatchingTrajectory::CalculateTrajectory(
    const Transform& currentTransform,
    const Float3& currentVelocity,
    const Float3& currentAcceleration,
    const Float3& currentAngularVelocity,
    const Float3& desiredVelocity,
    const Quaternion& desiredRotation,
    const Array<float>& sampleOffsets,
    float positionHalflife,
    float facingHalflife,
    float deltaTime
)
{
    // 1. Determine effective sample offsets
    const float* offsets = sampleOffsets.IsEmpty() ? DefaultOffsets : sampleOffsets.Get();
    const int32 pointCount = sampleOffsets.IsEmpty() ? 6 : sampleOffsets.Count();

    if (_points.Count() != pointCount)
        _points.Resize(pointCount);

    const Quaternion orientation = SpringDamper::NormalizeSafe(currentTransform.Orientation);

    // Current facing on the XZ plane (rotation-only, no translation involved).
    Float3 currentFacing = SpringDamper::RotateVector(orientation, Float3::Forward);
    currentFacing.Y = 0.0f;
    currentFacing = SpringDamper::NormalizeSafe(currentFacing, Float3::Forward);

    // 2. Record current state into circular history buffer
    _currentTime += deltaTime;
    const Float3 currentPos = (Float3)currentTransform.Translation;
    AddHistory(currentPos, currentFacing, _currentTime);

    // Inverse orientation once for world->local facing conversion (scale-free, normalized later).
    const Quaternion invOrientation = SpringDamper::Conjugate(orientation);

    // Bend the prediction's initial velocity toward intent.
    // History (past points) and @0 stay exact; only future spring rollout uses it.
    const Float3 bentVelocity = BendVelocityTowardsDesired(currentVelocity, desiredVelocity);

    // 3. Compute trajectory points in World Space
    for (int32 i = 0; i < pointCount; i++)
    {
        const float offset = offsets[i];
        Float3 worldPos;
        Float3 worldFacing;

        if (offset < -0.0001f)
        {
            // Past: exact position AND facing from history.
            SampleHistory(_currentTime + offset, worldPos, worldFacing);
        }
        else if (Math::Abs(offset) <= 0.0001f)
        {
            worldPos = currentPos;
            worldFacing = currentFacing;
        }
        else
        {
            // Steady-yaw curve: rotate the spring target by yawRate * t so a
            // constant turn predicts an arc. Angle scaled 0.5x (measured:
            // full angle over-bends — clips cut corners early, readable as
            // unfinished turns). Straight runs and idle skip
            // (BendMinSpeedSq rest-noise floor, 1e-4 rad/s yaw floor).
            const float desSq = desiredVelocity.X * desiredVelocity.X + desiredVelocity.Z * desiredVelocity.Z;
            Float3 curvedDesired = desiredVelocity;
            const float yawRate = currentAngularVelocity.Y;
            if (desSq > BendMinSpeedSq && (yawRate >= 1e-4f || yawRate <= -1e-4f))
                curvedDesired = RotateY(desiredVelocity, yawRate * offset * 0.5f);
            // Future: analytic 2nd-order spring in O(1). Planar only, Y untouched.
            Float3 predictedVel;
            SpringDamper::SpringPlanarCharacterPredict(
                currentPos,
                bentVelocity,
                currentAcceleration,
                curvedDesired,
                positionHalflife,
                offset,
                worldPos,
                predictedVel);

            const Quaternion predictedRot = SpringDamper::SpringQuaternionPredict(
                orientation,
                currentAngularVelocity,
                desiredRotation,
                facingHalflife,
                offset);

            worldFacing = SpringDamper::RotateVector(predictedRot, Float3::Forward);
            worldFacing.Y = 0.0f;
            worldFacing = SpringDamper::NormalizeSafe(worldFacing, currentFacing);
        }

        // 4. Convert to character-local space.
        Vector3 localPos;
        currentTransform.WorldToLocal(Vector3(worldPos), localPos);

        const Float3 localFacingRaw = SpringDamper::RotateVector(invOrientation, worldFacing);
        Float3 localFacing = localFacingRaw;
        localFacing.Y = 0.0f;
        localFacing = SpringDamper::NormalizeSafe(localFacing, Float3::Forward);

        _points[i].Position = (Float3)localPos;
        _points[i].Facing = localFacing;
    }

    return _points;
}

void MotionMatchingTrajectory::UpdateTrajectory(
    const Transform& currentTransform,
    const Float3& currentVelocity,
    const Float3& currentAcceleration,
    const Float3& currentAngularVelocity,
    const Float3& desiredVelocity,
    const Quaternion& desiredRotation,
    const Array<float>& sampleOffsets,
    float positionHalflife,
    float facingHalflife,
    float deltaTime)
{
    CalculateTrajectory(currentTransform, currentVelocity, currentAcceleration,
        currentAngularVelocity, desiredVelocity, desiredRotation,
        sampleOffsets, positionHalflife, facingHalflife, deltaTime);
}

int32 MotionMatchingTrajectory::GetPredictedCount() const
{
    return _points.Count();
}

bool MotionMatchingTrajectory::GetPredictedPoint(int32 index, TrajectoryPoint& outPoint) const
{
    if (index < 0 || index >= _points.Count())
    {
        outPoint.Position = Float3::Zero;
        outPoint.Facing = Float3::Forward;
        return false;
    }
    outPoint = _points[index];
    return true;
}

void MotionMatchingTrajectory::Reset(const Transform& transform)
{
    _currentTime = 0.0f;
    _historyHead = 0;
    _historyCount = 0;

    const Quaternion orientation = SpringDamper::NormalizeSafe(transform.Orientation);
    Float3 initialFacing = SpringDamper::RotateVector(orientation, Float3::Forward);
    initialFacing.Y = 0.0f;
    initialFacing = SpringDamper::NormalizeSafe(initialFacing, Float3::Forward);

    AddHistory((Float3)transform.Translation, initialFacing, 0.0f);

    for (int32 i = 0; i < _points.Count(); i++)
    {
        _points[i].Position = Float3::Zero;
        _points[i].Facing = Float3::Forward;
    }
}
