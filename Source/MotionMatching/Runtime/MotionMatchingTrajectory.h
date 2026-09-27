// MotionMatchingTrajectory.h: spring-predicted character trajectory with history (MotionMatchingTrajectory::CalculateTrajectory/UpdateTrajectory, HistorySample).
// Ownership: plugin runtime, no game/host references.
// Key invariants: past points come from the recorded history buffer, future points from the analytic 2nd-order spring rollout; UpdateTrajectory shares one core with CalculateTrajectory and publishes into caller-reused buffers via GetPredictedCount/GetPredictedPoint without managed allocation.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Transform.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Quaternion.h"
#include "Engine/Scripting/ScriptingObject.h"
#include "../Schema/TrajectoryPoint.h"

API_CLASS(Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingTrajectory : public ScriptingObject
{
    DECLARE_SCRIPTING_TYPE_WITH_CONSTRUCTOR_IMPL(MotionMatchingTrajectory, ScriptingObject);

private:
    struct HistorySample
    {
        Float3 Position;
        Float3 Facing;
        float Time;
    };

    static constexpr int32 HistoryCapacity = 128;
    HistorySample _historyBuffer[HistoryCapacity];
    int32 _historyHead = 0;
    int32 _historyCount = 0;

    Array<TrajectoryPoint> _points;
    float _currentTime = 0.0f;

    void AddHistory(const Float3& position, const Float3& facing, float time);
    const HistorySample& GetHistory(int32 index) const;
    void SampleHistory(float targetTime, Float3& outPosition, Float3& outFacing) const;

public:
    /// <summary>
    /// Calculates trajectory points using past history and 2nd-order analytic spring prediction (OrangeDuck).
    /// </summary>
    /// <param name="currentTransform">Current character transform in world space.</param>
    /// <param name="currentVelocity">Current actual character velocity (Y is ignored, planar XZ only).</param>
    /// <param name="currentAcceleration">Persistent planar spring acceleration from the locomotion controller.</param>
    /// <param name="currentAngularVelocity">Persistent angular velocity in rad/s (from look controller), NOT Zero.</param>
    /// <param name="desiredVelocity">Target/desired velocity from player input (planar XZ).</param>
    /// <param name="desiredRotation">Target facing rotation (camera/look yaw), NOT derived from desiredVelocity.</param>
    /// <param name="sampleOffsets">Relative time offsets (e.g. from schema or database).</param>
    /// <param name="positionHalflife">Spring half-life for position prediction.</param>
    /// <param name="facingHalflife">Spring half-life for facing prediction.</param>
    /// <param name="deltaTime">Delta time of current frame (drives the history clock).</param>
    /// <returns>Array of trajectory points in character's local space.</returns>
    API_FUNCTION() const Array<TrajectoryPoint>& CalculateTrajectory(
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
    );
    /// <summary>
    /// Zero-alloc trajectory update: same prediction as CalculateTrajectory
    /// but returns void (no managed array allocation). The predicted points
    /// are read back with GetPredictedCount/GetPredictedPoint below into
    /// caller-reused buffers. Kept in lockstep with CalculateTrajectory by
    /// sharing the same core (this method forwards to it and discards the
    /// reference; the native _points member holds the result).
    /// </summary>
    API_FUNCTION() void UpdateTrajectory(
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
    );
    /// <summary>
    /// Zero-alloc trajectory read: predicted point count (6 for the
    /// current schema). Bound: schema point count (6, hard cap 16).
    /// </summary>
    API_FUNCTION() int32 GetPredictedCount() const;
    /// <summary>
    /// Zero-alloc trajectory read: copies one predicted point into the
    /// caller struct (no managed array allocation). Returns false when the
    /// index is out of range (outPoint set to origin/forward).
    /// </summary>
    API_FUNCTION() bool GetPredictedPoint(int32 index, API_PARAM(Out) TrajectoryPoint& outPoint) const;

    /// <summary>
    /// Resets history buffer to a given start transform.
    /// </summary>
    API_FUNCTION() void Reset(const Transform& transform);
};
