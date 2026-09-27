// SpringDamper.h: closed-form critically-damped spring predictors (SpringDamper: HalflifeToDamping, SpringPlanarCharacterPredict, SpringQuaternionPredict, RotateVector).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the planar position/velocity spring solves the analytic 2nd-order response on XZ and never touches Y; the quaternion spring integrates scaled angle-axis error with a persistent angular velocity; degenerate half-lives fall back to linear extrapolation instead of dividing by zero.
#pragma once

#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Math/Quaternion.h"

// Closed-form critically-damped spring helpers in the style of OrangeDuck/Holden.
// - Planar (XZ) position/velocity spring: analytic 2nd-order solution. Y is never touched.
// - Quaternion spring via scaled angle-axis with a persistent angular velocity (rad/s).
namespace SpringDamper
{
    // Half-life (seconds for the error to halve) -> damping coefficient.
    FORCE_INLINE float HalflifeToDamping(float halflife, float eps = 1e-5f)
    {
        return (4.0f * 0.69314718056f) / (Math::Max(halflife, eps));
    }

    FORCE_INLINE Float3 NormalizeSafe(const Float3& v, const Float3& fallback)
    {
        const float lenSq = v.X * v.X + v.Y * v.Y + v.Z * v.Z;
        if (lenSq > 1e-10f)
        {
            const float invLen = 1.0f / Math::Sqrt(lenSq);
            return Float3(v.X * invLen, v.Y * invLen, v.Z * invLen);
        }
        return fallback;
    }

    FORCE_INLINE Quaternion NormalizeSafe(const Quaternion& q)
    {
        const float lenSq = q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W;
        if (lenSq > 1e-10f)
        {
            const float invLen = 1.0f / Math::Sqrt(lenSq);
            return Quaternion(q.X * invLen, q.Y * invLen, q.Z * invLen, q.W * invLen);
        }
        return Quaternion::Identity;
    }

    // Conjugate == inverse for unit quaternions.
    FORCE_INLINE Quaternion Conjugate(const Quaternion& q)
    {
        return Quaternion(-q.X, -q.Y, -q.Z, q.W);
    }

    // Rotate a vector by a (unit) quaternion: v + 2*w*cross(xyz,v) + 2*cross(xyz,cross(xyz,v)).
    FORCE_INLINE Float3 RotateVector(const Quaternion& q, const Float3& v)
    {
        const float cx = q.Y * v.Z - q.Z * v.Y;
        const float cy = q.Z * v.X - q.X * v.Z;
        const float cz = q.X * v.Y - q.Y * v.X;
        const float ccx = q.Y * cz - q.Z * cy;
        const float ccy = q.Z * cx - q.X * cz;
        const float ccz = q.X * cy - q.Y * cx;
        return Float3(
            v.X + 2.0f * (q.W * cx + ccx),
            v.Y + 2.0f * (q.W * cy + ccy),
            v.Z + 2.0f * (q.W * cz + ccz));
    }

    /// <summary>
    /// Analytic 2nd-order spring prediction on the XZ plane (Holden).
    /// currentAccel is the persistent spring state (kept on the locomotion controller).
    /// Y (gravity) is never touched: outPos.Y = currentPos.Y, outVel.Y = 0.
    /// </summary>
    FORCE_INLINE void SpringPlanarCharacterPredict(
        const Float3& currentPos,
        const Float3& currentVel,
        const Float3& currentAccel,
        const Float3& desiredVel,
        float halflife,
        float t,
        Float3& outPos,
        Float3& outVel)
    {
        const float y = HalflifeToDamping(halflife) * 0.5f;
        if (y < 1e-4f || t <= 0.0f)
        {
            // Degenerate halflife: fall back to linear extrapolation.
            outPos = Float3(currentPos.X + currentVel.X * t, currentPos.Y, currentPos.Z + currentVel.Z * t);
            outVel = Float3(currentVel.X, 0.0f, currentVel.Z);
            return;
        }

        const float y2 = y * y;
        const float eydt = Math::Exp(-y * t);

        const float j0x = currentVel.X - desiredVel.X;
        const float j1x = currentAccel.X + j0x * y;
        outPos.X = eydt * ((-j1x / y2) + (-j0x - j1x * t) / y) + (j1x / y2) + (j0x / y) + desiredVel.X * t + currentPos.X;
        outVel.X = eydt * (j0x + j1x * t) + desiredVel.X;

        const float j0z = currentVel.Z - desiredVel.Z;
        const float j1z = currentAccel.Z + j0z * y;
        outPos.Z = eydt * ((-j1z / y2) + (-j0z - j1z * t) / y) + (j1z / y2) + (j0z / y) + desiredVel.Z * t + currentPos.Z;
        outVel.Z = eydt * (j0z + j1z * t) + desiredVel.Z;

        outPos.Y = currentPos.Y;
        outVel.Y = 0.0f;
    }

    // Error quaternion -> scaled angle-axis (shortest path enforced via W flip).
    FORCE_INLINE Float3 QuaternionToScaledAngleAxis(const Quaternion& q)
    {
        Quaternion nq = q;
        if (nq.W < 0.0f)
        {
            nq.X = -nq.X;
            nq.Y = -nq.Y;
            nq.Z = -nq.Z;
            nq.W = -nq.W;
        }

        const float sinHalfAngle = Math::Sqrt(nq.X * nq.X + nq.Y * nq.Y + nq.Z * nq.Z);
        if (sinHalfAngle < 1e-5f)
            return Float3::Zero;

        const float halfAngle = Math::Atan2(sinHalfAngle, nq.W);
        const float factor = (2.0f * halfAngle) / sinHalfAngle;
        return Float3(nq.X * factor, nq.Y * factor, nq.Z * factor);
    }

    FORCE_INLINE Quaternion QuaternionFromScaledAngleAxis(const Float3& v)
    {
        const float angleSq = v.X * v.X + v.Y * v.Y + v.Z * v.Z;
        if (angleSq < 1e-10f)
            return Quaternion::Identity;

        const float angle = Math::Sqrt(angleSq);
        const float halfAngle = angle * 0.5f;
        const float factor = Math::Sin(halfAngle) / angle;
        return Quaternion(v.X * factor, v.Y * factor, v.Z * factor, Math::Cos(halfAngle));
    }

    /// <summary>
    /// Analytic quaternion spring prediction in O(1).
    /// currentAngVel must be the persistent angular velocity (rad/s), NOT Zero.
    /// </summary>
    FORCE_INLINE Quaternion SpringQuaternionPredict(
        const Quaternion& currentRot,
        const Float3& currentAngVel,
        const Quaternion& desiredRot,
        float halflife,
        float t)
    {
        const Quaternion c = NormalizeSafe(currentRot);
        const Quaternion g = NormalizeSafe(desiredRot);
        const float y = HalflifeToDamping(halflife) * 0.5f;

        // Error from goal to current, expressed so that FromAxis(offset) * goal == current at t=0.
        const Quaternion diff = NormalizeSafe(c * Conjugate(g));
        const Float3 j0 = QuaternionToScaledAngleAxis(diff);
        const Float3 j1 = currentAngVel + j0 * y;

        const float eydt = Math::Exp(-y * t);
        const Float3 offsetAngleAxis = (j0 + j1 * t) * eydt;
        return NormalizeSafe(QuaternionFromScaledAngleAxis(offsetAngleAxis) * g);
    }
}
