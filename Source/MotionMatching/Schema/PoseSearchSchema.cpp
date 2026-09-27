// PoseSearchSchema.cpp: schema validation, channel metadata, layout hash, contacts, serialization (PoseSearchSchema::TryValidate/DescribePoseChannels/DescribeTrajectoryChannels/GetFeatureLayoutHash/ValidateLayout/TryGetNormalizedUpAxis/ComputeContactHeight/IsPlantedContact).
// Ownership: plugin runtime, no game/host references.
// Key invariants: DescribePoseChannels/DescribeTrajectoryChannels are the single layout truth (mirrored feet/hands share groups, trajectory dims use identity groups); GetFeatureLayoutHash covers every channel field and 0 means unknown; contacts share one slow-and-low gate; Serialize/Deserialize round-trip the sampled schema values.
#include "PoseSearchSchema.h"

#include "Engine/Core/Math/Math.h"
#include "Engine/Serialization/ReadStream.h"
#include "Engine/Serialization/WriteStream.h"

#include <cmath>

namespace
{
    constexpr float MinimumAxisLengthSquared = 0.000001f;
    constexpr float TimeZeroTolerance = 0.0001f;

    bool IsFinite(const Float3 &value)
    {
        return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
    }
}

bool PoseSearchSchema::TryValidate(String &error) const
{
    error = String::Empty;

    if (SampleRate <= 0)
    {
        error = TEXT("SampleRate must be greater than zero.");
        return false;
    }

    if (!IsFinite(RootForwardAxis) || !IsFinite(RootUpAxis) ||
        RootForwardAxis.LengthSquared() <= MinimumAxisLengthSquared ||
        RootUpAxis.LengthSquared() <= MinimumAxisLengthSquared)
    {
        error = TEXT("RootForwardAxis and RootUpAxis must be finite and non-zero.");
        return false;
    }
    Float3 right;
    if (!TryGetRightAxis(right))
    {
        error = TEXT("RootForwardAxis and RootUpAxis cannot be parallel.");
        return false;
    }

    if (!std::isfinite(PoseWeight) || !std::isfinite(TrajectoryWeight) ||
        PoseWeight < 0.0f || TrajectoryWeight < 0.0f)
    {
        error = TEXT("PoseWeight and TrajectoryWeight must be finite and non-negative.");
        return false;
    }

    if (Skeleton.Root.IsEmpty() || Skeleton.Pelvis.IsEmpty() ||
        Skeleton.LeftFoot.IsEmpty() || Skeleton.RightFoot.IsEmpty())
    {
        error = TEXT("Skeleton requires Root, Pelvis, LeftFoot and RightFoot names. "
            "Configure the SkeletonProfile for your rig "
            "(scene → MotionMatchingController → Schema → Skeleton).");
        return false;
    }

    if (TrajectorySampleTimes.Count() != TrajectoryPointCount)
    {
        error = String::Format(
            TEXT("TrajectorySampleTimes must contain exactly {} values."),
            static_cast<int32>(TrajectoryPointCount));
        return false;
    }
    bool includesZero = false;
    for (int32 i = 0; i < TrajectoryPointCount; i++)
    {
        const float time = TrajectorySampleTimes[i];
        if (!std::isfinite(time) || (i > 0 && time <= TrajectorySampleTimes[i - 1]))
        {
            error = TEXT("TrajectorySampleTimes must be finite and strictly ascending.");
            return false;
        }
        if (std::fabs(time) <= TimeZeroTolerance)
            includesZero = true;
    }
    if (TrajectorySampleTimes[0] >= 0.0f ||
        TrajectorySampleTimes[TrajectoryPointCount - 1] <= 0.0f ||
        !includesZero)
    {
        error = TEXT("TrajectorySampleTimes must span past and future around time zero.");
        return false;
    }

    if (!ValidateLayout(error))
        return false;

    return true;
}

void PoseSearchSchema::DescribePoseChannels(Array<FeatureChannel>& outChannels)
{
    outChannels.Clear();
    // Fixed bone order matches GetPoseBoneIndices: [0] pelvis, [1] L foot,
    // [2] R foot, [3] spine, [4] head, [5] L hand, [6] R hand.
    const Char* boneNames[PoseBoneCount] = {
        TEXT("pelvis"), TEXT("left_foot"), TEXT("right_foot"),
        TEXT("spine"), TEXT("head"), TEXT("left_hand"), TEXT("right_hand")
    };
    // L/R limbs share one normalization group AND one semantic/symmetry
    // scope, never mixed with another bone or with velocities.
    const Char* posSemantics[PoseBoneCount] = {
        TEXT("position.pelvis"), TEXT("position.foot"), TEXT("position.foot"),
        TEXT("position.spine"), TEXT("position.head"), TEXT("position.hand"), TEXT("position.hand")
    };
    const Char* velSemantics[PoseBoneCount] = {
        TEXT("velocity.pelvis"), TEXT("velocity.foot"), TEXT("velocity.foot"),
        TEXT("velocity.spine"), TEXT("velocity.head"), TEXT("velocity.hand"), TEXT("velocity.hand")
    };
    const int32 posGroups[PoseBoneCount] = { 0, 1, 1, 2, 3, 4, 4 };
    const int32 velGroups[PoseBoneCount] = { 5, 6, 6, 7, 8, 9, 9 };
    const int32 positionBlockSize = GetPosePositionBlockSize();

    for (int32 bone = 0; bone < PoseBoneCount; bone++)
    {
        FeatureChannel pos;
        pos.Name = String(TEXT("pose.positions.")) + String(boneNames[bone]);
        pos.Block = 0;
        pos.Offset = bone * 3;
        pos.Dimension = 3;
        pos.Weight = 1.0f;
        pos.NormalizationGroup = posGroups[bone];
        pos.Kind = FeatureChannelKind::Vector;
        pos.SymmetryGroup = posGroups[bone];
        pos.Semantic = String(posSemantics[bone]);
        pos.ContactRole = FeatureContactRole::None;
        outChannels.Add(pos);

        FeatureChannel vel;
        vel.Name = String(TEXT("pose.velocities.")) + String(boneNames[bone]);
        vel.Block = 0;
        vel.Offset = positionBlockSize + bone * 3;
        vel.Dimension = 3;
        vel.Weight = 1.0f;
        vel.NormalizationGroup = velGroups[bone];
        vel.Kind = FeatureChannelKind::Vector;
        vel.SymmetryGroup = velGroups[bone];
        vel.Semantic = String(velSemantics[bone]);
        vel.ContactRole = FeatureContactRole::None;
        outChannels.Add(vel);
    }

    FeatureChannel contactL;
    contactL.Name = String(TEXT("pose.contacts.L"));
    contactL.Block = 0;
    contactL.Offset = GetContactFeatureOffset(0);
    contactL.Dimension = 1;
    contactL.Weight = 1.0f;
    contactL.NormalizationGroup = 10;
    contactL.Kind = FeatureChannelKind::Scalar;
    contactL.SymmetryGroup = 10;
    contactL.Semantic = String(TEXT("contact"));
    contactL.ContactRole = FeatureContactRole::Left;
    outChannels.Add(contactL);

    FeatureChannel contactR;
    contactR.Name = String(TEXT("pose.contacts.R"));
    contactR.Block = 0;
    contactR.Offset = GetContactFeatureOffset(1);
    contactR.Dimension = 1;
    contactR.Weight = 1.0f;
    contactR.NormalizationGroup = 10;
    contactR.Kind = FeatureChannelKind::Scalar;
    contactR.SymmetryGroup = 10;
    contactR.Semantic = String(TEXT("contact"));
    contactR.ContactRole = FeatureContactRole::Right;
    outChannels.Add(contactR);
}

void PoseSearchSchema::DescribeTrajectoryChannels(Array<FeatureChannel>& outChannels)
{
    outChannels.Clear();
    // Trajectory dims stay scalar with identity groups (each dim normalizes
    // independently). A future vectorization (pos_xy/facing_xy sharing one
    // deviation) would change group ids, the layout hash and the revision, and
    // must rebake + sync the manifest mirror.
    const Char* fieldNames[TrajectoryFloatsPerPoint] = {
        TEXT("position_x"), TEXT("position_y"),
        TEXT("facing_x"), TEXT("facing_y"), TEXT("yaw_rate")
    };
    const Char* fieldSemantics[TrajectoryFloatsPerPoint] = {
        TEXT("trajectory.position"), TEXT("trajectory.position"),
        TEXT("trajectory.facing"), TEXT("trajectory.facing"), TEXT("trajectory.yaw_rate")
    };
    for (int32 point = 0; point < TrajectoryPointCount; point++)
    {
        for (int32 field = 0; field < TrajectoryFloatsPerPoint; field++)
        {
            const int32 local = point * TrajectoryFloatsPerPoint + field;
            FeatureChannel ch;
            ch.Name = String(TEXT("trajectory.")) + String::Format(TEXT("{}"), point) +
                String(TEXT(".")) + String(fieldNames[field]);
            ch.Block = 1;
            ch.Offset = local;
            ch.Dimension = 1;
            ch.Weight = 1.0f;
            ch.NormalizationGroup = GetTrajectoryNormalizationGroup(local);
            ch.Kind = FeatureChannelKind::Scalar;
            ch.SymmetryGroup = ch.NormalizationGroup;
            ch.Semantic = String(fieldSemantics[field]);
            ch.ContactRole = FeatureContactRole::None;
            outChannels.Add(ch);
        }
    }
}

uint32 PoseSearchSchema::GetFeatureLayoutHash()
{
    // FNV-1a over the full channel metadata. 0 is reserved for unknown files without the tail int.
    uint32 hash = 2166136261u;
    auto mixByte = [&](uint32 b)
    {
        hash ^= (b & 0xFFu);
        hash *= 16777619u;
    };
    auto mixInt = [&](int32 v)
    {
        const uint32 u = static_cast<uint32>(v);
        mixByte(u);
        mixByte(u >> 8);
        mixByte(u >> 16);
        mixByte(u >> 24);
    };
    auto mixString = [&](const String& s)
    {
        mixInt(s.Length());
        for (int32 i = 0; i < s.Length(); i++)
        {
            const uint32 c = static_cast<uint32>(s[i]);
            mixByte(c);
            mixByte(c >> 8);
            mixByte(c >> 16);
            mixByte(c >> 24);
        }
    };
    auto mixChannels = [&](Array<FeatureChannel>& channels)
    {
        for (int32 i = 0; i < channels.Count(); i++)
        {
            const FeatureChannel& ch = channels[i];
            mixString(ch.Name);
            mixInt(ch.Block);
            mixInt(ch.Offset);
            mixInt(ch.Dimension);
            mixInt(ch.NormalizationGroup);
            mixInt(static_cast<int32>(ch.Kind));
            mixInt(ch.SymmetryGroup);
            mixString(ch.Semantic);
            mixInt(static_cast<int32>(ch.ContactRole));
        }
    };
    Array<FeatureChannel> pose;
    Array<FeatureChannel> trajectory;
    DescribePoseChannels(pose);
    DescribeTrajectoryChannels(trajectory);
    mixInt(GetPoseFeatureSize());
    mixInt(GetTrajectoryFeatureSize());
    mixChannels(pose);
    mixChannels(trajectory);
    if (hash == 0)
        hash = 1;
    return hash;
}

bool PoseSearchSchema::ValidateLayout(String& error)
{
    error = String::Empty;
    const int32 poseSize = GetPoseFeatureSize();
    const int32 trajectorySize = GetTrajectoryFeatureSize();

    Array<FeatureChannel> pose;
    Array<FeatureChannel> trajectory;
    DescribePoseChannels(pose);
    DescribeTrajectoryChannels(trajectory);

    bool poseUsed[PoseBoneCount * PoseFloatsPerBone + PoseExtraCount];
    for (int32 i = 0; i < poseSize; i++)
        poseUsed[i] = false;
    for (int32 i = 0; i < pose.Count(); i++)
    {
        const FeatureChannel& ch = pose[i];
        if (ch.Block != 0 || ch.Dimension <= 0 ||
            ch.Offset < 0 || ch.Offset + ch.Dimension > poseSize)
        {
            error = String::Format(TEXT("Feature channel '{}' is outside the pose block."), ch.Name);
            return false;
        }
        // Dim-level mapping must agree with the baked group function.
        for (int32 k = 0; k < ch.Dimension; k++)
        {
            if (GetPoseNormalizationGroup(ch.Offset + k) != ch.NormalizationGroup)
            {
                error = String::Format(TEXT("Feature channel '{}' disagrees with GetPoseNormalizationGroup."), ch.Name);
                return false;
            }
            if (poseUsed[ch.Offset + k])
            {
                error = String::Format(TEXT("Feature channel '{}' overlaps another pose channel."), ch.Name);
                return false;
            }
            poseUsed[ch.Offset + k] = true;
        }
    }
    for (int32 i = 0; i < poseSize; i++)
    {
        if (!poseUsed[i])
        {
            error = String::Format(TEXT("Pose dimension {} is not covered by any feature channel."), i);
            return false;
        }
    }

    bool trajectoryUsed[TrajectoryPointCount * TrajectoryFloatsPerPoint];
    for (int32 i = 0; i < trajectorySize; i++)
        trajectoryUsed[i] = false;
    for (int32 i = 0; i < trajectory.Count(); i++)
    {
        const FeatureChannel& ch = trajectory[i];
        if (ch.Block != 1 || ch.Dimension <= 0 ||
            ch.Offset < 0 || ch.Offset + ch.Dimension > trajectorySize)
        {
            error = String::Format(TEXT("Feature channel '{}' is outside the trajectory block."), ch.Name);
            return false;
        }
        for (int32 k = 0; k < ch.Dimension; k++)
        {
            if (GetTrajectoryNormalizationGroup(ch.Offset + k) != ch.NormalizationGroup)
            {
                error = String::Format(TEXT("Feature channel '{}' disagrees with GetTrajectoryNormalizationGroup."), ch.Name);
                return false;
            }
            if (trajectoryUsed[ch.Offset + k])
            {
                error = String::Format(TEXT("Feature channel '{}' overlaps another trajectory channel."), ch.Name);
                return false;
            }
            trajectoryUsed[ch.Offset + k] = true;
        }
    }
    for (int32 i = 0; i < trajectorySize; i++)
    {
        if (!trajectoryUsed[i])
        {
            error = String::Format(TEXT("Trajectory dimension {} is not covered by any feature channel."), i);
            return false;
        }
    }

    // Contact channels must sit exactly at the metadata offsets.
    int32 contactSeen = 0;
    for (int32 i = 0; i < pose.Count(); i++)
    {
        const FeatureChannel& ch = pose[i];
        if (ch.ContactRole == FeatureContactRole::None)
            continue;
        const int32 expected = GetContactFeatureOffset(static_cast<int32>(ch.ContactRole));
        if (ch.Dimension != 1 || ch.Offset != expected || ch.Kind != FeatureChannelKind::Scalar)
        {
            error = String::Format(TEXT("Contact channel '{}' is not at GetContactFeatureOffset."), ch.Name);
            return false;
        }
        contactSeen++;
    }
    if (contactSeen != PoseExtraCount ||
        GetContactFeatureOffset(0) != poseSize - PoseExtraCount ||
        GetContactFeatureOffset(1) != poseSize - PoseExtraCount + 1 ||
        GetContactFeatureOffset(-1) != -1 ||
        GetContactFeatureOffset(PoseExtraCount) != -1)
    {
        error = TEXT("Contact feature offsets are inconsistent.");
        return false;
    }

    return true;
}

bool PoseSearchSchema::TryGetNormalizedUpAxis(Float3 &up) const
{
    up = RootUpAxis;
    if (!IsFinite(up) || up.LengthSquared() <= MinimumAxisLengthSquared)
    {
        up = Float3::Zero;
        return false;
    }
    up.Normalize();
    return true;
}

float PoseSearchSchema::ComputeContactHeight(
    const Float3 &footPosition,
    const Float3 &rootPosition,
    const Float3 &normalizedUpAxis)
{
    const Float3 relativeFoot(
        footPosition.X - rootPosition.X,
        footPosition.Y - rootPosition.Y,
        footPosition.Z - rootPosition.Z);
    return Float3::Dot(relativeFoot, normalizedUpAxis);
}

bool PoseSearchSchema::IsPlantedContact(float speedSquared, float maxSpeed, float height, float maxHeight)
{
    const float maxSpeedSq = maxSpeed * maxSpeed;
    return speedSquared <= maxSpeedSq && height <= maxHeight;
}

bool PoseSearchSchema::TryGetRightAxis(Float3 &right) const
{
    Float3 forward = RootForwardAxis;
    Float3 up = RootUpAxis;
    if (forward.LengthSquared() <= MinimumAxisLengthSquared ||
        up.LengthSquared() <= MinimumAxisLengthSquared)
    {
        right = Float3::Zero;
        return false;
    }
    forward.Normalize();
    up.Normalize();
    right = Float3::Cross(forward, up);
    if (right.LengthSquared() <= MinimumAxisLengthSquared)
    {
        right = Float3::Zero;
        return false;
    }
    right.Normalize();
    return true;
}

bool PoseSearchSchema::operator==(const PoseSearchSchema &other) const
{
    if (TrajectorySampleTimes.Count() != other.TrajectorySampleTimes.Count())
        return false;

    for (int i = 0; i < TrajectorySampleTimes.Count(); i++)
        if (TrajectorySampleTimes[i] != other.TrajectorySampleTimes[i])
            return false;

    return SampleRate == other.SampleRate &&
           Float3::NearEqual(Float3::Normalize(RootForwardAxis), Float3::Normalize(other.RootForwardAxis), AxisTolerance) &&
           Float3::NearEqual(Float3::Normalize(RootUpAxis), Float3::Normalize(other.RootUpAxis), AxisTolerance) &&
           PoseWeight == other.PoseWeight &&
           TrajectoryWeight == other.TrajectoryWeight &&
           Skeleton.Root == other.Skeleton.Root &&
           Skeleton.Pelvis == other.Skeleton.Pelvis &&
           Skeleton.Spine == other.Skeleton.Spine &&
           Skeleton.Head == other.Skeleton.Head &&
           Skeleton.LeftFoot == other.Skeleton.LeftFoot &&
           Skeleton.RightFoot == other.Skeleton.RightFoot &&
           Skeleton.LeftHand == other.Skeleton.LeftHand &&
           Skeleton.RightHand == other.Skeleton.RightHand;
}

void PoseSearchSchema::Serialize(WriteStream &stream) const
{
    stream.WriteInt32(SampleRate);
    stream.Write(RootForwardAxis);
    stream.Write(RootUpAxis);
    stream.WriteFloat(PoseWeight);
    stream.WriteFloat(TrajectoryWeight);
    stream.Write(Skeleton.Root);
    stream.Write(Skeleton.Pelvis);
    stream.Write(Skeleton.Spine);
    stream.Write(Skeleton.Head);
    stream.Write(Skeleton.LeftFoot);
    stream.Write(Skeleton.RightFoot);
    stream.Write(Skeleton.LeftHand);
    stream.Write(Skeleton.RightHand);
    stream.Write(TrajectorySampleTimes);
}

void PoseSearchSchema::Deserialize(ReadStream &stream)
{
    stream.ReadInt32(&SampleRate);
    stream.Read(RootForwardAxis);
    stream.Read(RootUpAxis);
    stream.ReadFloat(&PoseWeight);
    stream.ReadFloat(&TrajectoryWeight);
    stream.Read(Skeleton.Root);
    stream.Read(Skeleton.Pelvis);
    stream.Read(Skeleton.Spine);
    stream.Read(Skeleton.Head);
    stream.Read(Skeleton.LeftFoot);
    stream.Read(Skeleton.RightFoot);
    stream.Read(Skeleton.LeftHand);
    stream.Read(Skeleton.RightHand);
    stream.Read(TrajectorySampleTimes);
}
