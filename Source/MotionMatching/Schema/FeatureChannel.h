// FeatureChannel.h: pose-search feature channel metadata (FeatureChannel, FeatureChannelKind, FeatureContactRole).
// Ownership: plugin runtime, no game/host references.
// Key invariants: one descriptor per channel is the single source of truth for block, offset, dimension, scalar/vector contract, normalization group, symmetry group, semantic and contact role (see PoseSearchSchema::DescribePoseChannels/DescribeTrajectoryChannels); contacts resolve only through PoseSearchSchema::GetContactFeatureOffset, never via size arithmetic.
#pragma once

#include "Engine/Core/Types/String.h"

/// <summary>
/// Scalar-vs-vector contract of one pose-search feature channel. Scalar channels normalize per-dimension (mean + population
/// variance); vector channels share one deviation (mean norm of the centered
/// vector, UE floor 0.1) across their dimensions.
/// </summary>
enum class FeatureChannelKind : int32
{
    Scalar = 0,
    Vector = 1,
};

/// <summary>
/// Foot-contact role of a channel. Only the two planted-flag channels carry
/// a side; everything else is None. Search decodes contacts through
/// PoseSearchSchema::GetContactFeatureOffset, never via size-2/-1.
/// </summary>
enum class FeatureContactRole : int32
{
    None = -1,
    Left = 0,
    Right = 1,
};

/// <summary>
/// Data-only description of one pose-search feature channel.
///
/// The native baker and database keep their existing flat-array API. This
/// descriptor is the metadata single source of truth for layout (block,
/// offset, dimension), scalar/vector contract, normalization group,
/// symmetry group, semantic and contact role. See
/// PoseSearchSchema::DescribePoseChannels/DescribeTrajectoryChannels.
/// </summary>
struct FeatureChannel
{
    String Name;
    /// <summary>Feature block: 0 = pose, 1 = trajectory.</summary>
    int32 Block = 0;
    /// <summary>Block-local offset of the first float of this channel.</summary>
    int32 Offset = 0;
    int32 Dimension = 0;
    float Weight = 1.0f;
    /// <summary>
    /// Normalization group: dimensions sharing a group share one deviation
    /// (vector) or one symmetry audit scope (scalar). Pose feet/hands share
    /// groups across L/R (same semantic only); trajectory groups are
    /// per-dimension identity (scalar contract).
    /// See PoseSearchSchema::GetPoseNormalizationGroup for the baked layout.
    /// </summary>
    int32 NormalizationGroup = 0;
    /// <summary>Scalar/vector contract (scalar: per-dimension; vector: shared deviation).</summary>
    FeatureChannelKind Kind = FeatureChannelKind::Scalar;
    /// <summary>
    /// Symmetry group. Only channels of the same semantic ever share a
    /// symmetry group (mirrored limbs must not skew turns).
    /// </summary>
    int32 SymmetryGroup = 0;
    /// <summary>Semantic tag, e.g. "position.foot", "velocity.pelvis", "contact".</summary>
    String Semantic;
    /// <summary>Contact role (None except the two planted-flag channels).</summary>
    FeatureContactRole ContactRole = FeatureContactRole::None;
};
