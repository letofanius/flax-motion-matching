// MotionMatchingDatabase.h: persistent motion-matching database asset (MotionMatchingDatabase, ValidationStatus).
// Ownership: plugin runtime, no game/host references.
// Key invariants: flat baked arrays (samples, pose/trajectory features, deviations, root motion) stay cache-friendly for FindBest; NormalizeTags/SplitTagSet/JoinTags/TagSeparator are the single tag-set contract; ValidateDataStatus enforces a three-way gate (Ok / RegistryOnlyStale with rebake / Corrupt load failure); stale provenance (NormalizationRevision, layout hash, schema/sampling/source/tool stamps) drops baked arrays but keeps the clip registry.
#pragma once

#include "Engine/Content/AssetReference.h"
#include "Engine/Content/Assets/Animation.h"
#include "Engine/Content/BinaryAsset.h"
#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Core/Types/String.h"
#include "../Schema/PoseSearchSchema.h"

/// <summary>
/// Native, persistent motion-matching database.
/// Sample data is stored in flat arrays for cache-friendly runtime searches.
/// The schema snapshot used for baking is stored with the data.
/// </summary>
API_CLASS(NoSpawn, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingDatabase : public BinaryAsset
{
    DECLARE_BINARY_ASSET_HEADER(MotionMatchingDatabase, 3);

public:
    /// <summary>
    /// Three-way validation outcome (provenance-gated load).
    /// Ok: data is usable. RegistryOnlyStale: the baked arrays cannot be
    /// trusted (provenance revision/hash mismatch or an older binary-format feature
    /// layout) but the clip registry is intact — drop the baked arrays (registry-only)
    /// and REBAKE. Corrupt: structural invariants (finite/count/index/duration)
    /// failed; the asset must not be loaded nor cleared-and-continued as valid.
    /// </summary>
    enum class ValidationStatus : int32
    {
        Ok = 0,
        RegistryOnlyStale = 1,
        Corrupt = 2,
    };

    /// <summary>
    /// Provenance contract versions stamped into baked data (0 is
    /// reserved for unknown files without the tail ints). SchemaContractRevision: bump whenever
    /// PoseSearchSchema serialization or feature semantics change.
    /// Rev 2 adds per-clip speed bands (onset one-shot gait filter).
    /// SamplingContractRevision: mixed into ComputeSamplingContractHash; bump
    /// when the shared sampler helper semantics change without a schema value
    /// change (the hash is otherwise derived from schema sampling values).
    /// BakeToolBuildId: bump on any bake-tool (MotionMatchingBaker) change.
    /// </summary>
    static constexpr int32 SchemaContractRevision = 2;
    static constexpr int32 SamplingContractRevision = 1;
    static constexpr int32 BakeToolBuildId = 1;
    /// <summary>
    /// Schema snapshot captured at bake time. Meaningful only when SampleCount > 0.
    /// </summary>
    API_PROPERTY() const PoseSearchSchema& GetSchema() const
    {
        return _schema;
    }

    API_PROPERTY() int32 GetSampleRate() const
    {
        return _schema.SampleRate;
    }

    API_PROPERTY() int32 GetPoseFeatureSize() const
    {
        return PoseSearchSchema::GetPoseFeatureSize();
    }

    API_PROPERTY() int32 GetTrajectoryFeatureSize() const
    {
        return PoseSearchSchema::GetTrajectoryFeatureSize();
    }

    /// <summary>
    /// True when the asset contains baked samples (and therefore a valid schema snapshot).
    /// </summary>
    API_PROPERTY() bool IsBaked() const
    {
        return _sampleTimes.Count() > 0;
    }

    API_PROPERTY() int32 GetAnimationClipsCount() const
    {
        return _sourceAnimations.Count();
    }

    API_PROPERTY() int32 GetClipCount() const
    {
        return _clipAnimations.Count();
    }

    API_PROPERTY() int32 GetSampleCount() const
    {
        return _sampleTimes.Count();
    }

    API_PROPERTY() const Array<String>& GetAnimationTags() const
    {
        return _tagNames;
    }

    API_PROPERTY() const Array<int32>& GetAnimationClipCountsByTag() const
    {
        return _tagClipCounts;
    }

    API_PROPERTY() const Array<float>& GetTrajectorySampleTimes() const
    {
        return _schema.TrajectorySampleTimes;
    }

    API_PROPERTY() const Array<float>& GetClipLengths() const
    {
        return _clipLengths;
    }

    API_PROPERTY() const Array<int32>& GetClipSampleCounts() const
    {
        return _clipSampleCounts;
    }

    API_PROPERTY() const Array<int32>& GetSampleClipIndices() const
    {
        return _sampleClipIndices;
    }

    API_PROPERTY() const Array<float>& GetSampleTimes() const
    {
        return _sampleTimes;
    }

    API_PROPERTY() const Array<float>& GetPoseFeatures() const
    {
        return _poseFeatures;
    }

    API_PROPERTY() const Array<float>& GetTrajectoryFeatures() const
    {
        return _trajectoryFeatures;
    }

    API_PROPERTY() const Array<Float3>& GetRootPositions() const
    {
        return _rootPositions;
    }

    API_PROPERTY() const Array<Float3>& GetRootVelocities() const
    {
        return _rootVelocities;
    }

    API_PROPERTY() const Array<Float3>& GetRootForwards() const
    {
        return _rootForwards;
    }

    /// <summary>
    /// Per-dimension deviation (mean centered-row-norm, floored at
    /// `dev > 0.1 ? dev : 1.0`). The only stored normalization statistic:
    /// value means are gone (centering lives inside the deviation calc) and
    /// baked/query features stay raw. The search applies
    /// sqrt(sum-normalized weight) / deviation once at cost time.
    /// </summary>
    API_PROPERTY() const Array<float>& GetPoseFeatureDeviations() const
    {
        return _poseFeatureDeviations;
    }

    API_PROPERTY() const Array<float>& GetTrajectoryFeatureDeviations() const
    {
        return _trajectoryFeatureDeviations;
    }

    API_FUNCTION()
    int32 GetAnimationClipCountForTag(const String& tag) const;

    API_FUNCTION()
    Animation* GetClipAnimation(int32 clipIndex) const;

    /// <summary>
    /// Normalized tag set of a baked clip (lowercase, sorted, unique).
    /// Empty = invalid index. Membership, not identity: a clip passes a tag
    /// filter when its set overlaps the allowed set.
    /// </summary>
    const Array<String>& GetClipTags(int32 clipIndex) const;

    /// <summary>
    /// Primary tag (element 0) for single-tag display contexts. Empty when none.
    /// </summary>
    API_FUNCTION()
    String GetClipTag(int32 clipIndex) const;

    API_FUNCTION()
    bool GetClipLoop(int32 clipIndex) const;

    /// <summary>
    /// Baked turn-like flag (name keywords turn/arc/reface/pivot/spin, set
    /// once at bake). Replaces runtime clip-name substring matching in the
    /// policy: renaming assets no longer blinds turn hold/replay. No runtime
    /// path fallback: a database without the baked flag is registry-only and
    /// must be rebaked (runtime name matching was removed; bake-time flags are authoritative).
    /// </summary>
    API_FUNCTION()
    bool GetClipTurnLike(int32 clipIndex) const;

    /// <summary>
    /// Baked gait speed band (Walk/Run/Sprint from the asset name, set once
    /// at bake via MotionMatchingBakeHeuristics::BandFromClipName). Lets
    /// onset queries restrict one-shots to the intended gait: at the onset
    /// instant the features carry almost no gait information. Loops are
    /// never band-cut. 0 = Any (turns and unknowns stay eligible everywhere).
    /// </summary>
    API_FUNCTION()
    int32 GetClipSpeedBand(int32 clipIndex) const;

    /// <summary>
    /// Measured total yaw travel (radians, SIGNED) of a clip, accumulated
    /// from baked root forwards with the same atan2(x, z) convention the C#
    /// audit uses. Powers the search-time straightness penalty: turn clips
    /// must lose to loops while moving straight. Recomputed from baked data
    /// at load/bake (never serialized), so no asset format change.
    /// </summary>
    float GetClipTurnAngle(int32 clipIndex) const;

    /// <summary>
    /// Bake-statistics revision of the loaded baked data (0 = unknown: files baked
    /// before the revision tail int). Used by the baker's persistence check to reject stale reads.
    /// </summary>
    int32 GetNormalizationRevision() const
    {
        return _normalizationRevision;
    }

    /// <summary>
    /// Feature-layout hash of the loaded baked data, cast from
    /// PoseSearchSchema::GetFeatureLayoutHash (0 = unknown: files without the hash tail). A
    /// mismatch against the current layout drops baked arrays to
    /// registry-only with REBAKE REQUIRED, same as a revision mismatch.
    /// </summary>
    int32 GetFeatureLayoutHash() const
    {
        return _featureLayoutHash;
    }

    /// <summary>
    /// Schema serialization/feature-semantics contract revision of the
    /// loaded baked data (0 = unknown: files without the provenance tails).
    /// </summary>
    API_PROPERTY() int32 GetSchemaContractRevision() const
    {
        return _schemaContractRevision;
    }

    /// <summary>
    /// Sampling contract hash of the loaded baked data (0 =
    /// unknown: files without the provenance tails), derived from the schema sampling values + the manual
    /// SamplingContractRevision. A mismatch drops the baked arrays.
    /// </summary>
    API_PROPERTY() int32 GetSamplingContractHash() const
    {
        return _samplingContractHash;
    }

    /// <summary>
    /// Source manifest identity hash (0 = unknown: files without the provenance tails): registered
    /// source asset IDs + tags + loop. A mismatch means the registry changed
    /// after bake, so the baked arrays drop to registry-only.
    /// </summary>
    API_PROPERTY() int32 GetSourceManifestHash() const
    {
        return _sourceManifestHash;
    }

    /// <summary>
    /// Bake-tool build id of the loaded baked data (0 = unknown: files without the provenance tails).
    /// </summary>
    API_PROPERTY() int32 GetBakeToolBuildId() const
    {
        return _bakeToolBuildId;
    }

    API_FUNCTION()
    int32 GetSourceClipCount() const;

    API_FUNCTION()
    Animation* GetSourceClipAnimation(int32 sourceIndex) const;

    /// <summary>
    /// Normalized tag set of a registered source clip. Empty = invalid index.
    /// </summary>
    const Array<String>& GetSourceClipTags(int32 sourceIndex) const;

    /// <summary>
    /// Full tag set, '+'-joined for the editor UI transport
    /// (split back with SplitTagSet). Empty when none.
    /// </summary>
    API_FUNCTION()
    String GetSourceClipTag(int32 sourceIndex) const;

    API_FUNCTION()
    bool GetSourceClipLoop(int32 sourceIndex) const;

    API_FUNCTION()
    bool ValidateData() const;

    /// <summary>
    /// Three-way validation. On RegistryOnlyStale, reason names the
    /// exact mismatching revision/hash plus the rebake path. Corrupt data must
    /// never be treated as valid (load fails instead of clearing and running).
    /// </summary>
    ValidationStatus ValidateDataStatus(String& reason) const;

    /// <summary>
    /// Display/transport separator for tag sets ("walk+turn_travel").
    /// Shared with trace, replay and the editor UI.
    /// </summary>
    static constexpr Char TagSeparator = '+';

    /// <summary>
    /// Normalizes raw tags: lowercase, drop empties, sort, unique.
    /// </summary>
    static Array<String> NormalizeTags(const Array<String>& tags);

    /// <summary>
    /// Splits a '+'-joined tag set, then normalizes.
    /// </summary>
    static Array<String> SplitTagSet(const String& joined);

    /// <summary>
    /// Joins a tag set with '+' for display/transport.
    /// </summary>
    static String JoinTags(const Array<String>& tags);

    /// <summary>
    /// Order-insensitive set equality (both sides assumed normalized).
    /// </summary>
    static bool TagSetsEqual(const Array<String>& a, const Array<String>& b);

#if USE_EDITOR
    API_FUNCTION()
    static bool AddClips(
        const String& databasePath,
        const Array<String>& tags,
        const Array<Animation*>& animations,
        bool loop);

    API_FUNCTION()
    static bool AddClipFiles(
        const String& databasePath,
        const Array<String>& tags,
        const Array<String>& animationPaths,
        bool loop,
        API_PARAM(Out) int32& addedCount);

    API_FUNCTION()
    static bool ClearClips(const String& databasePath);

    API_FUNCTION()
    static bool UpdateSourceClip(
        const String& databasePath,
        int32 sourceIndex,
        const Array<String>& tags,
        bool loop);

    API_FUNCTION()
    static bool RemoveSourceClip(
        const String& databasePath,
        int32 sourceIndex);

    /// <summary>
    /// Batch clip edits. Each updateTagSets entry is one '+'-joined tag set
    /// (C# boundary stays string[]; split happens inside).
    /// </summary>
    API_FUNCTION()
    static bool ApplyClipEdits(
        const String& databasePath,
        const Array<int32>& updateIndices,
        const Array<String>& updateTagSets,
        const Array<int32>& updateLoops,
        const Array<int32>& removeIndices);

    API_FUNCTION()
    static bool ValidateSavedDatabase(
        const String& databasePath,
        API_PARAM(Out) int32& animationClipsCount,
        API_PARAM(Out) int32& clipCount,
        API_PARAM(Out) int32& sampleCount,
        API_PARAM(Out) int32& poseFeaturesLength,
        API_PARAM(Out) int32& trajectoryFeaturesLength);
#endif

    void GetSourceData(
        Array<Animation*>& animations,
        Array<Array<String>>& tags,
        Array<int32>& loopingClips) const;

#if USE_EDITOR
    bool AddSourceClips(
        const Array<String>& tags,
        const Array<Animation*>& animations,
        bool loop);

    void ClearSourceClips();

    void SetBakedData(
        const PoseSearchSchema& schema,
        const Array<int32>& sourceClipIndices,
        const Array<float>& clipLengths,
        const Array<int32>& clipSampleCounts,
        const Array<int32>& sampleClipIndices,
        const Array<float>& sampleTimes,
        const Array<Float3>& rootPositions,
        const Array<Float3>& rootVelocities,
        const Array<Float3>& rootForwards,
        const Array<float>& poseFeatures,
        const Array<float>& trajectoryFeatures,
        const Array<float>& poseFeatureDeviations,
        const Array<float>& trajectoryFeatureDeviations);

    void GetReferences(Array<Guid>& assets, Array<String>& files) const override;
    bool Save(const StringView& path = StringView::Empty) override;
#endif

    void InitAsVirtual() override;
    uint64 GetMemoryUsage() const override;

protected:
    LoadResult load() override;
    void unload(bool isReloading) override;
    AssetChunksFlag getChunksToPreload() const override;

private:
    PoseSearchSchema _schema;

    Array<AssetReference<Animation>> _sourceAnimations;
    Array<Array<String>> _sourceTags;
    Array<int32> _sourceLooping;
    Array<String> _tagNames;
    Array<int32> _tagClipCounts;

    Array<AssetReference<Animation>> _clipAnimations;
    Array<Array<String>> _clipTags;
    Array<int32> _clipLooping;
    Array<int32> _clipTurnLike;
    Array<int32> _clipSpeedBands;
    Array<float> _clipTurnAngles;
    Array<float> _clipLengths;
    Array<int32> _clipSampleCounts;

    Array<int32> _sampleClipIndices;
    Array<float> _sampleTimes;
    Array<float> _poseFeatures;
    Array<float> _trajectoryFeatures;
    Array<Float3> _rootPositions;
    Array<Float3> _rootVelocities;
    Array<Float3> _rootForwards;
    Array<float> _poseFeatureDeviations;
    Array<float> _trajectoryFeatureDeviations;
    // Value means are no longer stored (centering is internal to the deviation
    // calc). The on-disk means slot is still written as an empty array purely so
    // older binary-format files (poseMean, poseDev, trajMean, trajDev) keep parsing
    // far enough to hit the revision gate; the stored means are read into a
    // throwaway local and discarded.
    // Bake-statistics revision: which normalization produced the
    // baked arrays. 0 = unknown (files without the tail int). Written as a tail int (conditional
    // read like _clipTurnLike); a mismatch drops baked data with a loud
    // "rebake required" instead of silently mismatching. Registry survives.
    int32 _normalizationRevision = 0;
    // Feature-layout hash, cast from PoseSearchSchema::GetFeatureLayoutHash
    // (0 = unknown: files without the tail int). Second tail int after the revision; same
    // registry-only stale handling. SetBakedData stamps the current hash,
    // ClearBakedData resets it.
    int32 _featureLayoutHash = 0;
    // Provenance tails, appended after the layout hash. All 0 =
    // unknown (files written before the provenance tails have no bytes left, so they read as 0 and
    // classify registry-only stale). SetBakedData stamps them, ClearBakedData
    // resets them. Conditional reads keep old .flax files loadable.
    int32 _schemaContractRevision = 0;
    int32 _samplingContractHash = 0;
    int32 _sourceManifestHash = 0;
    int32 _bakeToolBuildId = 0;

    void ClearBakedData();
    void RebuildTagCounts();
    void ComputeClipTurnAngles();

    // Tiny pure FNV-1a helpers (no engine state, shared sampler
    // semantics untouched). Sampling hash covers schema sampling values + the
    // manual SamplingContractRevision; source hash covers the registered
    // source asset identity (IDs + tags + loop). Neither ever returns 0.
    int32 ComputeSamplingContractHash() const;
    int32 ComputeSourceManifestHash() const;

};
