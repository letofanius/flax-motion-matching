// MotionMatchingDatabase.Editor.cpp: editor registry and persistence (MotionMatchingDatabase::AddSourceClips/ClearSourceClips/SetBakedData/Save).
// Ownership: plugin runtime, no game/host references.
// Key invariants: SetBakedData copies the source tag sets verbatim, derives turn-like flags and speed bands once via MotionMatchingBakeHeuristics, and stamps the full provenance (NormalizationRevision, layout hash, schema/sampling/source/tool); Save appends revision/hash/band tails last so conditional reads keep older .flax files loadable as registry-only.
#include "MotionMatchingDatabase.h"

#if USE_EDITOR

#include "Engine/Core/Log.h"
#include "Engine/Serialization/MemoryWriteStream.h"
#include "Engine/Threading/Threading.h"
#include "../Baker/MotionMatchingBakeHeuristics.h"

bool MotionMatchingDatabase::AddSourceClips(
    const Array<String>& tags,
    const Array<Animation*>& animations,
    bool loop)
{
    const Array<String> set = NormalizeTags(tags);
    if (set.IsEmpty())
        return false;

    bool changed = false;
    for (Animation* animation : animations)
    {
        if (animation == nullptr)
            continue;

        int32 existingIndex = -1;
        for (int32 i = 0; i < _sourceAnimations.Count(); i++)
        {
            if (_sourceAnimations[i].GetID() == animation->GetID() &&
                TagSetsEqual(_sourceTags[i], set))
            {
                existingIndex = i;
                break;
            }
        }

        if (existingIndex >= 0)
        {
            const int32 loopValue = loop ? 1 : 0;
            if (_sourceLooping[existingIndex] != loopValue)
            {
                _sourceLooping[existingIndex] = loopValue;
                changed = true;
            }
            continue;
        }

        _sourceAnimations.Add(animation);
        _sourceTags.Add(set);
        _sourceLooping.Add(loop ? 1 : 0);
        changed = true;
    }

    if (changed)
    {
        RebuildTagCounts();
        ClearBakedData();
    }
    return changed;
}

void MotionMatchingDatabase::ClearSourceClips()
{
    _sourceAnimations.Clear();
    _sourceTags.Clear();
    _sourceLooping.Clear();
    _tagNames.Clear();
    _tagClipCounts.Clear();
    ClearBakedData();
}

void MotionMatchingDatabase::SetBakedData(
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
    const Array<float>& trajectoryFeatureDeviations)
{
    ClearBakedData();
    _schema = schema;

    _clipAnimations.EnsureCapacity(sourceClipIndices.Count());
    _clipTags.EnsureCapacity(sourceClipIndices.Count());
    _clipLooping.EnsureCapacity(sourceClipIndices.Count());
    _clipTurnLike.EnsureCapacity(sourceClipIndices.Count());
    _clipSpeedBands.EnsureCapacity(sourceClipIndices.Count());
    for (int32 sourceIndex : sourceClipIndices)
    {
        _clipAnimations.Add(_sourceAnimations[sourceIndex]);
        _clipTags.Add(_sourceTags[sourceIndex]);
        _clipLooping.Add(_sourceLooping[sourceIndex]);
        // Tag-set copy: baked clips carry the same set the source had at
        // bake time (registry edits clear baked data, so no drift).
        Animation* animation = _sourceAnimations[sourceIndex].Get();
        const String clipPath = animation != nullptr ? animation->GetPath() : String::Empty;
        _clipTurnLike.Add(MotionMatchingBakeHeuristics::IsTurnLikePath(clipPath) ? 1 : 0);
        _clipSpeedBands.Add(static_cast<int32>(MotionMatchingBakeHeuristics::BandFromClipName(clipPath)));
    }

    _clipLengths = clipLengths;
    _clipSampleCounts = clipSampleCounts;
    _sampleClipIndices = sampleClipIndices;
    _sampleTimes = sampleTimes;
    _rootPositions = rootPositions;
    _rootVelocities = rootVelocities;
    _rootForwards = rootForwards;
    _poseFeatures = poseFeatures;
    _trajectoryFeatures = trajectoryFeatures;
    _poseFeatureDeviations = poseFeatureDeviations;
    _trajectoryFeatureDeviations = trajectoryFeatureDeviations;
    // Freshly baked by this build: stamp the current statistics revision and
    // the feature-layout hash.
    _normalizationRevision = PoseSearchSchema::NormalizationRevision;
    _featureLayoutHash = static_cast<int32>(PoseSearchSchema::GetFeatureLayoutHash());
    // Provenance stamps: schema contract revision (class const), sampling
    // contract hash (schema sampling values + SamplingContractRevision),
    // source manifest identity (registered source asset IDs/tags/loop) and the
    // bake-tool build id. ValidateDataStatus recomputes these and drops the
    // baked arrays to registry-only on any mismatch.
    _schemaContractRevision = SchemaContractRevision;
    _samplingContractHash = ComputeSamplingContractHash();
    _sourceManifestHash = ComputeSourceManifestHash();
    _bakeToolBuildId = BakeToolBuildId;
    ComputeClipTurnAngles();
}

void MotionMatchingDatabase::GetReferences(
    Array<Guid>& assets,
    Array<String>& files) const
{
    BinaryAsset::GetReferences(assets, files);
    for (const AssetReference<Animation>& animation : _sourceAnimations)
    {
        if (animation.GetID().IsValid())
            assets.Add(animation.GetID());
    }
}

bool MotionMatchingDatabase::Save(const StringView& path)
{
    if (OnCheckSave(path))
        return true;
    ScopeLock lock(Locker);

    if (!ValidateData())
    {
        LOG(Error, "Cannot save motion matching database: data validation failed.");
        return true;
    }

    MemoryWriteStream stream(4096);
    _schema.Serialize(stream);
    stream.Write(_sourceAnimations);
    stream.Write(_sourceTags);
    stream.Write(_sourceLooping);
    stream.Write(_tagNames);
    stream.Write(_tagClipCounts);
    stream.Write(_clipAnimations);
    stream.Write(_clipTags);
    stream.Write(_clipLooping);
    stream.Write(_clipLengths);
    stream.Write(_clipSampleCounts);
    stream.Write(_sampleClipIndices);
    stream.Write(_sampleTimes);
    stream.Write(_poseFeatures);
    stream.Write(_trajectoryFeatures);
    stream.Write(_rootPositions);
    stream.Write(_rootVelocities);
    stream.Write(_rootForwards);
    // Revision-3 binary-format compat: the retired value-mean slots stay in the layout
    // as empty arrays so the byte stream keeps its shape (load() still reads them)
    // and revision-3 files stay distinguishable from revision-2 by the stored revision.
    // Only deviations are persisted as real statistics.
    const Array<float> noMeans;
    stream.Write(noMeans);
    stream.Write(_poseFeatureDeviations);
    stream.Write(noMeans);
    stream.Write(_trajectoryFeatureDeviations);
    stream.Write(_clipTurnLike);
    stream.WriteInt32(_normalizationRevision);
    stream.WriteInt32(_featureLayoutHash);
    // Provenance tails (read back conditionally, so files written before the
    // provenance tails stay loadable as registry-only).
    stream.WriteInt32(_schemaContractRevision);
    stream.WriteInt32(_samplingContractHash);
    stream.WriteInt32(_sourceManifestHash);
    stream.WriteInt32(_bakeToolBuildId);
    // Per-clip speed bands (rev 2, SchemaContractRevision gate): appended
    // LAST so every older layout keeps parsing (load reads conditionally).
    stream.Write(_clipSpeedBands);

    FlaxChunk* chunk;
    if (IsVirtual())
    {
        _header.Chunks[0] = chunk = New<FlaxChunk>();
    }
    else
    {
        chunk = GetOrCreateChunk(0);
    }
    chunk->Data.Copy(ToSpan(stream));

    AssetInitData data;
    data.SerializedVersion = SerializedVersion;
    const bool failed = path.HasChars()
        ? SaveAsset(path, data)
        : SaveAsset(data, true);

    if (IsVirtual())
    {
        _header.Chunks[0] = nullptr;
        Delete(chunk);
    }

    if (failed)
        LOG(Error, "Failed to save motion matching database '{}'.", path);
    return failed;
}

#endif
