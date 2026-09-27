// MotionMatchingDatabase.cpp: database queries, hashing, validation, serialization (GetClipTurnLike/GetClipSpeedBand/GetClipTurnAngle, ComputeClipTurnAngles/ComputeSamplingContractHash/ComputeSourceManifestHash, ValidateDataStatus, load/unload).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the binary-asset factory key TypeName stays frozen for load compat (see docs/MIGRATION.md); turn angles derive from baked root forwards and are never serialized; ValidateDataStatus orders structural checks before the provenance gate so stale bakes drop to registry-only while corrupt data fails the load; tail ints read conditionally so older .flax files stay loadable.
#include "MotionMatchingDatabase.h"

// Logging is used by the runtime load-validation path as well, so it stays
// outside the editor guard (building the game target exposed the missing
// include: the old project only ever built the editor target).
#include "Engine/Core/Log.h"

#if USE_EDITOR
#include "Engine/Content/Content.h"
#include "Engine/Platform/FileSystem.h"
#include "Engine/Platform/StringUtils.h"
#endif

#include "Engine/Content/Factories/BinaryAssetFactory.h"
#include "Engine/Serialization/MemoryReadStream.h"

#include <cmath>
#include <cstring>
#if USE_EDITOR
#include "Engine/Content/Upgraders/BinaryAssetUpgrader.h"
#include "Engine/Core/Types/Guid.h"
#include "Engine/Serialization/MemoryWriteStream.h"
#endif

namespace
{
#if USE_EDITOR
    /// <summary>
    /// Upgrades old database files so registered clips survive serialization format changes.
    /// </summary>
    class MotionMatchingDatabaseUpgrader : public BinaryAssetUpgrader
    {
    public:
        MotionMatchingDatabaseUpgrader()
        {
            static const Upgrader upgraders[] =
            {
                { 1, 2, &Upgrade_1_To_2 },
                { 2, 3, &Upgrade_2_To_3 },
            };
            setup(upgraders, ARRAY_COUNT(upgraders));
        }

    private:
        /// <summary>
        /// v1: sampleRate + source clips + baked data (no schema).
        /// v2: PoseSearchSchema + source clips + baked data.
        /// Registered clips are kept; baked data is dropped because v1 did not store
        /// the axes/bones it was baked with, so the database must be re-baked.
        /// </summary>
        static bool Upgrade_1_To_2(AssetMigrationContext& context)
        {
            const FlaxChunk* chunk = context.Input.Header.Chunks[0];
            if (chunk == nullptr || !chunk->IsLoaded())
            {
                LOG(Warning, "Motion matching database upgrade failed: missing data chunk.");
                return true;
            }

            MemoryReadStream input(chunk->Get(), chunk->Size());
            int32 sampleRate = 0;
            Array<Guid> sourceAnimations;
            Array<String> sourceTags;
            Array<int32> sourceLooping;
            Array<String> tagNames;
            Array<int32> tagClipCounts;
            Array<Guid> clipAnimations;
            Array<String> clipTags;
            Array<int32> clipLooping;
            Array<float> clipLengths;
            Array<int32> clipSampleCounts;
            Array<float> trajectorySampleTimes;
            input.ReadInt32(&sampleRate);
            input.Read(sourceAnimations);
            input.Read(sourceTags);
            input.Read(sourceLooping);
            input.Read(tagNames);
            input.Read(tagClipCounts);
            input.Read(clipAnimations);
            input.Read(clipTags);
            input.Read(clipLooping);
            input.Read(clipLengths);
            input.Read(clipSampleCounts);
            input.Read(trajectorySampleTimes);
            if (input.HasError())
            {
                LOG(Warning, "Motion matching database upgrade failed: corrupted v1 data.");
                return true;
            }

            PoseSearchSchema schema;
            if (sampleRate > 0)
                schema.SampleRate = sampleRate;
            if (trajectorySampleTimes.HasItems())
                schema.TrajectorySampleTimes = trajectorySampleTimes;

            MemoryWriteStream output(4096);
            schema.Serialize(output);
            output.Write(sourceAnimations);
            output.Write(sourceTags);
            output.Write(sourceLooping);
            output.Write(tagNames);
            output.Write(tagClipCounts);
            // Empty baked data: clipAnimations, clipTags, clipLooping, clipLengths, clipSampleCounts,
            // sampleClipIndices, sampleTimes, poseFeatures, trajectoryFeatures, rootPositions,
            // rootVelocities, rootForwards, poseMeans, poseStd, trajectoryMeans, trajectoryStd.
            for (int32 i = 0; i < 16; i++)
                output.WriteInt32(0);

            if (context.AllocateChunk(0))
                return true;
            context.Output.Header.Chunks[0]->Data.Copy(ToSpan(output));

            LOG(
                Info,
                "Motion matching database upgraded to v2: kept {} registered clips, baked data cleared (re-bake required).",
                sourceAnimations.Count());
            return false;
        }

        /// <summary>
        /// v2: one tag string per clip. v3: tag SET per clip.
        /// Registry survives (each single tag becomes a 1-element set);
        /// baked data is dropped because per-clip tag identity feeds the
        /// search partitions, so the database must be re-baked.
        /// </summary>
        static bool Upgrade_2_To_3(AssetMigrationContext& context)
        {
            const FlaxChunk* chunk = context.Input.Header.Chunks[0];
            if (chunk == nullptr || !chunk->IsLoaded())
            {
                LOG(Warning, "Motion matching database upgrade failed: missing data chunk.");
                return true;
            }

            MemoryReadStream input(chunk->Get(), chunk->Size());
            PoseSearchSchema schema;
            schema.Deserialize(input);
            Array<Guid> sourceAnimations;
            Array<String> sourceTagsFlat;
            Array<int32> sourceLooping;
            Array<String> tagNames;
            Array<int32> tagClipCounts;
            input.Read(sourceAnimations);
            input.Read(sourceTagsFlat);
            input.Read(sourceLooping);
            input.Read(tagNames);
            input.Read(tagClipCounts);
            if (input.HasError() ||
                sourceAnimations.Count() != sourceTagsFlat.Count() ||
                sourceAnimations.Count() != sourceLooping.Count())
            {
                LOG(Warning, "Motion matching database upgrade failed: corrupted v2 registry.");
                return true;
            }

            Array<Array<String>> sourceTags;
            sourceTags.EnsureCapacity(sourceTagsFlat.Count());
            for (const String& tag : sourceTagsFlat)
            {
                Array<String> set;
                Array<String> raw;
                raw.Add(tag);
                set = MotionMatchingDatabase::NormalizeTags(raw);
                if (set.IsEmpty())
                    set.Add(tag);
                sourceTags.Add(set);
            }

            MemoryWriteStream output(4096);
            schema.Serialize(output);
            output.Write(sourceAnimations);
            output.Write(sourceTags);
            output.Write(sourceLooping);
            output.Write(tagNames);
            output.Write(tagClipCounts);
            // Empty baked data (same 16 slots the v1->v2 upgrader writes).
            for (int32 i = 0; i < 16; i++)
                output.WriteInt32(0);

            if (context.AllocateChunk(0))
                return true;
            context.Output.Header.Chunks[0]->Data.Copy(ToSpan(output));

            LOG(
                Info,
                "Motion matching database upgraded to v3 (tag sets): kept {} registered clips, baked data cleared (re-bake required).",
                sourceAnimations.Count());
            return false;
        }

    };
#endif

    /// <summary>
    /// Registers the asset factory safely when the Game module is hot-reloaded.
    /// Flax can keep the previous factory entry alive until the old module is
    /// released, so replacing the key is required before adding the new one.
    /// </summary>
    class MotionMatchingDatabaseFactory : public BinaryAssetFactory<MotionMatchingDatabase>
    {
    public:
        MotionMatchingDatabaseFactory()
        {
            auto& factories = IAssetFactory::Get();
            factories.Remove(MotionMatchingDatabase::TypeName);
            factories.Add(MotionMatchingDatabase::TypeName, this);
        }

        ~MotionMatchingDatabaseFactory()
        {
            auto& factories = IAssetFactory::Get();
            IAssetFactory* current = nullptr;
            if (factories.TryGet(MotionMatchingDatabase::TypeName, current) &&
                current == this)
            {
                factories.Remove(MotionMatchingDatabase::TypeName);
            }
        }

        bool SupportsVirtualAssets() const override
        {
            return true;
        }

#if USE_EDITOR
        IAssetUpgrader* GetUpgrader() const override
        {
            return &_upgrader;
        }

    private:
        mutable MotionMatchingDatabaseUpgrader _upgrader;
#endif
    };

#if USE_EDITOR
    MotionMatchingDatabase* LoadOrCreateDatabase(const String& databasePath)
    {
        if (databasePath.IsEmpty())
            return nullptr;

        const String directory = StringUtils::GetDirectoryName(databasePath);
        if (directory.HasChars() &&
            !FileSystem::DirectoryExists(directory) &&
            !FileSystem::CreateDirectory(directory))
        {
            LOG(Error, "Failed to create motion matching database directory '{}'.", directory);
            return nullptr;
        }

        if (!FileSystem::FileExists(databasePath))
            return Content::CreateVirtualAsset<MotionMatchingDatabase>();

        MotionMatchingDatabase* database =
            Content::Load<MotionMatchingDatabase>(databasePath);
        if (database == nullptr || database->WaitForLoaded())
        {
            LOG(Error, "Failed to load motion matching database '{}'.", databasePath);
            return nullptr;
        }
        return database;
    }
#endif
}

// Serialized asset type identity, FROZEN for load compatibility: baked
// .flax files store this exact UTF-16 string in their header, so renaming
// it would orphan every shipped database. The C#/scripting namespace moved
// to MotionMatching; only this binary-asset factory key keeps the legacy
// name. A future rename requires a rebake + header migration (see
// docs/MIGRATION.md), not a one-line edit.
const String MotionMatchingDatabase::TypeName = TEXT("Game.MotionMatchingDatabase");
static MotionMatchingDatabaseFactory CFactoryMotionMatchingDatabase;

namespace
{
    const Array<String>& EmptyTagSet()
    {
        static const Array<String> empty;
        return empty;
    }
}

Array<String> MotionMatchingDatabase::NormalizeTags(const Array<String>& tags)
{
    Array<String> out;
    out.EnsureCapacity(tags.Count());
    for (const String& tag : tags)
    {
        String t = tag.ToLower();
        if (!t.IsEmpty() && !out.Contains(t))
            out.Add(t);
    }
    // Insertion sort (sets are tiny; Flax Array has no Sort member here).
    for (int32 i = 1; i < out.Count(); i++)
    {
        String key = out[i];
        int32 j = i - 1;
        while (j >= 0 && out[j].Compare(key) > 0)
        {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
    return out;
}

Array<String> MotionMatchingDatabase::SplitTagSet(const String& joined)
{
    Array<String> parts;
    joined.Split(TagSeparator, parts);
    return NormalizeTags(parts);
}

String MotionMatchingDatabase::JoinTags(const Array<String>& tags)
{
    String out;
    for (int32 i = 0; i < tags.Count(); i++)
    {
        if (i > 0)
            out += TagSeparator;
        out += tags[i];
    }
    return out;
}

bool MotionMatchingDatabase::TagSetsEqual(const Array<String>& a, const Array<String>& b)
{
    if (a.Count() != b.Count())
        return false;
    for (const String& tag : a)
    {
        if (!b.Contains(tag))
            return false;
    }
    return true;
}

MotionMatchingDatabase::MotionMatchingDatabase(const SpawnParams& params, const AssetInfo* info)
    : BinaryAsset(params, info)
{
}

int32 MotionMatchingDatabase::GetAnimationClipCountForTag(const String& tag) const
{
    const int32 index = _tagNames.Find(tag);
    return index >= 0 ? _tagClipCounts[index] : 0;
}

Animation* MotionMatchingDatabase::GetClipAnimation(int32 clipIndex) const
{
    return _clipAnimations.IsValidIndex(clipIndex)
        ? _clipAnimations[clipIndex].Get()
        : nullptr;
}

const Array<String>& MotionMatchingDatabase::GetClipTags(int32 clipIndex) const
{
    return _clipTags.IsValidIndex(clipIndex)
        ? _clipTags[clipIndex]
        : EmptyTagSet();
}

String MotionMatchingDatabase::GetClipTag(int32 clipIndex) const
{
    const Array<String>& set = GetClipTags(clipIndex);
    return set.HasItems() ? set[0] : String::Empty;
}

bool MotionMatchingDatabase::GetClipLoop(int32 clipIndex) const
{
    return _clipLooping.IsValidIndex(clipIndex) &&
           _clipLooping[clipIndex] != 0;
}

bool MotionMatchingDatabase::GetClipTurnLike(int32 clipIndex) const
{
    if (_clipTurnLike.IsValidIndex(clipIndex))
        return _clipTurnLike[clipIndex] != 0;
    // No filename fallback at runtime. A database without the baked flag (older
    // binary format or stale: layout/revision mismatch dropped its baked arrays)
    // is registry-only / REBAKE REQUIRED; turning it into a turn-like
    // classification by looking at clip paths would reintroduce the runtime name
    // matching this flag exists to remove. Name matching stays bake-time only
    // (Baker/MotionMatchingBakeHeuristics.h).
    return false;
}

int32 MotionMatchingDatabase::GetClipSpeedBand(int32 clipIndex) const
{
    if (_clipSpeedBands.IsValidIndex(clipIndex))
        return _clipSpeedBands[clipIndex];
    // No band data (files written before the speed-band tail): Any keeps every one-shot eligible, exactly
    // the pre-band behavior. The SchemaContractRevision gate already forces
    // such files registry-only; this is only a safe default, never a bypass.
    return 0;
}

float MotionMatchingDatabase::GetClipTurnAngle(int32 clipIndex) const
{
    return _clipTurnAngles.IsValidIndex(clipIndex)
        ? _clipTurnAngles[clipIndex]
        : 0.0f;
}

// One-time per load/bake: signed total yaw travel per clip from baked root
// forwards (same atan2(x, z) convention as the C# audit, so bucket matching
// and native penalties agree). Derived data — never serialized, recomputed
// here, so old .flax files need no migration.
void MotionMatchingDatabase::ComputeClipTurnAngles()
{
    _clipTurnAngles.Clear();
    const int32 clipCount = _clipAnimations.Count();
    if (_clipSampleCounts.Count() != clipCount || _rootForwards.Count() == 0)
        return;
    _clipTurnAngles.EnsureCapacity(clipCount);
    int32 cursor = 0;
    const float twoPi = TWO_PI;
    for (int32 c = 0; c < clipCount; c++)
    {
        float yaw = 0.0f;
        const int32 n = _clipSampleCounts[c];
        for (int32 i = 1; i < n && cursor + i < _rootForwards.Count(); i++)
        {
            const Float3& a = _rootForwards[cursor + i - 1];
            const Float3& b = _rootForwards[cursor + i];
            const float aLenSq = a.X * a.X + a.Z * a.Z;
            const float bLenSq = b.X * b.X + b.Z * b.Z;
            if (aLenSq > 0.000001f && bLenSq > 0.000001f)
            {
                float dy = Math::Atan2(b.X, b.Z) - Math::Atan2(a.X, a.Z);
                while (dy > PI)
                    dy -= twoPi;
                while (dy < -PI)
                    dy += twoPi;
                yaw += dy;
            }
        }
        _clipTurnAngles.Add(yaw);
        cursor += n;
    }
}

// Tiny pure FNV-1a helpers. These deliberately do NOT call or change
// any shared sampler helper: the sampling hash is derived from the schema values
// the database already stores plus the manual SamplingContractRevision, and
// the source hash from the registered source identity. 0 is reserved for
// unknown (files without the provenance tails), so a computed 0 is remapped to 1.
namespace
{
    void MixHashByte(uint32& hash, uint32 b)
    {
        hash ^= (b & 0xFFu);
        hash *= 16777619u;
    }

    void MixHashInt(uint32& hash, int32 v)
    {
        const uint32 u = static_cast<uint32>(v);
        MixHashByte(hash, u);
        MixHashByte(hash, u >> 8);
        MixHashByte(hash, u >> 16);
        MixHashByte(hash, u >> 24);
    }

    void MixHashFloat(uint32& hash, float v)
    {
        uint32 u = 0;
        std::memcpy(&u, &v, sizeof(u));
        MixHashInt(hash, static_cast<int32>(u));
    }

    void MixHashString(uint32& hash, const String& s)
    {
        MixHashInt(hash, s.Length());
        for (int32 i = 0; i < s.Length(); i++)
        {
            const uint32 c = static_cast<uint32>(s[i]);
            MixHashByte(hash, c);
            MixHashByte(hash, c >> 8);
            MixHashByte(hash, c >> 16);
            MixHashByte(hash, c >> 24);
        }
    }

    int32 FinishHash(uint32 hash)
    {
        if (hash == 0u)
            hash = 1u;
        return static_cast<int32>(hash);
    }
}

int32 MotionMatchingDatabase::ComputeSamplingContractHash() const
{
    // Contract fields the database can observe: feature block sizes, sample
    // rate, trajectory sample times and the shared contact gate. A helper-only
    // semantic change is covered by bumping SamplingContractRevision.
    uint32 hash = 2166136261u;
    MixHashInt(hash, SamplingContractRevision);
    MixHashInt(hash, GetPoseFeatureSize());
    MixHashInt(hash, GetTrajectoryFeatureSize());
    MixHashInt(hash, _schema.SampleRate);
    MixHashFloat(hash, PoseSearchSchema::ContactMaxSpeed);
    MixHashFloat(hash, PoseSearchSchema::ContactMaxHeight);
    MixHashInt(hash, _schema.TrajectorySampleTimes.Count());
    for (int32 i = 0; i < _schema.TrajectorySampleTimes.Count(); i++)
        MixHashFloat(hash, _schema.TrajectorySampleTimes[i]);
    return FinishHash(hash);
}

int32 MotionMatchingDatabase::ComputeSourceManifestHash() const
{
    // Source manifest identity: every registered source asset's 16-byte ID
    // plus its tag and loop flag. Flax's runtime Asset API exposes no stable
    // source file timestamp, so the "asset id+timestamp" contract is carried
    // by this ID/tag/loop identity; the tool side is BakeToolBuildId. A change
    // here means the registry no longer matches the bake snapshot.
    uint32 hash = 2166136261u;
    MixHashInt(hash, _sourceAnimations.Count());
    for (int32 i = 0; i < _sourceAnimations.Count(); i++)
    {
        const Guid id = _sourceAnimations[i].GetID();
        for (int32 b = 0; b < 16; b++)
            MixHashByte(hash, id.Raw[b]);
        const Array<String>& set = GetSourceClipTags(i);
        MixHashInt(hash, set.Count());
        for (const String& tag : set)
            MixHashString(hash, tag);
        MixHashInt(hash, _sourceLooping.IsValidIndex(i) && _sourceLooping[i] != 0 ? 1 : 0);
    }
    return FinishHash(hash);
}

int32 MotionMatchingDatabase::GetSourceClipCount() const
{
    return _sourceAnimations.Count();
}

Animation* MotionMatchingDatabase::GetSourceClipAnimation(int32 sourceIndex) const
{
    return _sourceAnimations.IsValidIndex(sourceIndex)
        ? _sourceAnimations[sourceIndex].Get()
        : nullptr;
}

const Array<String>& MotionMatchingDatabase::GetSourceClipTags(int32 sourceIndex) const
{
    return _sourceTags.IsValidIndex(sourceIndex)
        ? _sourceTags[sourceIndex]
        : EmptyTagSet();
}

String MotionMatchingDatabase::GetSourceClipTag(int32 sourceIndex) const
{
    return JoinTags(GetSourceClipTags(sourceIndex));
}

bool MotionMatchingDatabase::GetSourceClipLoop(int32 sourceIndex) const
{
    return _sourceLooping.IsValidIndex(sourceIndex) &&
           _sourceLooping[sourceIndex] != 0;
}

#if USE_EDITOR

bool MotionMatchingDatabase::AddClips(
    const String& databasePath,
    const Array<String>& tags,
    const Array<Animation*>& animations,
    bool loop)
{
    const Array<String> set = NormalizeTags(tags);
    if (set.IsEmpty() || animations.IsEmpty())
    {
        LOG(Error, "Cannot add motion matching clips without tags and animations.");
        return false;
    }

    MotionMatchingDatabase* database = LoadOrCreateDatabase(databasePath);
    if (database == nullptr)
        return false;

    const bool changed = database->AddSourceClips(set, animations, loop);
    if (database->IsVirtual() && !changed)
        return false;
    if (changed &&
        (database->IsVirtual() ? database->Save(databasePath) : database->Save()))
        return false;

    LOG(
        Info,
        "Motion matching clips persisted: total={}, tags='{}'.",
        database->GetAnimationClipsCount(),
        JoinTags(set));
    return true;
}

bool MotionMatchingDatabase::AddClipFiles(
    const String& databasePath,
    const Array<String>& tags,
    const Array<String>& animationPaths,
    bool loop,
    int32& addedCount)
{
    addedCount = 0;
    Array<Animation*> animations;
    animations.EnsureCapacity(animationPaths.Count());
    for (const String& path : animationPaths)
    {
        Animation* animation = Content::LoadAsync<Animation>(path);
        if (animation != nullptr)
            animations.Add(animation);
    }

    addedCount = animations.Count();
    return addedCount > 0 && AddClips(databasePath, tags, animations, loop);
}

bool MotionMatchingDatabase::ClearClips(const String& databasePath)
{
    MotionMatchingDatabase* database = LoadOrCreateDatabase(databasePath);
    if (database == nullptr)
        return false;

    database->ClearSourceClips();
    if (database->IsVirtual() ? database->Save(databasePath) : database->Save())
        return false;

    LOG(Info, "Motion matching clip registry cleared and persisted.");
    return true;
}

bool MotionMatchingDatabase::UpdateSourceClip(
    const String& databasePath,
    int32 sourceIndex,
    const Array<String>& tags,
    bool loop)
{
    const Array<String> set = NormalizeTags(tags);
    if (set.IsEmpty())
    {
        LOG(Error, "Cannot update motion matching clip without tags.");
        return false;
    }

    MotionMatchingDatabase* database = LoadOrCreateDatabase(databasePath);
    if (database == nullptr)
        return false;
    if (sourceIndex < 0 || sourceIndex >= database->_sourceAnimations.Count())
    {
        LOG(Error, "Cannot update motion matching clip: source index {} out of range.", sourceIndex);
        return false;
    }

    database->_sourceTags[sourceIndex] = set;
    database->_sourceLooping[sourceIndex] = loop ? 1 : 0;
    database->RebuildTagCounts();
    // Source tags feed baked clips, so any baked data is stale now.
    database->ClearBakedData();
    if (database->IsVirtual() ? database->Save(databasePath) : database->Save())
        return false;

    LOG(
        Info,
        "Motion matching clip updated: index={}, tags='{}', loop={}. Rebake required.",
        sourceIndex,
        JoinTags(set),
        loop ? 1 : 0);
    return true;
}

bool MotionMatchingDatabase::RemoveSourceClip(
    const String& databasePath,
    int32 sourceIndex)
{
    MotionMatchingDatabase* database = LoadOrCreateDatabase(databasePath);
    if (database == nullptr)
        return false;
    if (sourceIndex < 0 || sourceIndex >= database->_sourceAnimations.Count())
    {
        LOG(Error, "Cannot remove motion matching clip: source index {} out of range.", sourceIndex);
        return false;
    }

    database->_sourceAnimations.RemoveAt(sourceIndex);
    database->_sourceTags.RemoveAt(sourceIndex);
    database->_sourceLooping.RemoveAt(sourceIndex);
    database->RebuildTagCounts();
    database->ClearBakedData();
    if (database->IsVirtual() ? database->Save(databasePath) : database->Save())
        return false;

    LOG(Info, "Motion matching clip removed: index={}. Rebake required.", sourceIndex);
    return true;
}

bool MotionMatchingDatabase::ApplyClipEdits(
    const String& databasePath,
    const Array<int32>& updateIndices,
    const Array<String>& updateTagSets,
    const Array<int32>& updateLoops,
    const Array<int32>& removeIndices)
{
    if (updateIndices.Count() != updateTagSets.Count() ||
        updateIndices.Count() != updateLoops.Count())
    {
        LOG(Error, "Cannot apply motion matching clip edits: mismatched array sizes.");
        return false;
    }
    if (updateIndices.IsEmpty() && removeIndices.IsEmpty())
        return true;

    MotionMatchingDatabase* database = LoadOrCreateDatabase(databasePath);
    if (database == nullptr)
        return false;

    const int32 sourceCount = database->_sourceAnimations.Count();
    Array<Array<String>> updateSets;
    updateSets.EnsureCapacity(updateIndices.Count());
    for (int32 i = 0; i < updateIndices.Count(); i++)
    {
        updateSets.Add(SplitTagSet(updateTagSets[i]));
        if (updateIndices[i] < 0 || updateIndices[i] >= sourceCount ||
            updateSets[i].IsEmpty())
        {
            LOG(Error, "Cannot apply motion matching clip edits: invalid update at position {}.", i);
            return false;
        }
    }
    for (int32 index : removeIndices)
    {
        if (index < 0 || index >= sourceCount)
        {
            LOG(Error, "Cannot apply motion matching clip edits: invalid remove index {}.", index);
            return false;
        }
    }

    for (int32 i = 0; i < updateIndices.Count(); i++)
    {
        database->_sourceTags[updateIndices[i]] = updateSets[i];
        database->_sourceLooping[updateIndices[i]] = updateLoops[i] != 0 ? 1 : 0;
    }

    if (!removeIndices.IsEmpty())
    {
        // Rebuild without the removed entries; updates above target original
        // indices, so they stay valid regardless of removal order.
        Array<AssetReference<Animation>> keptAnimations;
        Array<Array<String>> keptTags;
        Array<int32> keptLooping;
        keptAnimations.EnsureCapacity(sourceCount);
        keptTags.EnsureCapacity(sourceCount);
        keptLooping.EnsureCapacity(sourceCount);
        for (int32 i = 0; i < sourceCount; i++)
        {
            if (removeIndices.Contains(i))
                continue;
            keptAnimations.Add(database->_sourceAnimations[i]);
            keptTags.Add(database->_sourceTags[i]);
            keptLooping.Add(database->_sourceLooping[i]);
        }
        database->_sourceAnimations = keptAnimations;
        database->_sourceTags = keptTags;
        database->_sourceLooping = keptLooping;
    }

    database->RebuildTagCounts();
    // Source tags feed baked clips, so any baked data is stale now.
    database->ClearBakedData();
    if (database->IsVirtual() ? database->Save(databasePath) : database->Save())
        return false;

    LOG(
        Info,
        "Motion matching clip edits applied: updates={}, removes={}. Rebake required.",
        updateIndices.Count(),
        removeIndices.Count());
    return true;
}

bool MotionMatchingDatabase::ValidateSavedDatabase(
    const String& databasePath,
    int32& animationClipsCount,
    int32& clipCount,
    int32& sampleCount,
    int32& poseFeaturesLength,
    int32& trajectoryFeaturesLength)
{
    animationClipsCount = 0;
    clipCount = 0;
    sampleCount = 0;
    poseFeaturesLength = 0;
    trajectoryFeaturesLength = 0;

    if (!FileSystem::FileExists(databasePath))
    {
        LOG(Error, "Motion matching persistence check failed: '{}' does not exist.", databasePath);
        return false;
    }

    MotionMatchingDatabase* database =
        Content::Load<MotionMatchingDatabase>(databasePath);
    if (database == nullptr)
        return false;

    database->Reload();
    if (database->WaitForLoaded() || !database->ValidateData())
    {
        LOG(Error, "Motion matching persistence check failed while reloading '{}'.", databasePath);
        return false;
    }

    animationClipsCount = database->GetAnimationClipsCount();
    clipCount = database->GetClipCount();
    sampleCount = database->GetSampleCount();
    poseFeaturesLength = database->GetPoseFeatures().Count();
    trajectoryFeaturesLength = database->GetTrajectoryFeatures().Count();
    return poseFeaturesLength == sampleCount * database->GetPoseFeatureSize() &&
           trajectoryFeaturesLength == sampleCount * database->GetTrajectoryFeatureSize();
}

#endif

bool MotionMatchingDatabase::ValidateData() const
{
    String reason;
    return ValidateDataStatus(reason) == ValidationStatus::Ok;
}

MotionMatchingDatabase::ValidationStatus MotionMatchingDatabase::ValidateDataStatus(String& reason) const
{
    reason = String::Empty;
    const int32 poseFeatureSize = GetPoseFeatureSize();
    const int32 trajectoryFeatureSize = GetTrajectoryFeatureSize();

    // ---- Structural invariants: violations are CORRUPT ----
    const int32 sourceCount = _sourceAnimations.Count();
    if (_sourceTags.Count() != sourceCount ||
        _sourceLooping.Count() != sourceCount ||
        _tagNames.Count() != _tagClipCounts.Count())
    {
        reason = TEXT("source registry arrays have inconsistent sizes");
        return ValidationStatus::Corrupt;
    }

    // Multi-tag: one clip belongs to every tag in its set, so the derived
    // counts sum to the total memberships, not the source clip count.
    int32 tagCountTotal = 0;
    int32 membershipTotal = 0;
    for (int32 i = 0; i < sourceCount; i++)
    {
        if (!_sourceAnimations[i].GetID().IsValid() ||
            _sourceTags[i].IsEmpty() ||
            (_sourceLooping[i] != 0 && _sourceLooping[i] != 1))
        {
            reason = String::Format(TEXT("source clip {} is invalid"), i);
            return ValidationStatus::Corrupt;
        }
        membershipTotal += _sourceTags[i].Count();
    }
    for (int32 count : _tagClipCounts)
    {
        if (count < 0)
        {
            reason = TEXT("tag clip count is negative");
            return ValidationStatus::Corrupt;
        }
        tagCountTotal += count;
    }
    if (tagCountTotal != membershipTotal)
    {
        reason = TEXT("tag clip counts do not sum to the source tag memberships");
        return ValidationStatus::Corrupt;
    }

    const int32 clipCount = _clipAnimations.Count();
    const int32 sampleCount = _sampleTimes.Count();

    if (sampleCount == 0)
    {
        // Unbaked database: only the source clip registry is meaningful.
        if (clipCount != 0 ||
            !_poseFeatureDeviations.IsEmpty() ||
            !_trajectoryFeatureDeviations.IsEmpty())
        {
            reason = TEXT("unbaked database still carries baked arrays");
            return ValidationStatus::Corrupt;
        }
        return ValidationStatus::Ok;
    }

    // ---- Provenance / older binary-format layout: REGISTRY-ONLY STALE ----
    // Decided after the persistent source registry but before any baked-array
    // check: a stale bake from an older binary format is dropped wholesale to registry-only (even
    // if its arrays are also malformed), exactly like the previous gate, so a
    // layout/version change never destroys the clip list. Corrupt detection
    // below therefore only ever applies to data whose provenance is current.
    // Feature/stat array sizes baked under an older feature layout are a
    // provenance problem (the layout hash/revision gate drops them), not
    // corruption: the count mismatch is explained by the layout change.
    const bool legacyLayout =
        _poseFeatures.Count() != sampleCount * poseFeatureSize ||
        _trajectoryFeatures.Count() != sampleCount * trajectoryFeatureSize ||
        _poseFeatureDeviations.Count() != poseFeatureSize ||
        _trajectoryFeatureDeviations.Count() != trajectoryFeatureSize;
    const int32 samplingHash = ComputeSamplingContractHash();
    const int32 sourceHash = ComputeSourceManifestHash();
    const int32 currentLayoutHash = static_cast<int32>(PoseSearchSchema::GetFeatureLayoutHash());
    const bool statsStale = _normalizationRevision != PoseSearchSchema::NormalizationRevision;
    const bool layoutHashStale = _featureLayoutHash != currentLayoutHash;
    // 0 is never a valid stamped value (unknown: files without the tails), so
    // equality alone is not enough: an old file with no tails must classify as stale.
    const bool schemaStale =
        _schemaContractRevision == 0 || _schemaContractRevision != SchemaContractRevision;
    const bool samplingStale =
        _samplingContractHash == 0 || _samplingContractHash != samplingHash;
    const bool sourceStale =
        _sourceManifestHash == 0 || _sourceManifestHash != sourceHash;
    const bool bakeToolStale =
        _bakeToolBuildId == 0 || _bakeToolBuildId != BakeToolBuildId;
    if (legacyLayout || statsStale || layoutHashStale || schemaStale ||
        samplingStale || sourceStale || bakeToolStale)
    {
        reason = String::Format(
            TEXT("{} (statsRev {} vs {}, layoutHash {} vs {}, schemaRev {} vs {}, samplingHash {} vs {}, sourceHash {} vs {}, bakeTool {} vs {}). REBAKE REQUIRED: rebake from the Motion Matching bake panel."),
            String(legacyLayout ? TEXT("legacy feature layout") : TEXT("provenance mismatch")),
            _normalizationRevision,
            PoseSearchSchema::NormalizationRevision,
            _featureLayoutHash,
            currentLayoutHash,
            _schemaContractRevision,
            SchemaContractRevision,
            _samplingContractHash,
            samplingHash,
            _sourceManifestHash,
            sourceHash,
            _bakeToolBuildId,
            BakeToolBuildId);
        return ValidationStatus::RegistryOnlyStale;
    }

    // ---- Baked structural invariants (CORRUPT): provenance is current ----
    if (_clipTags.Count() != clipCount ||
        _clipLooping.Count() != clipCount ||
        (_clipTurnLike.Count() != clipCount && _clipTurnLike.Count() != 0) ||
        (_clipSpeedBands.Count() != clipCount && _clipSpeedBands.Count() != 0) ||
        (_clipTurnAngles.Count() != clipCount && _clipTurnAngles.Count() != 0) ||
        _clipLengths.Count() != clipCount ||
        _clipSampleCounts.Count() != clipCount)
    {
        reason = TEXT("baked clip arrays have inconsistent sizes");
        return ValidationStatus::Corrupt;
    }
    for (int32 i = 0; i < clipCount; i++)
    {
        if (!_clipAnimations[i].GetID().IsValid() ||
            _clipTags[i].IsEmpty() ||
            (_clipLooping[i] != 0 && _clipLooping[i] != 1) ||
            (_clipTurnLike.Count() != 0 && _clipTurnLike[i] != 0 && _clipTurnLike[i] != 1) ||
            (_clipSpeedBands.Count() != 0 && (_clipSpeedBands[i] < 0 || _clipSpeedBands[i] > 3)) ||
            _clipLengths[i] <= 0.0f)
        {
            reason = String::Format(TEXT("baked clip {} is invalid"), i);
            return ValidationStatus::Corrupt;
        }
    }
    if (_sampleClipIndices.Count() != sampleCount ||
        _rootPositions.Count() != sampleCount ||
        _rootVelocities.Count() != sampleCount ||
        _rootForwards.Count() != sampleCount)
    {
        reason = TEXT("baked sample arrays have inconsistent sizes");
        return ValidationStatus::Corrupt;
    }
    int32 clipSampleTotal = 0;
    for (int32 count : _clipSampleCounts)
    {
        if (count <= 0)
        {
            reason = TEXT("a clip sample count is not positive");
            return ValidationStatus::Corrupt;
        }
        clipSampleTotal += count;
    }
    if (clipSampleTotal != sampleCount)
    {
        reason = TEXT("clip sample counts do not sum to the sample count");
        return ValidationStatus::Corrupt;
    }
    for (int32 clipIndex : _sampleClipIndices)
    {
        if (clipIndex < 0 || clipIndex >= clipCount)
        {
            reason = TEXT("a sample clip index is out of range");
            return ValidationStatus::Corrupt;
        }
    }

    // The schema snapshot must be usable by the runtime query builder.
    String schemaError;
    if (!_schema.TryValidate(schemaError))
    {
        reason = String::Format(TEXT("baked schema is invalid: {}"), schemaError);
        return ValidationStatus::Corrupt;
    }

    for (float deviation : _poseFeatureDeviations)
    {
        if (!(deviation > 0.0f))
        {
            reason = TEXT("a pose feature deviation is not strictly positive");
            return ValidationStatus::Corrupt;
        }
    }
    for (float deviation : _trajectoryFeatureDeviations)
    {
        if (!(deviation > 0.0f))
        {
            reason = TEXT("a trajectory feature deviation is not strictly positive");
            return ValidationStatus::Corrupt;
        }
    }

    // Baked arrays must be finite — a NaN/Inf stat would poison every query
    // cost and silently break pruning lower bounds.
    for (float value : _poseFeatureDeviations)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("pose feature deviation is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (float value : _trajectoryFeatureDeviations)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("trajectory feature deviation is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (float value : _poseFeatures)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("pose feature value is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (float value : _trajectoryFeatures)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("trajectory feature value is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (float value : _sampleTimes)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("sample time is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (float value : _clipLengths)
    {
        if (!std::isfinite(value))
        {
            reason = TEXT("clip length is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (const Float3& value : _rootPositions)
    {
        if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z))
        {
            reason = TEXT("root position is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (const Float3& value : _rootVelocities)
    {
        if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z))
        {
            reason = TEXT("root velocity is non-finite");
            return ValidationStatus::Corrupt;
        }
    }
    for (const Float3& value : _rootForwards)
    {
        if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z))
        {
            reason = TEXT("root forward is non-finite");
            return ValidationStatus::Corrupt;
        }
    }

    return ValidationStatus::Ok;
}

void MotionMatchingDatabase::GetSourceData(
    Array<Animation*>& animations,
    Array<Array<String>>& tags,
    Array<int32>& loopingClips) const
{
    animations.Resize(_sourceAnimations.Count());
    for (int32 i = 0; i < _sourceAnimations.Count(); i++)
        animations[i] = _sourceAnimations[i].Get();
    tags = _sourceTags;
    loopingClips = _sourceLooping;
}

void MotionMatchingDatabase::ClearBakedData()
{
    _normalizationRevision = 0;
    _featureLayoutHash = 0;
    _schemaContractRevision = 0;
    _samplingContractHash = 0;
    _sourceManifestHash = 0;
    _bakeToolBuildId = 0;
    _schema = PoseSearchSchema();
    _clipAnimations.Clear();
    _clipTags.Clear();
    _clipLooping.Clear();
    _clipTurnLike.Clear();
    _clipSpeedBands.Clear();
    _clipTurnAngles.Clear();
    _clipLengths.Clear();
    _clipSampleCounts.Clear();
    _sampleClipIndices.Clear();
    _sampleTimes.Clear();
    _poseFeatures.Clear();
    _trajectoryFeatures.Clear();
    _rootPositions.Clear();
    _rootVelocities.Clear();
    _rootForwards.Clear();
    _poseFeatureDeviations.Clear();
    _trajectoryFeatureDeviations.Clear();
}

void MotionMatchingDatabase::RebuildTagCounts()
{
    _tagNames.Clear();
    _tagClipCounts.Clear();
    for (const Array<String>& set : _sourceTags)
    {
        for (const String& tag : set)
        {
            const int32 index = _tagNames.Find(tag);
            if (index >= 0)
            {
                _tagClipCounts[index]++;
            }
            else
            {
                _tagNames.Add(tag);
                _tagClipCounts.Add(1);
            }
        }
    }
}

void MotionMatchingDatabase::InitAsVirtual()
{
    BinaryAsset::InitAsVirtual();
    unload(false);
}

uint64 MotionMatchingDatabase::GetMemoryUsage() const
{
    uint64 result = BinaryAsset::GetMemoryUsage();
    result += sizeof(MotionMatchingDatabase) - sizeof(BinaryAsset);
    result += _sourceAnimations.Capacity() * sizeof(AssetReference<Animation>);
    result += _sourceTags.Capacity() * sizeof(Array<String>);
    for (const Array<String>& set : _sourceTags)
        result += set.Capacity() * sizeof(String);
    result += _sourceLooping.Capacity() * sizeof(int32);
    result += _tagNames.Capacity() * sizeof(String);
    result += _tagClipCounts.Capacity() * sizeof(int32);
    result += _clipAnimations.Capacity() * sizeof(AssetReference<Animation>);
    result += _clipTags.Capacity() * sizeof(Array<String>);
    for (const Array<String>& set : _clipTags)
        result += set.Capacity() * sizeof(String);
    result += _clipLooping.Capacity() * sizeof(int32);
    result += _clipTurnLike.Capacity() * sizeof(int32);
    result += _clipSpeedBands.Capacity() * sizeof(int32);
    result += _clipTurnAngles.Capacity() * sizeof(float);
    result += _clipLengths.Capacity() * sizeof(float);
    result += _clipSampleCounts.Capacity() * sizeof(int32);
    result += _schema.TrajectorySampleTimes.Capacity() * sizeof(float);
    result += _sampleClipIndices.Capacity() * sizeof(int32);
    result += _sampleTimes.Capacity() * sizeof(float);
    result += _poseFeatures.Capacity() * sizeof(float);
    result += _trajectoryFeatures.Capacity() * sizeof(float);
    result += _rootPositions.Capacity() * sizeof(Float3);
    result += _rootVelocities.Capacity() * sizeof(Float3);
    result += _rootForwards.Capacity() * sizeof(Float3);
    result += _poseFeatureDeviations.Capacity() * sizeof(float);
    result += _trajectoryFeatureDeviations.Capacity() * sizeof(float);
    return result;
}

Asset::LoadResult MotionMatchingDatabase::load()
{
    const FlaxChunk* chunk = GetChunk(0);
    if (chunk == nullptr || chunk->IsMissing())
        return LoadResult::MissingDataChunk;

    MemoryReadStream stream(chunk->Get(), chunk->Size());
    _schema.Deserialize(stream);
    stream.Read(_sourceAnimations);
    stream.Read(_sourceTags);
    stream.Read(_sourceLooping);
    stream.Read(_tagNames);
    stream.Read(_tagClipCounts);
    stream.Read(_clipAnimations);
    stream.Read(_clipTags);
    stream.Read(_clipLooping);
    stream.Read(_clipLengths);
    stream.Read(_clipSampleCounts);
    stream.Read(_sampleClipIndices);
    stream.Read(_sampleTimes);
    stream.Read(_poseFeatures);
    stream.Read(_trajectoryFeatures);
    stream.Read(_rootPositions);
    stream.Read(_rootVelocities);
    stream.Read(_rootForwards);
    // Revision-3 binary-format compat: value means are dropped in memory, but the
    // retired means slot remains in the byte layout (written empty by revision-3
    // saves). Reading it into a throwaway keeps revision-2 files (poseMean,
    // poseDev, trajMean, trajDev) parseable all the way to the revision gate; the
    // provenance gate then drops their stale stats to registry-only. Means are
    // never used at runtime.
    Array<float> legacyPoseMeans;
    Array<float> legacyTrajectoryMeans;
    stream.Read(legacyPoseMeans);
    stream.Read(_poseFeatureDeviations);
    stream.Read(legacyTrajectoryMeans);
    stream.Read(_trajectoryFeatureDeviations);
    // Tail fields appended after the v2 layout: databases written before the turn-like tail end above,
    // so read them only when bytes remain (then ValidateData tolerates the
    // empty array and GetClipTurnLike derives from clip paths).
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.Read(_clipTurnLike);
    // Normalization revision tail: absent in older files (= 0).
    int32 storedRevision = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedRevision);
    _normalizationRevision = storedRevision;
    // Feature-layout hash tail: absent in older files (= 0 = unknown).
    int32 storedLayoutHash = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedLayoutHash);
    _featureLayoutHash = storedLayoutHash;
    // Provenance tails, appended after the layout hash. Files written before the
    // provenance tails have no bytes left here, so every field reads as 0 = unknown and
    // the database classifies registry-only stale (never crashes, never runs).
    int32 storedSchemaRevision = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedSchemaRevision);
    _schemaContractRevision = storedSchemaRevision;
    int32 storedSamplingHash = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedSamplingHash);
    _samplingContractHash = storedSamplingHash;
    int32 storedSourceHash = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedSourceHash);
    _sourceManifestHash = storedSourceHash;
    int32 storedBakeToolId = 0;
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.ReadInt32(&storedBakeToolId);
    _bakeToolBuildId = storedBakeToolId;
    // Per-clip speed bands (rev 2): absent in older files (empty = Any for
    // every clip, the pre-band behavior). The SchemaContractRevision gate
    // above already classifies such files registry-only.
    if (!stream.HasError() && stream.GetPosition() < stream.GetLength())
        stream.Read(_clipSpeedBands);

    // Three-way gate. Registry-only stale (older binary-format layout or a
    // provenance revision/hash mismatch) keeps the clip registry but drops the
    // baked arrays so the runtime never runs stale data. Corrupt data fails
    // the load outright — it is never ClearBakedData-then-continued as valid.
    if (!stream.HasError() && !_sampleTimes.IsEmpty())
    {
        String reason;
        const ValidationStatus status = ValidateDataStatus(reason);
        if (status == ValidationStatus::Corrupt)
        {
            LOG(
                Error,
                "Motion matching database '{}' is corrupt ({}). Refusing to load baked data.",
                GetPath(),
                reason);
            unload(false);
            return LoadResult::InvalidData;
        }
        if (status == ValidationStatus::RegistryOnlyStale)
        {
            LOG(
                Warning,
                "Motion matching database baked data is stale ({}); keeping {} registered clips, baked data cleared (REBAKE REQUIRED).",
                reason,
                _sourceAnimations.Count());
            ClearBakedData();
        }
    }

    ComputeClipTurnAngles();

    if (stream.HasError() || !ValidateData())
    {
        unload(false);
        return LoadResult::InvalidData;
    }

    return LoadResult::Ok;
}

void MotionMatchingDatabase::unload(bool isReloading)
{
    _sourceAnimations.Clear();
    _sourceTags.Clear();
    _sourceLooping.Clear();
    _tagNames.Clear();
    _tagClipCounts.Clear();
    ClearBakedData();
}

AssetChunksFlag MotionMatchingDatabase::getChunksToPreload() const
{
    return GET_CHUNK_FLAG(0);
}
