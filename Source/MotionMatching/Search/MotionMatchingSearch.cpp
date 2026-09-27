// MotionMatchingSearch.cpp: search backends (FindBest/SearchPass/IndexedSearchPass, FullSampleCost/PoseBlockCost/TrajectoryBlockCost/ContinuityTerm/LoopPreferenceTerm/TurnPenalty, ComputeClassBests, EnsureTagPartition/TagPartitionCache, SampleCost/FindTopCandidates).
// Ownership: plugin runtime, no game/host references.
// Key invariants: raw query/baked values score through one per-dimension scale pass (sqrt of sum-normalized weight divided by the stored deviation); every prior except the continuing-bias and loop-preference bonuses stays non-negative so block pruning with explicit headroom is exact; ties break to the lowest sample index; the indexed path retrieves top-K then exact-reranks with shared term helpers and falls back to brute force on any miss.
#include "MotionMatchingSearch.h"

#include "../Database/MotionMatchingDatabase.h"
#include "../Runtime/MotionMatchingSampler.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

namespace
{
    // Revision 3: baked and query features are RAW. The only stored stat is a
    // positive per-dim deviation (baker floors it at `dev > 0.1 ? dev : 1.0`),
    // so the divide floor below is only a last-resort guard and never
    // participates for valid databases.
    constexpr float MinimumStandardDeviation = 0.000001f;
    constexpr float SelfTestMaxCost = 0.0001f;

    // Straightness prior (Game-owned tuning via settings, values preserved
    // from the 0.03/4.0 calibration): penalty = |clipTurnAngle| * Weight /
    // (1 + yawRate * YawScale). Recorded in the trace; brute-force, indexed
    // and replay share this one definition.
    // Measured verdict: grouped L/R normalization did
    // NOT retire this — with weight 0, turn one-shots flood steady strafe/
    // diagonal phases (suite 2/12 vs 5-6/12). The penalty is load-bearing
    // for pool purity beyond asymmetry compensation, so it stays (rescaled
    // to rev-3 cost units below: same ratio against typical dissimilarity).
    // (Attempt 1 of the penalty-0 retune failed cleanly; code path kept.)
    // Rev-3 scale (Revision-3 shrank dissimilarity ~100x, measured mean 0.095):
    // 180deg -> +0.094 straight — same blocking ratio as the old +9.4.

    // Straightness prior, SINGLE definition shared by the hot loop, the
    // bit-exact verifier and the test gate below. Always >= 0 (fabs), so
    // block pruning stays valid with no extra headroom anywhere it is used.
    // weight/scale ride the per-query settings (Game tuning, trace-
    // recorded); callers pass settings fields verbatim, never literals.
    float TurnPenalty(float clipAngle, float queryYawRate, float weight, float yawScale)
    {
        const float a = clipAngle >= 0.0f ? clipAngle : -clipAngle;
        const float w = weight >= 0.0f ? weight : 0.0f;
        const float s = yawScale >= 0.0f ? yawScale : 0.0f;
        return a * w / (1.0f + Math::Max(queryYawRate, 0.0f) * s);
    }

    // foot-lock search prior deleted (FootLockCost + FootLockL/R +
    // FirmLockL/R). The runtime always fed it zero (verified: 4004/4004
    // trace entries flock=(0,0) firm=(0,0)), so removal is query-identical.
    // Baked contact channels + feature extraction + DB stay untouched;
    // Game-side IK owns its own contact state.

    // N1 weight contract: schema weights are stored, never baked. The search
    // applies each block weight exactly ONCE. Negative inputs clamp to 0;
    // non-finite settings are rejected by the FindBest gate before the hot
    // loop.
    void GetSearchBlockWeights(
        const MotionMatchingSearchSettings& settings,
        float& outPoseWeight,
        float& outTrajectoryWeight)
    {
        outPoseWeight = Math::Max(settings.PoseWeight, 0.0f);
        outTrajectoryWeight = Math::Max(settings.TrajectoryWeight, 0.0f);
    }

    // Weight-scale core shared by the live pipeline (below) and the index
    // tree space: sum-normalized sqrt block weight per dim / deviation.
    void BuildWeightScales(
        const float* poseDeviations,
        int32 poseFeatureSize,
        const float* trajectoryDeviations,
        int32 trajectoryFeatureSize,
        float poseWeight,
        float trajectoryWeight,
        float* poseScale,
        float* trajectoryScale)
    {
        const float weightSum = poseWeight * static_cast<float>(poseFeatureSize) +
            trajectoryWeight * static_cast<float>(trajectoryFeatureSize);
        const float poseRoot = weightSum > 0.0f ? std::sqrt(poseWeight / weightSum) : 0.0f;
        const float trajectoryRoot = weightSum > 0.0f ? std::sqrt(trajectoryWeight / weightSum) : 0.0f;
        for (int32 i = 0; i < poseFeatureSize; i++)
        {
            const float deviation = Math::Max(poseDeviations[i], MinimumStandardDeviation);
            poseScale[i] = poseRoot / deviation;
        }
        for (int32 i = 0; i < trajectoryFeatureSize; i++)
        {
            const float deviation = Math::Max(trajectoryDeviations[i], MinimumStandardDeviation);
            trajectoryScale[i] = trajectoryRoot / deviation;
        }
    }

    // Revision 3 weight pipeline (per-dimension scale pipeline): per-dim
    // base weight = PoseWeight for pose dims, TrajectoryWeight for trajectory
    // dims; sum-normalize across ALL dims; sqrt each; divide by the stored
    // per-dim deviation. The raw query and raw baked values are then scored as
    // (scale * (query - baked))^2. Baked features are never pre-scaled, so a
    // weight retune never needs a rebake. Shared by the hot pass, top-N,
    // SampleCost, the exact self-test probes and the verifier.
    void BuildQueryScales(
        const MotionMatchingDatabase& database,
        const MotionMatchingSearchSettings& settings,
        float* poseScale,
        float* trajectoryScale)
    {
        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const Array<float>& poseDeviations = database.GetPoseFeatureDeviations();
        const Array<float>& trajectoryDeviations = database.GetTrajectoryFeatureDeviations();
        float poseWeight = 0.0f;
        float trajectoryWeight = 0.0f;
        GetSearchBlockWeights(settings, poseWeight, trajectoryWeight);
        BuildWeightScales(
            poseDeviations.Get(), poseFeatureSize,
            trajectoryDeviations.Get(), trajectoryFeatureSize,
            poseWeight, trajectoryWeight,
            poseScale, trajectoryScale);
    }

    // Multi-tag membership: a clip passes when its tag SET overlaps the
    // allowed set. Empty allowed = all (unchanged contract).
    bool TagAllowed(const Array<String>& allowedTags, const Array<String>& clipTags)
    {
        if (allowedTags.IsEmpty())
            return true;
        for (const String& tag : clipTags)
        {
            if (allowedTags.Contains(tag))
                return true;
        }
        return false;
    }

    // Single-tag overload for the partition prefilter (partition t passes
    // when its tag name is in the allowed set).
    bool TagAllowed(const Array<String>& allowedTags, const String& tag)
    {
        return allowedTags.IsEmpty() || allowedTags.Contains(tag);
    }

    // Multi-tag exclusion: a candidate is dropped when its tag set shares
    // ANY tag with the exclude set. Empty exclude disables outright.
    bool TagExcluded(const Array<String>& excludeTags, const Array<String>& clipTags)
    {
        if (excludeTags.IsEmpty())
            return false;
        for (const String& tag : clipTags)
        {
            if (excludeTags.Contains(tag))
                return true;
        }
        return false;
    }

    // Tag-partitioned sample lists for the coarse prefilter. Built lazily on
    // first query per database (in-memory, no DB format change): one
    // sample-index list per baked tag. The per-tag KD-trees (phase 1-2) were
    // REMOVED after measurement -- exact 74-D trees barely prune (1.9x) and a
    // trajectory-only tier regressed recall; the linear block prefilter won
    // (sub-1ms, 0/1500 divergence). Brute force stays the reference backend
    // permanently (fallback + verify harness).
    // Reference shape: retrieval + exact rerank (per-tag block prefilter, then full scoring)
    // (PoseSearchDerivedData.cpp / PoseSearchIndex.cpp); orangeduck motion
    // matching (https://github.com/orangeduck/Motion-Matching).
    struct TagPartitionCache
    {
        const MotionMatchingDatabase* Db = nullptr;
        int32 SampleCount = -1;
        int32 ClipCount = -1;
        Array<String> TagNames;
        Array<Array<int32>> SamplesByTag;
        Array<int32> ClipsByTag;
    };

    TagPartitionCache& GetTagPartitionStore()
    {
        static TagPartitionCache cache;
        return cache;
    }

    // Scratch scratch buffers: IndexedSearchPass runs per-frame, so it must
    // not allocate per query (GC churn showed up as a fat qms tail). Single
    // game-thread caller (same discipline as the static partition cache);
    // buffers grow once to the high-water mark and are reused.
    struct CoarseEntry
    {
        float Dist = MAX_float;
        int32 Sample = -1;
    };

    struct IndexedScratch
    {
        Array<int32> Candidates;
        Array<CoarseEntry> Coarse;
    };

    IndexedScratch& GetIndexedScratch()
    {
        static IndexedScratch scratch;
        return scratch;
    }

    // Verify counters: the indexed path
    // must reproduce brute-force winners before any flip; during verify the
    // BRUTE-FORCE result always wins (caller overwrites), this only counts.
    // Benchmark (phase 4): verify also accumulates per-path native scan time
    // (indexed vs unstrided-brute on the SAME queries, back-to-back) so the
    // speedup is measured without policy/diagnostics noise.
    struct IndexedVerifyStats
    {
        int32 Verifies = 0;
        int32 Diverges = 0;
        int32 Fallbacks = 0;
        int32 Logged = 0;
        int32 SampleCount = -1;
        double IndexedMs = 0.0;
        double BruteMs = 0.0;
    };

    IndexedVerifyStats& GetIndexedVerifyStats()
    {
        static IndexedVerifyStats stats;
        return stats;
    }

    void VerifyIndexedResult(
        const MotionMatchingDatabase& database,
        bool indexedOk,
        const MotionMatchingResult& indexed,
        const MotionMatchingResult& reference,
        double indexedMs,
        double bruteMs)
    {
        IndexedVerifyStats& stats = GetIndexedVerifyStats();
        if (stats.SampleCount != database.GetSampleCount())
        {
            stats.Verifies = 0;
            stats.Diverges = 0;
            stats.Fallbacks = 0;
            stats.Logged = 0;
            stats.SampleCount = database.GetSampleCount();
            stats.IndexedMs = 0.0;
            stats.BruteMs = 0.0;
            LOG(Info, "MotionMatching indexed verify: armed on {} clips / {} samples; brute force wins during verify.",
                database.GetClipCount(), database.GetSampleCount());
        }
        if (!indexedOk || !indexed.IsValid)
        {
            stats.Fallbacks++;
            if (stats.Logged < 16)
            {
                stats.Logged++;
                LOG(Warning, "MotionMatching indexed verify: FALLBACK (indexed produced no winner, brute force sample={}).",
                    reference.SampleIndex);
            }
            return;
        }
        stats.Verifies++;
        const bool sameWinner = indexed.IsValid == reference.IsValid &&
            indexed.SampleIndex == reference.SampleIndex &&
            (!indexed.IsValid || Math::Abs(indexed.Cost - reference.Cost) <= 0.0001f);
        if (!sameWinner)
        {
            stats.Diverges++;
            if (stats.Logged < 16)
            {
                stats.Logged++;
                LOG(Warning, "MotionMatching indexed verify DIVERGE #{}: coarse sample={} clip={} cost={:.6f} vs brute sample={} clip={} cost={:.6f}.",
                    stats.Verifies,
                    indexed.SampleIndex,
                    indexed.ClipIndex,
                    indexed.Cost,
                    reference.SampleIndex,
                    reference.ClipIndex,
                    reference.Cost);
            }
        }
        if (stats.Verifies % 500 == 0)
        {
            const double avgIndexed = stats.Verifies > 0 ? stats.IndexedMs / (double)stats.Verifies : 0.0;
            const double avgBrute = stats.Verifies > 0 ? stats.BruteMs / (double)stats.Verifies : 0.0;
            LOG(Info, "MotionMatching indexed verify: {}/{} diverge ({} fallbacks) over {} queries; native scan avg indexed={:.3f}ms brute={:.3f}ms.",
                stats.Diverges, stats.Verifies, stats.Fallbacks, stats.Verifies + stats.Fallbacks,
                avgIndexed, avgBrute);
        }
        stats.IndexedMs += indexedMs;
        stats.BruteMs += bruteMs;
    }

    const TagPartitionCache& EnsureTagPartition(const MotionMatchingDatabase& database)
    {
        TagPartitionCache& cache = GetTagPartitionStore();
        const int32 sampleCount = database.GetSampleCount();
        const int32 clipCount = database.GetClipCount();
        if (cache.Db == &database && cache.SampleCount == sampleCount && cache.ClipCount == clipCount)
            return cache;

        const auto buildStart = std::chrono::high_resolution_clock::now();
        cache.Db = &database;
        cache.SampleCount = sampleCount;
        cache.ClipCount = clipCount;
        cache.TagNames = database.GetAnimationTags();
        const int32 tagCount = cache.TagNames.Count();
        cache.SamplesByTag.Resize(tagCount);
        cache.ClipsByTag.Resize(tagCount);
        for (int32 t = 0; t < tagCount; t++)
        {
            cache.SamplesByTag[t].Clear();
            cache.ClipsByTag[t] = 0;
        }
        const Array<int32>& sampleClipIndices = database.GetSampleClipIndices();
        for (int32 s = 0; s < sampleCount && s < sampleClipIndices.Count(); s++)
        {
            const int32 clipIndex = sampleClipIndices[s];
            if (clipIndex < 0 || clipIndex >= clipCount)
                continue;
            // Multi-tag: a sample belongs to EVERY tag in its clip's set.
            const Array<String>& set = database.GetClipTags(clipIndex);
            for (int32 t = 0; t < tagCount; t++)
            {
                if (set.Contains(cache.TagNames[t]))
                    cache.SamplesByTag[t].Add(s);
            }
        }
        // Clip counts per tag: one clip counts into every tag it carries.
        for (int32 c = 0; c < clipCount; c++)
        {
            const Array<String>& set = database.GetClipTags(c);
            for (int32 t = 0; t < tagCount; t++)
            {
                if (set.Contains(cache.TagNames[t]))
                    cache.ClipsByTag[t]++;
            }
        }
        const auto buildEnd = std::chrono::high_resolution_clock::now();
        const double buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
        int32 accounted = 0;
        for (int32 t = 0; t < tagCount; t++)
            accounted += cache.SamplesByTag[t].Count();
        LOG(Info, "MotionMatching tag partition: {} clips / {} samples / {} tags, built in {:.2f} ms ({} samples accounted).",
            clipCount, sampleCount, tagCount, buildMs, accounted);
        for (int32 t = 0; t < tagCount; t++)
        {
            LOG(Info, "MotionMatching tag partition: tag='{}' clips={} samples={}.",
                cache.TagNames[t], cache.ClipsByTag[t], cache.SamplesByTag[t].Count());
        }
        return cache;
    }

    float TimeDistance(const MotionMatchingDatabase& database, int32 clipIndex, float a, float b)
    {
        float distance = Math::Abs(a - b);
        if (clipIndex < 0 || !database.GetClipLoop(clipIndex))
            return distance;
        const Array<float>& lengths = database.GetClipLengths();
        if (clipIndex >= lengths.Count() || lengths[clipIndex] <= 0.0f)
            return distance;
        const float length = lengths[clipIndex];
        distance = Math::Mod(distance, length);
        return Math::Min(distance, length - distance);
    }

    // ---- Shared cost terms (one definition per term) ----
    // One definition per term. SearchPass (FindBest), FindTopCandidates,
    // FullSampleCost (SampleCost + the brute-force verifier) all build their
    // costs from these helpers, so no path can drift from another. The turn
    // term is the existing TurnPenalty above (foot-lock prior deleted).
    //
    // Pruning invariant kept: every prior is non-negative (turn +
    // continuity switch penalty) so a pruned candidate can never earn cost
    // back; the only negative terms (continuing bias, loop preference) get
    // explicit headroom added to the prune bound in SearchPass.

    // Pose block. With a finite pruneLimit it stops once the partial block
    // alone already exceeds the bound (remaining terms are >= 0). The scale
    // array already folds in sqrt(sum-normalized weight)/deviation, so the raw
    // query/baked delta is squared directly.
    float PoseBlockCost(
        const float* queryPose,
        const float* bakedPose,
        const float* poseScale,
        int32 poseFeatureSize,
        float pruneLimit)
    {
        float poseCost = 0.0f;
        for (int32 i = 0; i < poseFeatureSize; i++)
        {
            const float delta = poseScale[i] * (queryPose[i] - bakedPose[i]);
            poseCost += delta * delta;
            if (poseCost > pruneLimit)
                return poseCost;
        }
        return poseCost;
    }

    // Trajectory block, pruned against spentPoseCost + the bound.
    float TrajectoryBlockCost(
        const float* queryTrajectory,
        const float* bakedTrajectory,
        const float* trajectoryScale,
        int32 trajectoryFeatureSize,
        float spentPoseCost,
        float pruneLimit)
    {
        float trajectoryCost = 0.0f;
        for (int32 i = 0; i < trajectoryFeatureSize; i++)
        {
            const float delta = trajectoryScale[i] * (queryTrajectory[i] - bakedTrajectory[i]);
            trajectoryCost += delta * delta;
            if (spentPoseCost + trajectoryCost > pruneLimit)
                return trajectoryCost;
        }
        return trajectoryCost;
    }

    // Continuity block folds the switch penalty and the continuing bias so
    // Pose + Trajectory + Continuity equals Cost (telemetry + self-test).
    float ContinuityTerm(
        bool continuityApplies,
        bool clipDiffers,
        bool isContinuation,
        float continuityWeight,
        float bias)
    {
        if (continuityApplies && clipDiffers)
            return continuityWeight;
        if (isContinuation && bias > 0.0f)
            return -bias;
        return 0.0f;
    }

    // Rest-loop preference bonus (<= 0; SearchPass adds loopPreference to the
    // prune bound so a pruned candidate cannot lose to an unseen loop bonus).
    float LoopPreferenceTerm(bool isLoop, float loopPreference)
    {
        return (isLoop && loopPreference > 0.0f) ? -loopPreference : 0.0f;
    }

    // Natural continuation (same clip, near the expected playback time). One
    // definition shared by the hot loop and the exact reference.
    bool IsContinuationSample(
        const MotionMatchingDatabase& database,
        const MotionMatchingSearchSettings& settings,
        bool useTimeTerms,
        int32 currentClipIndex,
        int32 clipIndex,
        float sampleTime,
        float biasWindow)
    {
        return useTimeTerms && clipIndex == currentClipIndex && biasWindow > 0.0f &&
            TimeDistance(database, clipIndex, settings.ExpectedTime, sampleTime) <= biasWindow;
    }
    // One-shot progression guard (sprint->run fix): admissibility predicate
    // for the CURRENT non-loop clip. Returns true (candidate stays ranked)
    // for other clips, loop clips, and pure searches without playback
    // context (CurrentTime < 0 or ExpectedTime < 0). For the current
    // one-shot, only samples within ContinuingWindow of ExpectedTime are
    // admissible. ContinuingBias is deliberately NOT consulted: bias=0 must
    // still hold progression. No SampleRate fallback: a zero window admits
    // only the expected sample itself.
    bool IsNonLoopContinuation(
        const MotionMatchingDatabase& database,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        int32 clipIndex,
        float sampleTime)
    {
        if (clipIndex != currentClipIndex)
            return true;
        if (currentClipIndex < 0)
            return true;
        if (database.GetClipLoop(clipIndex))
            return true;
        if (settings.CurrentTime < 0.0f || settings.ExpectedTime < 0.0f)
            return true;
        const float window = Math::Max(settings.ContinuingWindow, 0.0f);
        return Math::Abs(sampleTime - settings.ExpectedTime) <= window;
    }

    // Horizon rule: furthest future trajectory target from the schema. A non-loop
    // sample whose future stencil reaches past the clip end baked a clamped
    // "stopped" future for playback; search must not rank it (see
    // MotionMatchingSampler::IsNonLoopFutureOutside). Loop clips return 0.
    float GetTrajectoryHorizon(const MotionMatchingDatabase& database)
    {
        float horizon = 0.0f;
        const Array<float>& times = database.GetTrajectorySampleTimes();
        for (int32 i = 0; i < times.Count(); i++)
            horizon = Math::Max(horizon, times[i]);
        return horizon;
    }

    // All-terms cost of one baked sample for a raw query. Pose + Trajectory +
    // Continuity/bias + turn + foot-lock + loop, built from the shared term
    // helpers. This is the exact reference the self-test verifier compares
    // FindBest against, and the body of the C# fast-path SampleCost (called
    // there with continuity/loop/foot disabled).
    float FullSampleCost(
        const MotionMatchingDatabase& database,
        const MotionMatchingSearchSettings& settings,
        const float* queryPose,
        const float* queryTrajectory,
        int32 currentClipIndex,
        int32 sampleIndex,
        bool applyTurnPenalty)
    {
        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const Array<float>& poseFeatures = database.GetPoseFeatures();
        const Array<float>& trajectoryFeatures = database.GetTrajectoryFeatures();
        const Array<int32>& sampleClipIndices = database.GetSampleClipIndices();
        const Array<float>& sampleTimes = database.GetSampleTimes();
        const int32 clipCount = database.GetClipCount();

        // One scale pass per reference call: weight pipeline is O(dims) and
        // this path is used by the exact verifier/SampleCost only.
        float stackPoseScale[PoseSearchSchema::GetPoseFeatureSize()];
        float stackTrajectoryScale[PoseSearchSchema::GetTrajectoryFeatureSize()];
        Array<float> heapPoseScale, heapTrajectoryScale;
        float* poseScale = stackPoseScale;
        float* trajectoryScale = stackTrajectoryScale;
        if (poseFeatureSize > PoseSearchSchema::GetPoseFeatureSize())
        {
            heapPoseScale.Resize(poseFeatureSize);
            poseScale = heapPoseScale.Get();
        }
        if (trajectoryFeatureSize > PoseSearchSchema::GetTrajectoryFeatureSize())
        {
            heapTrajectoryScale.Resize(trajectoryFeatureSize);
            trajectoryScale = heapTrajectoryScale.Get();
        }
        BuildQueryScales(database, settings, poseScale, trajectoryScale);

        const float continuityWeight = Math::Max(settings.ContinuityWeight, 0.0f);
        const bool continuityApplies = currentClipIndex >= 0 && currentClipIndex < clipCount;
        const bool useTimeTerms = continuityApplies && settings.ExpectedTime >= 0.0f;
        const float bias = Math::Max(settings.ContinuingBias, 0.0f);
        const float biasWindow = Math::Max(settings.ContinuingWindow, 0.0f);
        const float loopPreference = Math::Max(settings.LoopPreference, 0.0f);

        const int32 clipIndex = (sampleIndex >= 0 && sampleIndex < sampleClipIndices.Count())
            ? sampleClipIndices[sampleIndex]
            : -1;
        const float* bakedPose = &poseFeatures[sampleIndex * poseFeatureSize];
        const float* bakedTrajectory = &trajectoryFeatures[sampleIndex * trajectoryFeatureSize];
        const bool isContinuation = IsContinuationSample(
            database, settings, useTimeTerms, currentClipIndex, clipIndex,
            sampleTimes[sampleIndex], biasWindow);
        const float poseCost = PoseBlockCost(
            queryPose, bakedPose, poseScale, poseFeatureSize, MAX_float);
        const float trajectoryCost = TrajectoryBlockCost(
            queryTrajectory, bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
            poseCost, MAX_float);
        float continuityCost = ContinuityTerm(
            continuityApplies, clipIndex != currentClipIndex, isContinuation,
            continuityWeight, bias);
        if (applyTurnPenalty && settings.TurnPenaltyWeight > 0.0f)
            continuityCost += TurnPenalty(database.GetClipTurnAngle(clipIndex), settings.QueryYawRate,
                settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale);
        const float loopCost = LoopPreferenceTerm(database.GetClipLoop(clipIndex), loopPreference);
        return poseCost + trajectoryCost + continuityCost + loopCost;
    }

    // Array wrapper for the self-test exact probes (they build raw arrays).
    float FullSampleCost(
        const MotionMatchingDatabase& database,
        const Array<float>& queryPose,
        const Array<float>& queryTrajectory,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        int32 sampleIndex,
        bool applyTurnPenalty)
    {
        return FullSampleCost(
            database, settings, queryPose.Get(), queryTrajectory.Get(),
            currentClipIndex, sampleIndex, applyTurnPenalty);
    }

    // Exact per-class minima over the tag+filter scope: full scan, no stride,
    // no pruning, raw pose+trajectory blocks only (no continuity, bias,
    // preference or margin -- this is the tune surface for those terms, so it
    // must not contain them). Respects the loop filter (a filtered-out class
    // reports untouched init values) and tags; ignores lock, exclusion and
    // stride, which are playback concerns, not tune data. Costs ~one extra
    // pruned scan per query; the hot path stays pruned.
    void ComputeClassBests(
        const MotionMatchingDatabase& database,
        const float* queryPose,
        const float* queryTrajectory,
        const float* poseScale,
        const float* trajectoryScale,
        const Array<String>& allowedTags,
        const Array<String>& excludeTags,
        MotionLoopFilter loopFilter,
        int32 oneShotBand,
        float& bestLoopCost,
        int32& bestLoopSample,
        int32& bestLoopClip,
        int32& loopTotalCount,
        float& bestOneShotCost,
        int32& bestOneShotSample,
        int32& bestOneShotClip,
        int32& oneShotTotalCount)
    {
        bestLoopCost = MAX_float;
        bestLoopSample = -1;
        bestLoopClip = -1;
        loopTotalCount = 0;
        bestOneShotCost = MAX_float;
        bestOneShotSample = -1;
        bestOneShotClip = -1;
        oneShotTotalCount = 0;

        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const int32 sampleCount = database.GetSampleCount();
        const int32 clipCount = database.GetClipCount();
        const Array<int32>& sampleClipIndices = database.GetSampleClipIndices();
        const Array<float>& poseFeatures = database.GetPoseFeatures();
        const Array<float>& trajectoryFeatures = database.GetTrajectoryFeatures();

        int32 cachedClip = -1;
        bool cachedLoop = false;
        int32 cachedBand = 0;
        // Tags are per-clip; samples arrive clip-ordered, so resolve the
        // tag check once per clip instead of per sample. Empty include scope
        // AND empty exclude set disable the check outright.
        const bool checkTags = !allowedTags.IsEmpty() || !excludeTags.IsEmpty();
        bool cachedTagOk = true;
        for (int32 sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            const int32 clipIndex = sampleClipIndices[sampleIndex];
            if (clipIndex < 0 || clipIndex >= clipCount)
                continue;
            if (clipIndex != cachedClip)
            {
                cachedClip = clipIndex;
                cachedLoop = database.GetClipLoop(clipIndex);
                cachedBand = database.GetClipSpeedBand(clipIndex);
                const Array<String>& clipTags = database.GetClipTags(clipIndex);
                cachedTagOk = !checkTags ||
                    (TagAllowed(allowedTags, clipTags) && !TagExcluded(excludeTags, clipTags));
            }
            if (loopFilter == MotionLoopFilter::LoopOnly && !cachedLoop)
                continue;
            if (loopFilter == MotionLoopFilter::OneShotOnly && cachedLoop)
                continue;
            if (oneShotBand != 0 && !cachedLoop && cachedBand != 0 &&
                cachedBand != oneShotBand)
                continue;
            if (!cachedTagOk)
                continue;

            // Raw blocks through the shared term helpers (no prior, no prune):
            // this is the class tune surface and must match FullSampleCost's
            // pose/trajectory terms exactly.
            const float* bakedPose = &poseFeatures[sampleIndex * poseFeatureSize];
            const float* bakedTrajectory = &trajectoryFeatures[sampleIndex * trajectoryFeatureSize];
            const float poseCost = PoseBlockCost(
                queryPose, bakedPose, poseScale, poseFeatureSize, MAX_float);
            const float trajectoryCost = TrajectoryBlockCost(
                queryTrajectory, bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
                poseCost, MAX_float);
            const float cost = poseCost + trajectoryCost;

            // Strictly-less keeps the deterministic lowest-index tie-break.
            if (cachedLoop)
            {
                loopTotalCount++;
                if (cost < bestLoopCost)
                {
                    bestLoopCost = cost;
                    bestLoopSample = sampleIndex;
                    bestLoopClip = clipIndex;
                }
            }
            else
            {
                oneShotTotalCount++;
                if (cost < bestOneShotCost)
                {
                    bestOneShotCost = cost;
                    bestOneShotSample = sampleIndex;
                    bestOneShotClip = clipIndex;
                }
            }
        }
    }

    // Lightweight per-query readiness check. Full ValidateData() walks every
    // sample/tag array (62k samples in the GASP DB) and WaitForLoaded() blocks
    // the game thread by default -- both are wrong on the 10 Hz hot path.
    // The editor self-test and bake still run full ValidateData().
    bool IsReadyForQuery(const MotionMatchingDatabase& database, String& error)
    {
        if (database.WaitForLoaded(0))
        {
            error = TEXT("Motion matching search skipped: database is still loading.");
            return false;
        }
        if (!database.IsBaked())
        {
            error = TEXT("Motion matching search failed: database has no baked samples. Bake it first.");
            return false;
        }
        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const int32 sampleCount = database.GetSampleCount();
        const int32 clipCount = database.GetClipCount();
        if (sampleCount <= 0 || clipCount <= 0 ||
            database.GetPoseFeatures().Count() != sampleCount * poseFeatureSize ||
            database.GetTrajectoryFeatures().Count() != sampleCount * trajectoryFeatureSize ||
            database.GetPoseFeatureDeviations().Count() != poseFeatureSize ||
            database.GetTrajectoryFeatureDeviations().Count() != trajectoryFeatureSize ||
            database.GetSampleClipIndices().Count() != sampleCount ||
            database.GetSampleTimes().Count() != sampleCount)
        {
            error = TEXT("Motion matching search failed: database dimensions are inconsistent (rebake required).");
            return false;
        }
        return true;
    }

    void SearchPass(
        const MotionMatchingDatabase& database,
        const float* queryPose,
        const float* queryTrajectory,
        const float* poseScale,
        const float* trajectoryScale,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        bool lockSwitch,
        const Array<String>& allowedTags,
        int32 stride,
        MotionMatchingResult& result)
    {
        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const int32 sampleCount = database.GetSampleCount();
        const int32 clipCount = database.GetClipCount();
        const Array<int32>& sampleClipIndices = database.GetSampleClipIndices();
        const Array<float>& sampleTimes = database.GetSampleTimes();
        const Array<float>& poseFeatures = database.GetPoseFeatures();
        const Array<float>& trajectoryFeatures = database.GetTrajectoryFeatures();

        const float continuityWeight = Math::Max(settings.ContinuityWeight, 0.0f);
        const bool continuityApplies = currentClipIndex >= 0 && currentClipIndex < clipCount;
        const bool useExclusion = continuityApplies && settings.CurrentTime >= 0.0f &&
            settings.SameClipExclusionWindow > 0.0f;
        const float exclusionWindow = Math::Max(settings.SameClipExclusionWindow, 0.0f);
        // Time-based terms need a current clip AND a valid expectation.
        // Without playback (ExpectedTime < 0) the search is pure cost.
        const bool useTimeTerms = continuityApplies && settings.ExpectedTime >= 0.0f;
        const float bias = Math::Max(settings.ContinuingBias, 0.0f);
        const float biasWindow = Math::Max(settings.ContinuingWindow, 0.0f);
        const float switchMargin = Math::Max(settings.SwitchMargin, 0.0f);
        const float loopPreference = Math::Max(settings.LoopPreference, 0.0f);
        // Exclusion bans (all exclusion-only: they can never lower a
        // cost, so block pruning needs no extra headroom).
        const float twinBan = Math::Max(settings.TwinBanWindow, 0.0f);
        const float tailExclusion = Math::Max(settings.LoopTailExclusion, 0.0f);
        const bool useReselectBan = settings.ReselectClip >= 0 && settings.ReselectClip < clipCount &&
            settings.ReselectBanWindow > 0.0f;
        const float reselectWindow = Math::Max(settings.ReselectBanWindow, 0.0f);
        const Array<float>& clipLengths = database.GetClipLengths();
        const bool haveLengths = clipLengths.Count() == clipCount;
        // Structural-validity rule: non-loop candidates whose future
        // stencil reaches past the clip end baked a clamped "stopped" future,
        // so exclude them from ranking. Structural, never a tunable; loop
        // clips are cycle-aware and unaffected.
        const float trajectoryHorizon = GetTrajectoryHorizon(database);
        const bool excludeNonLoopTail = settings.ExcludeNonLoopTail;
        // Per-frame rotating stride: consecutive queries cover disjoint
        // subsets ((frame % stride) start) so full coverage completes over
        // `stride` frames instead of quantizing on a cadence. Deterministic:
        // the cursor rides settings.QueryFrame (policy frame counter, reset
        // per play session). Continuation samples are exempt below -- the hold
        // must survive every frame regardless of the cursor. (Rebuild tick.)
        const int32 queryFrame = settings.QueryFrame >= 0 ? settings.QueryFrame : 0;
        const int32 strideOffset = (stride > 1) ? (queryFrame % stride) : 0;

        float bestCost = MAX_float;
        float secondBestCost = MAX_float;
        float bestPoseCost = MAX_float;
        float bestTrajectoryCost = MAX_float;
        float bestContinuityCost = 0.0f;
        float bestLoopCost = 0.0f;
        int32 bestSample = -1;
        // Best in-window same-clip continuation (final cost, bias included).
        float bestContinueCost = MAX_float;
        float bestContinuePoseCost = 0.0f;
        float bestContinueTrajectoryCost = 0.0f;
        float bestContinueContinuityCost = 0.0f;
        float bestContinueLoopCost = 0.0f;
        int32 bestContinueSample = -1;
        // Loop flag is per-clip; samples arrive clip-ordered, so resolve once
        // per clip instead of per sample (one native call per clip total).
        // Loop-mode prefilter runs before tags: cheapest structural cut.
        // Tags resolve on the same clip-change tick (one String copy +
        // compare per clip); empty scope disables the check outright.
        const bool checkTags = !allowedTags.IsEmpty() || !settings.ExcludeTags.IsEmpty();
        int32 cachedLoopClip = -1;
        bool cachedLoopFlag = false;
        bool cachedTagOk = true;
        float cachedTurnAngle = 0.0f;
        int32 cachedSpeedBand = 0;
        // Start of the non-searchable tail for the cached clip (-1 = none).
        float cachedTailStart = -1.0f;
        const bool wantLoop = settings.LoopFilter == MotionLoopFilter::LoopOnly;
        const bool wantOneShot = settings.LoopFilter == MotionLoopFilter::OneShotOnly;
        // Onset gait filter: restrict ONE-SHOTs to the intended band (loops
        // never band-cut). 0 = Any.
        const int32 oneShotBand = settings.OneShotBand;
        // Scored-member counts per class (completed both blocks). Pruned
        // members never land here. Class MINIMA come from ComputeClassBests
        // (unpruned full scan), never from this loop.
        int32 loopScoredCount = 0;
        int32 oneShotScoredCount = 0;

        for (int32 sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            const int32 clipIndex = sampleClipIndices[sampleIndex];
            if (clipIndex < 0 || clipIndex >= clipCount)
                continue;
            if (clipIndex != cachedLoopClip)
            {
                cachedLoopClip = clipIndex;
                cachedLoopFlag = database.GetClipLoop(clipIndex);
                cachedTurnAngle = database.GetClipTurnAngle(clipIndex);
                cachedSpeedBand = database.GetClipSpeedBand(clipIndex);
                const Array<String>& clipTagSet = database.GetClipTags(clipIndex);
                cachedTagOk = !checkTags || (TagAllowed(allowedTags, clipTagSet) && !TagExcluded(settings.ExcludeTags, clipTagSet));
                // Loop-tail cutoff for this clip (non-loop only, runtime from
                // lengths + sample times, no rebake). Clips shorter than the
                // exclusion keep full scope so short one-shots stay winnable.
                cachedTailStart = -1.0f;
                if (tailExclusion > 0.0f && !cachedLoopFlag && haveLengths &&
                    clipLengths[clipIndex] > tailExclusion)
                    cachedTailStart = clipLengths[clipIndex] - tailExclusion;
            }
            if ((wantLoop && !cachedLoopFlag) || (wantOneShot && cachedLoopFlag))
                continue;
            // Onset gait filter: one-shots outside the intended band never
            // rank (at the onset instant the features carry almost no gait
            // information, so walk/run starts would otherwise coin-flip).
            if (oneShotBand != 0 && !cachedLoopFlag && cachedSpeedBand != 0 &&
                cachedSpeedBand != oneShotBand)
                continue;
            if (lockSwitch && clipIndex != currentClipIndex)
                continue;
            if (!cachedTagOk)
                continue;
            // Loop-tail non-searchable (tail exclusion):
            // end stances of one-shots can never re-win on exact pose match.
            if (cachedTailStart >= 0.0f && sampleTimes[sampleIndex] > cachedTailStart)
                continue;
            // Structural-validity rule: a non-loop future target outside [0, length] is
            // excluded (T1 helper), never ranked: the baked feature is a
            // clamped "stopped" future. Loop is unaffected (cycle-aware).
            if (excludeNonLoopTail && haveLengths &&
                MotionMatchingSampler::IsNonLoopFutureOutside(
                    sampleTimes[sampleIndex], trajectoryHorizon, clipLengths[clipIndex], cachedLoopFlag))
                continue;
            // Reselect-history ban (reselect history): the recently
            // played (clip,sample) neighborhood cannot re-win. Kills A->B->A
            // chatter between near-twins; legitimate returns at another phase
            // (outside the window) still win fairly.
            if (useReselectBan && clipIndex == settings.ReselectClip &&
                TimeDistance(database, clipIndex, settings.ReselectSampleTime, sampleTimes[sampleIndex]) <= reselectWindow)
                continue;
            // Natural continuation: same clip, near the expected playback
            // time. Exempt from temporal exclusion (it IS the frame playback
            // is heading to, not a reseek) and rewarded with the bias.
            const bool isContinuation = IsContinuationSample(
                database, settings, useTimeTerms, currentClipIndex, clipIndex,
                sampleTimes[sampleIndex], biasWindow);
            // Rotating-stride cursor: skip non-continuation samples outside
            // this query's subset. Continuations are exempt so the hold never
            // flickers on subset rotation; everything else is covered over
            // `stride` consecutive queries.
            if (stride > 1 && !isContinuation &&
                (sampleIndex % stride) != strideOffset)
                continue;
            // Same-segment twin ban (twin-jump threshold): same-clip
            // non-continuations just outside the continuation core (near-twin
            // frames of where playback is heading) are skipped. The true
            // continuation above stays exempt, so steady play-through is
            // unaffected -- only hops to your own twin frames die.
            if (useTimeTerms && twinBan > 0.0f && clipIndex == currentClipIndex && !isContinuation &&
                TimeDistance(database, clipIndex, settings.ExpectedTime, sampleTimes[sampleIndex]) <= biasWindow + twinBan)
                continue;
            // Temporal exclusion: same-clip samples too close in time to the
            // current playback position can never win. Without this, a query
            // every 0.1s reseeks to a nearby frame on the same clip (often the
            // same phase +/-1 sample) and playback micro-stutters forever instead
            // of playing through. Real discontinuities (large jumps, clip
            // changes, end recovery) are outside the window and unaffected.
            if (useExclusion && clipIndex == currentClipIndex && !isContinuation)
            {
                if (TimeDistance(database, clipIndex, settings.CurrentTime, sampleTimes[sampleIndex]) <= exclusionWindow)
                    continue;
            }
            // One-shot progression: far same-clip one-shot samples are not
            // admissible (only the natural continuation is). Loops, other
            // clips and pure searches pass through (see IsNonLoopContinuation).
            if (!IsNonLoopContinuation(
                    database, settings, currentClipIndex, clipIndex,
                    sampleTimes[sampleIndex]))
                continue;

            // Query is pre-normalized once by the caller: baked features are
            // already z-scored, so this is a plain squared distance. Prune
            // against secondBestCost (not bestCost): pruning at bestCost
            // discards candidates that could still be the correct second-best,
            // which made SecondBestCost telemetry wrong whenever the true
            // runner-up shared the pose block with the winner.
            // Bias+margin+loop-aware pruning: a continuation's final cost can
            // dip below its block sum by up to bias, the switch margin lets it
            // win even when it is not top-2 on blocks alone, and loop clips
            // gain up to loopPreference -- so those prune limits are raised by
            // exactly the applicable bonuses (see margin post-mortem: without
            // headroom an early exact winner drops secondBest near zero and
            // the arbitration never sees the candidates). Others keep tight.
            const float pruneLimit = secondBestCost +
                (isContinuation ? (bias > 0.0f ? bias : 0.0f) + switchMargin : 0.0f) +
                (cachedLoopFlag && loopPreference > 0.0f ? loopPreference : 0.0f);
            const float* bakedPose = &poseFeatures[sampleIndex * poseFeatureSize];
            const float* bakedTrajectory = &trajectoryFeatures[sampleIndex * trajectoryFeatureSize];
            // Shared terms: the pose block alone can already disqualify
            // against the runner-up threshold because every remaining term is
            // non-negative except the bias/loop bonuses, whose headroom is in
            // pruneLimit above.
            const float poseCost = PoseBlockCost(
                queryPose, bakedPose, poseScale, poseFeatureSize, pruneLimit);
            if (poseCost > pruneLimit)
                continue;
            const float trajectoryCost = TrajectoryBlockCost(
                queryTrajectory, bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
                poseCost, pruneLimit);
            if (poseCost + trajectoryCost > pruneLimit)
                continue;

            // Continuity block folds penalty AND bias so Pose + Trajectory +
            // Continuity still equals Cost (self-test invariant + telemetry).
            // Straightness + foot-lock priors fold in too (both >= 0, so
            // pruning stays valid with no extra headroom).
            float continuityCost = ContinuityTerm(
                continuityApplies, clipIndex != currentClipIndex, isContinuation,
                continuityWeight, bias);
            if (settings.TurnPenaltyWeight > 0.0f)
                continuityCost += TurnPenalty(cachedTurnAngle, settings.QueryYawRate,
                    settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale);
            // Rest-loop bonus lives in its own telemetry block (a loop clip at
            // rest outranks one-shot end stances structurally, not via continuity).
            const float loopCost = LoopPreferenceTerm(cachedLoopFlag, loopPreference);
            const float cost = poseCost + trajectoryCost + continuityCost + loopCost;

            // Class COUNTS only: how many members of each class survived
            // pruning to full scoring. The best-per-class VALUES come from
            // ComputeClassBests (unpruned full scan below) — a pruned member's
            // true cost is unknowable here, so this loop must not report minima.
            if (cachedLoopFlag)
                loopScoredCount++;
            else
                oneShotScoredCount++;

            // Strictly-less comparison: the first (lowest-index) sample wins
            // ties, which keeps replay deterministic down to the sample.
            if (cost < bestCost)
            {
                secondBestCost = bestCost;
                bestCost = cost;
                bestSample = sampleIndex;
                bestPoseCost = poseCost;
                bestTrajectoryCost = trajectoryCost;
                bestContinuityCost = continuityCost;
                bestLoopCost = loopCost;
            }
            else if (cost < secondBestCost)
            {
                secondBestCost = cost;
            }
            if (isContinuation && cost < bestContinueCost)
            {
                bestContinueCost = cost;
                bestContinueSample = sampleIndex;
                bestContinuePoseCost = poseCost;
                bestContinueTrajectoryCost = trajectoryCost;
                bestContinueContinuityCost = continuityCost;
                bestContinueLoopCost = loopCost;
            }
        }

        // Switch margin: an other-clip winner replaces the best continuation
        // only by a clear margin. bestCost <= secondBestCost always holds, so
        // the demoted winner becomes the new second-best exactly.
        if (bestSample >= 0 && bestContinueSample >= 0 &&
            sampleClipIndices[bestSample] != currentClipIndex &&
            switchMargin > 0.0f && bestCost + switchMargin >= bestContinueCost)
        {
            secondBestCost = bestCost;
            bestCost = bestContinueCost;
            bestSample = bestContinueSample;
            bestPoseCost = bestContinuePoseCost;
            bestTrajectoryCost = bestContinueTrajectoryCost;
            bestContinuityCost = bestContinueContinuityCost;
            bestLoopCost = bestContinueLoopCost;
        }

        if (bestSample < 0)
            return;

        result.IsValid = true;
        result.SampleIndex = bestSample;
        result.ClipIndex = sampleClipIndices[bestSample];
        result.SampleTime = sampleTimes[bestSample];
        result.Cost = bestCost;
        result.SecondBestCost = secondBestCost;
        result.PoseCost = bestPoseCost;
        result.TrajectoryCost = bestTrajectoryCost;
        result.ContinuityCost = bestContinuityCost;
        result.LoopCost = bestLoopCost;
        // Class minima come from the unpruned full scan (exact over the
        // tag+filter scope), never from the pruned loop above.
        if (settings.CollectClassDiagnostics)
        {
            ComputeClassBests(
                database,
                queryPose,
                queryTrajectory,
                poseScale,
                trajectoryScale,
                allowedTags,
                settings.ExcludeTags,
                settings.LoopFilter,
                settings.OneShotBand,
                result.BestLoopCost,
                result.BestLoopSample,
                result.BestLoopClip,
                result.LoopTotalCount,
                result.BestOneShotCost,
                result.BestOneShotSample,
                result.BestOneShotClip,
                result.OneShotTotalCount);
        }
        result.LoopScoredCount = loopScoredCount;
        result.OneShotScoredCount = oneShotScoredCount;
    }

    // Coarse-prefilter query: linear block retrieval over allowed tags +
    // EXACT rerank with the same terms/filters as SearchPass (shared helpers
    // above: Pose/TrajectoryBlockCost, ContinuityTerm, TurnPenalty,
    // LoopPreferenceTerm, IsContinuationSample). Anything this
    // path cannot decide falls back to SearchPass (caller). Differences from
    // SearchPass, all deliberate:
    //  - scope = per-tag block top-512 pooled to a global top retrieve
    //    instead of the full scan; stride is IGNORED (full tag scope, better
    //    coverage).
    //  - playback-neighborhood safety samples are added explicitly so margin
    //    arbitration always sees the hold candidate even if retrieval missed
    //    it.
    //  - candidates are scored in ascending sample order so strictly-less
    //    tie-breaks match the brute-force scan exactly.
    // Returns true when a valid winner was scored (result filled).
    bool IndexedSearchPass(
        const MotionMatchingDatabase& database,
        const TagPartitionCache& partition,
        const float* queryPose,
        const float* queryTrajectory,
        const float* poseScale,
        const float* trajectoryScale,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        bool lockSwitch,
        const Array<String>& allowedTags,
        MotionMatchingResult& result)
    {
        result = MotionMatchingResult();

        const int32 poseFeatureSize = database.GetPoseFeatureSize();
        const int32 trajectoryFeatureSize = database.GetTrajectoryFeatureSize();
        const int32 sampleCount = database.GetSampleCount();
        const int32 clipCount = database.GetClipCount();
        if (sampleCount <= 0 || clipCount <= 0)
            return false;

        const Array<int32>& sampleClipIndices = database.GetSampleClipIndices();
        const Array<float>& sampleTimes = database.GetSampleTimes();
        const Array<float>& poseFeatures = database.GetPoseFeatures();
        const Array<float>& trajectoryFeatures = database.GetTrajectoryFeatures();

        // Scratch buffers (phase 4): no per-query allocation on this path.
        IndexedScratch& scratch = GetIndexedScratch();

        // Shared precompute (same gates as SearchPass).
        const float continuityWeight = Math::Max(settings.ContinuityWeight, 0.0f);
        const bool continuityApplies = currentClipIndex >= 0 && currentClipIndex < clipCount;
        const bool useExclusion = continuityApplies && settings.CurrentTime >= 0.0f &&
            settings.SameClipExclusionWindow > 0.0f;
        const float exclusionWindow = Math::Max(settings.SameClipExclusionWindow, 0.0f);
        const bool useTimeTerms = continuityApplies && settings.ExpectedTime >= 0.0f;
        const float bias = Math::Max(settings.ContinuingBias, 0.0f);
        const float biasWindow = Math::Max(settings.ContinuingWindow, 0.0f);
        const float switchMargin = Math::Max(settings.SwitchMargin, 0.0f);
        const float loopPreference = Math::Max(settings.LoopPreference, 0.0f);
        const float twinBan = Math::Max(settings.TwinBanWindow, 0.0f);
        const float tailExclusion = Math::Max(settings.LoopTailExclusion, 0.0f);
        const bool useReselectBan = settings.ReselectClip >= 0 && settings.ReselectClip < clipCount &&
            settings.ReselectBanWindow > 0.0f;
        const float reselectWindow = Math::Max(settings.ReselectBanWindow, 0.0f);
        const Array<float>& clipLengths = database.GetClipLengths();
        const bool haveLengths = clipLengths.Count() == clipCount;
        const float trajectoryHorizon = GetTrajectoryHorizon(database);
        const bool excludeNonLoopTail = settings.ExcludeNonLoopTail;
        const bool wantLoop = settings.LoopFilter == MotionLoopFilter::LoopOnly;
        const bool wantOneShot = settings.LoopFilter == MotionLoopFilter::OneShotOnly;
        const int32 oneShotBand = settings.OneShotBand;

        // Retrieval: trajectory-coarse prefilter (replaces the 74-D tree
        // query -- exact trees barely prune in high dimensions, measured 1.9x
        // only). Two cheap linear tiers over allowed-tag members:
        //   tier 1: exact pose+trajectory blocks with LIVE scales, per-tag
        //           top tier1Keep (same ordering as the rerank blocks, so no
        //           signal is ever dropped early -- a trajectory-only tier was
        //           tried and REGRESSED recall to 1.7% on flat trajectories);
        //   tier 2: global top retrieve ? unchanged full scoring loop below
        //           applies every filter + term. TopKCandidates keeps its
        //           meaning (x6 finalists).
        int32 retrieve = settings.TopKCandidates > 0 ? settings.TopKCandidates * 6 : 192;
        if (retrieve < 16)
            retrieve = 16;
        if (retrieve > 512)
            retrieve = 512;
        // Tier-1 width tracks the finalist set (measured: TopK16 ? 96/288
        // gives sub-1ms scans with 0/1500 divergence; the linear scan costs
        // the same either way, this only sizes the selection).
        int32 tier1Keep = retrieve * 3;
        if (tier1Keep < 128)
            tier1Keep = 128;
        if (tier1Keep > 768)
            tier1Keep = 768;

        Array<CoarseEntry>& coarse = scratch.Coarse;
        coarse.Clear();
        Array<int32>& candidates = scratch.Candidates;
        candidates.Clear();
        const int32 tagCount = partition.TagNames.Count();
        int32 rangeBase[32];
        int32 rangeKept[32];
        int32 rangeCount = 0;
        // Perf phase 2: per-tag proportional tier-1 keep. A flat keep gives
        // the 590-sample tag the same 288 slots as the 22144-sample tag
        // (49% vs 1.3% coverage). Distribute the same total budget
        // (tier1Keep x allowed tags) by member count with a floor so tiny
        // tags still surface winners; the global top-retrieve below and the
        // full linear tier-1 scan are unchanged.
        int32 tagMembers[32];
        int32 tagTotal = 0;
        for (int32 t = 0; t < tagCount && rangeCount < 32; t++)
        {
            if (!TagAllowed(allowedTags, partition.TagNames[t]))
                continue;
            const int32 n = partition.SamplesByTag[t].Count();
            if (n <= 0)
                continue;
            tagMembers[rangeCount] = n;
            tagTotal += n;
            rangeCount++;
        }
        const int32 keepBudget = tier1Keep * rangeCount;
        int32 tagCursor = 0;
        for (int32 t = 0; t < tagCount && tagCursor < rangeCount; t++)
        {
            if (!TagAllowed(allowedTags, partition.TagNames[t]))
                continue;
            const Array<int32>& members = partition.SamplesByTag[t];
            const int32 n = members.Count();
            if (n <= 0)
                continue;
            // Proportional share, floor 16 (tiny tags keep a voice), cap n.
            int32 k = tagTotal > 0
                ? (int32)(((int64)keepBudget * tagMembers[tagCursor]) / tagTotal)
                : tier1Keep;
            if (k < 16)
                k = 16;
            if (k > n)
                k = n;
            tagCursor++;
            const int32 base = coarse.Count();
            // Excluded clips sink to MAX so they never take a retrieval slot
            // that a valid candidate needs (rerank drops them regardless).
            const bool coarseExclude = !settings.ExcludeTags.IsEmpty();
            for (int32 m = 0; m < n; m++)
            {
                const int32 s = members[m];
                if (coarseExclude)
                {
                    const int32 cs = sampleClipIndices[s];
                    if (cs >= 0 && cs < clipCount &&
                        TagExcluded(settings.ExcludeTags, database.GetClipTags(cs)))
                    {
                        CoarseEntry e;
                        e.Dist = MAX_float;
                        e.Sample = s;
                        coarse.Add(e);
                        continue;
                    }
                }
                const float* bakedPose = &poseFeatures[s * poseFeatureSize];
                const float* bakedTrajectory = &trajectoryFeatures[s * trajectoryFeatureSize];
                const float pose = PoseBlockCost(queryPose, bakedPose, poseScale, poseFeatureSize, MAX_float);
                CoarseEntry e;
                e.Dist = pose + TrajectoryBlockCost(
                    queryTrajectory, bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
                    pose, MAX_float);
                e.Sample = s;
                coarse.Add(e);
            }
            CoarseEntry* begin = coarse.Get() + base;
            std::nth_element(begin, begin + k, begin + n,
                [](const CoarseEntry& a, const CoarseEntry& b) { return a.Dist < b.Dist; });
            rangeBase[tagCursor - 1] = base;
            rangeKept[tagCursor - 1] = k;
        }
        // Compact per-tag tops to the front (writes stay below reads: kept
        // totals never exceed the source range starts), then keep the global
        // top retrieve by the already-exact block score -- no recompute.
        int32 pooled = 0;
        for (int32 r = 0; r < rangeCount; r++)
        {
            CoarseEntry* src = coarse.Get() + rangeBase[r];
            for (int32 i = 0; i < rangeKept[r]; i++)
                coarse[pooled++] = src[i];
        }
        if (pooled > 0)
        {
            CoarseEntry* pool = coarse.Get();
            int32 k = retrieve < pooled ? retrieve : pooled;
            std::nth_element(pool, pool + k, pool + pooled,
                [](const CoarseEntry& a, const CoarseEntry& b) { return a.Dist < b.Dist; });
            for (int32 i = 0; i < k; i++)
                candidates.Add(pool[i].Sample);
        }

        // Continuation safety set: the playback-relevant neighborhood on the
        // current clip is always exactly ranked, so the hold + margin
        // arbitration survive retrieval misses (rank data: same-clip twins
        // reordered by turn/foot terms the tree space cannot see). Window =
        // bias + twin-ban + exclusion reach: excluded/banned members are
        // still scored then filtered, so widening is harmless.
        if (useTimeTerms && currentClipIndex >= 0 && currentClipIndex < clipCount)
        {
            const float safetyWindow = biasWindow + twinBan + exclusionWindow;
            if (safetyWindow > 0.0f)
            {
                const Array<int32>& clipSampleCounts = database.GetClipSampleCounts();
                if (currentClipIndex < clipSampleCounts.Count())
                {
                    int32 start = 0;
                    for (int32 c = 0; c < currentClipIndex; c++)
                        start += clipSampleCounts[c];
                    const int32 end = start + clipSampleCounts[currentClipIndex];
                    for (int32 s = start; s < end && s < sampleCount; s++)
                    {
                        if (TimeDistance(database, currentClipIndex, settings.ExpectedTime, sampleTimes[s]) <= safetyWindow)
                            candidates.Add(s);
                    }
                }
            }
        }
        if (candidates.Count() == 0)
            return false;

        // Ascending sample order: strictly-less tie-breaks match SearchPass.
        // Dedup in place (no resize: the tail past uniqueCount is ignored).
        std::sort(candidates.Get(), candidates.Get() + candidates.Count());
        int32 uniqueCount = 0;
        for (int32 i = 0; i < candidates.Count(); i++)
        {
            if (i == 0 || candidates[i] != candidates[i - 1])
                candidates[uniqueCount++] = candidates[i];
        }

        float bestCost = MAX_float;
        float secondBestCost = MAX_float;
        float bestPoseCost = MAX_float;
        float bestTrajectoryCost = MAX_float;
        float bestContinuityCost = 0.0f;
        float bestLoopCost = 0.0f;
        int32 bestSample = -1;
        float bestContinueCost = MAX_float;
        float bestContinuePoseCost = 0.0f;
        float bestContinueTrajectoryCost = 0.0f;
        float bestContinueContinuityCost = 0.0f;
        float bestContinueLoopCost = 0.0f;
        int32 bestContinueSample = -1;
        int32 cachedLoopClip = -1;
        bool cachedLoopFlag = false;
        float cachedTurnAngle = 0.0f;
        int32 cachedSpeedBand = 0;
        float cachedTailStart = -1.0f;
        // Tags resolve on the clip-change tick below (candidates arrive in
        // ascending sample order, still clip-grouped); empty scope skips it.
        const bool checkTags = !allowedTags.IsEmpty() || !settings.ExcludeTags.IsEmpty();
        bool cachedTagOk = true;
        int32 loopScoredCount = 0;
        int32 oneShotScoredCount = 0;

        for (int32 ci = 0; ci < uniqueCount; ci++)
        {
            const int32 sampleIndex = candidates[ci];
            const int32 clipIndex = sampleClipIndices[sampleIndex];
            if (clipIndex < 0 || clipIndex >= clipCount)
                continue;
            if (clipIndex != cachedLoopClip)
            {
                cachedLoopClip = clipIndex;
                cachedLoopFlag = database.GetClipLoop(clipIndex);
                cachedTurnAngle = database.GetClipTurnAngle(clipIndex);
                cachedSpeedBand = database.GetClipSpeedBand(clipIndex);
                const Array<String>& clipTagSet = database.GetClipTags(clipIndex);
                cachedTagOk = !checkTags || (TagAllowed(allowedTags, clipTagSet) && !TagExcluded(settings.ExcludeTags, clipTagSet));
                cachedTailStart = -1.0f;
                if (tailExclusion > 0.0f && !cachedLoopFlag && haveLengths &&
                    clipLengths[clipIndex] > tailExclusion)
                    cachedTailStart = clipLengths[clipIndex] - tailExclusion;
            }
            if ((wantLoop && !cachedLoopFlag) || (wantOneShot && cachedLoopFlag))
                continue;
            if (oneShotBand != 0 && !cachedLoopFlag && cachedSpeedBand != 0 &&
                cachedSpeedBand != oneShotBand)
                continue;
            if (lockSwitch && clipIndex != currentClipIndex)
                continue;
            if (!cachedTagOk)
                continue;
            if (cachedTailStart >= 0.0f && sampleTimes[sampleIndex] > cachedTailStart)
                continue;
            if (excludeNonLoopTail && haveLengths &&
                MotionMatchingSampler::IsNonLoopFutureOutside(
                    sampleTimes[sampleIndex], trajectoryHorizon, clipLengths[clipIndex], cachedLoopFlag))
                continue;
            if (useReselectBan && clipIndex == settings.ReselectClip &&
                TimeDistance(database, clipIndex, settings.ReselectSampleTime, sampleTimes[sampleIndex]) <= reselectWindow)
                continue;
            const bool isContinuation = IsContinuationSample(
                database, settings, useTimeTerms, currentClipIndex, clipIndex,
                sampleTimes[sampleIndex], biasWindow);
            if (useTimeTerms && twinBan > 0.0f && clipIndex == currentClipIndex && !isContinuation &&
                TimeDistance(database, clipIndex, settings.ExpectedTime, sampleTimes[sampleIndex]) <= biasWindow + twinBan)
                continue;
            if (useExclusion && clipIndex == currentClipIndex && !isContinuation)
            {
                if (TimeDistance(database, clipIndex, settings.CurrentTime, sampleTimes[sampleIndex]) <= exclusionWindow)
                    continue;
            }
            // One-shot progression (same guard as SearchPass): the indexed
            // rerank must rank what brute force ranks.
            if (!IsNonLoopContinuation(
                    database, settings, currentClipIndex, clipIndex,
                    sampleTimes[sampleIndex]))
                continue;

            const float pruneLimit = secondBestCost +
                (isContinuation ? (bias > 0.0f ? bias : 0.0f) + switchMargin : 0.0f) +
                (cachedLoopFlag && loopPreference > 0.0f ? loopPreference : 0.0f);
            const float* bakedPose = &poseFeatures[sampleIndex * poseFeatureSize];
            const float* bakedTrajectory = &trajectoryFeatures[sampleIndex * trajectoryFeatureSize];
            const float poseCost = PoseBlockCost(
                queryPose, bakedPose, poseScale, poseFeatureSize, pruneLimit);
            if (poseCost > pruneLimit)
                continue;
            const float trajectoryCost = TrajectoryBlockCost(
                queryTrajectory, bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
                poseCost, pruneLimit);
            if (poseCost + trajectoryCost > pruneLimit)
                continue;

            float continuityCost = ContinuityTerm(
                continuityApplies, clipIndex != currentClipIndex, isContinuation,
                continuityWeight, bias);
            if (settings.TurnPenaltyWeight > 0.0f)
                continuityCost += TurnPenalty(cachedTurnAngle, settings.QueryYawRate,
                    settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale);
        const float loopCost = LoopPreferenceTerm(cachedLoopFlag, loopPreference);
            const float cost = poseCost + trajectoryCost + continuityCost + loopCost;

            if (cachedLoopFlag)
                loopScoredCount++;
            else
                oneShotScoredCount++;

            if (cost < bestCost)
            {
                secondBestCost = bestCost;
                bestCost = cost;
                bestSample = sampleIndex;
                bestPoseCost = poseCost;
                bestTrajectoryCost = trajectoryCost;
                bestContinuityCost = continuityCost;
                bestLoopCost = loopCost;
            }
            else if (cost < secondBestCost)
            {
                secondBestCost = cost;
            }
            if (isContinuation && cost < bestContinueCost)
            {
                bestContinueCost = cost;
                bestContinueSample = sampleIndex;
                bestContinuePoseCost = poseCost;
                bestContinueTrajectoryCost = trajectoryCost;
                bestContinueContinuityCost = continuityCost;
                bestContinueLoopCost = loopCost;
            }
        }

        if (bestSample >= 0 && bestContinueSample >= 0 &&
            sampleClipIndices[bestSample] != currentClipIndex &&
            switchMargin > 0.0f && bestCost + switchMargin >= bestContinueCost)
        {
            secondBestCost = bestCost;
            bestCost = bestContinueCost;
            bestSample = bestContinueSample;
            bestPoseCost = bestContinuePoseCost;
            bestTrajectoryCost = bestContinueTrajectoryCost;
            bestContinuityCost = bestContinueContinuityCost;
            bestLoopCost = bestContinueLoopCost;
        }

        if (bestSample < 0)
            return false;

        result.IsValid = true;
        result.SampleIndex = bestSample;
        result.ClipIndex = sampleClipIndices[bestSample];
        result.SampleTime = sampleTimes[bestSample];
        result.Cost = bestCost;
        result.SecondBestCost = secondBestCost;
        result.PoseCost = bestPoseCost;
        result.TrajectoryCost = bestTrajectoryCost;
        result.ContinuityCost = bestContinuityCost;
        result.LoopCost = bestLoopCost;
        if (settings.CollectClassDiagnostics)
        {
            ComputeClassBests(
                database,
                queryPose,
                queryTrajectory,
                poseScale,
                trajectoryScale,
                allowedTags,
                settings.ExcludeTags,
                settings.LoopFilter,
                settings.OneShotBand,
                result.BestLoopCost,
                result.BestLoopSample,
                result.BestLoopClip,
                result.LoopTotalCount,
                result.BestOneShotCost,
                result.BestOneShotSample,
                result.BestOneShotClip,
                result.OneShotTotalCount);
        }
        result.LoopScoredCount = loopScoredCount;
        result.OneShotScoredCount = oneShotScoredCount;
        return true;
    }
}

void MotionMatchingSearch::FindTopCandidates(
    MotionMatchingDatabase* database,
    const Array<float>& queryPose,
    const Array<float>& queryTrajectory,
    const MotionMatchingSearchSettings& settings,
    int32 currentClipIndex,
    const Array<String>& allowedTags,
    int32 topCount,
    Array<float>& packed,
    String& error)
{
    packed.Clear();
    error = String::Empty;

    if (database == nullptr)
    {
        error = TEXT("Motion matching top-N failed: database is not loaded.");
        return;
    }
    if (!IsReadyForQuery(*database, error))
        return;

    // Partition stats: build (or reuse) the tag-partition stats. Return value unused:
    // the hot pass still scans everything; this only counts + logs.
    (void)EnsureTagPartition(*database);

    const int32 poseFeatureSize = database->GetPoseFeatureSize();
    const int32 trajectoryFeatureSize = database->GetTrajectoryFeatureSize();
    if (queryPose.Count() != poseFeatureSize || queryTrajectory.Count() != trajectoryFeatureSize)
    {
        error = TEXT("Motion matching top-N failed: query dimensions do not match schema.");
        return;
    }
    for (int32 i = 0; i < queryPose.Count(); i++)
    {
        if (!std::isfinite(queryPose[i]))
        {
            error = TEXT("Motion matching top-N failed: query pose contains a non-finite value.");
            return;
        }
    }
    for (int32 i = 0; i < trajectoryFeatureSize; i++)
    {
        if (!std::isfinite(queryTrajectory[i]))
        {
            error = TEXT("Motion matching top-N failed: query trajectory contains a non-finite value.");
            return;
        }
    }

    // Revision 3: raw query + one per-dim scale pass (same pipeline as FindBest).
    float stackPoseScale[PoseSearchSchema::GetPoseFeatureSize()];
    float stackTrajectoryScale[PoseSearchSchema::GetTrajectoryFeatureSize()];
    Array<float> heapPoseScale, heapTrajectoryScale;
    float* poseScale = stackPoseScale;
    float* trajectoryScale = stackTrajectoryScale;
    if (poseFeatureSize > PoseSearchSchema::GetPoseFeatureSize())
    {
        heapPoseScale.Resize(poseFeatureSize);
        poseScale = heapPoseScale.Get();
    }
    if (trajectoryFeatureSize > PoseSearchSchema::GetTrajectoryFeatureSize())
    {
        heapTrajectoryScale.Resize(trajectoryFeatureSize);
        trajectoryScale = heapTrajectoryScale.Get();
    }
    BuildQueryScales(*database, settings, poseScale, trajectoryScale);

    const int32 n = Math::Clamp(topCount, 1, 32);
    constexpr int32 CandidateStride = 10;
    // Fixed stack covers the max clamp above; larger requests are clamped, so
    // no heap path is needed here.
    float top[32 * CandidateStride];
    for (int32 i = 0; i < 32 * CandidateStride; i++)
        top[i] = MAX_float;
    int32 collected = 0;

    const int32 clipCount = database->GetClipCount();
    const int32 sampleCount = database->GetSampleCount();
    const Array<int32>& sampleClipIndices = database->GetSampleClipIndices();
    const Array<float>& sampleTimes = database->GetSampleTimes();
    const Array<float>& poseFeatures = database->GetPoseFeatures();
    const Array<float>& trajectoryFeatures = database->GetTrajectoryFeatures();

    const float continuityWeight = Math::Max(settings.ContinuityWeight, 0.0f);
    const bool continuityApplies = currentClipIndex >= 0 && currentClipIndex < clipCount;
    const bool useExclusion = continuityApplies && settings.CurrentTime >= 0.0f &&
        settings.SameClipExclusionWindow > 0.0f;
    const float exclusionWindow = Math::Max(settings.SameClipExclusionWindow, 0.0f);
    const bool useTimeTerms = continuityApplies && settings.ExpectedTime >= 0.0f;
    const float bias = Math::Max(settings.ContinuingBias, 0.0f);
    const float biasWindow = Math::Max(settings.ContinuingWindow, 0.0f);
    const float loopPreference = Math::Max(settings.LoopPreference, 0.0f);
    const bool wantLoop = settings.LoopFilter == MotionLoopFilter::LoopOnly;
    const bool wantOneShot = settings.LoopFilter == MotionLoopFilter::OneShotOnly;
    // Same Exclusion bans as the hot pass (top-N must rank what runtime plays).
    const float twinBan = Math::Max(settings.TwinBanWindow, 0.0f);
    const float tailExclusion = Math::Max(settings.LoopTailExclusion, 0.0f);
    const bool useReselectBan = settings.ReselectClip >= 0 && settings.ReselectClip < clipCount &&
        settings.ReselectBanWindow > 0.0f;
    const float reselectWindow = Math::Max(settings.ReselectBanWindow, 0.0f);
    const Array<float>& clipLengths = database->GetClipLengths();
    const bool haveLengths = clipLengths.Count() == clipCount;
    // Same structural-validity cut as the hot pass: top-N must
    // rank what runtime could actually play, so a non-loop clamped-future
    // candidate is excluded here too.
    const float trajectoryHorizon = GetTrajectoryHorizon(*database);
    const bool excludeNonLoopTail = settings.ExcludeNonLoopTail;
    const int32 oneShotBand = settings.OneShotBand;

    int32 cachedLoopClip = -1;
    bool cachedLoopFlag = false;
    float cachedTurnAngle = 0.0f;
    int32 cachedSpeedBand = 0;
    float cachedTailStart = -1.0f;
    // Tooling path too: tags resolve once per clip, not per sample.
    const bool checkTags = !allowedTags.IsEmpty();
    bool cachedTagOk = true;
    for (int32 sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        const int32 clipIndex = sampleClipIndices[sampleIndex];
        if (clipIndex < 0 || clipIndex >= clipCount)
            continue;
        if (clipIndex != cachedLoopClip)
        {
            cachedLoopClip = clipIndex;
            cachedLoopFlag = database->GetClipLoop(clipIndex);
            cachedTurnAngle = database->GetClipTurnAngle(clipIndex);
            cachedSpeedBand = database->GetClipSpeedBand(clipIndex);
            const Array<String>& clipTagSet = database->GetClipTags(clipIndex);
            cachedTagOk = !checkTags || (TagAllowed(allowedTags, clipTagSet) && !TagExcluded(settings.ExcludeTags, clipTagSet));
            cachedTailStart = -1.0f;
            if (tailExclusion > 0.0f && !cachedLoopFlag && haveLengths &&
                clipLengths[clipIndex] > tailExclusion)
                cachedTailStart = clipLengths[clipIndex] - tailExclusion;
        }
        if ((wantLoop && !cachedLoopFlag) || (wantOneShot && cachedLoopFlag))
            continue;
        if (oneShotBand != 0 && !cachedLoopFlag && cachedSpeedBand != 0 &&
            cachedSpeedBand != oneShotBand)
            continue;
        if (!cachedTagOk)
            continue;
        if (cachedTailStart >= 0.0f && sampleTimes[sampleIndex] > cachedTailStart)
            continue;
        if (excludeNonLoopTail && haveLengths &&
            MotionMatchingSampler::IsNonLoopFutureOutside(
                sampleTimes[sampleIndex], trajectoryHorizon, clipLengths[clipIndex], cachedLoopFlag))
            continue;
        if (useReselectBan && clipIndex == settings.ReselectClip &&
            TimeDistance(*database, clipIndex, settings.ReselectSampleTime, sampleTimes[sampleIndex]) <= reselectWindow)
            continue;
        const bool isContinuation = IsContinuationSample(
            *database, settings, useTimeTerms, currentClipIndex, clipIndex,
            sampleTimes[sampleIndex], biasWindow);
        if (useExclusion && clipIndex == currentClipIndex && !isContinuation)
        {
            if (TimeDistance(*database, clipIndex, settings.CurrentTime, sampleTimes[sampleIndex]) <= exclusionWindow)
                continue;
        }
        if (useTimeTerms && twinBan > 0.0f && clipIndex == currentClipIndex && !isContinuation &&
            TimeDistance(*database, clipIndex, settings.ExpectedTime, sampleTimes[sampleIndex]) <= biasWindow + twinBan)
            continue;
        // One-shot progression (same guard as the hot pass): top-N must
        // rank what runtime could actually play.
        if (!IsNonLoopContinuation(
                *database, settings, currentClipIndex, clipIndex,
                sampleTimes[sampleIndex]))
            continue;

        // Exact scan: no pruning here (tooling path, ~1ms for 68k samples),
        // so the shared term helpers run with an infinite prune bound.
        const float* bakedPose = &poseFeatures[sampleIndex * poseFeatureSize];
        const float* bakedTrajectory = &trajectoryFeatures[sampleIndex * trajectoryFeatureSize];
        const float poseCost = PoseBlockCost(
            queryPose.Get(), bakedPose, poseScale, poseFeatureSize, MAX_float);
        const float trajectoryCost = TrajectoryBlockCost(
            queryTrajectory.Get(), bakedTrajectory, trajectoryScale, trajectoryFeatureSize,
            poseCost, MAX_float);
        float continuityCost = ContinuityTerm(
            continuityApplies, clipIndex != currentClipIndex, isContinuation,
            continuityWeight, bias);
        if (settings.TurnPenaltyWeight > 0.0f)
            continuityCost += TurnPenalty(cachedTurnAngle, settings.QueryYawRate,
                settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale);
        const float loopCost = LoopPreferenceTerm(cachedLoopFlag, loopPreference);
        const float cost = poseCost + trajectoryCost + continuityCost + loopCost;

        // Sorted insertion, strictly-less: ties keep the lower sample index
        // because the scan itself is ascending.
        int32 slot = collected < n ? collected : n;
        while (slot > 0 && cost < top[(slot - 1) * CandidateStride + 3])
            slot--;
        if (slot >= n)
            continue;
        if (collected < n)
            collected++;
        for (int32 k = collected - 1; k > slot; k--)
        {
            for (int32 f = 0; f < CandidateStride; f++)
                top[k * CandidateStride + f] = top[(k - 1) * CandidateStride + f];
        }
        top[slot * CandidateStride + 0] = (float)sampleIndex;
        top[slot * CandidateStride + 1] = (float)clipIndex;
        top[slot * CandidateStride + 2] = sampleTimes[sampleIndex];
        top[slot * CandidateStride + 3] = cost;
        top[slot * CandidateStride + 4] = poseCost;
        top[slot * CandidateStride + 5] = trajectoryCost;
        top[slot * CandidateStride + 6] = continuityCost;
        top[slot * CandidateStride + 7] = loopCost;
        top[slot * CandidateStride + 8] = isContinuation ? 1.0f : 0.0f;
        top[slot * CandidateStride + 9] = cachedLoopFlag ? 1.0f : 0.0f;
    }

    packed.Resize(collected * CandidateStride);
    for (int32 i = 0; i < collected * CandidateStride; i++)
        packed[i] = top[i];
    if (collected == 0)
        error = TEXT("Motion matching top-N found no valid candidate. Check tag filters.");
}

void MotionMatchingSearch::FindBest(
    MotionMatchingDatabase* database,
    const Array<float>& queryPose,
    const Array<float>& queryTrajectory,
    const MotionMatchingSearchSettings& settings,
    int32 currentClipIndex,
    float timeSinceSwitch,
    const Array<String>& allowedTags,
    MotionMatchingResult& result,
    String& error)
{
    result = MotionMatchingResult();
    error = String::Empty;

    if (database == nullptr)
    {
        error = TEXT("Motion matching search failed: database is not loaded.");
        return;
    }
    if (!IsReadyForQuery(*database, error))
        return;

    // Index bootstrap: tag-partition stats + per-tag KD-trees (built once per DB).
    const TagPartitionCache& partition = EnsureTagPartition(*database);

    const int32 poseFeatureSize = database->GetPoseFeatureSize();
    const int32 trajectoryFeatureSize = database->GetTrajectoryFeatureSize();
    if (queryPose.Count() != poseFeatureSize || queryTrajectory.Count() != trajectoryFeatureSize)
    {
        error = String::Format(
            TEXT("Motion matching search failed: query dimensions (pose={}, trajectory={}) do not match schema ({}, {})."),
            queryPose.Count(),
            queryTrajectory.Count(),
            poseFeatureSize,
            trajectoryFeatureSize);
        return;
    }
    for (int32 i = 0; i < queryPose.Count(); i++)
    {
        if (!std::isfinite(queryPose[i]))
        {
            error = TEXT("Motion matching search failed: query pose contains a non-finite value.");
            return;
        }
    }
    for (int32 i = 0; i < queryTrajectory.Count(); i++)
    {
        if (!std::isfinite(queryTrajectory[i]))
        {
            error = TEXT("Motion matching search failed: query trajectory contains a non-finite value.");
            return;
        }
    }
    if (!std::isfinite(settings.PoseWeight) || !std::isfinite(settings.TrajectoryWeight) ||
        !std::isfinite(settings.ContinuityWeight) || !std::isfinite(settings.MinSwitchTime) ||
        !std::isfinite(settings.CurrentTime) || !std::isfinite(settings.SameClipExclusionWindow) ||
        !std::isfinite(settings.ExpectedTime) || !std::isfinite(settings.ContinuingBias) ||
        !std::isfinite(settings.ContinuingWindow) || !std::isfinite(settings.SwitchMargin) ||
        !std::isfinite(settings.LoopPreference) || !std::isfinite(settings.QueryYawRate) ||
        !std::isfinite(settings.ReselectSampleTime) || !std::isfinite(settings.ReselectBanWindow) ||
        !std::isfinite(settings.TwinBanWindow) || !std::isfinite(settings.LoopTailExclusion))
    {
        error = TEXT("Motion matching search failed: settings contain a non-finite value.");
        return;
    }
    if (settings.LoopFilter != MotionLoopFilter::Any &&
        settings.LoopFilter != MotionLoopFilter::LoopOnly &&
        settings.LoopFilter != MotionLoopFilter::OneShotOnly)
    {
        error = TEXT("Motion matching search failed: unknown loop filter.");
        return;
    }

    const int32 sampleCount = database->GetSampleCount();
    // Brute-force reference scans everything by default (MaxCandidates <= 0).
    // A positive budget strides the scan; stride > 1 is an explicit quality
    // trade, never the default.
    int32 stride = 1;
    if (settings.MaxCandidates > 0 && settings.MaxCandidates < sampleCount)
        stride = (sampleCount + settings.MaxCandidates - 1) / settings.MaxCandidates;

    const int32 clipCount = database->GetClipCount();
    const bool hasCurrent = currentClipIndex >= 0 && currentClipIndex < clipCount;
    const bool lockSwitch = hasCurrent && timeSinceSwitch < settings.MinSwitchTime;

    // Revision 3: the query stays raw. Build one per-dim scale array
    // (sqrt(sum-normalized weight)/deviation) ONCE; the hot loop then squares
    // scale*(raw query - raw baked). Fixed stack caps cover the v3 layout
    // (44 + 30); larger future schemas fall back to heap. Shared with the
    // top-N/tooling path.
    float stackPoseScale[PoseSearchSchema::GetPoseFeatureSize()];
    float stackTrajectoryScale[PoseSearchSchema::GetTrajectoryFeatureSize()];
    Array<float> heapPoseScale, heapTrajectoryScale;
    float* poseScale = stackPoseScale;
    float* trajectoryScale = stackTrajectoryScale;
    if (poseFeatureSize > PoseSearchSchema::GetPoseFeatureSize())
    {
        heapPoseScale.Resize(poseFeatureSize);
        poseScale = heapPoseScale.Get();
    }
    if (trajectoryFeatureSize > PoseSearchSchema::GetTrajectoryFeatureSize())
    {
        heapTrajectoryScale.Resize(trajectoryFeatureSize);
        trajectoryScale = heapTrajectoryScale.Get();
    }
    BuildQueryScales(*database, settings, poseScale, trajectoryScale);
    const float* rawPose = queryPose.Get();
    const float* rawTrajectory = queryTrajectory.Get();

    // Indexed retrieval: indexed retrieval + exact rerank behind a default-off flag.
    // Falls back to the brute-force reference whenever it cannot score a
    // valid winner (missing tag tree, filters exhausting top-K, ...).
    // Verify: with VerifyIndexedSearch the reference ALSO runs, divergence
    // is counted to the log, and the BRUTE-FORCE winner is kept (behavior
    // unchanged during verify).
    MotionMatchingResult indexedResult;
    const bool wantVerify = settings.UseIndexedSearch && settings.VerifyIndexedSearch;
    bool indexedOk = false;
    double indexedMs = 0.0;
    double bruteMs = 0.0;
    if (settings.UseIndexedSearch)
    {
        const auto indexedStart = std::chrono::high_resolution_clock::now();
        indexedOk = IndexedSearchPass(
            *database,
            partition,
            rawPose,
            rawTrajectory,
            poseScale,
            trajectoryScale,
            settings,
            currentClipIndex,
            lockSwitch,
            allowedTags,
            indexedResult);
        if (wantVerify)
            indexedMs = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - indexedStart).count();
    }
    if (!indexedOk || wantVerify)
    {
        // Verify compares against the EXACT reference: stride would hide the
        // global best from brute force (by design) and fake divergence, so
        // the reference always scans unstrided here. The live strided path is
        // unaffected (wantVerify is false outside audits).
        const auto bruteStart = std::chrono::high_resolution_clock::now();
        SearchPass(
            *database,
            rawPose,
            rawTrajectory,
            poseScale,
            trajectoryScale,
            settings,
            currentClipIndex,
            lockSwitch,
            allowedTags,
            wantVerify ? 1 : stride,
            result);
        if (wantVerify)
            bruteMs = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - bruteStart).count();
    }
    else
    {
        result = indexedResult;
    }
    if (wantVerify)
        VerifyIndexedResult(*database, indexedOk, indexedResult, result, indexedMs, bruteMs);

    if (!result.IsValid && lockSwitch)
    {
        // Continuity lock filtered out every candidate (e.g. the allowed tags
        // exclude the current clip). Forced-interruption semantics: search
        // again with switching permitted rather than returning nothing.
        SearchPass(
            *database,
            rawPose,
            rawTrajectory,
            poseScale,
            trajectoryScale,
            settings,
            currentClipIndex,
            false,
            allowedTags,
            stride,
            result);
    }

    if (!result.IsValid)
        error = TEXT("Motion matching search found no valid candidate. Check tag filters and continuity settings.");
}

bool MotionMatchingSearch::SampleCost(
    MotionMatchingDatabase* database,
    const Array<float>& queryPose,
    const Array<float>& queryTrajectory,
    int32 sampleIndex,
    float poseWeight,
    float trajectoryWeight,
    float queryYawRate,
    bool applyTurnPenalty,
    float turnPenaltyWeight,
    float turnPenaltyYawScale,
    float& cost,
    String& error)
{
    cost = 0.0f;
    error = String::Empty;

    if (database == nullptr)
    {
        error = TEXT("Motion matching sample cost failed: database is not loaded.");
        return false;
    }
    if (!IsReadyForQuery(*database, error))
        return false;

    const int32 poseFeatureSize = database->GetPoseFeatureSize();
    const int32 trajectoryFeatureSize = database->GetTrajectoryFeatureSize();
    if (queryPose.Count() != poseFeatureSize || queryTrajectory.Count() != trajectoryFeatureSize)
    {
        error = TEXT("Motion matching sample cost failed: query dimensions do not match schema.");
        return false;
    }
    if (sampleIndex < 0 || sampleIndex >= database->GetSampleCount())
    {
        error = TEXT("Motion matching sample cost failed: sample index out of range.");
        return false;
    }
    for (int32 i = 0; i < queryPose.Count(); i++)
    {
        if (!std::isfinite(queryPose[i]))
        {
            error = TEXT("Motion matching sample cost failed: query pose contains a non-finite value.");
            return false;
        }
    }
    for (int32 i = 0; i < queryTrajectory.Count(); i++)
    {
        if (!std::isfinite(queryTrajectory[i]))
        {
            error = TEXT("Motion matching sample cost failed: query trajectory contains a non-finite value.");
            return false;
        }
    }

    // Term parity: the fast-path estimate uses the same FullSampleCost term
    // definitions as the hot pass, with no playback context (continuity/loop
    // disabled exactly as before: raw pose + trajectory + straightness).
    // turn prior weights ride explicit params (trace-recorded tuning).
    // foot-lock prior deleted (was zero here too).
    MotionMatchingSearchSettings costSettings;
    costSettings.PoseWeight = poseWeight;
    costSettings.TrajectoryWeight = trajectoryWeight;
    costSettings.QueryYawRate = queryYawRate;
    costSettings.TurnPenaltyWeight = turnPenaltyWeight;
    costSettings.TurnPenaltyYawScale = turnPenaltyYawScale;
    costSettings.ContinuityWeight = 0.0f;
    costSettings.CurrentTime = -1.0f;
    costSettings.ExpectedTime = -1.0f;
    costSettings.ContinuingBias = 0.0f;
    costSettings.LoopPreference = 0.0f;

    // Revision 3: raw query; FullSampleCost builds its own scale pass.
    cost = FullSampleCost(
        *database, costSettings, queryPose.Get(), queryTrajectory.Get(),
        -1, sampleIndex, applyTurnPenalty);
    return true;
}

// RunSelfTest implementation moved to MotionMatchingSearch.SelfTest.cpp
// (editor-only, #if USE_EDITOR guarded header + impl). Game Release contains
// no self-test code or strings. This translation unit keeps only the live
// search (FindBest/SampleCost/FindTopCandidates + indexed/scratch/verify).
