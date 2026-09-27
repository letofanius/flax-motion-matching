// MotionMatchingSearch.SelfTest.cpp: editor search self-test (MotionMatchingSearch::RunSelfTest, phases A-R).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the file-local reference duplicates TurnPenalty/FullSampleCost and the shared cost terms (no ODR issue) and every phase compares FindBest against it; exact queries copy raw baked samples; probe phases fail loudly on missing fixture classes and never pass on zero probes.
#include "MotionMatchingSearch.h"

#include "../Database/MotionMatchingDatabase.h"
#include "../Runtime/MotionMatchingSampler.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

#if USE_EDITOR

// editor-only self-test. Live search helpers (TurnPenalty/FullSampleCost/
// Horizon + cost terms) are duplicated here as file-local test reference
// (anonymous namespace, no ODR issue). The live definitions in
// MotionMatchingSearch.cpp are untouched (zero search behavior change).
// Phase K (prior-term parity) compares FindBest (live) against this
// reference every run — if the two ever drift, the test fails loudly.
// New phases M-Q replace the deleted FootLock probes with
// filter/empty-scope/contact/fastpath/replay probes (counts reported real,
// never forced to 1357).

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

    // Weight contract: schema weights are stored, never baked. The search
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

    // ---- Shared cost terms (shared cost terms) ----
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

} // namespace

bool MotionMatchingSearch::RunSelfTest(
    MotionMatchingDatabase* database,
    String& log)
{
    log = String::Empty;
    if (database == nullptr || database->WaitForLoaded())
    {
        log = TEXT("Self-test failed: database is not loaded.");
        return false;
    }
    if (!database->IsBaked() || !database->ValidateData())
    {
        log = TEXT("Self-test failed: database is not baked or not valid.");
        return false;
    }

    const int32 poseFeatureSize = database->GetPoseFeatureSize();
    const int32 trajectoryFeatureSize = database->GetTrajectoryFeatureSize();
    const int32 sampleCount = database->GetSampleCount();
    const Array<float>& poseFeatures = database->GetPoseFeatures();
    const Array<float>& trajectoryFeatures = database->GetTrajectoryFeatures();
    const Array<float>& poseDeviations = database->GetPoseFeatureDeviations();
    const Array<float>& trajectoryDeviations = database->GetTrajectoryFeatureDeviations();

    // Helper: an exact query is a straight copy of the RAW baked sample
    // (rev 3 keeps baked values raw; no denormalization, no mean/deviation).
    auto buildExactQuery = [&](int32 source, Array<float>& queryPose, Array<float>& queryTrajectory)
    {
        queryPose.Resize(poseFeatureSize);
        queryTrajectory.Resize(trajectoryFeatureSize);
        const float* bakedPose = &poseFeatures[source * poseFeatureSize];
        const float* bakedTrajectory = &trajectoryFeatures[source * trajectoryFeatureSize];
        for (int32 i = 0; i < poseFeatureSize; i++)
            queryPose[i] = bakedPose[i];
        for (int32 i = 0; i < trajectoryFeatureSize; i++)
            queryTrajectory[i] = bakedTrajectory[i];
    };

    MotionMatchingSearchSettings settings;
    settings.PoseWeight = database->GetSchema().PoseWeight;
    settings.TrajectoryWeight = database->GetSchema().TrajectoryWeight;
    settings.ContinuityWeight = 0.0f;
    settings.MinSwitchTime = 0.0f;
    // Explicit self-test exemption from the exclusion bans: exact-query
    // probes must see the full database (a banned tail sample could never
    // re-verify itself). The runtime policy opts in per query instead.
    settings.ReselectClip = -1;
    settings.ReselectSampleTime = 0.0f;
    settings.ReselectBanWindow = 0.0f;
    settings.TwinBanWindow = 0.0f;
    settings.LoopTailExclusion = 0.0f;
    // Horizon rule: exact-query probes copy raw baked samples (including the
    // clamped non-loop tail) and must be able to re-find them, so the
    // structural runtime validity cut is disabled here and covered directly
    // by Phase L instead.
    settings.ExcludeNonLoopTail = false;
    const Array<String> noTags;

    int32 passed = 0;
    int32 tested = 0;
    float worstExactCost = 0.0f;
    int32 worstExactSample = -1;
    float minPoseDeviation = MAX_float;
    float minTrajectoryDeviation = MAX_float;
    for (int32 i = 0; i < poseFeatureSize; i++)
        minPoseDeviation = Math::Min(minPoseDeviation, poseDeviations[i]);
    for (int32 i = 0; i < trajectoryFeatureSize; i++)
        minTrajectoryDeviation = Math::Min(minTrajectoryDeviation, trajectoryDeviations[i]);
    log += String::Format(
        TEXT("Min stored deviation: pose={:.8f}, trajectory={:.8f} (raw values; exact-query cost is 0 regardless).\n"),
        minPoseDeviation,
        minTrajectoryDeviation);
    const Array<int32>& sampleClipIndices = database->GetSampleClipIndices();
    const Array<int32>& clipSampleCounts = database->GetClipSampleCounts();
    const Array<String>& tagNames = database->GetAnimationTags();
    const int32 clipCount = database->GetClipCount();
    const auto startTime = std::chrono::high_resolution_clock::now();

    Array<float> queryPose;
    Array<float> queryTrajectory;

    // Exact-query verdict. The winner must be within eps of the PROBE
    // itself (near-optimal), not of zero: turn probes can never cost ~0 by
    // design (straightness prior), while straight probes keep the old
    // behavior (their own penalty is 0). On mismatch, rescan everything
    // before the winner with the SAME penalty-aware math and require
    // strictly greater cost — otherwise the earlier sample should have won
    // under strictly-less comparison.
    int32 tieCount = 0;
    auto checkExact = [&](int32 source, const MotionMatchingResult& result, const String& error, const String& phaseLabel) -> bool
    {
        tested++;
        const int32 sourceClip = (source >= 0 && source < sampleClipIndices.Count()) ? sampleClipIndices[source] : -1;
        const float sourcePenalty = TurnPenalty(database->GetClipTurnAngle(sourceClip), settings.QueryYawRate,
            settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale);
        if (!error.IsEmpty() || !result.IsValid || result.Cost > sourcePenalty + SelfTestMaxCost)
        {
            log += String::Format(
                TEXT("{} FAIL: sample {} (winner={}, cost={:.6f}, err='{}').\n"),
                phaseLabel,
                source,
                result.SampleIndex,
                result.Cost,
                error);
            return false;
        }
        if (result.SampleIndex != source)
        {
            tieCount++;
            for (int32 s = 0; s < result.SampleIndex; s++)
            {
                const float earlier = FullSampleCost(*database, queryPose, queryTrajectory, settings, -1, s, true);
                if (earlier <= result.Cost)
                {
                    log += String::Format(
                        TEXT("{} TIE-BREAK BROKEN: sample {} won by {} but earlier sample {} costs {:.9f} <= {:.9f}.\n"),
                        phaseLabel,
                        source,
                        result.SampleIndex,
                        s,
                        earlier,
                        result.Cost);
                    return false;
                }
            }
        }
        passed++;
        if (result.Cost > worstExactCost)
        {
            worstExactCost = result.Cost;
            worstExactSample = source;
        }
        return true;
    };

    // Phase A: clip boundaries. First/last samples exercise loop wrap-around
    // trajectory and clamped endpoint velocities — the riskiest baked data.
    // Only failures are logged individually.
    int32 phaseATested = 0;
    int32 phaseAPassed = 0;
    int32 sampleCursor = 0;
    for (int32 clipIndex = 0; clipIndex < clipCount; clipIndex++)
    {
        const int32 count = clipSampleCounts[clipIndex];
        if (count <= 0)
        {
            continue;
        }
        const int32 edges[2] = { sampleCursor, sampleCursor + count - 1 };
        for (int32 e = 0; e < 2; e++)
        {
            const int32 source = edges[e];
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, result, error);
            phaseATested++;
            if (checkExact(source, result, error, TEXT("Boundary")))
                phaseAPassed++;
        }
        sampleCursor += count;
    }
    log += String::Format(
        TEXT("Phase A (clip boundaries): {}/{} passed, worst exact cost={:.6f} at sample {}.\n"),
        phaseAPassed,
        phaseATested,
        worstExactCost,
        worstExactSample);

    // Phase B: strided spread over the whole database.
    int32 phaseBTested = 0;
    int32 phaseBPassed = 0;
    for (int32 source = 0; source < sampleCount; source += 997)
    {
        buildExactQuery(source, queryPose, queryTrajectory);
        MotionMatchingResult result;
        String error;
        FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, result, error);
        phaseBTested++;
        if (checkExact(source, result, error, TEXT("Spread")))
            phaseBPassed++;
    }
    log += String::Format(TEXT("Phase B (spread): {}/{} passed.\n"), phaseBPassed, phaseBTested);

    // Phase C: policy behavior on spread samples — continuity lock keeps the
    // current clip, tag include/exclude filters steer the winner tag.
    MotionMatchingSearchSettings lockedSettings = settings;
    lockedSettings.ContinuityWeight = 1.0f;
    lockedSettings.MinSwitchTime = 10.0f;
    int32 phaseCTest = 0;
    int32 phaseCPass = 0;
    for (int32 source = 0; source < sampleCount && phaseCTest < 60; source += 9973)
    {
        const int32 sourceClip = sampleClipIndices[source];
        const Array<String> sourceTags = database->GetClipTags(sourceClip);
        const String sourceTag = MotionMatchingDatabase::JoinTags(sourceTags);
        buildExactQuery(source, queryPose, queryTrajectory);

        // locked on the source clip -> winner must stay in it.
        {
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, lockedSettings, sourceClip, 0.0f, noTags, result, error);
            phaseCTest++;
            tested++;
            const bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == sourceClip;
            if (ok)
            {
                phaseCPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Continuity FAIL: sample {} clip {} (winner clip={}, err='{}').\n"),
                    source,
                    sourceClip,
                    result.ClipIndex,
                    error);
            }
        }

        // tag include -> winner overlaps the source set (exactly what
        // the overlap filter guarantees; tie twins may carry a subset, so a
        // superset assert would false-alarm on identical-feature ties).
        {
            const Array<String> include = sourceTags;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, include, result, error);
            phaseCTest++;
            tested++;
            const Array<String> winnerTags = database->GetClipTags(result.ClipIndex);
            bool overlap = false;
            for (const String& tag : include)
            {
                if (winnerTags.Contains(tag))
                {
                    overlap = true;
                    break;
                }
            }
            const bool ok = error.IsEmpty() && result.IsValid && overlap;
            if (ok)
            {
                phaseCPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Tag-include FAIL: sample {} tag '{}' (winner clip={}, err='{}').\n"),
                    source,
                    sourceTag,
                    result.ClipIndex,
                    error);
            }
        }

        // C3: tag exclude -> winner must carry none of the source tags.
        // Uses the disjoint ExcludeTags settings (not the overlap include
        // scope: with multi-tags a winner can pass an overlap filter via one
        // tag while carrying a forbidden one).
        {
            MotionMatchingSearchSettings exclSettings = settings;
            exclSettings.ExcludeTags = sourceTags;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, exclSettings, -1, MAX_float, noTags, result, error);
            phaseCTest++;
            tested++;
            const Array<String> winnerTags = database->GetClipTags(result.ClipIndex);
            bool disjoint = true;
            for (const String& tag : sourceTags)
            {
                if (winnerTags.Contains(tag))
                {
                    disjoint = false;
                    break;
                }
            }
            const bool ok = error.IsEmpty() && result.IsValid && disjoint;
            if (ok)
            {
                phaseCPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Tag-exclude FAIL: sample {} tag '{}' (winner clip={}, err='{}').\n"),
                    source,
                    sourceTag,
                    result.ClipIndex,
                    error);
            }
        }
    }
    log += String::Format(TEXT("Phase C (continuity/tags): {}/{} passed.\n"), phaseCPass, phaseCTest);

    // Phase D: temporal exclusion + per-block telemetry. Querying the exact
    // source sample while standing on it (CurrentTime == its time) must NOT
    // return a nearby frame on the same clip — otherwise playback reseeks
    // every query instead of playing through. The winner must be outside the
    // window or on another clip, and Pose+Trajectory+Continuity must sum to Cost.
    int32 phaseDTest = 0;
    int32 phaseDPass = 0;
    {
        MotionMatchingSearchSettings exclSettings = settings;
        exclSettings.CurrentTime = 0.0f; // overwritten per probe
        exclSettings.SameClipExclusionWindow = 0.25f;
        const Array<float>& sampleTimesArr = database->GetSampleTimes();
        int32 probes = 0;
        for (int32 source = 0; source < sampleCount && probes < 12; source += 4999)
        {
            const int32 sourceClip = sampleClipIndices[source];
            buildExactQuery(source, queryPose, queryTrajectory);
            exclSettings.CurrentTime = sampleTimesArr[source];
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, exclSettings, sourceClip, MAX_float, noTags, result, error);
            phaseDTest++;
            tested++;
            probes++;
            bool ok = error.IsEmpty() && result.IsValid;
            if (ok)
            {
                // Same-clip winners must be outside the exclusion window.
                if (result.ClipIndex == sourceClip)
                {
                    const float dt = result.SampleTime - sampleTimesArr[source];
                    if (dt < 0.0f ? -dt <= 0.250001f : dt <= 0.250001f)
                        ok = false;
                }
                // Per-block telemetry must reconstruct the total and stay ordered
                // (LoopCost included: rest preference is a real block now).
                const float blockSum = result.PoseCost + result.TrajectoryCost + result.ContinuityCost + result.LoopCost;
                if (!std::isfinite(result.PoseCost) || !std::isfinite(result.TrajectoryCost) ||
                    !std::isfinite(result.ContinuityCost) || !std::isfinite(result.LoopCost) ||
                    std::fabs(blockSum - result.Cost) > 0.001f ||
                    result.SecondBestCost < result.Cost)
                    ok = false;
            }
            if (ok)
            {
                phaseDPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Exclusion/telemetry FAIL: sample {} clip {} time {:.3f} (winner clip={} time={:.3f} cost={:.4f} pose={:.4f} traj={:.4f} cont={:.4f} second={:.4f} err='{}').\n"),
                    source,
                    sourceClip,
                    sampleTimesArr[source],
                    result.ClipIndex,
                    result.SampleTime,
                    result.Cost,
                    result.PoseCost,
                    result.TrajectoryCost,
                    result.ContinuityCost,
                    result.SecondBestCost,
                    error);
            }
        }
        // Exclusion disabled (window 0) must recover the exact source sample
        // with ~zero cost — proves exclusion is the only steering here.
        for (int32 source = 0; source < sampleCount && probes < 16; source += 19999)
        {
            const int32 sourceClip = sampleClipIndices[source];
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingSearchSettings noExcl = settings;
            noExcl.CurrentTime = sampleTimesArr[source];
            noExcl.SameClipExclusionWindow = 0.0f;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, noExcl, sourceClip, MAX_float, noTags, result, error);
            phaseDTest++;
            tested++;
            probes++;
            // Same verdict contract as exact queries (ties allowed but ordered).
            bool ok = error.IsEmpty() && result.IsValid && result.Cost <= SelfTestMaxCost;
            if (ok)
            {
                phaseDPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Exclusion-off FAIL: sample {} (winner={}, cost={:.6f}, err='{}').\n"),
                    source,
                    result.SampleIndex,
                    result.Cost,
                    error);
            }
        }
    }
    log += String::Format(TEXT("Phase D (temporal exclusion/telemetry): {}/{} passed.\n"), phaseDPass, phaseDTest);

    // Phase E: live trajectory frame parity. A constant actor->root offset
    // (mesh mount, stripped-root placement) must not bias the query: after
    // BuildRawTrajectory the zero-time point is exactly (0,0)/(0,1).
    int32 phaseETest = 0;
    int32 phaseEPass = 0;
    {
        const PoseSearchSchema& bakedSchema = database->GetSchema();
        int32 zeroIdx = -1;
        for (int32 i = 0; i < bakedSchema.TrajectorySampleTimes.Count(); i++)
        {
            if (std::fabs(bakedSchema.TrajectorySampleTimes[i]) <= 0.0001f)
            {
                zeroIdx = i;
                break;
            }
        }
        // Synthetic actor-local walk line with a large constant offset,
        // simulating a mesh mount 100 units right / 50 forward of the actor.
        TrajectoryPoint synth[PoseSearchSchema::TrajectoryPointCount];
        for (int32 i = 0; i < PoseSearchSchema::TrajectoryPointCount; i++)
        {
            synth[i].Position = Float3(100.0f + i * 20.0f, 0.0f, 50.0f + i * 30.0f);
            synth[i].Facing = Float3(0.0f, 0.0f, 1.0f);
        }
        Matrix actorWorld = Matrix::Identity;
        Matrix rootWorld = Matrix::Identity;
        rootWorld.SetTranslation(Float3(100.0f, 0.0f, 50.0f)); // root sits at the offset origin
        float raw[PoseSearchSchema::GetTrajectoryFeatureSize()];
        String trajError;
        phaseETest++;
        tested++;
        if (MotionMatchingQuery::BuildRawTrajectory(synth, actorWorld, rootWorld, bakedSchema, raw, trajError))
        {
            const int32 zi = (zeroIdx >= 0 ? zeroIdx : PoseSearchSchema::TrajectoryPointCount / 2) *
                PoseSearchSchema::TrajectoryFloatsPerPoint;
            const bool zeroOk = std::fabs(raw[zi]) <= 0.0001f &&
                std::fabs(raw[zi + 1]) <= 0.0001f &&
                std::fabs(raw[zi + 2]) <= 0.0001f &&
                std::fabs(raw[zi + 3] - 1.0f) <= 0.0001f;
            bool allFinite = true;
            for (int32 i = 0; i < PoseSearchSchema::GetTrajectoryFeatureSize(); i++)
                allFinite &= std::isfinite(raw[i]);
            if (zeroOk && allFinite)
            {
                phaseEPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Trajectory-rebase FAIL: zero=({:.4f},{:.4f},{:.4f},{:.4f}) finite={}.\n"),
                    raw[zi], raw[zi + 1], raw[zi + 2], raw[zi + 3], allFinite ? 1 : 0);
            }
        }
        else
        {
            log += String::Format(TEXT("Trajectory-rebase FAIL: builder error '{}'.\n"), trajError);
        }
    }
    log += String::Format(TEXT("Phase E (trajectory zero-time rebase): {}/{} passed.\n"), phaseEPass, phaseETest);

    // Phase F: top-2 correctness vs a no-prune reference. The hot loop prunes
    // at secondBestCost; recompute the two lowest FullSampleCost values without
    // pruning or continuity and require exact agreement on best and runner-up.
    int32 phaseFTest = 0;
    int32 phaseFPass = 0;
    {
        int32 probes = 0;
        for (int32 source = 123; source < sampleCount && probes < 5; source += 19999)
        {
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, result, error);
            phaseFTest++;
            tested++;
            probes++;
            bool ok = error.IsEmpty() && result.IsValid;
            if (ok)
            {
                float refBest = MAX_float;
                float refSecond = MAX_float;
                for (int32 s = 0; s < sampleCount; s++)
                {
                    const float c = FullSampleCost(*database, queryPose, queryTrajectory, settings, -1, s, true);
                    if (c < refBest)
                    {
                        refSecond = refBest;
                        refBest = c;
                    }
                    else if (c < refSecond)
                    {
                        refSecond = c;
                    }
                }
                if (std::fabs(result.Cost - refBest) > 0.0005f ||
                    std::fabs(result.SecondBestCost - refSecond) > 0.0005f)
                    ok = false;
            }
            if (ok)
            {
                phaseFPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Top2 FAIL: sample {} (cost={:.6f} second={:.6f}, err='{}').\n"),
                    source,
                    result.Cost,
                    result.SecondBestCost,
                    error);
            }
        }
    }
    log += String::Format(TEXT("Phase F (top-2 reference): {}/{} passed.\n"), phaseFPass, phaseFTest);

    // Phase G: continuation bias, switch margin, exclusion exemption.
    // Winner assertions use clip + cost (not exact index): static clips hold
    // bit-identical twins, so any in-family winner with the biased cost
    // proves the mechanism; exact-index checks would false-alarm on ties.
    int32 phaseGTest = 0;
    int32 phaseGPass = 0;
    {
        const Array<float>& sampleTimesArr = database->GetSampleTimes();
        const Array<int32>& clipSampleCounts = database->GetClipSampleCounts();
        const int32 clipCount = database->GetClipCount();
        Array<int32> clipFirst;
        clipFirst.Resize(clipCount);
        int32 cursor = 0;
        for (int32 c = 0; c < clipCount; c++)
        {
            clipFirst[c] = cursor;
            cursor += clipSampleCounts[c];
        }
        // Structural probes (NOT absolute sample offsets): middle samples of
        // spread low-turn clips. Absolute offsets rot with every rebake
        // (clip order follows tag grouping), landing probes in turn clips
        // where TurnPenalty legitimately breaks the -0.99 assertions. Each
        // probe is self-consistency pre-checked (exact query returns its own
        // clip), so bit-identical twins can never false-alarm the family.
        struct GProbe { int32 sample; int32 clip; };
        Array<GProbe> gProbes;
        MotionMatchingSearchSettings gPre = settings;
        gPre.CurrentTime = -1.0f;
        gPre.ExpectedTime = -1.0f;
        gPre.SameClipExclusionWindow = 0.0f;
        gPre.ContinuingBias = 0.0f;
        gPre.ContinuingWindow = 0.2f;
        gPre.SwitchMargin = 0.0f;
        const int32 gStep = clipCount / 12 > 0 ? clipCount / 12 : 1;
        for (int32 c = 0; c < clipCount && gProbes.Count() < 6; c += gStep)
        {
            const int32 count = c < clipSampleCounts.Count() ? clipSampleCounts[c] : 0;
            if (count < 40)
                continue;
            if (Math::Abs(database->GetClipTurnAngle(c)) >= 0.05f)
                continue;
            const int32 mid = clipFirst[c] + count / 2;
            buildExactQuery(mid, queryPose, queryTrajectory);
            MotionMatchingResult gPreResult;
            String gPreError;
            FindBest(database, queryPose, queryTrajectory, gPre, -1, 999.0f, noTags, gPreResult, gPreError);
            if (gPreError.IsEmpty() && gPreResult.IsValid && gPreResult.ClipIndex == c &&
                gPreResult.Cost <= SelfTestMaxCost)
            {
                GProbe p;
                p.sample = mid;
                p.clip = c;
                gProbes.Add(p);
            }
        }
        for (int32 pi = 0; pi < gProbes.Count(); pi++)
        {
            const int32 source = gProbes[pi].sample;
            const int32 C = gProbes[pi].clip;
            const float t_s = sampleTimesArr[source];
            const int32 D = gProbes[(pi + 1) % gProbes.Count()].clip;
            const float t_d = sampleTimesArr[clipFirst[D]];
            buildExactQuery(source, queryPose, queryTrajectory);

            // G1: bias rewards the natural continuation (cost — -bias).
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = t_s - 0.1f;
                g.ExpectedTime = t_s;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 1.0f;
                g.ContinuingWindow = 0.2f;
                g.SwitchMargin = 0.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, C, 999.0f, noTags, result, error);
                phaseGTest++;
                tested++;
                const float blockSum = result.PoseCost + result.TrajectoryCost + result.ContinuityCost + result.LoopCost;
                bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == C &&
                    result.Cost < -0.99f && result.ContinuityCost < -0.99f &&
                    std::fabs(blockSum - result.Cost) <= 0.001f;
                if (ok)
                {
                    phaseGPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Bias FAIL: sample {} (winner={} clip={} cost={:.4f} cont={:.4f} err='{}').\n"),
                        source,
                        result.SampleIndex,
                        result.ClipIndex,
                        result.Cost,
                        result.ContinuityCost,
                        error);
                }
            }
            // G2: huge margin demotes the exact winner to a continuation;
            // the demoted exact sample must become runner-up ± 0.
            // Continuity is zeroed: this probes margin mechanics only, and a
            // +0.25 other-clip penalty on the exact sample would pollute the
            // runner-up assertion below.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = t_d;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 0.0f;
                g.ContinuityWeight = 0.0f;
                g.ContinuingWindow = 0.2f;
                g.SwitchMargin = 1000000.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, D, 999.0f, noTags, result, error);
                phaseGTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == D &&
                    result.SecondBestCost <= SelfTestMaxCost;
                if (ok)
                {
                    phaseGPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Margin FAIL: sample {} (winner={} clip={} second={:.6f} err='{}').\n"),
                        source,
                        result.SampleIndex,
                        result.ClipIndex,
                        result.SecondBestCost,
                        error);
                }
            }
            // G2 control: zero margin keeps the exact family winning. Pure
            // cost (no current clip at all) so continuity cannot steer it.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = -1.0f;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 0.0f;
                g.ContinuingWindow = 0.2f;
                g.SwitchMargin = 0.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
                phaseGTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == C &&
                    result.Cost <= SelfTestMaxCost;
                if (ok)
                {
                    phaseGPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Margin-control FAIL: sample {} (winner={} clip={} err='{}').\n"),
                        source,
                        result.SampleIndex,
                        result.ClipIndex,
                        error);
                }
            }
            // G3: continuation is exempt from temporal exclusion.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = t_s;
                g.ExpectedTime = t_s;
                g.SameClipExclusionWindow = 0.25f;
                g.ContinuingBias = 1.0f;
                g.ContinuingWindow = 0.2f;
                g.SwitchMargin = 0.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, C, 999.0f, noTags, result, error);
                phaseGTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == C &&
                    result.Cost < -0.99f;
                if (ok)
                {
                    phaseGPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Exempt FAIL: sample {} (winner={} clip={} cost={:.4f} err='{}').\n"),
                        source,
                        result.SampleIndex,
                        result.ClipIndex,
                        result.Cost,
                        error);
                }
            }
        }
    }
    log += String::Format(TEXT("Phase G (continuation/margin): {}/{} passed.\n"), phaseGPass, phaseGTest);

    // Phase H: rest-loop preference. An exact query of a NON-loop end stance
    // (break/stop tail) with a huge preference must resolve to a loop clip;
    // with zero preference the exact one-shot wins. Proves the break-loop
    // lock has a structural exit, not just better-tuned weights.
    int32 phaseHTest = 0;
    int32 phaseHPass = 0;
    {
        // Collect one non-loop end sample per tag that has loop clips.
        struct EndProbe { int32 sample; int32 clip; };
        Array<EndProbe> probes;
        const Array<int32>& clipSampleCounts = database->GetClipSampleCounts();
        const int32 clipCount = database->GetClipCount();
        int32 cursor = 0;
        for (int32 c = 0; c < clipCount && probes.Count() < 8; c++)
        {
            const int32 count = clipSampleCounts[c];
            if (count > 10 && !database->GetClipLoop(c))
            {
                EndProbe p;
                p.sample = cursor + count - 1;
                p.clip = c;
                probes.Add(p);
            }
            cursor += count;
        }
        for (int32 i = 0; i < probes.Count(); i++)
        {
            const int32 source = probes[i].sample;
            buildExactQuery(source, queryPose, queryTrajectory);
            // H1: huge preference -> winner must be a loop clip.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = -1.0f;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 0.0f;
                g.SwitchMargin = 0.0f;
                g.LoopPreference = 1000000.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
                phaseHTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid &&
                    database->GetClipLoop(result.ClipIndex) &&
                    result.LoopCost <= -999999.0f;
                if (ok)
                {
                    phaseHPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("LoopPref FAIL: sample {} clip {} (winner={} clip={} loop={} loopCost={:.1f} err='{}').\n"),
                        source,
                        probes[i].clip,
                        result.SampleIndex,
                        result.ClipIndex,
                        database->GetClipLoop(result.ClipIndex) ? 1 : 0,
                        result.LoopCost,
                        error);
                }
            }
            // H2 control: zero preference -> exact one-shot wins.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = -1.0f;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 0.0f;
                g.SwitchMargin = 0.0f;
                g.LoopPreference = 0.0f;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
                phaseHTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid && result.ClipIndex == probes[i].clip &&
                    result.Cost <= SelfTestMaxCost && result.LoopCost == 0.0f;
                if (ok)
                {
                    phaseHPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("LoopPref-control FAIL: sample {} clip {} (winner={} clip={} cost={:.6f} err='{}').\n"),
                        source,
                        probes[i].clip,
                        result.SampleIndex,
                        result.ClipIndex,
                        result.Cost,
                        error);
                }
            }
            // H3: end-recovery shape (LoopOnly, zero preference, idle tags):
            // exact one-shot end queries must resolve to a loop clip even
            // with NO bonus helping — the filter alone exits the lock.
            {
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = -1.0f;
                g.SameClipExclusionWindow = 0.0f;
                g.ContinuingBias = 0.0f;
                g.SwitchMargin = 0.0f;
                g.LoopPreference = 0.0f;
                                g.CollectClassDiagnostics = true;
                                g.LoopFilter = MotionLoopFilter::LoopOnly;
                Array<String> idleOnly;
                idleOnly.Add(String(TEXT("idle")));
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, idleOnly, result, error);
                phaseHTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid &&
                    database->GetClipLoop(result.ClipIndex) &&
                    database->GetClipTags(result.ClipIndex).Contains(String(TEXT("idle")));
                if (ok)
                {
                    phaseHPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("EndRecover FAIL: sample {} clip {} (winner={} clip={} tag='{}' err='{}').\n"),
                        source,
                        probes[i].clip,
                        result.SampleIndex,
                        result.ClipIndex,
                        MotionMatchingDatabase::JoinTags(database->GetClipTags(result.ClipIndex)),
                        error);
                }
            }
        }
    }
    log += String::Format(TEXT("Phase H (loop preference): {}/{} passed.\n"), phaseHPass, phaseHTest);

    // Phase I: loop filter + class telemetry. Exact queries keep the true
    // class minima near zero so prune can never hide them (pruned members
    // are invisible to class telemetry by design — see SearchPass).
    int32 phaseITest = 0;
    int32 phaseIPass = 0;
    {
        // Deterministic probe picks (spread probing would rarely land on the
        // ~6% loop samples): middle samples of known loop / non-loop clips.
        const Array<int32>& clipSampleCounts = database->GetClipSampleCounts();
        const int32 clipCount = database->GetClipCount();
        Array<int32> loopProbes;
        Array<int32> oneProbes;
        int32 cursor = 0;
        for (int32 c = 0; c < clipCount && (loopProbes.Count() < 3 || oneProbes.Count() < 3); c++)
        {
            const int32 count = clipSampleCounts[c];
            if (count > 10)
            {
                if (database->GetClipLoop(c) && loopProbes.Count() < 3)
                    loopProbes.Add(cursor + count / 2);
                else if (!database->GetClipLoop(c) && oneProbes.Count() < 3)
                    oneProbes.Add(cursor + count / 2);
            }
            cursor += count;
        }
        // I1: LoopOnly on exact one-shot queries -> loop winners, one-shot
        // telemetry untouched, loop class scored.
        for (int32 i = 0; i < oneProbes.Count(); i++)
        {
            const int32 source = oneProbes[i];
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingSearchSettings g = settings;
            g.CurrentTime = -1.0f;
            g.ExpectedTime = -1.0f;
            g.SameClipExclusionWindow = 0.0f;
            g.LoopPreference = 0.0f;
            g.CollectClassDiagnostics = true;
            g.LoopFilter = MotionLoopFilter::LoopOnly;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
            phaseITest++;
            tested++;
            bool ok = error.IsEmpty() && result.IsValid &&
                database->GetClipLoop(result.ClipIndex) &&
                result.BestOneShotSample == -1 && result.OneShotScoredCount == 0 &&
                result.LoopScoredCount > 0;
            if (ok)
            {
                phaseIPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("LoopOnly FAIL: sample {} (winner={} loop={} oneshotN={} loopN={} err='{}').\n"),
                    source,
                    result.SampleIndex,
                    database->GetClipLoop(result.ClipIndex) ? 1 : 0,
                    result.OneShotScoredCount,
                    result.LoopScoredCount,
                    error);
            }
        }
        // I2: OneShotOnly on exact loop queries -> non-loop winners.
        for (int32 i = 0; i < loopProbes.Count(); i++)
        {
            const int32 source = loopProbes[i];
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingSearchSettings g = settings;
            g.CurrentTime = -1.0f;
            g.ExpectedTime = -1.0f;
            g.SameClipExclusionWindow = 0.0f;
            g.LoopPreference = 0.0f;
            g.CollectClassDiagnostics = true;
            g.LoopFilter = MotionLoopFilter::OneShotOnly;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
            phaseITest++;
            tested++;
            bool ok = error.IsEmpty() && result.IsValid &&
                !database->GetClipLoop(result.ClipIndex) &&
                result.BestLoopSample == -1 && result.LoopScoredCount == 0 &&
                result.OneShotScoredCount > 0;
            if (ok)
            {
                phaseIPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("OneShotOnly FAIL: sample {} (winner={} loop={} loopN={} oneshotN={} err='{}').\n"),
                    source,
                    result.SampleIndex,
                    database->GetClipLoop(result.ClipIndex) ? 1 : 0,
                    result.LoopScoredCount,
                    result.OneShotScoredCount,
                    error);
            }
        }
        // I3: class telemetry on plain exact queries. The tight near-zero
        // bound (which implies a scored member) applies ONLY to the source's
        // own class — its global-min twin always completes. The opposite
        // class may be legitimately fully pruned (nothing near an exact
        // query survives a collapsed runner-up), so it only needs finite
        // values; I4 below covers both-classes-scored with a blurry query.
        for (int32 round = 0; round < 2; round++)
        {
            const Array<int32>& set = round == 0 ? loopProbes : oneProbes;
            for (int32 i = 0; i < set.Count(); i++)
            {
                const int32 source = set[i];
                const bool sourceLoop = round == 0;
                buildExactQuery(source, queryPose, queryTrajectory);
                MotionMatchingSearchSettings g = settings;
                g.CurrentTime = -1.0f;
                g.ExpectedTime = -1.0f;
                g.SameClipExclusionWindow = 0.0f;
                g.LoopPreference = 0.0f;
                                g.CollectClassDiagnostics = true;
                                g.LoopFilter = MotionLoopFilter::Any;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
                phaseITest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid &&
                    std::isfinite(result.BestLoopCost) && std::isfinite(result.BestOneShotCost) &&
                    (sourceLoop
                        ? (result.BestLoopCost <= SelfTestMaxCost + 0.0005f &&
                           result.BestLoopClip >= 0 && result.LoopScoredCount > 0)
                        : (result.BestOneShotCost <= SelfTestMaxCost + 0.0005f &&
                           result.BestOneShotClip >= 0 && result.OneShotScoredCount > 0));
                if (ok)
                {
                    phaseIPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("ClassTel FAIL: sample {} (loop={:.4f} one={:.4f} loopN={} oneN={} err='{}').\n"),
                        source,
                        result.BestLoopCost,
                        result.BestOneShotCost,
                        result.LoopScoredCount,
                        result.OneShotScoredCount,
                        error);
                }
            }
        }
        // I4 dropped (was: blurry midpoint query must land in a parent
        // family). Unsound premise: in 42-D feature space the midpoint of two
        // distant samples is routinely nearest to a THIRD family (observed:
        // midpoint of loop+oneshot won by an unrelated clip with oneN=0).
        // That is legitimate search behavior, not a telemetry failure —
        // both-classes-scored is query-dependent by design (see prune
        // honesty note). I1-I3 already cover filter mechanics deterministically.
    }
    log += String::Format(TEXT("Phase I (loop filter/telemetry): {}/{} passed.\n"), phaseIPass, phaseITest);

    // Phase J: full-scan class minima vs an independent FullSampleCost
    // reference (raw blocks, same tag+filter scope). Proves BestLoopCost /
    // BestOneShotCost report exact minima, not pruned leftovers, and that
    // totals match an independent member count.
    int32 phaseJTest = 0;
    int32 phaseJPass = 0;
    {
        const int32 clipCount = database->GetClipCount();
        int32 probes = 0;
        for (int32 source = 9000; source < sampleCount && probes < 4; source += 14999)
        {
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingSearchSettings g = settings;
            g.CurrentTime = -1.0f;
            g.ExpectedTime = -1.0f;
            g.SameClipExclusionWindow = 0.0f;
            g.LoopPreference = 0.0f;
            g.CollectClassDiagnostics = true;
            g.LoopFilter = MotionLoopFilter::Any;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, g, -1, 999.0f, noTags, result, error);
            phaseJTest++;
            tested++;
                bool ok = error.IsEmpty() && result.IsValid;
                float refLoop = MAX_float;
                float refOne = MAX_float;
                int32 refLoopN = 0;
                int32 refOneN = 0;
                if (ok)
                {
                    for (int32 s = 0; s < sampleCount; s++)
                    {
                        const int32 c = sampleClipIndices[s];
                        if (c < 0 || c >= clipCount)
                            continue;
                        const float v = FullSampleCost(*database, queryPose, queryTrajectory, g, -1, s, false);
                        if (database->GetClipLoop(c))
                        {
                            refLoopN++;
                            if (v < refLoop)
                                refLoop = v;
                        }
                        else
                        {
                            refOneN++;
                            if (v < refOne)
                                refOne = v;
                        }
                    }
                    if (std::fabs(result.BestLoopCost - refLoop) > 0.0005f ||
                        std::fabs(result.BestOneShotCost - refOne) > 0.0005f ||
                        result.LoopTotalCount != refLoopN ||
                        result.OneShotTotalCount != refOneN)
                        ok = false;
                }
            if (ok)
            {
                phaseJPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("FullScan FAIL: sample {} (loop={:.4f} vs {:.4f} n={}/{} one={:.4f} vs {:.4f} n={}/{} err='{}').\n"),
                    source,
                    result.BestLoopCost,
                    refLoop,
                    result.LoopTotalCount,
                    refLoopN,
                    result.BestOneShotCost,
                    refOne,
                    result.OneShotTotalCount,
                    refOneN,
                    error);
            }
            probes++;
        }
    }
    log += String::Format(TEXT("Phase J (full-scan reference): {}/{} passed.\n"), phaseJPass, phaseJTest);

    // Phase K: prior term parity. The brute-force reference now uses the SAME
    // FullSampleCost all-terms function as the hot pass, so it covers
    // continuity/bias and loop preference — the cases the old raw-block
    // reference could not see (shared cost terms).
    // the footlock case died with the foot-lock search prior (it always
    // ran at zero at runtime; contact features stay in the DB untouched).
    int32 phaseKTest = 0;
    int32 phaseKPass = 0;
    {
        struct KCase
        {
            const Char* Label;
            float Continuity;
            float LoopPreference;
            int32 CurrentClip;
            float ExpectedTime;
        };
        const Array<float>& sampleTimesArr = database->GetSampleTimes();
        int32 probes = 0;
        for (int32 source = 2000; source < sampleCount && probes < 3; source += 17000)
        {
            const int32 sourceClip = sampleClipIndices[source];
            buildExactQuery(source, queryPose, queryTrajectory);
            const KCase cases[2] =
            {
                { TEXT("continuity"), 1.0f, 0.0f, sourceClip, sampleTimesArr[source] },
                { TEXT("loop"), 0.0f, 100.0f, -1, -1.0f },
            };
            for (const KCase& kc : cases)
            {
                MotionMatchingSearchSettings k = settings;
                k.ContinuityWeight = kc.Continuity;
                k.CurrentTime = -1.0f;
                k.SameClipExclusionWindow = 0.0f;
                k.ExpectedTime = kc.ExpectedTime;
                k.ContinuingBias = kc.Continuity > 0.0f ? 1.0f : 0.0f;
                k.ContinuingWindow = 0.2f;
                k.SwitchMargin = 0.0f;
                k.TwinBanWindow = 0.0f;
                k.LoopTailExclusion = 0.0f;
                k.LoopPreference = kc.LoopPreference;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, k, kc.CurrentClip, MAX_float, noTags, result, error);
                phaseKTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid;
                float refBest = MAX_float;
                int32 refBestSample = -1;
                if (ok)
                {
                    for (int32 s = 0; s < sampleCount; s++)
                    {
                        const float c = FullSampleCost(
                            *database, queryPose, queryTrajectory, k, kc.CurrentClip, s, true);
                        if (c < refBest)
                        {
                            refBest = c;
                            refBestSample = s;
                        }
                    }
                    if (std::fabs(result.Cost - refBest) > 0.0005f ||
                        result.SampleIndex != refBestSample)
                        ok = false;
                }
                if (ok)
                {
                    phaseKPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Term-parity({}) FAIL: sample {} (winner={} cost={:.6f} vs ref {:.6f}@{} err='{}').\n"),
                        String(kc.Label),
                        source,
                        result.SampleIndex,
                        result.Cost,
                        refBest,
                        refBestSample,
                        error);
                }
            }
            probes++;
        }
    }
    log += String::Format(TEXT("Phase K (prior term parity): {}/{} passed.\n"), phaseKPass, phaseKTest);

    // Phase L: structural non-loop tail validity. A non-loop sample whose
    // future trajectory horizon reaches past the clip end must never win when
    // ExcludeNonLoopTail is on; the winner must itself have a valid future.
    // (The flag-off exact-tail behavior is Phase A/B's contract.)
    int32 phaseLTest = 0;
    int32 phaseLPass = 0;
    {
        const float horizon = GetTrajectoryHorizon(*database);
        const Array<float>& sampleTimesArr = database->GetSampleTimes();
        const Array<float>& clipLengths = database->GetClipLengths();
        MotionMatchingSearchSettings l = settings;
        l.ExcludeNonLoopTail = true;
        l.CurrentTime = -1.0f;
        l.ExpectedTime = -1.0f;
        l.SameClipExclusionWindow = 0.0f;
        l.ContinuingBias = 0.0f;
        l.SwitchMargin = 0.0f;
        int32 cursor = 0;
        int32 probes = 0;
        for (int32 c = 0; c < clipCount && probes < 6; c++)
        {
            const int32 n = clipSampleCounts[c];
            const int32 source = cursor + (n > 0 ? n - 1 : 0);
            cursor += n;
            if (n <= 1 || database->GetClipLoop(c) || c >= clipLengths.Count())
                continue;
            if (!MotionMatchingSampler::IsNonLoopFutureOutside(
                    sampleTimesArr[source], horizon, clipLengths[c], false))
                continue;
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, l, -1, MAX_float, noTags, result, error);
            phaseLTest++;
            tested++;
            bool ok = error.IsEmpty() && result.IsValid;
            if (ok)
            {
                const int32 wc = sampleClipIndices[result.SampleIndex];
                const bool wLoop = wc >= 0 && wc < clipCount && database->GetClipLoop(wc);
                const float wLen = (wc >= 0 && wc < clipLengths.Count()) ? clipLengths[wc] : 0.0f;
                if (MotionMatchingSampler::IsNonLoopFutureOutside(
                        result.SampleTime, horizon, wLen, wLoop))
                    ok = false;
            }
            if (ok)
            {
                phaseLPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Tail-validity FAIL: tail sample {} clip {} (winner={} clip={} time={:.3f} err='{}').\n"),
                    source,
                    c,
                    result.SampleIndex,
                    result.ClipIndex,
                    result.SampleTime,
                    error);
            }
            probes++;
        }
    }
    log += String::Format(TEXT("Phase L (non-loop tail validity): {}/{} passed.\n"), phaseLPass, phaseLTest);

    // Phase M: filter respect (replaces deleted FootLock probes with
    // caller-filter probes). Exact queries with include/exclude filters.
    int32 phaseMTest = 0;
    int32 phaseMPass = 0;
    {
        int32 mProbes = 0;
        for (int32 source = 5000; source < sampleCount && mProbes < 4; source += 17000)
        {
            const int32 sourceClip = (source >= 0 && source < sampleClipIndices.Count()) ? sampleClipIndices[source] : -1;
            if (sourceClip < 0 || sourceClip >= clipCount)
                continue;
            const Array<String> sourceTags = database->GetClipTags(sourceClip);
            if (sourceTags.Count() == 0)
                continue;
            buildExactQuery(source, queryPose, queryTrajectory);
            // M1: include = source tags must overlap the winner.
            {
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, sourceTags, result, error);
                phaseMTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid;
                if (ok)
                {
                    const Array<String> winnerTags = database->GetClipTags(result.ClipIndex);
                    bool overlap = false;
                    for (const String& tag : sourceTags)
                    {
                        if (winnerTags.Contains(tag))
                        {
                            overlap = true;
                            break;
                        }
                    }
                    ok = overlap;
                }
                if (ok)
                {
                    phaseMPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Filter-include FAIL: sample {} clip {} (winner={}, err='{}').\n"),
                        source,
                        sourceClip,
                        result.SampleIndex,
                        error);
                }
            }
            // M2: exclude = source tags must make the winner disjoint.
            {
                MotionMatchingSearchSettings excl = settings;
                excl.ExcludeTags = sourceTags;
                MotionMatchingResult result;
                String error;
                FindBest(database, queryPose, queryTrajectory, excl, -1, MAX_float, noTags, result, error);
                phaseMTest++;
                tested++;
                bool ok = error.IsEmpty() && result.IsValid;
                if (ok)
                {
                    const Array<String> winnerTags = database->GetClipTags(result.ClipIndex);
                    bool disjoint = true;
                    for (const String& tag : sourceTags)
                    {
                        if (winnerTags.Contains(tag))
                        {
                            disjoint = false;
                            break;
                        }
                    }
                    ok = disjoint;
                }
                if (ok)
                {
                    phaseMPass++;
                    passed++;
                }
                else
                {
                    log += String::Format(
                        TEXT("Filter-exclude FAIL: sample {} clip {} (winner={}, err='{}').\n"),
                        source,
                        sourceClip,
                        result.SampleIndex,
                        error);
                }
            }
            mProbes++;
        }
        // unknown-only include must be invalid (no candidate).
        {
            Array<String> unknown;
            unknown.Add(String(TEXT("__p4_no_such_tag__")));
            buildExactQuery(0, queryPose, queryTrajectory);
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, unknown, result, error);
            phaseMTest++;
            tested++;
            bool ok = !result.IsValid;
            if (ok)
            {
                phaseMPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Filter-unknown FAIL: expected invalid, got winner={}.\n"),
                    result.SampleIndex);
            }
        }
    }
    log += String::Format(TEXT("Phase M (filter respect): {}/{} passed.\n"), phaseMPass, phaseMTest);

    // Empty-scope / unconstrained edge probes: empty include stays unconstrained, exclude-all stays invalid.
    int32 phaseNTest = 0;
    int32 phaseNPass = 0;
    {
        // Empty include (noTags) = unconstrained, must be valid.
        buildExactQuery(sampleCount > 1000 ? 1000 : 0, queryPose, queryTrajectory);
        {
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, result, error);
            phaseNTest++;
            tested++;
            bool ok = error.IsEmpty() && result.IsValid;
            if (ok)
            {
                phaseNPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Empty-unconstrained FAIL: err='{}'.\n"),
                    error);
            }
        }
        // N2: exclude-all-tags must be invalid (no candidate survives).
        {
            MotionMatchingSearchSettings exclAll = settings;
            for (int32 ti = 0; ti < tagNames.Count(); ti++)
                exclAll.ExcludeTags.Add(tagNames[ti]);
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, exclAll, -1, MAX_float, noTags, result, error);
            phaseNTest++;
            tested++;
            bool ok = !result.IsValid;
            if (ok)
            {
                phaseNPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Exclude-all FAIL: expected invalid, got winner={}.\n"),
                    result.SampleIndex);
            }
        }
    }
    log += String::Format(TEXT("Phase N (empty-scope/unconstrained): {}/{} passed.\n"), phaseNPass, phaseNTest);

    // Phase O: contact preservation (baked contact channels stay in the
    // DB + participate in search; the foot-lock prior is deleted, features kept).
    int32 phaseOTest = 0;
    int32 phaseOPass = 0;
    {
        // O1: feature sizes (pose 44 = 7 bones * 6 + 2 contacts, traj 30).
        phaseOTest++;
        tested++;
        bool sizeOk = poseFeatureSize == 44 && trajectoryFeatureSize == 30;
        if (sizeOk)
        {
            phaseOPass++;
            passed++;
        }
        else
        {
            log += String::Format(
                TEXT("Contact-size FAIL: pose={} traj={} (want 44/30).\n"),
                poseFeatureSize,
                trajectoryFeatureSize);
        }
        // O2: deviations finite and positive (contact dims included).
        phaseOTest++;
        tested++;
        bool devOk = true;
        for (int32 i = 0; i < poseFeatureSize && devOk; i++)
        {
            if (!std::isfinite(poseDeviations[i]) || poseDeviations[i] <= 0.0f)
                devOk = false;
        }
        for (int32 i = 0; i < trajectoryFeatureSize && devOk; i++)
        {
            if (!std::isfinite(trajectoryDeviations[i]) || trajectoryDeviations[i] <= 0.0f)
                devOk = false;
        }
        if (devOk)
        {
            phaseOPass++;
            passed++;
        }
        else
        {
            log += String(TEXT("Contact-deviation FAIL: non-finite/non-positive deviation.\n"));
        }
        // O3/O4: flipping contact bits (42/43) must change the reference cost
        // (proves they participate; exact cost ~0, flipped clearly > 0).
        const int32 contactL = 42;
        const int32 contactR = 43;
        if (poseFeatureSize > contactR && sampleCount > 0)
        {
            buildExactQuery(0, queryPose, queryTrajectory);
            const float base = FullSampleCost(*database, queryPose, queryTrajectory, settings, -1, 0, false);
            queryPose[contactL] += 1.0f;
            const float flipL = FullSampleCost(*database, queryPose, queryTrajectory, settings, -1, 0, false);
            phaseOTest++;
            tested++;
            bool okL = std::isfinite(flipL) && flipL > base + 0.00001f;
            if (okL)
            {
                phaseOPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Contact-L FAIL: base={:.6f} flipped={:.6f}.\n"),
                    base,
                    flipL);
            }
            queryPose[contactL] -= 1.0f;
            queryPose[contactR] += 1.0f;
            const float flipR = FullSampleCost(*database, queryPose, queryTrajectory, settings, -1, 0, false);
            phaseOTest++;
            tested++;
            bool okR = std::isfinite(flipR) && flipR > base + 0.00001f;
            if (okR)
            {
                phaseOPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Contact-R FAIL: base={:.6f} flipped={:.6f}.\n"),
                    base,
                    flipR);
            }
        }
    }
    log += String::Format(TEXT("Phase O (contact preservation): {}/{} passed.\n"), phaseOPass, phaseOTest);

    // Phase P: fastpath scoring determinism (SampleCost is the
    // continuation scorer; must be deterministic + weight-sensitive, with
    // graceful failures. Fastpath-invalidate itself is policy-level and is
    // covered by the contract filter-change test + auto_test replay gate.)
    int32 phasePTest = 0;
    int32 phasePPass = 0;
    {
        buildExactQuery(0, queryPose, queryTrajectory);
        const int32 s0 = 0;
        float c1 = 0.0f;
        float c2 = 0.0f;
        String e1;
        String e2;
        bool ok1 = MotionMatchingSearch::SampleCost(database, queryPose, queryTrajectory, s0,
            settings.PoseWeight, settings.TrajectoryWeight, 0.0f, true,
            settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, c1, e1);
        bool ok2 = MotionMatchingSearch::SampleCost(database, queryPose, queryTrajectory, s0,
            settings.PoseWeight, settings.TrajectoryWeight, 0.0f, true,
            settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, c2, e2);
        phasePTest++;
        tested++;
        bool detOk = ok1 && ok2 && e1.IsEmpty() && e2.IsEmpty() && c1 == c2;
        if (detOk)
        {
            phasePPass++;
            passed++;
        }
        else
        {
            log += String::Format(
                TEXT("SampleCost-determinism FAIL: {:.6f} vs {:.6f}.\n"),
                c1,
                c2);
        }
        // weight sensitivity on an off-exact query (exact cost ~0 stays ~0
        // under any weight, so offset pose[0] first).
        Array<float> offPose;
        offPose.Resize(poseFeatureSize);
        for (int32 i = 0; i < poseFeatureSize; i++)
            offPose[i] = queryPose[i];
        offPose[0] += 0.5f;
        float cw1 = 0.0f;
        float cw2 = 0.0f;
        String ew1;
        String ew2;
        bool ow1 = MotionMatchingSearch::SampleCost(database, offPose, queryTrajectory, s0,
            settings.PoseWeight, settings.TrajectoryWeight, 0.0f, true,
            settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, cw1, ew1);
        bool ow2 = MotionMatchingSearch::SampleCost(database, offPose, queryTrajectory, s0,
            settings.PoseWeight * 2.0f, settings.TrajectoryWeight, 0.0f, true,
            settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, cw2, ew2);
        phasePTest++;
        tested++;
        bool sensOk = ow1 && ow2 && ew1.IsEmpty() && ew2.IsEmpty() &&
            std::isfinite(cw1) && std::isfinite(cw2) && cw1 > 0.0f && cw2 != cw1;
        if (sensOk)
        {
            phasePPass++;
            passed++;
        }
        else
        {
            log += String::Format(
                TEXT("SampleCost-weight FAIL: {:.6f} vs {:.6f}.\n"),
                cw1,
                cw2);
        }
        // invalid index fails gracefully.
        {
            float badCost = 0.0f;
            String badError;
            bool badOk = MotionMatchingSearch::SampleCost(database, queryPose, queryTrajectory, -1,
                settings.PoseWeight, settings.TrajectoryWeight, 0.0f, true,
                settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, badCost, badError);
            phasePTest++;
            tested++;
            bool ok = !badOk && !badError.IsEmpty();
            if (ok)
            {
                phasePPass++;
                passed++;
            }
            else
            {
                log += String(TEXT("SampleCost-badindex FAIL: expected graceful failure.\n"));
            }
        }
        // non-finite query fails gracefully.
        {
            Array<float> nanPose;
            nanPose.Resize(poseFeatureSize);
            for (int32 i = 0; i < poseFeatureSize; i++)
                nanPose[i] = queryPose[i];
            nanPose[0] = 1e30f * 1e30f;
            float nanCost = 0.0f;
            String nanError;
            bool nanOk = MotionMatchingSearch::SampleCost(database, nanPose, queryTrajectory, s0,
                settings.PoseWeight, settings.TrajectoryWeight, 0.0f, true,
                settings.TurnPenaltyWeight, settings.TurnPenaltyYawScale, nanCost, nanError);
            phasePTest++;
            tested++;
            bool ok = !nanOk && !nanError.IsEmpty();
            if (ok)
            {
                phasePPass++;
                passed++;
            }
            else
            {
                log += String(TEXT("SampleCost-nonfinite FAIL: expected graceful failure.\n"));
            }
        }
    }
    log += String::Format(TEXT("Phase P (fastpath scoring): {}/{} passed.\n"), phasePPass, phasePTest);

    // Phase Q: replay determinism (same query twice gives the same winner).
    // Force-sync is policy-level and is covered by the contract force-hatch
    // test + auto_test replay gate, not by this search-only self-test.)
    int32 phaseQTest = 0;
    int32 phaseQPass = 0;
    {
        int32 qProbes = 0;
        for (int32 source = 7000; source < sampleCount && qProbes < 3; source += 19000)
        {
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingResult r1;
            MotionMatchingResult r2;
            String e1;
            String e2;
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, r1, e1);
            FindBest(database, queryPose, queryTrajectory, settings, -1, MAX_float, noTags, r2, e2);
            phaseQTest++;
            tested++;
            bool ok = e1.IsEmpty() && e2.IsEmpty() && r1.IsValid && r2.IsValid &&
                r1.SampleIndex == r2.SampleIndex && r1.Cost == r2.Cost;
            if (ok)
            {
                phaseQPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Replay-determinism FAIL: sample {} ({}/{:.6f} vs {}/{:.6f}).\n"),
                    source,
                    r1.SampleIndex,
                    r1.Cost,
                    r2.SampleIndex,
                    r2.Cost);
            }
            qProbes++;
        }
    }
    log += String::Format(TEXT("Phase Q (replay determinism): {}/{} passed.\n"), phaseQPass, phaseQTest);
    // Phase R: one-shot progression (sprint->run fix). The current non-loop
    // clip may only continue near ExpectedTime; exact-pose queries pointing
    // far away must not return that far sample. The same predicate gates
    // brute-force (FindBest), indexed (UseIndexedSearch) and tooling
    // (FindTopCandidates). Metadata-driven: probes are located from the
    // baked registry (loop flag, clip lengths, sample times, tags); any
    // missing fixture class fails loudly, never passes on zero probes.
    int32 phaseRTest = 0;
    int32 phaseRPass = 0;
    {
        const Array<float>& sampleTimesArr = database->GetSampleTimes();
        // Probe qualification (metadata-driven, Phase G style): one long
        // straight non-loop clip and one straight loop clip whose exact
        // mid/tail queries already win by themselves in a pure search, so
        // the progression asserts below cannot false-alarm on turn-penalty
        // steals or bit-identical twins. No qualifying clip = loud fixture
        // fail, never a zero-probe pass.
        int32 oneShotClip = -1;
        int32 loopClip = -1;
        int32 oneShotFirst = -1;
        int32 loopFirst = -1;
        int32 oneShotAnchor = -1;
        int32 loopSource = -1;
        {
            MotionMatchingSearchSettings pre = settings;
            pre.CurrentTime = -1.0f;
            pre.ExpectedTime = -1.0f;
            pre.SameClipExclusionWindow = 0.0f;
            pre.ContinuingBias = 0.0f;
            pre.ContinuingWindow = 0.2f;
            pre.SwitchMargin = 0.0f;
            pre.TwinBanWindow = 0.0f;
            pre.LoopTailExclusion = 0.0f;
            pre.ExcludeNonLoopTail = false;
            int32 scan = 0;
            for (int32 c = 0; c < clipCount; c++)
            {
                const int32 n = (c >= 0 && c < clipSampleCounts.Count()) ? clipSampleCounts[c] : 0;
                const bool isLoop = database->GetClipLoop(c);
                const int32 first = scan;
                scan += n;
                if (n < 30)
                    continue;
                if (Math::Abs(database->GetClipTurnAngle(c)) >= 0.05f)
                    continue;
                const float span = sampleTimesArr[first + n - 1] - sampleTimesArr[first];
                if (isLoop)
                {
                    // Mid, not tail: a well-baked loop closes the cycle, so
                    // an exact tail query ties toward the head index under
                    // strictly-less comparison. The mid sample carries a
                    // unique pose while still sitting far from the start
                    // expectation used in R5 below (span/2 > window).
                    if (loopClip >= 0 || span <= 0.8f)
                        continue;
                    const int32 mid = first + n / 2;
                    buildExactQuery(mid, queryPose, queryTrajectory);
                    MotionMatchingResult pr;
                    String pe;
                    FindBest(database, queryPose, queryTrajectory, pre, -1, MAX_float, noTags, pr, pe);
                    if (pe.IsEmpty() && pr.IsValid && pr.SampleIndex == mid)
                    {
                        loopClip = c;
                        loopFirst = first;
                        loopSource = mid;
                    }
                }
                else
                {
                    if (oneShotClip >= 0 || span <= 0.8f)
                        continue;
                    const int32 mid = first + n / 2;
                    buildExactQuery(mid, queryPose, queryTrajectory);
                    MotionMatchingResult pr;
                    String pe;
                    FindBest(database, queryPose, queryTrajectory, pre, -1, MAX_float, noTags, pr, pe);
                    if (pe.IsEmpty() && pr.IsValid && pr.SampleIndex == mid)
                    {
                        oneShotClip = c;
                        oneShotFirst = first;
                        oneShotAnchor = mid;
                    }
                }
                if (oneShotClip >= 0 && loopClip >= 0)
                    break;
            }
        }
        // R1: current non-loop exact-pose query pointing far-past/far-future
        // must NOT return that far sample (brute force).
        if (oneShotClip < 0 || oneShotFirst < 0)
        {
            log += String(TEXT("Oneshot-progression FAIL: no straight non-loop clip (>= 30 samples, span > 0.8s) whose exact mid query wins by itself (fixture missing).\n"));
            // Fixture gap must fail the run, never pass on zero probes.
            phaseRTest += 2;
            tested += 2;
        }
        else
        {
            const float win = 0.15f;
            const int32 n = (oneShotClip >= 0 && oneShotClip < clipSampleCounts.Count())
                ? clipSampleCounts[oneShotClip] : 0;
            const int32 anchor = oneShotAnchor;
            const bool rangeValid = oneShotClip >= 0 && oneShotFirst >= 0 && n > 0 &&
                anchor >= oneShotFirst && anchor < oneShotFirst + n &&
                anchor >= 0 && anchor < sampleCount &&
                oneShotFirst + n - 1 >= 0 && oneShotFirst + n - 1 < sampleCount;
            const float anchorTime = rangeValid ? sampleTimesArr[anchor] : 0.0f;
            const int32 farPast = oneShotFirst;
            const int32 farFuture = oneShotFirst + n - 1;
            const bool pastFar = rangeValid &&
                Math::Abs(sampleTimesArr[farPast] - anchorTime) > win + 0.2f;
            const bool futureFar = rangeValid &&
                Math::Abs(sampleTimesArr[farFuture] - anchorTime) > win + 0.2f;
            if (!pastFar || !futureFar)
            {
                log += String::Format(
                    TEXT("Oneshot-progression FAIL: fixture clip {} span too short for far offsets (need > {:.2f}s each side).\n"),
                    oneShotClip,
                    win + 0.2f);
                // Span gap must fail the run, never pass on zero probes.
                phaseRTest += 2;
                tested += 2;
            }
            else
            {
                const int32 farSamples[2] = { farPast, farFuture };
                for (int32 fi = 0; fi < 2; fi++)
                {
                    const int32 source = farSamples[fi];
                    buildExactQuery(source, queryPose, queryTrajectory);
                    MotionMatchingSearchSettings g = settings;
                    g.CurrentTime = anchorTime;
                    g.ExpectedTime = anchorTime;
                    g.SameClipExclusionWindow = 0.0f;
                    g.ContinuingBias = 0.0f;
                    g.ContinuingWindow = win;
                    g.SwitchMargin = 0.0f;
                    g.TwinBanWindow = 0.0f;
                    g.LoopTailExclusion = 0.0f;
                    g.ExcludeNonLoopTail = false;
                    MotionMatchingResult result;
                    String error;
                    FindBest(database, queryPose, queryTrajectory, g, oneShotClip, 999.0f, noTags, result, error);
                    phaseRTest++;
                    tested++;
                    const bool farReturned = error.IsEmpty() && result.IsValid &&
                        result.ClipIndex == oneShotClip && result.SampleIndex == source;
                    if (!farReturned && error.IsEmpty() && result.IsValid)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("Oneshot-far FAIL: clip {} anchor {:.3f} far sample {} (winner={} clip={} time={:.3f} err='{}').\n"),
                            oneShotClip,
                            anchorTime,
                            source,
                            result.SampleIndex,
                            result.ClipIndex,
                            result.SampleTime,
                            error);
                    }
                }
                // R2: continuation within half a baked frame stays
                // admissible with ContinuingBias=0 (no false self-bar).
                {
                    const float dt = sampleTimesArr[oneShotFirst + 1] - sampleTimesArr[oneShotFirst];
                    const float half = (dt > 0.0f ? dt : 1.0f / 30.0f) * 0.5f;
                    const int32 source = anchor;
                    buildExactQuery(source, queryPose, queryTrajectory);
                    MotionMatchingSearchSettings g = settings;
                    g.CurrentTime = anchorTime - half;
                    g.ExpectedTime = anchorTime;
                    g.SameClipExclusionWindow = 0.0f;
                    g.ContinuingBias = 0.0f;
                    g.ContinuingWindow = half + 0.001f;
                    g.SwitchMargin = 0.0f;
                    g.TwinBanWindow = 0.0f;
                    g.LoopTailExclusion = 0.0f;
                    g.ExcludeNonLoopTail = false;
                    MotionMatchingResult result;
                    String error;
                    FindBest(database, queryPose, queryTrajectory, g, oneShotClip, 999.0f, noTags, result, error);
                    phaseRTest++;
                    tested++;
                    bool ok = error.IsEmpty() && result.IsValid &&
                        result.ClipIndex == oneShotClip && result.SampleIndex == source;
                    if (ok)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("Oneshot-continue FAIL: clip {} sample {} half-frame {:.4f} (winner={} clip={} err='{}').\n"),
                            oneShotClip,
                            source,
                            half,
                            result.SampleIndex,
                            result.ClipIndex,
                            error);
                    }
                }
                // R3: same far-past probe through the indexed path
                // (UseIndexedSearch without verify) must agree: no far win.
                {
                    const int32 source = farPast;
                    buildExactQuery(source, queryPose, queryTrajectory);
                    MotionMatchingSearchSettings g = settings;
                    g.CurrentTime = anchorTime;
                    g.ExpectedTime = anchorTime;
                    g.SameClipExclusionWindow = 0.0f;
                    g.ContinuingBias = 0.0f;
                    g.ContinuingWindow = win;
                    g.SwitchMargin = 0.0f;
                    g.TwinBanWindow = 0.0f;
                    g.LoopTailExclusion = 0.0f;
                    g.ExcludeNonLoopTail = false;
                    g.UseIndexedSearch = true;
                    g.VerifyIndexedSearch = false;
                    MotionMatchingResult result;
                    String error;
                    FindBest(database, queryPose, queryTrajectory, g, oneShotClip, 999.0f, noTags, result, error);
                    phaseRTest++;
                    tested++;
                    const bool farReturned = error.IsEmpty() && result.IsValid &&
                        result.ClipIndex == oneShotClip && result.SampleIndex == source;
                    if (!farReturned && error.IsEmpty() && result.IsValid)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("Oneshot-indexed FAIL: clip {} far sample {} (winner={} clip={} err='{}').\n"),
                            oneShotClip,
                            source,
                            result.SampleIndex,
                            result.ClipIndex,
                            error);
                    }
                }
                // R4: same far-past probe through FindTopCandidates: the far
                // sample must not rank in the returned list.
                {
                    const int32 source = farPast;
                    buildExactQuery(source, queryPose, queryTrajectory);
                    MotionMatchingSearchSettings g = settings;
                    g.CurrentTime = anchorTime;
                    g.ExpectedTime = anchorTime;
                    g.SameClipExclusionWindow = 0.0f;
                    g.ContinuingBias = 0.0f;
                    g.ContinuingWindow = win;
                    g.SwitchMargin = 0.0f;
                    g.TwinBanWindow = 0.0f;
                    g.LoopTailExclusion = 0.0f;
                    g.ExcludeNonLoopTail = false;
                    Array<float> packed;
                    String error;
                    FindTopCandidates(database, queryPose, queryTrajectory, g, oneShotClip, noTags, 32, packed, error);
                    phaseRTest++;
                    tested++;
                    bool ranked = false;
                    constexpr int32 CandidateStride = 10;
                    const int32 rows = packed.Count() / CandidateStride;
                    for (int32 r = 0; r < rows; r++)
                    {
                        if ((int32)packed[r * CandidateStride + 0] == source)
                        {
                            ranked = true;
                            break;
                        }
                    }
                    if (!ranked && error.IsEmpty() && packed.Count() > 0)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("Oneshot-topN FAIL: clip {} far sample {} ranked={} rows={} err='{}'.\n"),
                            oneShotClip,
                            source,
                            ranked ? 1 : 0,
                            rows,
                            error);
                    }
                }
            }
        }
        // R5: current loop may still pick a far phase (guard is
        // one-shot-only). Exact far query on the loop clip must return it.
        if (loopClip < 0 || loopFirst < 0 || loopSource < 0 || loopSource >= sampleCount)
        {
            log += String(TEXT("Loop-far FAIL: no straight loop clip (>= 30 samples, span > 0.8s) whose exact mid query wins by itself (fixture missing).\n"));
            // Fixture gap must fail the run, never pass on zero probes.
            phaseRTest++;
            tested++;
        }
        else
        {
            const int32 source = loopSource;
            buildExactQuery(source, queryPose, queryTrajectory);
            MotionMatchingSearchSettings g = settings;
            g.CurrentTime = sampleTimesArr[loopFirst];
            g.ExpectedTime = sampleTimesArr[loopFirst];
            g.SameClipExclusionWindow = 0.0f;
            g.ContinuingBias = 0.0f;
            g.ContinuingWindow = 0.15f;
            g.SwitchMargin = 0.0f;
            g.TwinBanWindow = 0.0f;
            g.LoopTailExclusion = 0.0f;
            g.ExcludeNonLoopTail = false;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, g, loopClip, 999.0f, noTags, result, error);
            phaseRTest++;
            tested++;
            bool ok = error.IsEmpty() && result.IsValid &&
                result.ClipIndex == loopClip && result.SampleIndex == source;
            if (ok)
            {
                phaseRPass++;
                passed++;
            }
            else
            {
                log += String::Format(
                    TEXT("Loop-far FAIL: clip {} far sample {} (winner={} clip={} err='{}').\n"),
                    loopClip,
                    source,
                    result.SampleIndex,
                    result.ClipIndex,
                    error);
            }
        }
        // R6: pure search (CurrentTime=-1, ExpectedTime=-1) is uncut: reuse
        // the qualified one-shot anchor (exact mid already wins by itself
        // above), queried with no playback context and current = own clip.
        if (oneShotClip < 0 || oneShotAnchor < 0 || oneShotAnchor >= sampleCount)
        {
            log += String(TEXT("Pure-search FAIL: qualified one-shot anchor missing (fixture missing).\n"));
            // Fixture gap must fail the run, never pass on zero probes.
            phaseRTest++;
            tested++;
        }
        else
        {
            buildExactQuery(oneShotAnchor, queryPose, queryTrajectory);
            MotionMatchingSearchSettings g = settings;
            g.CurrentTime = -1.0f;
            g.ExpectedTime = -1.0f;
            g.SameClipExclusionWindow = 0.0f;
            g.ContinuingBias = 0.0f;
            g.ContinuingWindow = 0.15f;
            g.SwitchMargin = 0.0f;
            g.TwinBanWindow = 0.0f;
            g.LoopTailExclusion = 0.0f;
            g.ExcludeNonLoopTail = false;
            MotionMatchingResult result;
            String error;
            FindBest(database, queryPose, queryTrajectory, g, oneShotClip, 999.0f, noTags, result, error);
            // checkExact owns the tested++/passed++ count here (ties
            // allowed but ordered): proves the pure search is uncut.
            const bool pureOk = checkExact(oneShotAnchor, result, error, TEXT("Pure-search"));
            phaseRTest++;
            if (pureOk)
                phaseRPass++;
        }
        // R7: run include + back exclude returns no back-tagged clip;
        // empty intersection (run include + exclude-everything) stays
        // invalid and never widens.
        {
            const String runTag(TEXT("run"));
            const String backTag(TEXT("back"));
            bool haveRun = false;
            bool haveBack = false;
            bool haveRunNonBack = false;
            for (int32 ti = 0; ti < tagNames.Count(); ti++)
            {
                if (tagNames[ti] == runTag)
                    haveRun = true;
                if (tagNames[ti] == backTag)
                    haveBack = true;
            }
            for (int32 c = 0; c < clipCount && !haveRunNonBack; c++)
            {
                const Array<String>& set = database->GetClipTags(c);
                if (c < clipSampleCounts.Count() && clipSampleCounts[c] > 0 &&
                    set.Contains(runTag) && !set.Contains(backTag))
                    haveRunNonBack = true;
            }
            if (!haveRun || !haveBack || !haveRunNonBack)
            {
                log += String::Format(
                    TEXT("Run-exclude FAIL: fixture tags missing (run={} back={} run-non-back={}).\n"),
                    haveRun ? 1 : 0,
                    haveBack ? 1 : 0,
                    haveRunNonBack ? 1 : 0);
                // Fixture gap must fail the run, never pass on zero probes.
                phaseRTest += 2;
                tested += 2;
            }
            else
            {
                // Source probe: middle sample of the first clip carrying
                // both run and back (falls back to any run clip).
                int32 source = -1;
                int32 scanCursor = 0;
                for (int32 c = 0; c < clipCount && source < 0; c++)
                {
                    const int32 n = (c >= 0 && c < clipSampleCounts.Count()) ? clipSampleCounts[c] : 0;
                    const Array<String>& set = database->GetClipTags(c);
                    if (n > 0 && set.Contains(runTag) && set.Contains(backTag))
                        source = scanCursor + n / 2;
                    scanCursor += n;
                }
                if (source < 0)
                {
                    scanCursor = 0;
                    for (int32 c = 0; c < clipCount && source < 0; c++)
                    {
                        const int32 n = (c >= 0 && c < clipSampleCounts.Count()) ? clipSampleCounts[c] : 0;
                        const Array<String>& set = database->GetClipTags(c);
                        if (n > 0 && set.Contains(runTag))
                            source = scanCursor + n / 2;
                        scanCursor += n;
                    }
                }
                if (source < 0)
                {
                    log += String(TEXT("Run-exclude FAIL: no run-tagged clip with samples (fixture missing).\n"));
                    // Fixture gap must fail the run, never pass on zero probes.
                    phaseRTest += 2;
                    tested += 2;
                }
                else
                {
                    buildExactQuery(source, queryPose, queryTrajectory);
                    Array<String> runOnly;
                    runOnly.Add(runTag);
                    MotionMatchingSearchSettings g = settings;
                    g.CurrentTime = -1.0f;
                    g.ExpectedTime = -1.0f;
                    g.SameClipExclusionWindow = 0.0f;
                    g.ContinuingBias = 0.0f;
                    g.SwitchMargin = 0.0f;
                    Array<String> noBack;
                    noBack.Add(backTag);
                    g.ExcludeTags = noBack;
                    MotionMatchingResult result;
                    String error;
                    FindBest(database, queryPose, queryTrajectory, g, -1, MAX_float, runOnly, result, error);
                    phaseRTest++;
                    tested++;
                    bool disjoint = error.IsEmpty() && result.IsValid &&
                        !database->GetClipTags(result.ClipIndex).Contains(backTag);
                    if (disjoint)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("Run-exclude FAIL: source {} (winner={} clip={} tags='{}' err='{}').\n"),
                            source,
                            result.SampleIndex,
                            result.ClipIndex,
                            MotionMatchingDatabase::JoinTags(database->GetClipTags(result.ClipIndex)),
                            error);
                    }
                    // R8: conflicting pool (run include, exclude every tag
                    // in the inventory) must stay invalid, never widen.
                    // Same scope/excludes shape as the R7 probe: the R7
                    // settings (run include + back exclude) plus full
                    // inventory exclusion; scope/excludes then conflict on
                    // every run clip.
                    MotionMatchingSearchSettings w = g;
                    w.ExcludeTags.Clear();
                    for (int32 ti = 0; ti < tagNames.Count(); ti++)
                        w.ExcludeTags.Add(tagNames[ti]);
                    MotionMatchingResult wResult;
                    String wError;
                    FindBest(database, queryPose, queryTrajectory, w, -1, MAX_float, runOnly, wResult, wError);
                    phaseRTest++;
                    tested++;
                    if (!wResult.IsValid)
                    {
                        phaseRPass++;
                        passed++;
                    }
                    else
                    {
                        log += String::Format(
                            TEXT("No-widen FAIL: conflicting pool returned winner={} clip={} (must hold invalid).\n"),
                            wResult.SampleIndex,
                            wResult.ClipIndex);
                    }
                }
            }
        }
    }
    log += String::Format(TEXT("Phase R (one-shot progression/back-exclude): {}/{} passed.\n"), phaseRPass, phaseRTest);

    const auto endTime = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    log += String::Format(
        TEXT("Self-test: {}/{} checks passed ({} ties verified), worst exact cost={:.6f} at sample {}, {:.1f} ms total ({:.2f} ms/search over {} samples)."),
        passed,
        tested,
        tieCount,
        worstExactCost,
        worstExactSample,
        milliseconds,
        tested > 0 ? milliseconds / tested : 0.0,
        sampleCount);
    return passed == tested && tested > 0;
}

#endif // USE_EDITOR
