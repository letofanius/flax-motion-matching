// MotionMatchingSearch.h: pose search backends (MotionMatchingSearch::FindBest/SampleCost/FindTopCandidates/RunSelfTest).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the brute-force pass is the reference backend (strictly-less comparison keeps ties on the lowest sample index); the indexed path retrieves top-K then exact-reranks with the same term helpers and falls back on any miss; SampleCost scores one baked sample with the hot-pass model for the policy continuation check; FindTopCandidates reports the unpruned raw cost landscape for tooling.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Types/String.h"
#include "MotionMatchingQuery.h"

class MotionMatchingDatabase;

/// <summary>
/// Brute-force pose search over the baked database.
/// This is the reference backend: every future index (KD/ANN/learned) must
/// reproduce its winners. Deterministic by construction — ties break toward
/// the lowest sample index via strictly-less comparison, so a captured query
/// always returns the same sample.
/// </summary>
API_CLASS(Static, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingSearch
{
    DECLARE_SCRIPTING_TYPE_NO_SPAWN(MotionMatchingSearch);

public:
    /// <summary>
    /// Finds the lowest-cost sample for raw (unnormalized) query features.
    /// Applies the per-dimension weight pipeline (sqrt of sum-normalized per-dim weight
    /// divided by the stored deviation) to the raw query vs raw baked values,
    /// an optional tag filter, a clip-change continuity penalty and a minimum
    /// switch lock. Early-exits blocks once the partial cost exceeds the best.
    /// </summary>
    /// <param name="database">Baked database. Must pass ValidateData.</param>
    /// <param name="queryPose">44 raw pose features (see MotionMatchingQuery).</param>
    /// <param name="queryTrajectory">30 raw trajectory features.</param>
    /// <param name="settings">Per-query weights, continuity and candidate budget.</param>
    /// <param name="currentClipIndex">Baked clip index currently playing, or -1 for none.</param>
    /// <param name="timeSinceSwitch">Seconds since the last clip switch. Below
    /// MinSwitchTime, clip changes are rejected unless nothing else is valid.</param>
    /// <param name="allowedTags">Baked clip tags allowed to win. Empty means all.</param>
    /// <param name="result">Winner (IsValid = false when nothing qualified).</param>
    /// <param name="error">Actionable reason when the search could not run.</param>
    API_FUNCTION()
    static void FindBest(
        MotionMatchingDatabase* database,
        const Array<float>& queryPose,
        const Array<float>& queryTrajectory,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        float timeSinceSwitch,
        const Array<String>& allowedTags,
        API_PARAM(Out) MotionMatchingResult& result,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Editor-only self-test without playback: rebuilds raw queries from
    /// baked samples (a straight copy, since the database stores raw values) and
    /// requires the search to return the source sample with ~zero cost.
    /// Proves the weight pipeline, cost math and tie-breaks.
    /// Guarded by USE_EDITOR (verified Flax 1.12 macro: Engine/Core/Config.h
    /// defaults it to 0, FlaxEditor.Build.cs defines it for editor targets;
    /// FLAX_EDITOR has zero hits in engine source). The implementation lives
    /// in MotionMatchingSearch.SelfTest.cpp (also USE_EDITOR-guarded), so
    /// Game Release contains no self-test code or strings.
    /// </summary>
    /// <returns>True when every probe passed. Details are appended to log.</returns>
#if USE_EDITOR
    API_FUNCTION()
    static bool RunSelfTest(
        MotionMatchingDatabase* database,
        API_PARAM(Out) String& log);
#endif

    /// <summary>
    /// Continuing fast-path: raw cost of ONE baked sample
    /// (same model as the hot pass: pose + trajectory blocks with the given
    /// weights plus the straightness prior, no continuity/loop terms). The
    /// policy scores the expected next sample of the playing clip every tick
    /// and skips the full scan while it stays within threshold.
    /// Turn-prior weights ride explicit params (provider tuning, trace-
    /// recorded); callers pass them verbatim, never literals.
    /// </summary>
    /// <returns>False with an actionable error on bad database, sizes or index.</returns>
    API_FUNCTION()
    static bool SampleCost(
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
        API_PARAM(Out) float& cost,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Diagnostics: exact top-N cost ranking for one query (tuning panel,
    /// trace analysis). Same filters and final-cost model as the hot pass
    /// (tags, loop filter, temporal exclusion, continuation bias, loop
    /// preference) but NO pruning, NO stride, NO continuity lock and NO
    /// switch-margin arbitration — this is the raw cost landscape. The live
    /// winner (margin/lock applied) may rank below top-1 here; the caller
    /// marks it by matching SampleIndex against the recorded result.
    /// Each candidate packs 10 floats: sampleIndex, clipIndex, sampleTime,
    /// cost, poseCost, trajectoryCost, continuityCost, loopCost,
    /// isContinuation (0/1), isLoop (0/1). Sorted ascending by cost; ties
    /// keep the lower sample index (same scan order as the hot pass).
    /// </summary>
    /// <param name="database">Baked database (samples, features, tags).</param>
    /// <param name="queryPose">Live pose feature vector.</param>
    /// <param name="queryTrajectory">Live trajectory feature vector.</param>
    /// <param name="settings">Cost weights and filter knobs.</param>
    /// <param name="currentClipIndex">Playback clip for continuation bias, -1 when idle.</param>
    /// <param name="allowedTags">Tag scope for this search.</param>
    /// <param name="topCount">Candidates to return, clamped to 1..32.</param>
    /// <param name="packed">10 floats per candidate (see above).</param>
    /// <param name="error">Actionable reason when the ranking could not run.</param>
    API_FUNCTION()
    static void FindTopCandidates(
        MotionMatchingDatabase* database,
        const Array<float>& queryPose,
        const Array<float>& queryTrajectory,
        const MotionMatchingSearchSettings& settings,
        int32 currentClipIndex,
        const Array<String>& allowedTags,
        int32 topCount,
        API_PARAM(Out) Array<float>& packed,
        API_PARAM(Out) String& error);
};
