// MotionMatchingQuery.h: search settings, result, and raw query-feature builders (MotionLoopFilter, MotionMatchingSearchSettings, MotionMatchingResult, MotionMatchingQuery).
// Ownership: plugin runtime, no game/host references.
// Key invariants: schema weights stay unbaked and ride per-query settings; raw pose (44) / trajectory (30) features mirror the baker convention exactly; temporal exclusion, reselect/twin bans, tail validity, stride cursor and index route compose so replay from a trace entry stays bit-exact.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector2.h"
#include "Engine/Core/Types/String.h"
#include "../Schema/PoseSearchSchema.h"
#include "../Schema/TrajectoryPoint.h"

class AnimatedModel;
class MotionMatchingDatabase;

/// <summary>
/// General loop-mode filter for the search. Carries no gait/tag semantics:
/// the chooser combines it with tags (e.g. rest recovery = same tags +
/// LoopOnly instead of a hardcoded "idle" tag).
/// </summary>
API_ENUM()
enum class MotionLoopFilter
{
    Any = 0,
    LoopOnly = 1,
    OneShotOnly = 2,
};

/// <summary>
/// Search-time weights and policy. Defaults mirror the initial search contract.
/// These are per-query overrides: the C# policy layer is responsible for seeding
/// them from the database schema snapshot (schema weights are stored, never baked).
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API MotionMatchingSearchSettings
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(MotionMatchingSearchSettings);

    API_FIELD() float PoseWeight = 1.0f;
    API_FIELD() float TrajectoryWeight = 2.0f;
    API_FIELD() float ContinuityWeight = 0.25f;
    API_FIELD() float MinSwitchTime = 0.12f;

    /// <summary>Maximum samples evaluated per search. 0 means no budget (full scan).</summary>
    API_FIELD() int32 MaxCandidates = 0;

    /// <summary>
    /// Current playback time (seconds) inside currentClipIndex. Negative means
    /// unknown (first play, no clip). Used for same-clip temporal exclusion:
    /// candidates on the current clip within SameClipExclusionWindow seconds of
    /// this time are skipped, so a query never reseeks to a nearby frame every
    /// tick. Real discontinuities (large time jumps, clip changes, end recovery)
    /// are outside the window and still win fairly.
    /// </summary>
    API_FIELD() float CurrentTime = -1.0f;

    /// <summary>
    /// Half-width (seconds) of the same-clip exclusion window around CurrentTime.
    /// 0 disables temporal exclusion (no exclusion window). Default 0.25s covers the
    /// query cadence (0.1s) plus playback advance between queries.
    /// </summary>
    API_FIELD() float SameClipExclusionWindow = 0.25f;

    /// <summary>
    /// Expected playback time (seconds) at the next query, i.e. CurrentTime +
    /// query cadence. Negative disables all time-based terms (continuing bias,
    /// switch margin). The policy sets this from the playing clip; callers
    /// without playback leave it negative for pure cost search.
    /// </summary>
    API_FIELD() float ExpectedTime = -1.0f;

    // Fallback defaults when a caller leaves these unset — the C# policy
    // overrides all three from live scene fields, so retune there, not here.
    /// <summary>
    /// GASP-style continuing bias: subtracted from the cost of same-clip
    /// samples within ContinuingWindow of ExpectedTime (the natural next
    /// frames). Rewards playing on instead of switching for a marginally
    /// better pose elsewhere. Folded into ContinuityCost telemetry so
    /// Pose + Trajectory + Continuity still equals Cost.
    /// </summary>
    API_FIELD() float ContinuingBias = 0.25f;

    /// <summary>Half-width (seconds) around ExpectedTime where ContinuingBias applies.</summary>
    API_FIELD() float ContinuingWindow = 0.15f;

    /// <summary>
    /// Switch hysteresis on cost: an other-clip winner replaces the best
    /// in-window same-clip continuation only if it wins by more than this
    /// margin. Kills turn/stop flicker where near-twin clips trade the lead
    /// every query. 0 disables (pure lowest-cost wins).
    /// </summary>
    API_FIELD() float SwitchMargin = 0.1f;

    /// <summary>
    /// Rest-loop preference: subtracted from the cost of loop-flagged clips
    /// when positive. The policy enables it only at rest-calm so a finished
    /// one-shot (break/stop end stance) cannot re-win on exact pose match
    /// forever — the loop outranks it structurally. 0 disables.
    /// </summary>
    API_FIELD() float LoopPreference = 0.0f;

    /// <summary>
    /// Loop-mode prefilter, applied before tags. Lets recovery paths demand
    /// loop clips structurally (LoopOnly) without naming any gait tag.
    /// </summary>
    API_FIELD() MotionLoopFilter LoopFilter = MotionLoopFilter::Any;

    /// <summary>
    /// Reselect-history ban (UE PoseReselectHistory analogue, cost-domain):
    /// candidates on ReselectClip within ReselectBanWindow seconds (sample
    /// time, modulo for loops) of ReselectSampleTime can never win. The C#
    /// policy owns the 0.3s wall-clock lifetime (it clears ReselectClip to -1
    /// once the ban expires) and records the applied pair in the trace, so
    /// replay stays bit-exact. ReselectClip &lt; 0 or window &lt;= 0 disables.
    /// Default off so direct callers (self-test) see the uncut scope.
    /// </summary>
    API_FIELD() int32 ReselectClip = -1;
    API_FIELD() float ReselectSampleTime = 0.0f;
    API_FIELD() float ReselectBanWindow = 0.0f;

    /// <summary>
    /// Same-segment twin ban (UE PoseJumpThresholdTime analogue): same-clip,
    /// non-continuation candidates whose distance to ExpectedTime falls in
    /// (ContinuingWindow, ContinuingWindow + TwinBanWindow] are skipped.
    /// Kills hops to near-twin frames just outside the continuation core
    /// while the true continuation stays exempt. Requires playback context
    /// (same gate as the time terms); 0 disables. Default off (self-test safe).
    /// </summary>
    API_FIELD() float TwinBanWindow = 0.0f;

    /// <summary>
    /// Loop-tail exclusion (UE ExcludeFromDatabase (0,-0.3) analogue): the
    /// last LoopTailExclusion seconds of NON-loop clips are not searchable,
    /// computed at runtime from clip lengths + sample times (no rebake).
    /// Clips shorter than the exclusion keep full scope. 0 disables.
    /// Default off (self-test safe); the runtime policy sets 0.3s.
    /// </summary>
    API_FIELD() float LoopTailExclusion = 0.0f;

    /// <summary>
    /// Structural candidate validity: a non-loop candidate whose
    /// future trajectory target (sampleTime + the schema's furthest trajectory
    /// time) extends past the clip end baked a clamped "stopped" future for
    /// playback; such a candidate is excluded from search rather than merely
    /// penalized (the bake has no honest feature to rank). Loop candidates are
    /// never affected: their TimeDistance is already cycle-aware. This is a
    /// structural rule, not a tuning knob. Default ON for runtime; the offline
    /// self-test disables it to probe the raw baked tail.
    /// </summary>
    API_FIELD() bool ExcludeNonLoopTail = true;

    /// <summary>
    /// Body yaw rate (rad/s) at query time. Caller-provided (Game owns yaw
    /// meaning): drives the straightness prior below. Reconstructed from
    /// trace yaw on replay, so determinism is preserved.
    /// </summary>
    API_FIELD() float QueryYawRate = 0.0f;

    /// <summary>
    /// Straightness prior (previously hardcoded 0.03/4.0 in MotionMatchingSearch):
    /// penalty = |clipTurnAngle| * TurnPenaltyWeight /
    /// (1 + QueryYawRate * TurnPenaltyYawScale). Game-owned tuning (values
    /// preserved from the hardcoded calibration); recorded in the trace so
    /// brute-force, indexed and replay share one formula.
    /// </summary>
    API_FIELD() float TurnPenaltyWeight = 0.03f;
    API_FIELD() float TurnPenaltyYawScale = 4.0f;

    /// <summary>
    /// Onset one-shot gait filter (0 = Any). When set (Walk/Run/Sprint),
    /// ONE-SHOT candidates whose baked speed band disagrees are skipped, so a
    /// walk onset cannot coin-flip a run start: at the onset instant the
    /// features carry almost no gait information, only intent does. Loops are
    /// never band-cut. The policy sets this from desired speed on
    /// speed-increase onsets only; stale databases without band data report
    /// Any for every clip (no behavior change, still gated stale by
    /// SchemaContractRevision).
    /// </summary>
    API_FIELD() int32 OneShotBand = 0;

    /// <summary>
    /// Per-frame rotating-stride cursor. When MaxCandidates budgets a strided
    /// scan, each query starts at (QueryFrame % stride) so consecutive frames
    /// cover disjoint subsets and full coverage completes over `stride`
    /// frames. The policy passes its monotonic frame counter (reset per play
    /// session), keeping replay bit-exact. Continuation samples are exempt
    /// from striding (the hold must survive every frame).
    /// </summary>
    API_FIELD() int32 QueryFrame = 0;

    /// <summary>
    /// Indexed search: route FindBest through the per-tag index
    /// (top-K retrieval + exact rerank) instead of the brute-force
    /// SearchPass. Default OFF: brute force stays the reference backend and
    /// the indexed path must reproduce its winners (indexed-verify harness) before
    /// any flip. The indexed path ignores MaxCandidates stride (it searches
    /// the full tag scope, strictly better coverage) and falls back to
    /// SearchPass whenever it cannot produce a valid winner.
    /// </summary>
    API_FIELD() bool UseIndexedSearch = false;

    /// <summary>
    /// Candidates retrieved per tag from the index before exact rerank
    /// (over-retrieved x3 internally so post-retrieval filters rarely exhaust
    /// the set). Clamped to 1..256 at query time. Ignored unless
    /// UseIndexedSearch is set.
    /// </summary>
    API_FIELD() int32 TopKCandidates = 32;

    /// <summary>
    /// Indexed-verify harness: run the brute-force reference alongside the
    /// indexed path on every query, count winner divergences to the log, and
    /// keep the BRUTE-FORCE winner (behavior unchanged). Requires
    /// UseIndexedSearch. Permanent debug harness (safe to enable after any flip:
    /// setting it then audits the live path against the reference).
    /// </summary>
    API_FIELD() bool VerifyIndexedSearch = false;

    /// <summary>Collects exact loop/one-shot diagnostics with an extra full scan.</summary>
    API_FIELD() bool CollectClassDiagnostics = false;

    /// <summary>
    /// Multi-tag exclusion set: candidates carrying ANY of these tags are
    /// skipped (disjoint enforcement), independent of the overlap-based
    /// include scope above. Empty disables (default: no behavior change).
    /// Include and exclude compose: a candidate must overlap the allowed set
    /// (when non-empty) AND carry none of the excluded tags.
    /// Plain C++ member (NOT API_FIELD: the scripting marshaller cannot
    /// marshal Array&lt;String&gt; struct fields). C++ callers only; C#
    /// mirrors never see it and always marshal it as empty.
    /// </summary>
    Array<String> ExcludeTags;
};

/// <summary>
/// Winner of one pose search. SecondBestCost is telemetry for tuning.
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API MotionMatchingResult
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(MotionMatchingResult);

    API_FIELD() bool IsValid = false;
    API_FIELD() int32 SampleIndex = -1;
    API_FIELD() int32 ClipIndex = -1;
    API_FIELD() float SampleTime = 0.0f;
    API_FIELD() float Cost = MAX_float;
    API_FIELD() float SecondBestCost = MAX_float;

    /// <summary>Weighted pose-block contribution to Cost (telemetry).</summary>
    API_FIELD() float PoseCost = MAX_float;
    /// <summary>Weighted trajectory-block contribution to Cost (telemetry).</summary>
    API_FIELD() float TrajectoryCost = MAX_float;
    /// <summary>Continuity penalty contribution to Cost (telemetry).</summary>
    API_FIELD() float ContinuityCost = 0.0f;
    /// <summary>Loop-preference bonus applied to Cost, negative or zero (telemetry).</summary>
    API_FIELD() float LoopCost = 0.0f;

    /// <summary>
    /// Best raw-blocks cost among loop clips in the tag+filter scope, from an
    /// UNPRUNED full scan (exact — never pruned, no continuity/bias/preference).
    /// This is the tune surface for LoopPreference. MAX_float = class absent.
    /// </summary>
    API_FIELD() float BestLoopCost = MAX_float;
    API_FIELD() int32 BestLoopSample = -1;
    API_FIELD() int32 BestLoopClip = -1;
    /// <summary>Same for one-shot clips.</summary>
    API_FIELD() float BestOneShotCost = MAX_float;
    API_FIELD() int32 BestOneShotSample = -1;
    API_FIELD() int32 BestOneShotClip = -1;
    /// <summary>Tag+filter members per class (full-scan scope size).</summary>
    API_FIELD() int32 LoopTotalCount = 0;
    API_FIELD() int32 OneShotTotalCount = 0;
    /// <summary>Fully-scored members per class in the pruned hot pass.</summary>
    API_FIELD() int32 LoopScoredCount = 0;
    API_FIELD() int32 OneShotScoredCount = 0;
};

/// <summary>
/// Builds raw (unnormalized) query features in the exact convention the baker used,
/// so MotionMatchingSearch can scale them with the stored per-dimension deviations.
/// Pose: 44 floats, 7 bones [pelvis, feet, spine, head, hands] positions (21)
/// then velocities (21) plus 2 foot-contact flags, all root-relative.
/// Trajectory: 30 floats, 6 points x [posRight, posForward, dirRight, dirForward, yawRate].
/// The pointer core avoids remarshalling; the API wrappers adapt C#-owned
/// float arrays. No bone math is duplicated in C#.
/// </summary>
API_CLASS(Static, Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingQuery
{
    DECLARE_SCRIPTING_TYPE_NO_SPAWN(MotionMatchingQuery);

public:
    /// <summary>
    /// Samples live bone transforms and writes 44 raw pose features.
    /// prevPositions persists across frames (7 root-relative positions);
    /// the first call seeds it and reports zero velocity.
    /// </summary>
    /// <returns>False with an actionable error when the model or nodes are unusable.</returns>
    static bool SamplePoseFeatures(
        AnimatedModel* model,
        const NodeIndices& nodes,
        float deltaTime,
        Float3* prevPositions,
        float* outPose,
        String& error);

    /// <summary>
    /// Schema-aware core: same convention as SamplePoseFeatures but the
    /// contact height uses schema.RootUpAxis (normalized) through the shared
    /// PoseSearchSchema helper instead of hardcoded .Y. The default-axis overload
    /// delegates here with a default schema (up = (0,0,-1)) so existing
    /// callers (policy) keep compiling unchanged.
    /// </summary>
    /// <returns>False with an actionable error when the model, nodes or axis are unusable.</returns>
    static bool SamplePoseFeatures(
        AnimatedModel* model,
        const NodeIndices& nodes,
        float deltaTime,
        Float3* prevPositions,
        const PoseSearchSchema& schema,
        float* outPose,
        String& error);

    /// <summary>
    /// Converts 6 actor-local trajectory points (X = right, Z = forward, as produced
    /// by MotionMatchingTrajectory) into 30 raw trajectory features in the root
    /// node's local frame, mirroring the baker's planar projection exactly
    /// (plus the yaw-rate channel derived from facing deltas).
    /// </summary>
    /// <returns>False with an actionable error when the schema axes are degenerate.</returns>
    static bool BuildRawTrajectory(
        const TrajectoryPoint* points,
        const Matrix& actorWorld,
        const Matrix& rootWorld,
        const PoseSearchSchema& schema,
        float* outTrajectory,
        String& error);

    /// <summary>
    /// C#-facing live pose sampler. Reads the animated body's current bone
    /// transforms and writes 44 raw pose features plus the root world matrix
    /// (needed to map the trajectory into the root frame). prevPositions is
    /// 21 floats (7 root-relative xyz) owned by the caller across frames.
    /// </summary>
    API_FUNCTION()
    static bool SampleLivePose(
        AnimatedModel* animatedModel,
        const NodeIndices& nodes,
        float deltaTime,
        const Array<float>& prevPositions,
        API_PARAM(Out) Array<float>& outPose,
        API_PARAM(Out) Array<float>& outPrevPositions,
        API_PARAM(Out) Matrix& outRootWorld,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Schema-aware live sampler: same as SampleLivePose but contact uses
    /// the provided schema up axis. Kept as a separately-named API entry so
    /// the existing policy call-site compiles untouched; new callers should
    /// prefer this and pass the database/baked schema snapshot.
    /// </summary>
    API_FUNCTION()
    static bool SampleLivePoseWithSchema(
        AnimatedModel* animatedModel,
        const NodeIndices& nodes,
        float deltaTime,
        const Array<float>& prevPositions,
        const PoseSearchSchema& schema,
        API_PARAM(Out) Array<float>& outPose,
        API_PARAM(Out) Array<float>& outPrevPositions,
        API_PARAM(Out) Matrix& outRootWorld,
        API_PARAM(Out) String& error);

    /// <summary>
    /// Query source: samples features from the playback's saved
    /// pre-layer/pre-IK search snapshot (bone-LOCAL matrices) instead of the
    /// displayed pose, so the layer and foot IK never feed back into the
    /// next frame's candidate costs. Composes with the shared sampler FK, then
    /// runs the same feature core as the live path (no duplicated math).
    /// Falls back to SampleLivePoseWithSchema when no snapshot exists yet.
    /// </summary>
    API_FUNCTION()
    static bool SampleSearchPose(
        AnimatedModel* animatedModel,
        const Array<Matrix>& searchLocals,
        const NodeIndices& nodes,
        float deltaTime,
        const Array<float>& prevPositions,
        const PoseSearchSchema& schema,
        API_PARAM(Out) Array<float>& outPose,
        API_PARAM(Out) Array<float>& outPrevPositions,
        API_PARAM(Out) Matrix& outRootWorld,
        API_PARAM(Out) String& error);

    /// <summary>
    /// C#-facing trajectory builder. points packs 6 actor-local points as
    /// 36 floats (pos xyz + facing xyz each), matching TrajectoryPoint layout.
    /// </summary>
    API_FUNCTION()
    static bool BuildLiveTrajectory(
        const Array<float>& points,
        const Matrix& actorWorld,
        const Matrix& rootWorld,
        const PoseSearchSchema& schema,
        API_PARAM(Out) Array<float>& outTrajectory,
        API_PARAM(Out) String& error);
};
