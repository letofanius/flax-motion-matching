// MotionMatchingRuntimePolicy.h: per-tick policy interface (MotionMatchingRuntimePolicy::Initialize/Shutdown/Tick/ForcePlayClip, MotionMatchingPlayerSnapshot, MotionMatchingPolicyTuning, FastPathSignature).
// Ownership: plugin runtime, no game/host references.
// Key invariants: the core executes the provider caller filter verbatim (tag scope, LoopFilter, CandidateBand, yaw rate) and never infers gameplay; RunQuery anchors a fast-path signature that TryFastPath revalidates every tick; force-play is the only clip hatch besides search winners; ExportClipAudit/FlushTelemetry are explicit tooling-only file writers.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Math/Matrix.h"
#include "Engine/Core/Math/Vector2.h"
#include "Engine/Core/Math/Vector3.h"
#include "Engine/Scripting/ScriptingObject.h"
#include "MotionMatchingPlayback.h"
#include "MotionMatchingTrace.h"
#include "../Schema/MotionMatchingTypes.h"
#include "../Schema/PoseSearchSchema.h"
#include "../Schema/TrajectoryPoint.h"
#include "../Search/MotionMatchingQuery.h"

class AnimatedModel;
class MotionMatchingDatabase;

/// <summary>
/// Per-tick caller scope for the policy (plain data, supplied by the provider
/// selection policy through the C# frame). The core executes the caller
/// filter verbatim — it never infers locomotion, gait, intent, rest, or
/// airborne meaning. The provider computes the trajectory before the call and
/// owns all gameplay interpretation. Only the tag scope crosses.
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API MotionMatchingPlayerSnapshot
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(MotionMatchingPlayerSnapshot);

    /// <summary>
    /// Provider-decided tag scope, '+'-joined (e.g. "run+turn_travel").
    /// Always read (no fallback chooser). An EMPTY set is
    /// an explicit no-match: no candidate may win and the current clip
    /// holds (never fall back to the full pool). Plain String transport:
    /// the scripting marshaller cannot marshal Array&lt;String&gt; struct
    /// fields.
    /// </summary>
    API_FIELD() String AllowedTagSet;
    /// <summary>
    /// Controller-decided exclusions, '+'-joined. Merged into the search
    /// settings every query. Empty disables.
    /// </summary>
    API_FIELD() String ExcludedTagSet;
};

/// <summary>
/// Live tunables pushed by the C# controller every tick (plain data):
/// neutral search/playback tuning only (weights, budgets, anti-flicker,
/// straightness prior). The provider selection policy owns onset/fair-fight/
/// settle/rest and passes only the resulting LoopFilter/CandidateBand per tick
/// (Tick params, never inferred).
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API MotionMatchingPolicyTuning
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(MotionMatchingPolicyTuning);

    API_FIELD() float PoseWeight = 1.0f;
    API_FIELD() float TrajectoryWeight = 2.0f;
    API_FIELD() float SampleRate = 30.0f;
    API_FIELD() float MinSwitchTime = 0.12f;
    API_FIELD() float SameClipExclusion = 0.25f;
    API_FIELD() float SameClipSeekThreshold = 0.25f;
    API_FIELD() float ContinuingBias = 0.012f;
    API_FIELD() float ContinuingWindow = 0.15f;
    API_FIELD() float SwitchMargin = 0.008f;
    API_FIELD() int32 FrameSearchBudget = 12000;
    API_FIELD() bool SearchEveryFrame = true;
    API_FIELD() float QueryInterval = 0.1f;
    API_FIELD() bool UseIndexedSearch = true;
    API_FIELD() int32 TopKCandidates = 16;
    API_FIELD() bool VerifyIndexedSearch = false;
    API_FIELD() bool IndexedVerifyAuto = false;
    API_FIELD() bool EffectiveTelemetry = false;
    API_FIELD() float ContinuityWeight = 0.0025f;
    API_FIELD() float CostFuseThreshold = 2.0f;
    API_FIELD() float TurnPenaltyWeight = 0.03f;
    API_FIELD() float TurnPenaltyYawScale = 4.0f;

};

/// <summary>
/// C# animation command layer. Owns when to search, which
/// tags may win, and continuity — but never touches movement authority: it
/// only tells MotionMatchingPlayback which clip to show.
/// States are intentionally coarse (Locomotion/Airborne); start/stop/turn
/// emerge from the search itself.
/// </summary>
API_CLASS(Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingRuntimePolicy : public ScriptingObject
{
    DECLARE_SCRIPTING_TYPE_WITH_CONSTRUCTOR_IMPL(MotionMatchingRuntimePolicy, ScriptingObject);

public:
    API_FIELD() String PolicyState = TEXT("Disabled");
    API_FIELD() String ActiveTag;
    API_FIELD() int32 WinnerSample = -1;
    API_FIELD() int32 WinnerClip = -1;
    API_FIELD() String WinnerClipName;
    API_FIELD() float WinnerCost = MAX_float;
    API_FIELD() float SecondBestCost = MAX_float;
    API_FIELD() float WinnerPoseCost = MAX_float;
    API_FIELD() float WinnerTrajectoryCost = MAX_float;
    API_FIELD() float WinnerContinuityCost = 0.0f;
    API_FIELD() float WinnerLoopCost = 0.0f;
    API_FIELD() bool WinnerIsLoop = false;
    API_FIELD() float LoopPreferenceApplied = 0.0f;
    API_FIELD() float BestLoopCost = MAX_float;
    API_FIELD() int32 BestLoopClip = -1;
    API_FIELD() String BestLoopClipName;
    API_FIELD() float BestOneShotCost = MAX_float;
    API_FIELD() int32 BestOneShotClip = -1;
    API_FIELD() String BestOneShotClipName;
    API_FIELD() int32 LoopScoredCount = 0;
    API_FIELD() int32 OneShotScoredCount = 0;
    API_FIELD() int32 LoopTotalCount = 0;
    API_FIELD() int32 OneShotTotalCount = 0;
    API_FIELD() int32 SwitchCount = 0;
    API_FIELD() int32 SeekCount = 0;
    API_FIELD() float LastQueryMs = 0.0f;
    API_FIELD() String LastSwitchReason;
    API_FIELD() String ChooserStatusText;
    API_FIELD() int32 FastPathHits = 0;
    API_FIELD() float LastFastPathAge = 0.0f;
    API_FIELD() String ForcedReason;
    API_FIELD() float LastTimeError = 0.0f;
    // Additive profiling counters (record-only; never steer search or
    // playback). LastQueryMs above already measures the full-scan block
    // (FindBest + retries); the fields below split the remaining phases so
    // before/after runs can compare update/search/fast-path/I-O cost.
    // Whole-Tick wall time (pose sample + query/fast-path + playback).
    API_FIELD() float LastTickMs = 0.0f;
    // Continuation SampleCost block inside TryFastPath (hit path only).
    API_FIELD() float LastFastPathMs = 0.0f;
    // Synchronous telemetry file I/O (LogTelemetryLine per query).
    // Runtime query only enqueues a
    // structured record into a bounded in-memory ring (below); JSON
    // formatting + file write happen ONLY in the explicit FlushTelemetry
    // tooling call (session end, never in Tick/RunQuery). Telemetry-off
    // ticks record nothing (single branch, no allocation beyond the fixed
    // ring). Telemetry on/off never steers search (enqueue is record-only
    // after the winner is decided; CollectClassDiagnostics only fills
    // diagnostics, never changes the winner).
    API_FIELD() float LastTelemetryMs = 0.0f;
    API_FIELD() int32 TickCount = 0;
    API_FIELD() int32 FullSearchCount = 0;
    API_FIELD() float TotalTickMs = 0.0f;
    API_FIELD() float TotalFullSearchMs = 0.0f;
    API_FIELD() float TotalFastPathMs = 0.0f;
    API_FIELD() float TotalTelemetryMs = 0.0f;

    API_PROPERTY() bool GetIsReady() const;

    ~MotionMatchingRuntimePolicy();

    API_FUNCTION() void Initialize(
        MotionMatchingDatabase* database,
        AnimatedModel* body,
        const PoseSearchSchema& schema,
        const String& cacheFolder,
        float defaultInertial);
    API_FUNCTION() void Shutdown();
    API_FUNCTION() void ResetOnTeleport();
    API_FUNCTION() void Tick(
        float deltaTime,
        const MotionMatchingPlayerSnapshot& state,
        const Array<TrajectoryPoint>& trajectoryPoints,
        const Matrix& actorWorld,
        const Array<float>& trajectorySampleTimes,
        const MotionMatchingPolicyTuning& tuning,
        int32 callerLoopFilter,
        int32 callerCandidateBand,
        float callerYawRate,
        float callerPlanarSpeed);
    API_FUNCTION() MotionMatchingPlayback* GetPlayback();
    API_FUNCTION() MotionMatchingTraceLog* GetTrace();
    API_FUNCTION() MotionMatchingTraceEntry GetLastEntry() const;
    /// <summary>
    /// Single authoritative force-clip path (the only hatch besides the
    /// search-winner Plays inside RunQuery). Synchronizes playback clip/time,
    /// policy current clip/time, continuity baseline, fast-path invalidation,
    /// inertial state (via Play), telemetry reason and trace command entry.
    /// Search resumes immediately on the next tick with the caller filter.
    /// </summary>
    API_FUNCTION() void ForcePlayClip(int32 clipIndex, float startTime, const String& reason);
    /// <summary>
    /// Explicit editor/tool command: scans the baked database and writes
    /// Cache/MMClipAudit.txt on demand (index|name|tags|loop|turn|travel|net|
    /// yaw|len|speed). Static so it runs in edit mode with no policy/playback
    /// instance: init/play never call this and never produce the file.
    /// Returns false with an error when the database is not baked, the cache
    /// folder is unset, or the file cannot be written.
    /// </summary>
    API_FUNCTION() static bool ExportClipAudit(MotionMatchingDatabase* database,
        const String& cacheFolder, float sampleRate, API_PARAM(Out) String& error);
    /// <summary>
    /// Explicit telemetry flush (the ONLY writer of MMTelemetry.jsonl).
    /// Formats the bounded in-memory ring as JSONL + writes the file.
    /// Tooling/session-end only: Tick/RunQuery never do file I/O, they only
    /// RecordTelemetry (enqueue). Returns false with a visible error when
    /// the ring is empty, the cache folder is unset, or the file cannot be
    /// written. Overwritten drops are counted in GetTelemetryDropped.
    /// </summary>
    API_FUNCTION() bool FlushTelemetry(API_PARAM(Out) String& error);
    /// <summary>
    /// Explicit telemetry clear (no file deletion: init never touches
    /// MMTelemetry.jsonl; tools read whatever the last flush left behind).
    /// Resets the ring head/count/dropped to the bounded disabled state.
    /// </summary>
    API_FUNCTION() void ClearTelemetry();
    API_FUNCTION() int32 GetTelemetryCount() const;
    API_FUNCTION() int32 GetTelemetryDropped() const;

private:
    struct FastPathSignature
    {
        int32 AllowedHash = 0;
        int32 ExcludedHash = 0;
        int32 LoopFilter = 0;
        int32 CandidateBand = 0;
        int32 Clip = -1;
        int32 ClipTime = 0;
        int32 DatabaseId = 0;
    };

    MotionMatchingDatabase* _database = nullptr;
    AnimatedModel* _body = nullptr;
    MotionMatchingPlayback* _playback = nullptr;
    MotionMatchingTraceLog* _trace = nullptr;
    PoseSearchSchema _liveSchema;
    String _cacheFolder;
    NodeIndices _nodes;

    bool _samplingReady = false;
    Array<float> _prevPositions;
    bool _hasPrevPositions = false;
    bool _usedSearchSnapshot = false;
    int32 _searchSnapshotLength = 0;
    Array<float> _latestPose;
    Array<float> _packedTrajectory;
    Matrix _latestRootWorld = Matrix::Identity;
    bool _poseValid = false;
    Array<float> _tickTrajectory;
    float _lastScanTotal = MAX_float;
    float _lastFullScanTime = -99.0f;
    int32 _lastFullScanFrame = -1;
    int32 _consecutiveContinuationMisses = 0;
    FastPathSignature _lastScanSignature;
    bool _hasLastScanSignature = false;
    bool _fastPathReseed = true;
    int32 _policyFrame = 0;
    int64 _nextSearchFrameId = 1;
    int64 _lastObservedSampleFrameId = -1;
    int32 _fastPathFrames = 0;
    Array<int32> _clipSampleStarts;
    bool _hasSampleStarts = false;

    float _queryTimer = 0.0f;
    float _timeSinceSwitch = 999.0f;
    float _timeSinceInit = 0.0f;
    int32 _currentClip = -1;
    bool _hasLastActorPos = false;
    Float3 _lastActorPos = Float3::Zero;
    // Play-rate clip-speed cache (planar travel / length per clip, 0 when
    // unmeasurable). Built once per database, not per tick. Reuses the
    // audit path (baked root positions + clip lengths, no new bake).
    Array<float> _clipSpeeds;
    int32 _clipSpeedsDbId = 0;
    bool _warnedOnce = false;
    bool _bindingWarned = false;

    bool _chooserReseed = false;

    // Caller filter (provider selection policy output, Tick params): the ONLY
    // LoopFilter/CandidateBand/YawRate the core uses. Never inferred.
    int32 _callerLoopFilter = 0;
    int32 _callerCandidateBand = 0;
    float _callerYawRate = 0.0f;
    // Play-rate character speed (Game-side planar u/s, explicit Tick param).
    float _callerPlanarSpeed = 0.0f;

    // Init-time tag cache only (no motion scan, no audit file).
    Array<Array<String>> _clipTags;

    float _lastSpikeWarn = -99.0f;
    int32 _reselectClip = -1;
    float _reselectSampleTime = 0.0f;
    float _reselectStartTime = -99.0f;


    MotionMatchingPlayerSnapshot _snapshot;
    MotionMatchingPolicyTuning _tuning;
    MotionMatchingTraceEntry _lastEntry;
    // Steady-state reuse (all bounded, all record-only, never steer search):
    // - _queryTagsKey/_queryTagsCache: last AllowedTagSet split for RunQuery
    //   only (TryFastPath uses its own _fastAllowedKey/_fastAllowedCache
    //   below, never this pair). Bound: tag count per query
    //   (<=8 on the reference DB). Rebuilt only when the joined scope string
    //   changes (exact match); steady-state (same scope) reuses without
    //   Split per query/tick.
    // - _queryExcludedKey/_queryExcludedCache/_queryExcludedJoin: same rule
    //   for the ExcludedTagSet split and its '+'-joined form for RunQuery
    //   settings + trace/telemetry entries (TryFastPath uses its own
    //   _fastExcludedKey/_fastExcludedCache/_fastExcludedJoin below).
    // - _poseScratch/_prevScratch: Tick pose-sample scratch (44/21 floats).
    //   Passed to the samplers instead of per-tick locals; Resize only when
    //   the feature size changes (schema-fixed sizes).
    // - _tickTrajScratch: BuildLiveTrajectory scratch (30 floats). Same rule.
    // The per-tick member trajectory copies stay deleted (RunQuery uses
    // its params + _tickTrajectory). _snapshot itself is assigned field-wise
    // only on change (Tick) so steady-state copies no native Strings.
    String _queryTagsKey;
    Array<String> _queryTagsCache;
    String _queryExcludedKey;
    Array<String> _queryExcludedCache;
    String _queryExcludedJoin;
    // TryFastPath incumbent-check caches: SEPARATE keys from the RunQuery
    // pair above on purpose. The fast path runs before RunQuery on the same
    // tick; sharing one key would let the incumbent check consume the
    // scope-change signal, so RunQuery would then see a false cache hit and
    // skip the ActiveTag/ChooserStatus rebuild (stale chooser after a filter
    // change or reset). Same bound + same change-only rule.
    String _fastAllowedKey;
    Array<String> _fastAllowedCache;
    String _fastExcludedKey;
    Array<String> _fastExcludedCache;
    String _fastExcludedJoin;
    Array<float> _poseScratch;
    Array<float> _prevScratch;
    Array<float> _tickTrajScratch;

    static float ClampF(float value, float minValue, float maxValue);
    static float MaxF(float a, float b);
    static float MinF(float a, float b);
    static float AbsF(float value);
    static int32 MaxI(int32 a, int32 b);

    float SelectPlayRate(float characterPlanarSpeed);
    void EnsureClipSpeedCache();
    // Init caches tag SETS only (EnsureClipTagCache) — the motion scan
    // and accel log live inside the explicit static ExportClipAudit command,
    // never in Initialize/Tick.
    void EnsureClipTagCache();
    void CheckSchemaBinding();
    static bool AxesNear(const Float3& a, const Float3& b);
    static bool TimesEqual(const Array<float>& a, const Array<float>& b);
    void RecordPolicyEvent(const char* kind, int32 clipIndex, const String& reason);
    void RunQuery(const Array<TrajectoryPoint>& trajectoryPoints, const Matrix& actorWorld, float frameDt);
    bool TryFastPath(float frameDt);
    bool MissFastPath(const char* reason);
    static int32 Quantize(float value, float step);
    static int32 HashTagSet(const String& tagSet);
    FastPathSignature CaptureFastPathSignature() const;
    static bool FastPathSignaturesDiffer(const FastPathSignature& a, const FastPathSignature& b);
    // Change-only scope caches (RunQuery pair + fast-path-only twins +
    // record paths): Split/Join run only when the joined snapshot string changes.
    // Returned references stay valid until the next scope change; callers
    // must not hold them across a Tick that can change the scope.
    const Array<String>& GetCachedAllowed();
    const Array<String>& GetCachedExcluded();
    const String& GetCachedExcludedJoin();
    // Fast-path-only twins (see the member comment above): the incumbent
    // eligibility check must never consume RunQuery's change signal.
    const Array<String>& GetFastAllowed();
    const Array<String>& GetFastExcluded();
    const String& GetFastExcludedJoin();
    void NotifySearch();
    void EnsureSampleStarts();
    void RecordTrace(float speed, float yawRate, const Array<String>& tags,
        const Array<float>& trajectory, const MotionMatchingSearchSettings& settings,
        int32 continuityClip, float sinceSwitch, int32 retryKind, bool forcedEdge, bool reseeded,
        bool fairFight, bool oneshotWindow, float frameDt, float playRate,
        bool valid, const MotionMatchingResult& result, const String& reason);
    static float Wrap180(float a);
    void WarnOnce(const String& message);
    // Bounded telemetry ring (no per-query file I/O): runtime query only
    // I/O). Runtime query only copies numbers + one Tags string into the
    // fixed ring (no JSON formatting, no file I/O in Tick). FlushTelemetry
    // (explicit tooling) formats + writes. Overflow overwrites the oldest
    // and counts drops (visible via GetTelemetryDropped + flush log).
    struct TelemetryRecord
    {
        float Time = 0.0f;
        float YawRate = 0.0f;
        int32 ClipIndex = -1;
        int32 SampleIndex = -1;
        float Cost = MAX_float;
        float PoseCost = MAX_float;
        float TrajectoryCost = MAX_float;
        float ContinuityCost = 0.0f;
        float LoopCost = 0.0f;
        float SecondBestCost = MAX_float;
        float QueryMs = 0.0f;
        int32 SwitchCount = 0;
        int32 SeekCount = 0;
        int32 CurrentClip = -1;
        float PlaybackTime = -1.0f;
        float PlayRate = 1.0f;
        float EffectiveDt = 0.0f;
        int32 Cycle = 0;
        float TimeError = 0.0f;
        int32 BestLoopClip = -1;
        float BestLoopCost = MAX_float;
        int32 BestOneShotClip = -1;
        float BestOneShotCost = MAX_float;
        int32 LoopScored = 0;
        int32 OneShotScored = 0;
        int32 LoopTotal = 0;
        int32 OneShotTotal = 0;
        float LoopPreference = 0.0f;
        Float3 ActorPos = Float3::Zero;
        String Tags;
    };
    static constexpr int32 TelemetryCapacity = 4096;
    TelemetryRecord _telemetryRing[TelemetryCapacity];
    int32 _telemetryHead = 0;
    int32 _telemetryCount = 0;
    int32 _telemetryDropped = 0;
    void RecordTelemetry(float yawRate, const Array<String>& tags,
        const MotionMatchingResult& result, const Float3& actorPos, float timeError);
    String GetClipName(int32 clipIndex);
    // Normalized tag SET snapshot (EnsureClipTagCache at init) with DB fallback.
    const Array<String>& ClipTags(int32 clipIndex);
};
