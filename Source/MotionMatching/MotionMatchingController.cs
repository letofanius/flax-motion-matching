using System;
using System.IO;
using FlaxEngine;

namespace MotionMatching;

/// <summary>
/// Scene-detached runtime for the native motion-matching system.
///
/// Purpose: own the animation database, the native policy/playback session
/// lifecycle, and core-only search tuning.
/// Ownership: plugin runtime object. The host facade owns this object, feeds
/// it a prebuilt <see cref="MotionMatchingFrameInput"/> every tick (it calls
/// the provider's BuildInput itself) and drives the overlay post-pass + pose
/// write.
/// Tick/lifecycle role: Initialize loads the database; Update runs one
/// policy + playback tick; Shutdown releases the native session while this
/// object stays reusable. Everything locomotion (gait scope, trajectory
/// prediction, gait speeds) arrives through the frame argument.
/// This file must never reference game/player/sample types or host drivers
/// (player state, coordinators, automation, overlays): those live host-side.
/// </summary>
public class MotionMatchingController : IDisposable
{
    /// <summary>
    /// Bake contract authored by the host. A snapshot is stored inside the database on bake;
    /// </summary>
    public PoseSearchSchema Schema = CreateDefaultSchema();

    public SkinnedModel PlayerModel;

    public string DatabasePath = "MotionMatching/MotionMatchingDatabase.flax";

    public MotionMatchingDatabase Database;

    public int AnimationClipsCount;

    public int SampleCount;

    public bool PersistenceValid;

    public string[] AnimationTags = Array.Empty<string>();

    public int[] AnimationClipCountsByTag = Array.Empty<int>();

    // Host-assigned session config (set by the host facade — this class
    // never reads host automation itself).
    public string CacheFolder = string.Empty;
    public bool TelemetryOverride = false;
    public bool IndexedVerifyAuto = false;
    // Trace override for test runs: replay trace stays on even when the
    // telemetry file is off, so on/off search equivalence can be proven by
    // comparing winners. Normal play (no flags) leaves both false → no
    // per-query allocations beyond the fixed rings.
    public bool TraceOverride = false;

    public AnimatedModel CharacterBody;

    public float QueryInterval = 0.1f;

    // Per-frame search: the query gate runs every warmed tick instead of on
    // the QueryInterval cadence, so input latency never waits for a tick
    // boundary. Full scans stay budgeted (FrameSearchBudget) with a rotating
    // stride + continuation exemption, and intent-change queries always scan
    // unstrided — same total CPU as cadence, minimal worst-case latency.
    public bool SearchEveryFrame = true;

    // Database samples scored per full scan (rotating stride covers the rest over
    // following frames). 12000 ≈ 1ms on the reference machine. 0 = unstrided.
    public int FrameSearchBudget = 12000;

    // Indexed path is live by default (earlier audit: 0.33%
    // same-clip-twin divergence, suite parity gate below).
    // Brute force stays as fallback + permanent verify harness.
    public bool UseIndexedSearch = true;

    // Candidates retrieved per tag before exact rerank. 16 (measured
    // 2026-09-14: sub-1ms scans, 0/1500 verify divergence, 12/12 suite).
    public int TopKCandidates = 16;

    // Run brute force alongside the indexed path, count divergences, keep
    // the brute-force winner. Permanent debug harness.
    public bool VerifyIndexedSearch = false;

    // Foot-IK solving lives host-side (single-publish pipeline through the
    // host facade); the core keeps only search contact features (bake +
    // query), which are untouched.

    // NOTE: layer decisions (procedural angle, bone gate) live host-side
    // now (post-pass in the host facade): the provider snapshot still
    // carries the host-authored angle/gate for the host pass, but nothing
    // is forwarded into the core anymore.

    // Switch penalty added to any different-clip winner (continuity block).
    // Rev-3 cost units (N1b shrank dissimilarity ~100x): 0.0025 restores the
    // old ratio against typical winner costs (~0.01-0.2 measured).
    public float ContinuityWeight = 0.0025f;

    public float MinSwitchTime = 0.12f;

    // Continuation stickiness (live tuning, no rebake): bias rewards the
    // natural next frames on the current clip, margin forces other clips to
    // beat them clearly. Rev-3 cost units: N1b shrank dissimilarity ~100x
    // (measured mean winner cost 0.095), so both scale down equally to keep
    // the old hold/switch ratios. UE philosophy is the same direction (tiny
    // -0.01 bias + interrupt logic instead of giant margins).
    public float ContinuingBias = 0.012f;

    public float ContinuingWindow = 0.15f;

    public float SwitchMargin = 0.008f;

    // Onset/fair-fight accumulation lives in the host selection policy,
    // which passes only the resulting filter per tick. Previously tuned
    // values are preserved host-side (40/0.25).

    // Heuristic thresholds below were hardcoded constants: exposed live so
    // tuning moves numbers instead of code. No rebake needed. (The old
    // settle-delay branch is deleted; settle is owned by the host policy.)

    public float SameClipExclusion = 0.25f;

    public float SameClipSeekThreshold = 0.25f;

    // Turn-driver config moved to the host facade (turn angle / lookahead /
    // settle rate / patient angle / patient time). The native turn machine
    // is deleted and never reads these; the host-side turn driver is the
    // only reader. TurnMinSwitchTime died with them: it was never pushed
    // into the native tuning (no reader in the native headers) and no
    // driver reads it.

    // Cost fuse: a switched winner above this is treated as a corrupted query
    // (observed: single-query pose spike mid steady walk, self-heals next
    // query) — hold current instead of cascading. Rev-3 units: legit switches
    // peak ~0.9 measured, so 2.0 keeps headroom without ever firing legally.
    public float CostFuseThreshold = 2.0f;

    // Straightness prior (was hardcoded 0.03/4.0 in MotionMatchingSearch):
    // host-owned tuning, values preserved from the calibration. Recorded in
    // the trace so brute-force, indexed and replay share one formula.
    public float TurnPenaltyWeight = 0.03f;
    public float TurnPenaltyYawScale = 4.0f;

    // Inertialization decay time (replaces linear-crossfade BlendDuration;
    // single path, no parallel blend modes).
    public float InertializationDuration = 0.25f;

    public bool TelemetryLog = false;

    // Effective switch: the host field above never changes at runtime —
    // the host flips TelemetryOverride (e.g. from file-driven automation),
    // so the scene is never dirtied by a play session.
    public bool EffectiveTelemetry => TelemetryLog || TelemetryOverride;

    // Telemetry mirrors (getters: written only by Update below, read by the
    // game facade / editor service / bridge).
    public string PolicyState { get; private set; } = "Disabled";
    public int WinnerSample { get; private set; } = -1;
    public int WinnerClip { get; private set; } = -1;
    public string WinnerClipName { get; private set; } = string.Empty;
    public float WinnerCost { get; private set; } = float.MaxValue;
    public float SecondBestCost { get; private set; } = float.MaxValue;
    public float WinnerPoseCost { get; private set; } = float.MaxValue;
    public float WinnerTrajectoryCost { get; private set; } = float.MaxValue;
    public float WinnerContinuityCost { get; private set; }
    public float WinnerLoopCost { get; private set; }
    public int SwitchCount { get; private set; }
    public int SeekCount { get; private set; }
    public float LastQueryMs { get; private set; }
    public string ActiveTag { get; private set; } = string.Empty;

    // Turn-driver activity/yaw mirrors were deleted with the native turn
    // machine. Committed-turn facing is owned host-side (facade driver
    // facing); the core never steers facing anymore.

    public bool WinnerIsLoop { get; private set; }
    public float LoopPreferenceApplied { get; private set; }
    public float BestLoopCost { get; private set; } = float.MaxValue;
    public string BestLoopClipName { get; private set; } = string.Empty;
    public float BestOneShotCost { get; private set; } = float.MaxValue;
    public string BestOneShotClipName { get; private set; } = string.Empty;
    public int LoopScoredCount { get; private set; }
    public int OneShotScoredCount { get; private set; }
    public int LoopTotalCount { get; private set; }
    public int OneShotTotalCount { get; private set; }
    public string LastSwitchReason { get; private set; } = string.Empty;
    public string ChooserStatus { get; private set; } = string.Empty;
    public int TraceEntries { get; private set; }
    // Additive profiling mirrors (record-only; copied from the native
    // policy after each Update, never steer anything).
    // LastTelemetryMs measures bounded-ring enqueue (ns), not file I/O —
    // Tick never does file I/O; FlushTelemetry (explicit, session end) is
    // the only file writer. TelemetryCount/Dropped mirror the ring.
    public float LastTickMs { get; private set; }
    public float LastFastPathMs { get; private set; }
    public float LastTelemetryMs { get; private set; }
    public int TickCount { get; private set; }
    public int FullSearchCount { get; private set; }
    public float TotalTickMs { get; private set; }
    public float TotalFullSearchMs { get; private set; }
    public float TotalFastPathMs { get; private set; }
    public float TotalTelemetryMs { get; private set; }
    public int TelemetryCount { get; private set; }
    public int TelemetryDropped { get; private set; }

    // Machine/overlay/IK telemetry was deleted with their producers
    // (HoldCount, TurnReplayCount, SettleCount, RecoveryCount, JumpCount,
    // LayerAngle/Primary/Secondary/GateApplied, FootLockL/R, FootCorrL/R).
    // Search telemetry above (switches, seeks, query ms, fast path) stays.
    public int FastPathHits { get; private set; }

    private MotionMatchingRuntimePolicy _policy;
    private bool _telemetryWarned;
    private MotionMatchingDatabase _boundDatabase;
    private AnimatedModel _boundBody;
    private SkinnedModel _boundModel;
    private PoseSearchSchema _boundSchema;
    private string _loadedDatabaseFile = string.Empty;

    public float[] TrajectorySampleTimes
    {
        get
        {
            ReleaseStaleBinding();
            // The Database getter is a native binding that allocates a
            // managed float[] per call; the baked times never change under a
            // loaded database in play, so cache the array and refresh only
            // when the database reference changes. Callers (facade tick +
            // Update, 2x per tick) share the instance; nobody mutates it.
            // The null-database fallback (unbaked: the policy never ticks)
            // reads live with no caching.
            if (!Database)
                return Schema.TrajectorySampleTimes ?? Array.Empty<float>();
            if (!ReferenceEquals(Database, _sampleTimesDb))
            {
                _sampleTimesDb = Database;
                _sampleTimesCache = Database.TrajectorySampleTimes ?? Schema.TrajectorySampleTimes;
            }
            return _sampleTimesCache;
        }
    }

    private MotionMatchingDatabase _sampleTimesDb;
    private float[] _sampleTimesCache = Array.Empty<float>();

    public bool IsReady
    {
        get
        {
            ReleaseStaleBinding();
            return _policy && _policy.IsReady;
        }
    }

    /// <summary>
    /// Live playback handle for game-side integrations (e.g. pose overlays
    /// attach here). Null until the first update initializes the policy.
    /// Borrowed handle: reacquire after every rebind/shutdown; never destroy it.
    /// </summary>
    public MotionMatchingPlayback Playback => IsReady ? _policy.GetPlayback() : null;

    /// <summary>
    /// Publishes the existing search-pose path without touching Update.
    /// Returns the playback's current search locals plus the current winner
    /// (false + defaults when the policy is not ready).
    /// </summary>
    public bool TryGetBasePose(out Matrix[] locals, out int winnerClip, out float winnerCost)
    {
        locals = System.Array.Empty<Matrix>();
        winnerClip = -1;
        winnerCost = float.MaxValue;
        if (!IsReady)
            return false;
        var playback = _policy.GetPlayback();
        if (playback == null)
            return false;
        locals = playback.GetSearchLocals() ?? System.Array.Empty<Matrix>();
        winnerClip = _policy.WinnerClip;
        winnerCost = _policy.WinnerCost;
        return true;
    }

    /// <summary>
    /// Zero-alloc base-pose read: copies the playback search locals into
    /// the caller-owned buffer (no managed array allocation, unlike
    /// TryGetBasePose which allocates via GetSearchLocals). The caller
    /// reuses its buffer and resizes only when the bone count changes
    /// (read it via Playback.GetSearchLocalsCount first). Bound: model bone
    /// count (91 on the reference rig, callers cap at 512). Same snapshot
    /// data as TryGetBasePose (post-inertial search pose); the native
    /// history is never aliased with the caller buffer (per-bone copy).
    /// Returns false (no copy) when the policy is not ready, the buffer is
    /// null, or its length differs from the bone count.
    /// </summary>
    public bool TryGetBasePoseInto(Matrix[] buffer, out int winnerClip, out float winnerCost)
    {
        winnerClip = -1;
        winnerCost = float.MaxValue;
        if (!IsReady || buffer == null)
            return false;
        var playback = _policy.GetPlayback();
        if (playback == null)
            return false;
        int n = playback.GetSearchLocalsCount();
        if (n <= 0 || n > 512 || buffer.Length != n)
            return false;
        for (int i = 0; i < n; i++)
        {
            if (!playback.GetSearchLocal(i, out buffer[i]))
                return false;
        }
        winnerClip = _policy.WinnerClip;
        winnerCost = _policy.WinnerCost;
        return true;
    }

    /// <summary>
    /// Single authoritative force-clip hatch (the only hatch besides the
    /// search-winner Plays inside the native RunQuery). Forwards to the
    /// native policy ForcePlayClip, which synchronizes playback clip/time,
    /// policy current clip/time, continuity baseline, fast-path
    /// invalidation, inertial state, telemetry reason and the trace command
    /// entry in one call. Search resumes immediately on the next tick with
    /// the caller filter (no hold). Logs the reason on EVERY call.
    /// No production callers yet; host drivers are the intended callers.
    /// </summary>
    public void ForcePlayClip(int clipIndex, float startTime, string reason)
    {
        string safeReason = reason ?? string.Empty;
        Debug.Log("MotionMatchingController.ForcePlayClip: clip=" + clipIndex
            + " startTime=" + startTime + " reason=" + safeReason);
        if (!IsReady)
            return;
        try
        {
            _policy.ForcePlayClip(clipIndex, startTime, safeReason);
        }
        catch
        {
            // The force-clip command itself must never throw to drivers.
        }
    }

    /// <summary>
    /// Explicit telemetry flush (the ONLY writer of MMTelemetry.jsonl).
    /// Formats the bounded native ring as JSONL + writes the file. Host
    /// session-end only (export path), never per-tick. Returns false with a
    /// visible error when the ring is empty, the cache folder is unset, or
    /// the file cannot be written. Overwritten drops are counted in
    /// TelemetryDropped.
    /// </summary>
    public bool FlushTelemetry(out string error)
    {
        error = string.Empty;
        if (!IsReady)
        {
            error = "Telemetry flush stopped: policy is not ready.";
            return false;
        }
        try
        {
            return _policy.FlushTelemetry(out error);
        }
        catch (System.Exception ex)
        {
            error = "Telemetry flush stopped: " + ex.Message;
            return false;
        }
    }

    /// <summary>
    /// Explicit telemetry clear (no file deletion: init never touches
    /// MMTelemetry.jsonl; tools read whatever the last flush left behind).
    /// Resets the ring head/count/dropped to the bounded disabled state.
    /// </summary>
    public void ClearTelemetry()
    {
        ReleaseStaleBinding();
        if (_policy == null)
            return;
        try
        {
            _policy.ClearTelemetry();
        }
        catch
        {
        }
    }

    /// <summary>
    /// Explicit audit command (the ONLY writer of MMClipAudit.txt).
    /// Static native export: runs in edit mode with no play/policy instance.
    /// Runtime init/play never scans the database or writes the file — call
    /// this from the editor service / bridge clip_audit op when tools need
    /// per-clip travel/turn/speed data.
    /// </summary>
    public bool GenerateClipAudit(out string error)
    {
        error = string.Empty;
        if (Database == null || !Database.IsBaked)
        {
            error = "Clip audit stopped: database is not baked.";
            return false;
        }
        try
        {
            return MotionMatchingRuntimePolicy.ExportClipAudit(
                Database, CacheFolder ?? string.Empty, Schema.SampleRate, out error);
        }
        catch (System.Exception ex)
        {
            error = "Clip audit stopped: " + ex.Message;
            return false;
        }
    }

    /// <summary>
    /// Explicit init (replaces the Script OnEnable): the host facade
    /// assigns Schema/PlayerModel/DatabasePath/CharacterBody + session
    /// config first, then calls this. Content.Load works outside Scripts —
    /// Globals remain usable here.
    /// </summary>
    public void Initialize()
    {
        ReleaseStaleBinding();

        // Unified load path (no editor partial): Content.Load works in both
        // editor and game builds; the editor service refreshes explicitly
        // after bake/registry edits.
        EnsureDatabaseLoadedForGame();
    }

    /// <summary>
    /// Releases the owned native policy, playback and trace immediately, then
    /// schedules destruction of the policy object. Idempotent; call on the
    /// Flax main thread on host disable/destroy. Export traces/telemetry first.
    /// Configuration and host-owned assets are retained for editor tooling and
    /// reuse: a subsequent Update starts a fresh session without requiring Initialize.
    /// Previously borrowed Playback/Trace handles are invalid after this call.
    /// </summary>
    public void Shutdown()
    {
        var policy = _policy;
        _policy = null;
        _boundDatabase = null;
        _boundBody = null;
        _boundModel = null;
        _boundSchema = default;
        _sampleTimesDb = null;
        _sampleTimesCache = Array.Empty<float>();
        _telemetryWarned = false;
        ResetRuntimeStatus();
        if (policy)
        {
            try
            {
                policy.Shutdown();
            }
            finally
            {
                FlaxEngine.Object.Destroy(policy);
            }
        }
    }

    /// <summary>Equivalent to Shutdown; the configured controller remains reusable.</summary>
    public void Dispose() => Shutdown();

    private void ReleaseStaleBinding()
    {
        // Keep public fields (including nested struct edits) compatible with
        // existing hosts/tools. Validate before any native read or command.
        if (ReferenceEquals(_policy, null))
            return;
        if (!_policy || !Database || !CharacterBody ||
            !ReferenceEquals(Database, _boundDatabase) ||
            !ReferenceEquals(CharacterBody, _boundBody) ||
            !ReferenceEquals(CharacterBody.SkinnedModel, _boundModel) ||
            (!string.IsNullOrEmpty(_loadedDatabaseFile) &&
             !string.Equals(DatabaseFile, _loadedDatabaseFile, StringComparison.OrdinalIgnoreCase)) ||
            !SchemaBindingEquals(Schema, _boundSchema) || !_policy.IsReady)
            Shutdown();
    }

    private static bool SchemaBindingEquals(in PoseSearchSchema a, in PoseSearchSchema b)
    {
        // Weights are query-time tuning, not binding identity.
        var identity = a;
        identity.PoseWeight = b.PoseWeight;
        identity.TrajectoryWeight = b.TrajectoryWeight;
        return SchemaEquals(identity, b);
    }

    private void ResetRuntimeStatus()
    {
        PolicyState = "Disabled";
        WinnerSample = WinnerClip = -1;
        WinnerClipName = ActiveTag = BestLoopClipName = BestOneShotClipName = string.Empty;
        LastSwitchReason = ChooserStatus = string.Empty;
        WinnerCost = SecondBestCost = WinnerPoseCost = WinnerTrajectoryCost = float.MaxValue;
        BestLoopCost = BestOneShotCost = float.MaxValue;
        WinnerContinuityCost = WinnerLoopCost = LoopPreferenceApplied = 0.0f;
        WinnerIsLoop = false;
        SwitchCount = SeekCount = LoopScoredCount = OneShotScoredCount = 0;
        LoopTotalCount = OneShotTotalCount = TraceEntries = FastPathHits = 0;
        LastQueryMs = LastTickMs = LastFastPathMs = LastTelemetryMs = 0.0f;
        TickCount = FullSearchCount = TelemetryCount = TelemetryDropped = 0;
        TotalTickMs = TotalFullSearchMs = TotalFastPathMs = TotalTelemetryMs = 0.0f;
    }

    // Note: the old scope reflection-compat shim for the legacy native
    // scope gate is deleted (the native headers no longer carry it).

    /// <summary>
    /// Clears native playback state. The host facade resets its provider
    /// (scope + predictor) and drivers separately — this class no longer
    /// owns any provider reference. Call after teleport/respawn.
    /// </summary>
    public void ResetPlayback()
    {
        ReleaseStaleBinding();
        _policy?.ResetOnTeleport();
    }

    /// <summary>
    /// Policy + playback tick. The host facade builds the provider frame
    /// itself (provider.BuildInput) and passes it in with the actor world
    /// transform — this class never calls a provider. Loads the database on
    /// first tick if Initialize found none.
    /// </summary>
    public void Update(float deltaTime, Transform actorWorld, MotionMatchingFrameInput input)
    {
        ReleaseStaleBinding();
        if (!CharacterBody)
            return;

        if (_policy == null)
        {
            EnsureDatabaseLoadedForGame();
            if (!Database || !Database.IsBaked || !CharacterBody.SkinnedModel)
                return;
            _policy = new MotionMatchingRuntimePolicy();
            // No file deletion in init/update (the offline gate forbids file
            // deletion on the core path). The native Initialize clears the
            // bounded telemetry ring to the disabled state; FlushTelemetry
            // (explicit, session end) is the only file writer. Tools read
            // whatever the last flush left behind — init never touches
            // MMTelemetry.jsonl.
            _policy.Initialize(Database, CharacterBody, Schema,
                CacheFolder ?? string.Empty, InertializationDuration);
            if (!_policy.IsReady)
            {
                Shutdown();
                return;
            }
            _boundDatabase = Database;
            _boundBody = CharacterBody;
            _boundModel = CharacterBody.SkinnedModel;
            _boundSchema = Schema;
            // Snapshot array contents too: hosts can edit sample times in place.
            _boundSchema.TrajectorySampleTimes = Schema.TrajectorySampleTimes != null
                ? (float[])Schema.TrajectorySampleTimes.Clone() : Array.Empty<float>();
        }
        if (!_policy.IsReady)
            return;

        // Debug-only trace gate with a visible warning: the host
        // TelemetryLog field is never written at runtime; the host flips
        // only TelemetryOverride above.
        // Trace stays on for test runs (TraceOverride) even when the
        // telemetry file is off, so on/off search equivalence can be proven
        // by comparing trace winners. Normal play (no flags) leaves both
        // false → no per-query allocations beyond the fixed rings.
        if (_policy.GetTrace() != null)
            _policy.GetTrace().Enabled = EffectiveTelemetry || TraceOverride;
        if (EffectiveTelemetry && !_telemetryWarned)
        {
            _telemetryWarned = true;
            Debug.LogWarning("Motion matching telemetry/trace is ACTIVE (debug-only: per-query clones + bounded-ring enqueue, file flush at session end only). Turn off TelemetryLog for shipping builds.");
        }

        // The snapshot carries the caller scope only. LoopFilter/
        // CandidateBand/YawRate ride explicit Tick params (never inferred);
        // the core uses them verbatim. No new native struct field (bindings
        // cap 64).
        var snapshot = new MotionMatchingPlayerSnapshot
        {
            AllowedTagSet = input.AllowedTagSet ?? string.Empty,
            ExcludedTagSet = input.ExcludedTagSet ?? string.Empty,
        };
        var tuning = new MotionMatchingPolicyTuning
        {
            PoseWeight = Schema.PoseWeight,
            TrajectoryWeight = Schema.TrajectoryWeight,
            SampleRate = Schema.SampleRate,
            MinSwitchTime = MinSwitchTime,
            SameClipExclusion = SameClipExclusion,
            SameClipSeekThreshold = SameClipSeekThreshold,
            ContinuingBias = ContinuingBias,
            ContinuingWindow = ContinuingWindow,
            SwitchMargin = SwitchMargin,
            FrameSearchBudget = FrameSearchBudget,
            SearchEveryFrame = SearchEveryFrame,
            QueryInterval = QueryInterval,
            UseIndexedSearch = UseIndexedSearch,
            TopKCandidates = TopKCandidates,
            VerifyIndexedSearch = VerifyIndexedSearch,
            IndexedVerifyAuto = IndexedVerifyAuto,
            EffectiveTelemetry = EffectiveTelemetry,
            ContinuityWeight = ContinuityWeight,
            CostFuseThreshold = CostFuseThreshold,
            TurnPenaltyWeight = TurnPenaltyWeight,
            TurnPenaltyYawScale = TurnPenaltyYawScale,
        };
        int callerLoop = input.LoopFilter >= 0 && input.LoopFilter <= 2 ? input.LoopFilter : 0;
        int callerBand = input.CandidateBand >= 0 && input.CandidateBand <= 3 ? input.CandidateBand : 0;
        float callerYaw = input.QueryYawRate >= 0.0f && input.QueryYawRate < 1e29f ? input.QueryYawRate : 0.0f;
        float callerPlanar = input.CallerPlanarSpeed >= 0.0f && input.CallerPlanarSpeed < 1e29f ? input.CallerPlanarSpeed : 0.0f;
        _policy.Tick(deltaTime, snapshot,
            input.Points ?? Array.Empty<TrajectoryPoint>(),
            actorWorld.GetWorld(), TrajectorySampleTimes, tuning,
            callerLoop, callerBand, callerYaw, callerPlanar);

        PolicyState = _policy.PolicyState;
        WinnerSample = _policy.WinnerSample;
        WinnerClip = _policy.WinnerClip;
        WinnerClipName = _policy.WinnerClipName;
        WinnerCost = _policy.WinnerCost;
        SecondBestCost = _policy.SecondBestCost;
        WinnerPoseCost = _policy.WinnerPoseCost;
        WinnerTrajectoryCost = _policy.WinnerTrajectoryCost;
        WinnerContinuityCost = _policy.WinnerContinuityCost;
        SwitchCount = _policy.SwitchCount;
        SeekCount = _policy.SeekCount;
        LastQueryMs = _policy.LastQueryMs;
        ActiveTag = _policy.ActiveTag;
        WinnerLoopCost = _policy.WinnerLoopCost;
        WinnerIsLoop = _policy.WinnerIsLoop;
        LoopPreferenceApplied = _policy.LoopPreferenceApplied;
        BestLoopCost = _policy.BestLoopCost;
        BestLoopClipName = _policy.BestLoopClipName;
        BestOneShotCost = _policy.BestOneShotCost;
        BestOneShotClipName = _policy.BestOneShotClipName;
        LoopScoredCount = _policy.LoopScoredCount;
        OneShotScoredCount = _policy.OneShotScoredCount;
        LoopTotalCount = _policy.LoopTotalCount;
        OneShotTotalCount = _policy.OneShotTotalCount;
        LastSwitchReason = _policy.LastSwitchReason;
        ChooserStatus = _policy.ChooserStatusText;
        TraceEntries = _policy.GetTrace() != null ? _policy.GetTrace().Count : 0;
        FastPathHits = _policy.FastPathHits;
        // Profiling mirrors (record-only).
        LastTickMs = _policy.LastTickMs;
        LastFastPathMs = _policy.LastFastPathMs;
        LastTelemetryMs = _policy.LastTelemetryMs;
        TickCount = _policy.TickCount;
        FullSearchCount = _policy.FullSearchCount;
        TotalTickMs = _policy.TotalTickMs;
        TotalFullSearchMs = _policy.TotalFullSearchMs;
        TotalFastPathMs = _policy.TotalFastPathMs;
        TotalTelemetryMs = _policy.TotalTelemetryMs;
        TelemetryCount = _policy.GetTelemetryCount();
        TelemetryDropped = _policy.GetTelemetryDropped();
    }

    // The policy's query trace (null until the first play tick). The
    // tuning window reads it for top-N, export, and replay.
    // Borrowed handle, with the same lifetime as Playback.
    public MotionMatchingTraceLog Trace => IsReady ? _policy.GetTrace() : null;
    public MotionMatchingTraceEntry LastTraceEntry => IsReady ? _policy.GetLastEntry() : new MotionMatchingTraceEntry();

    private void EnsureDatabaseLoadedForGame()
    {
        string databaseFile = DatabaseFile;
        // A host may inject an already-loaded database (editor tooling and
        // isolated contract controllers do this). Once that reference is
        // adopted, the normal binding checks below own rebind detection.
        if (Database && (string.IsNullOrEmpty(_loadedDatabaseFile) ||
            string.Equals(_loadedDatabaseFile, databaseFile, StringComparison.OrdinalIgnoreCase)))
            return;
        if (_policy)
            Shutdown();
        Database = Content.Load<MotionMatchingDatabase>(databaseFile);
        _loadedDatabaseFile = Database ? databaseFile : string.Empty;
        RefreshDatabaseStatus();
    }

    /// <summary>
    /// Resolved database asset path. Public so editor services and host
    /// tooling can operate on the same asset without duplicating the
    /// fallback logic.
    /// </summary>
    public string DatabaseFile
    {
        get
        {
            string path = string.IsNullOrWhiteSpace(DatabasePath)
                ? "MotionMatching/MotionMatchingDatabase.flax"
                : DatabasePath.Trim();
            return StringUtils.NormalizePath(Path.IsPathRooted(path)
                ? path
                : Path.Combine(Globals.ProjectContentFolder, path));
        }
    }

    public static PoseSearchSchema CreateDefaultSchema()
    {
        // Skeleton ships EMPTY on purpose (no baked-in bone names): the host
        // owning this scene overrides every bone name for its own rig
        // (Schema → Skeleton). PoseSearchSchema.Default already carries the
        // empty profile, so nothing is filled in here.
        var schema = PoseSearchSchema.Default;
        schema.TrajectorySampleTimes = new float[]{ -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f};
        return schema;
    }

    public static bool SchemaEquals(in PoseSearchSchema a, in PoseSearchSchema b)
    {
        float AxisTolerance = 0.001f;

        // The baker stores normalized axes, so compare directions rather than raw vectors.
        if (a.SampleRate != b.SampleRate ||
            !Float3.NearEqual(Float3.Normalize(a.RootForwardAxis), Float3.Normalize(b.RootForwardAxis), AxisTolerance) ||
            !Float3.NearEqual(Float3.Normalize(a.RootUpAxis), Float3.Normalize(b.RootUpAxis), AxisTolerance) ||
            a.PoseWeight != b.PoseWeight ||
            a.TrajectoryWeight != b.TrajectoryWeight)
            return false;

        if (a.Skeleton.Root != b.Skeleton.Root ||
            a.Skeleton.Pelvis != b.Skeleton.Pelvis ||
            a.Skeleton.Spine != b.Skeleton.Spine ||
            a.Skeleton.Head != b.Skeleton.Head ||
            a.Skeleton.LeftFoot != b.Skeleton.LeftFoot ||
            a.Skeleton.RightFoot != b.Skeleton.RightFoot ||
            a.Skeleton.LeftHand != b.Skeleton.LeftHand ||
            a.Skeleton.RightHand != b.Skeleton.RightHand)
            return false;

        var ta = a.TrajectorySampleTimes ?? Array.Empty<float>();
        var tb = b.TrajectorySampleTimes ?? Array.Empty<float>();
        if (ta.Length != tb.Length)
            return false;
        for (int i = 0; i < ta.Length; i++)
        {
            if (ta[i] != tb[i])
                return false;
        }
        return true;
    }

    public void RefreshDatabaseStatus()
    {
        // Explicit editor reload/bake notification, including same-asset reloads.
        Shutdown();
        _loadedDatabaseFile = Database ? DatabaseFile : string.Empty;
        // Non-blocking check: WaitForLoaded defaults to a 30s main-thread
        // block, and Content.Load above already waited for the settled state.
        if (!Database || Database.WaitForLoaded(0))
        {
            ClearDatabaseStatus();
            return;
        }

        AnimationClipsCount = Database.AnimationClipsCount;
        SampleCount = Database.SampleCount;
        AnimationTags = Database.AnimationTags ?? Array.Empty<string>();
        AnimationClipCountsByTag = Database.AnimationClipCountsByTag ?? Array.Empty<int>();
        PersistenceValid = Database.ValidateData();
        // Gait-config validation lives in the provider now: it validates
        // against the inventory handed over in the tick context.
    }

    public void ClearDatabaseStatus()
    {
        Shutdown();
        _loadedDatabaseFile = string.Empty;
        Database = null;
        AnimationClipsCount = 0;
        SampleCount = 0;
        AnimationTags = Array.Empty<string>();
        AnimationClipCountsByTag = Array.Empty<int>();
        PersistenceValid = false;
    }
}
