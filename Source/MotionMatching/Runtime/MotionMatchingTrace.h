// MotionMatchingTrace.h: deterministic query trace ring (MotionMatchingTraceEntry, MotionMatchingTraceLog::Record/ExportJson/ImportJson/ExportCsv/VerifyReplay).
// Ownership: plugin runtime, no game/host references.
// Key invariants: every entry carries the complete effective request (weights, filters, retry kind, budget/stride, tail rule, fuse threshold, frame dt, play rate, database identity) so VerifyReplay re-runs search/fastpath bit-exactly; the seven boolean request flags travel packed in TraceFlags because the bindings generator caps API_STRUCT fields at 64; v3 appends fields without reordering so v2 files still import.
#pragma once

#include "Engine/Core/Collections/Array.h"
#include "Engine/Core/Types/String.h"
#include "Engine/Core/Types/StringBuilder.h"
#include "Engine/Scripting/ScriptingObject.h"
#include "../Search/MotionMatchingQuery.h"

class MotionMatchingDatabase;

// Trace-flag bits for MotionMatchingTraceEntry::TraceFlags.
// Flax 1.12 BindingsGenerator.Cpp indexes per-field wrapper caches with
// fixed bool/string[64] arrays, so a non-POD API_STRUCT with more than 64
// instance fields crashes binding generation (IndexOutOfRange inside
// GenerateCpp). The seven boolean request flags below therefore travel as
// ONE int32 bitmask; the JSON exporter still expands them to named keys so
// tools never see the packing. Keep the TOTAL instance field count of
// MotionMatchingTraceEntry at or below 64 (currently 61: 59 base fields + 2
// straightness-prior tuning fields).
constexpr int32 TraceFlag_ForcedEdge = 1 << 0;
constexpr int32 TraceFlag_Reseeded = 1 << 1;
constexpr int32 TraceFlag_FairFight = 1 << 2;
constexpr int32 TraceFlag_OneshotWindow = 1 << 3;
constexpr int32 TraceFlag_UseIndexedSearch = 1 << 4;
constexpr int32 TraceFlag_VerifyIndexedSearch = 1 << 5;
constexpr int32 TraceFlag_ExcludeNonLoopTail = 1 << 6;

/// <summary>
/// One recorded search: everything needed to re-run it bit-exactly.
/// Kind separates search from policy/playback events:
/// "search" ran FindBest (replayable via FindBest);
/// "fastpath" held the continuation without a full scan (replayable via
/// SampleCost from the recorded feature vectors);
/// "command" is a ForcePlayClip request (structural check only);
/// "reset" is a ResetOnTeleport/teleport discontinuity (structural check);
/// "hold" skipped the query (historical turn-hold event);
/// "replay" restarted a clip with no search (historical turn-replay event).
/// Replay verify re-runs "search"/"fastpath", structurally checks
/// "command"/"reset", and reports the rest as skipped with reasons.
/// The entry carries the COMPLETE effective request — every runtime
/// input that can change the winner (OneShotBand, exclusions, retry kind,
/// force/reset flags, intent windows, budget/stride, index route, tail
/// rule, fuse threshold, frame dt, play rate, database identity,
/// fast-path threshold, command clip/time).
/// </summary>
API_STRUCT(Namespace="MotionMatching")
struct MOTIONMATCHING_API MotionMatchingTraceEntry
{
    DECLARE_SCRIPTING_TYPE_MINIMAL(MotionMatchingTraceEntry);

    API_FIELD() String Kind = TEXT("search");
    API_FIELD() float Time = 0.0f;
    API_FIELD() String State;
    API_FIELD() String Tags;
    API_FIELD() float Speed = 0.0f;
    API_FIELD() float YawRate = 0.0f;
    API_FIELD() bool Grounded = true;
    API_FIELD() bool Sprinting = false;
    API_FIELD() float DesiredSpeed = 0.0f;
    API_FIELD() Array<float> Pose;
    API_FIELD() Array<float> Trajectory;
    API_FIELD() float PoseWeight = 1.0f;
    API_FIELD() float TrajectoryWeight = 2.0f;
    API_FIELD() float ContinuityWeight = 0.25f;
    API_FIELD() float MinSwitchTime = 0.12f;
    API_FIELD() float CurrentTime = -1.0f;
    API_FIELD() float ExclusionWindow = 0.25f;
    API_FIELD() float ExpectedTime = -1.0f;
    API_FIELD() float ContinuingBias = 0.25f;
    API_FIELD() float ContinuingWindow = 0.15f;
    API_FIELD() float SwitchMargin = 0.1f;
    API_FIELD() float LoopPreference = 0.0f;
    API_FIELD() int32 LoopFilter = 0;
    API_FIELD() int32 ReselectClip = -1;
    API_FIELD() float ReselectSampleTime = 0.0f;
    API_FIELD() float ReselectBanWindow = 0.0f;
    API_FIELD() float TwinBanWindow = 0.0f;
    API_FIELD() float LoopTailExclusion = 0.0f;
    API_FIELD() int32 ContinuityClip = -1;
    API_FIELD() float SinceSwitch = 999.0f;
    API_FIELD() bool Recovered = false;
    API_FIELD() bool Valid = false;
    API_FIELD() int32 ClipIndex = -1;
    API_FIELD() int32 SampleIndex = -1;
    API_FIELD() float SampleTime = 0.0f;
    API_FIELD() float Cost = MAX_float;
    API_FIELD() float PoseCost = MAX_float;
    API_FIELD() float TrajectoryCost = MAX_float;
    API_FIELD() float ContinuityCost = 0.0f;
    API_FIELD() float LoopCost = 0.0f;
    API_FIELD() float SecondBestCost = MAX_float;
    API_FIELD() String Reason;
    API_FIELD() String Chooser;
    API_FIELD() float QueryMs = 0.0f;
    // Complete-effective-request fields (trace v3, appended — never
    // reordered). All are recorded, never read by search/playback.
    // Booleans travel packed in TraceFlags (see bit constants above):
    // the Flax 1.12 bindings generator cannot handle more than 64 fields
    // on one API_STRUCT.
    // Controller-decided exclusions, '+'-joined (merged into search).
    API_FIELD() String ExcludedTags;
    // Onset one-shot gait band actually applied (0 = Any).
    API_FIELD() int32 OneShotBand = 0;
    // Retry path taken after the first FindBest: 0 none, 1 end-recovery
    // LoopOnly retry, 2 onset-fallback full-pool retry.
    API_FIELD() int32 RetryKind = 0;
    // Packed request flags (TraceFlag_* bits): ForcedEdge (grounded/
    // airborne edge bypassed continuity), Reseeded (first query after
    // teleport, LoopOnly), FairFight + OneshotWindow (intent windows),
    // UseIndexedSearch + VerifyIndexedSearch (index route),
    // ExcludeNonLoopTail (structural tail rule).
    // Firm-lock bits are retired (the foot-contact search prior recorded 0; kept as spare bits).
    API_FIELD() int32 TraceFlags = 0;
    // Budgeted-search stride cursor (0 = unstrided full scan).
    API_FIELD() int32 MaxCandidates = 0;
    API_FIELD() int32 QueryFrame = 0;
    // Index route breadth (meaningful with the TraceFlags indexed bit).
    API_FIELD() int32 TopKCandidates = 32;
    // Post-search fuse threshold (a switched winner above this holds).
    API_FIELD() float CostFuseThreshold = 2.0f;
    // Frame dt + play rate behind ExpectedTime for this query.
    API_FIELD() float FrameDt = 0.0f;
    API_FIELD() float PlayRate = 1.0f;
    // Database identity at record time (invalidate on rebake/reload).
    API_FIELD() int32 DbClipCount = 0;
    API_FIELD() int32 DbSampleCount = 0;
    // Fast-path continuation threshold actually applied (hit iff cost <= this).
    API_FIELD() float FastThreshold = MAX_float;
    // Command/reset entries: requested clip/time (structural check).
    API_FIELD() int32 CommandClip = -1;
    API_FIELD() float CommandTime = 0.0f;
    // Straightness-prior tuning actually applied (provider-owned values,
    // preserved from the 0.03/4.0 calibration). Recorded so brute-force,
    // indexed and replay share one formula.
    API_FIELD() float TurnPenaltyWeight = 0.03f;
    API_FIELD() float TurnPenaltyYawScale = 4.0f;
};

/// <summary>
/// In-memory ring of recent queries plus JSON dump/load, CSV statistics
/// export, and deterministic replay verify (re-runs the native search from
/// recorded inputs and compares the winner sample). Runtime-safe (no editor
/// types); the tuning window drives it.
/// </summary>
API_CLASS(Namespace="MotionMatching")
class MOTIONMATCHING_API MotionMatchingTraceLog : public ScriptingObject
{
    DECLARE_SCRIPTING_TYPE_WITH_CONSTRUCTOR_IMPL(MotionMatchingTraceLog, ScriptingObject);

public:
    // Ring capacity: 4096 entries ≈ 60s at 60 Hz queries. Memory stays
    // modest for one tuned character.
    static constexpr int32 Capacity = 4096;
    // v3 adds the complete effective request (exclusions, OneShotBand,
    // retry kind, force/reset flags, intent windows, budget/stride, index
    // route, tail rule, fuse threshold, frame dt, play rate, DB identity,
    // fast-path threshold, command clip/time) plus replayable fast-path
    // vectors and command/reset entries. v2 files still import.
    static constexpr int32 SchemaVersion = 3;

private:
    MotionMatchingTraceEntry _entries[Capacity];
    int32 _head = 0;
    int32 _count = 0;

    static void FormatFloat(StringBuilder& sb, float value);
    static void FormatFloatText(float value, char* buffer, int32 capacity);
    static void EscapeInto(StringBuilder& sb, const String& value);
    static void AppendFloats(StringBuilder& sb, const Array<float>& values);
    static bool ParseEntry(const String& line, MotionMatchingTraceEntry& entry);
    static float ParseFloatToken(const Char* text, int32 start, int32 end);
    static int32 FindKey(const Char* text, int32 length, const char* key);
    static float GetFloat(const Char* text, int32 length, const char* key);
    static String GetString(const Char* text, int32 length, const char* key);
    static Array<float> GetFloatArray(const Char* text, int32 length, const char* key);

public:
    // Debug-only gate: recording costs a pose+trajectory clone per query.
    // The policy enables this only from EffectiveTelemetry.
    API_FIELD() bool Enabled = true;

    API_PROPERTY() int32 GetCount() const;
    API_FUNCTION() void Clear();
    // NOTE: pose/trajectory arrays travel as function params, NOT struct
    // fields — C#->C++ struct Array fields do not round-trip (verified:
    // file exports came out empty). Function Array params marshal fine
    // (same path as FindBest). The entry keeps the fields for reads.
    API_FUNCTION() void Record(const MotionMatchingTraceEntry& entry, const Array<float>& pose, const Array<float>& trajectory);
    API_FUNCTION() Array<MotionMatchingTraceEntry> Snapshot() const;
    API_FUNCTION() bool ExportJson(const String& path, API_PARAM(Out) String& error);
    API_FUNCTION() bool ImportJson(const String& path, API_PARAM(Out) String& error);
    API_FUNCTION() bool ExportCsv(const String& path, API_PARAM(Out) String& error, API_PARAM(Out) String& summary);
    API_FUNCTION() bool VerifyReplay(MotionMatchingDatabase* database, int32 maxEntries, API_PARAM(Out) String& report);
};
