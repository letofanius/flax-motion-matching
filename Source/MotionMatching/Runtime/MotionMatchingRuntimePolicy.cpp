// MotionMatchingRuntimePolicy.cpp: per-tick policy (Tick/RunQuery/TryFastPath, ForcePlayClip, ExportClipAudit, SelectPlayRate, HashTagSet/CaptureFastPathSignature).
// Ownership: plugin runtime, no game/host references.
// Key invariants: Tick samples the live pose and builds the trajectory every tick, then dispatches pure search (cadence-gated) or the cheap continuation; RunQuery applies the provider caller filter verbatim with LoopOnly reseed after teleports and a loop-only end-recovery retry; the fast-path signature (tags, filters, clip/time, database identity) invalidates the hold on any scope change; telemetry/trace recording never steers the winner.
#include "MotionMatchingRuntimePolicy.h"

#include "../Database/MotionMatchingDatabase.h"
#include "../Search/MotionMatchingSearch.h"
#include "Engine/Content/Assets/Animation.h"
#include "Engine/Content/Assets/SkinnedModel.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Engine/Time.h"
#include "Engine/Level/Actors/AnimatedModel.h"
#include "Engine/Level/Actor.h"
#include "Engine/Platform/FileSystem.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{
    constexpr float WarmupTime = 0.5f;
    constexpr float TeleportDistance = 2000.0f;
    constexpr float TravelingDistance = 50.0f;
    constexpr float MaxFastPathAge = 0.5f;
    constexpr int32 MaxFastPathFrames = 5;
    constexpr float ReselectHistoryDuration = 0.6f;
    constexpr float ReselectSampleWindow = 0.35f;
    constexpr bool EnableReselectBan = true;
    constexpr bool EnableLoopTailExclusion = false;
    constexpr float SameSegmentTwinBan = 0.25f;
    constexpr float NonLoopTailExclusion = 0.3f;
    constexpr float PiF = 3.14159265358979323846f;

    void FormatFloatText(float value, char* buffer, int32 capacity)
    {
        if (std::isinf(value))
        {
            snprintf(buffer, (size_t)capacity, "%s", value > 0.0f ? "1e30" : "-1e30");
            return;
        }
        if (std::isnan(value))
        {
            snprintf(buffer, (size_t)capacity, "0");
            return;
        }
        snprintf(buffer, (size_t)capacity, "%.9g", (double)value);
        for (int32 i = 0; buffer[i] != '\0'; i++)
        {
            if (buffer[i] == ',')
                buffer[i] = '.';
        }
    }

    // pack the seven boolean request flags into MotionMatchingTraceEntry
    // bitmask (see TraceFlag_* in MotionMatchingTrace.h). Record-only.
    // FirmLockL/R died with the foot-lock search prior (always false).
    int32 PackTraceFlags(bool forcedEdge, bool reseeded, bool fairFight, bool oneshotWindow,
        bool useIndexed, bool verifyIndexed, bool exclTail)
    {
        int32 flags = 0;
        if (forcedEdge)
            flags |= TraceFlag_ForcedEdge;
        if (reseeded)
            flags |= TraceFlag_Reseeded;
        if (fairFight)
            flags |= TraceFlag_FairFight;
        if (oneshotWindow)
            flags |= TraceFlag_OneshotWindow;
        if (useIndexed)
            flags |= TraceFlag_UseIndexedSearch;
        if (verifyIndexed)
            flags |= TraceFlag_VerifyIndexedSearch;
        if (exclTail)
            flags |= TraceFlag_ExcludeNonLoopTail;
        return flags;
    }
}

float MotionMatchingRuntimePolicy::ClampF(float value, float minValue, float maxValue)
{
    if (value < minValue)
        return minValue;
    if (value > maxValue)
        return maxValue;
    return value;
}

float MotionMatchingRuntimePolicy::MaxF(float a, float b)
{
    return a > b ? a : b;
}

float MotionMatchingRuntimePolicy::MinF(float a, float b)
{
    return a < b ? a : b;
}

float MotionMatchingRuntimePolicy::AbsF(float value)
{
    return value < 0.0f ? -value : value;
}

int32 MotionMatchingRuntimePolicy::MaxI(int32 a, int32 b)
{
    return a > b ? a : b;
}

bool MotionMatchingRuntimePolicy::GetIsReady() const
{
    return _samplingReady;
}

MotionMatchingRuntimePolicy::~MotionMatchingRuntimePolicy()
{
    Shutdown();
}

MotionMatchingPlayback* MotionMatchingRuntimePolicy::GetPlayback()
{
    return _playback;
}

MotionMatchingTraceLog* MotionMatchingRuntimePolicy::GetTrace()
{
    return _trace;
}

MotionMatchingTraceEntry MotionMatchingRuntimePolicy::GetLastEntry() const
{
    return _lastEntry;
}

void MotionMatchingRuntimePolicy::Initialize(
    MotionMatchingDatabase* database,
    AnimatedModel* body,
    const PoseSearchSchema& schema,
    const String& cacheFolder,
    float defaultInertial)
{
    Shutdown();
    _database = database;
    _body = body;
    _liveSchema = schema;
    _cacheFolder = cacheFolder;
    _playback = New<MotionMatchingPlayback>();
    _playback->InertializationDuration = defaultInertial > 0.01f ? defaultInertial : 0.25f;
    _trace = New<MotionMatchingTraceLog>();
    SkinnedModel* fallbackModel = nullptr;
    Float3 controllerUp(0.0f, 0.0f, -1.0f);
    if (body != nullptr)
    {
        fallbackModel = body->SkinnedModel.Get();
        _playback->Setup(database, fallbackModel, schema.Skeleton, schema.RootUpAxis);
    }
    else
    {
        _playback->Setup(database, nullptr, schema.Skeleton, schema.RootUpAxis);
    }
    _queryTimer = 0.0f;
    _timeSinceSwitch = 999.0f;
    _timeSinceInit = 0.0f;
    _reselectClip = -1;
    _reselectStartTime = -99.0f;
    _lastScanTotal = MAX_float;
    _lastFullScanTime = -99.0f;
    _lastFullScanFrame = -1;
    _consecutiveContinuationMisses = 0;
    _fastPathFrames = 0;
    _hasLastScanSignature = false;
    _fastPathReseed = true;
    ForcedReason = String::Empty;
    _policyFrame = 0;
    _nextSearchFrameId = 1;
    _lastObservedSampleFrameId = -1;
    _hasSampleStarts = false;
    _tickTrajectory.Clear();
    FastPathHits = 0;
    LastFastPathAge = 0.0f;
    _currentClip = -1;
    _hasLastActorPos = false;
    _poseValid = false;
    _latestPose.Clear();
    _latestRootWorld = Matrix::Identity;

    if (database == nullptr || body == nullptr)
        return;

    // Manual-pose ownership: a bound AnimGraph would overwrite every
    // SetCurrentPose write. Warn once; do not silently reconfigure.
    if (body->AnimationGraph.Get() != nullptr)
        LOG(Warning, "Motion matching playback expects no AnimGraph on the body: the graph would overwrite manually written poses every frame.");

    SkinnedModel* model = body->SkinnedModel.Get();
    if (model == nullptr)
    {
        LOG(Warning, "Motion matching policy disabled: no SkinnedModel on the body.");
        return;
    }

    String resolveError;
    if (!MotionMatchingSkeleton::TryResolve(schema.Skeleton, model, _nodes, resolveError))
    {
        const StringAnsi errorAnsi(resolveError);
        char message[512];
        snprintf(message, sizeof(message), "Motion matching policy disabled: %s", errorAnsi.GetText());
        LOG(Warning, "{}", String(message));
        return;
    }

    _prevPositions.Resize(21);
    for (int32 i = 0; i < 21; i++)
        _prevPositions[i] = 0.0f;
    _packedTrajectory.Resize(36);
    for (int32 i = 0; i < 36; i++)
        _packedTrajectory[i] = 0.0f;
    _hasPrevPositions = false;
    _usedSearchSnapshot = false;
    _searchSnapshotLength = 0;
    // init caches only the per-clip tag SETS (cheap, per-clip copy for
    // the fast-path incumbent check). The full motion scan (travel/turn/net
    // over every sample) and the audit file write moved to the explicit
    // GenerateClipAudit editor/tool command below — init never scans the DB
    // or touches MMClipAudit.txt.
    EnsureClipTagCache();
    _samplingReady = true;
    PolicyState = String(TEXT("Locomotion"));
    _trace->Clear();
    _telemetryHead = 0;
    _telemetryCount = 0;
    _telemetryDropped = 0;
    _lastEntry = MotionMatchingTraceEntry();
    LastSwitchReason = String(TEXT("init"));
    ChooserStatusText = String::Empty;
}

void MotionMatchingRuntimePolicy::Shutdown()
{
    _samplingReady = false;
    _database = nullptr;
    _body = nullptr;
    _liveSchema = PoseSearchSchema();
    _cacheFolder = String::Empty;
    _nodes = NodeIndices();
    _prevPositions.Clear();
    _packedTrajectory.Clear();
    _latestPose.Clear();
    _tickTrajectory.Clear();
    _poseScratch.Clear();
    _prevScratch.Clear();
    _tickTrajScratch.Clear();
    _hasPrevPositions = false;
    _usedSearchSnapshot = false;
    _searchSnapshotLength = 0;
    _poseValid = false;
    _hasLastActorPos = false;
    _hasSampleStarts = false;
    _hasLastScanSignature = false;
    _fastPathReseed = true;
    _chooserReseed = false;
    if (_playback != nullptr)
    {
        Delete(_playback);
        _playback = nullptr;
    }
    if (_trace != nullptr)
    {
        Delete(_trace);
        _trace = nullptr;
    }
    _clipTags.Clear();
    _clipSpeeds.Clear();
    _clipSpeedsDbId = 0;
    _telemetryHead = 0;
    _telemetryCount = 0;
    _telemetryDropped = 0;
    _queryTagsKey = String::Empty;
    _queryTagsCache.Clear();
    _queryExcludedKey = String::Empty;
    _queryExcludedCache.Clear();
    _queryExcludedJoin = String::Empty;
    _fastAllowedKey = String::Empty;
    _fastAllowedCache.Clear();
    _fastExcludedKey = String::Empty;
    _fastExcludedCache.Clear();
    _fastExcludedJoin = String::Empty;
    PolicyState = String(TEXT("Disabled"));
}

void MotionMatchingRuntimePolicy::EnsureClipTagCache()
{
    // init-time cache only — per-clip tag SET copies, no sample scan.
    // Every per-clip/per-frame reader goes through ClipTags() and never
    // touches native strings again. The motion scan (travel/turn/net over
    // every sample) lives in GenerateClipAudit below and never runs at init.
    _clipTags.Clear();
    MotionMatchingDatabase* db = _database;
    if (db == nullptr || !db->IsBaked())
        return;
    const int32 clipCount = db->GetClipCount();
    _clipTags.Resize(clipCount);
    for (int32 c = 0; c < clipCount; c++)
        _clipTags[c] = db->GetClipTags(c);
}

bool MotionMatchingRuntimePolicy::ExportClipAudit(MotionMatchingDatabase* db, const String& cacheFolder, float sampleRate, String& error)
{
    // explicit editor/tool command (the ONLY writer of MMClipAudit.txt).
    // Static: runs in edit mode with no policy/playback instance. Init/play
    // never call this — tools run it on demand (editor service / bridge
    // clip_audit op). Same file format as the old init-time write
    // (index|name|tags|loop|turn|travel|net|yaw|len|speed), computed into
    // locals — no scan arrays survive on the runtime.
    error = String::Empty;
    if (db == nullptr || !db->IsBaked())
    {
        error = String(TEXT("Clip audit stopped: database is not baked."));
        return false;
    }
    if (cacheFolder.IsEmpty())
    {
        error = String(TEXT("Clip audit stopped: cache folder is not set."));
        return false;
    }
    const int32 clipCount = db->GetClipCount();
    const Array<int32>& counts = db->GetClipSampleCounts();
    const Array<Float3>& positions = db->GetRootPositions();
    const Array<Float3>& forwards = db->GetRootForwards();
    if (counts.Count() == 0 || positions.Count() == 0 || forwards.Count() == 0)
    {
        error = String(TEXT("Clip audit stopped: database motion arrays are empty."));
        return false;
    }
    Array<float> travel;
    Array<float> turnAngle;
    Array<float> net;
    travel.Resize(clipCount);
    turnAngle.Resize(clipCount);
    net.Resize(clipCount);
    int32 cursor = 0;
    for (int32 c = 0; c < clipCount && cursor < positions.Count(); c++)
    {
        float path = 0.0f;
        float yaw = 0.0f;
        const int32 n = c < counts.Count() ? counts[c] : 0;
        Float3 first = cursor < positions.Count() ? positions[cursor] : Float3::Zero;
        Float3 last = first;
        for (int32 i = 1; i < n && cursor + i < positions.Count() && cursor + i < forwards.Count(); i++)
        {
            const Float3 dp = positions[cursor + i] - positions[cursor + i - 1];
            path += Float2(dp.X, dp.Z).Length();
            last = positions[cursor + i];
            const Float2 a(forwards[cursor + i - 1].X, forwards[cursor + i - 1].Z);
            const Float2 b(forwards[cursor + i].X, forwards[cursor + i].Z);
            if (a.LengthSquared() > 0.000001f && b.LengthSquared() > 0.000001f)
            {
                // Signed yaw travel (radians, engine convention): same
                // atan2(x, z) formula everywhere.
                const float ya = atan2f(a.X, a.Y);
                const float yb = atan2f(b.X, b.Y);
                float dy = yb - ya;
                while (dy > PiF)
                    dy -= 2.0f * PiF;
                while (dy < -PiF)
                    dy += 2.0f * PiF;
                yaw += dy;
            }
        }
        const float netDist = Float2(last.X - first.X, last.Z - first.Z).Length();
        travel[c] = path;
        turnAngle[c] = yaw;
        net[c] = netDist;
        cursor += n;
    }
    LOG(Info, "Motion matching clip audit: {} clips measured.", clipCount);
    // Accel audit (was LogAccelAudit at init): per-tag peak/avg planar
    // acceleration from baked root velocities, log only.
    {
        const Array<Float3>& velocities = db->GetRootVelocities();
        if (velocities.Count() > 0)
        {
            const float dt = 1.0f / MaxF(sampleRate, 1.0f);
            Array<String> tagNames;
            Array<float> tagPeak;
            Array<double> tagSum;
            Array<int32> tagNum;
            int32 vcursor = 0;
            for (int32 c = 0; c < clipCount && vcursor < velocities.Count(); c++)
            {
                const int32 n = c < counts.Count() ? counts[c] : 0;
                // Multi-tag: one sample's accel counts into EVERY tag its
                // clip carries (membership counting, same as RebuildTagCounts).
                const Array<String>& tagSet = db->GetClipTags(c);
                for (int32 i = 1; i < n && vcursor + i < velocities.Count(); i++)
                {
                    const Float3 dv = velocities[vcursor + i] - velocities[vcursor + i - 1];
                    const float a = Float2(dv.X, dv.Z).Length() / dt;
                    for (const String& tag : tagSet)
                    {
                        int32 found = -1;
                        for (int32 k = 0; k < tagNames.Count(); k++)
                        {
                            if (tagNames[k] == tag)
                            {
                                found = k;
                                break;
                            }
                        }
                        if (found < 0)
                        {
                            tagNames.Add(tag);
                            tagPeak.Add(a);
                            tagSum.Add((double)a);
                            tagNum.Add(1);
                        }
                        else
                        {
                            if (a > tagPeak[found])
                                tagPeak[found] = a;
                            tagSum[found] += (double)a;
                            tagNum[found]++;
                        }
                    }
                }
                vcursor += n;
            }
            char message[1024];
            int32 offset = snprintf(message, sizeof(message), "Motion matching accel audit (u/s^2 by tag): ");
            for (int32 k = 0; k < tagNames.Count() && offset < 900; k++)
            {
                const StringAnsi nameAnsi(tagNames[k]);
                const double avg = tagSum[k] / (double)MaxI(tagNum[k], 1);
                offset += snprintf(message + offset, sizeof(message) - (size_t)offset, "%s%s peak=%.0f avg=%.0f",
                    k > 0 ? "; " : "", nameAnsi.GetText(), (double)tagPeak[k], avg);
            }
            LOG(Info, "{}", String(message));
        }
    }
    const Array<float>& lengths = db->GetClipLengths();
    StringBuilder sb;
    char line[512];
    for (int32 c = 0; c < clipCount; c++)
    {
        Animation* anim = db->GetClipAnimation(c);
        String name;
        if (anim != nullptr)
        {
            const StringAnsi pathAnsi(anim->GetPath());
            const char* path = pathAnsi.GetText();
            const int32 length = pathAnsi.Length();
            int32 start = 0;
            int32 dot = -1;
            for (int32 i = 0; i < length; i++)
            {
                if (path[i] == '/' || path[i] == '\\')
                    start = i + 1;
                else if (path[i] == '.')
                    dot = i;
            }
            const int32 end = (dot > start) ? dot : length;
            if (end > start)
                name = String(path + start, end - start);
        }
        if (name.IsEmpty())
        {
            char fallback[32];
            snprintf(fallback, sizeof(fallback), "#%d", c);
            name = String(fallback);
        }
        const float len = lengths.Count() > c ? lengths[c] : 0.0f;
        const float spd = len > 0.01f ? travel[c] / len : 0.0f;
        // Audit file tag column is '+'-joined (tools split it back).
        const String tag = MotionMatchingDatabase::JoinTags(db->GetClipTags(c));
        const bool loop = db->GetClipLoop(c);
        const bool turnLike = db->GetClipTurnLike(c);
        const StringAnsi nameAnsi(name);
        const StringAnsi tagAnsi(tag);
        char fTravel[32], fNet[32], fYaw[32], fLen[32], fSpd[32];
        FormatFloatText(travel[c], fTravel, sizeof(fTravel));
        FormatFloatText(net[c], fNet, sizeof(fNet));
        FormatFloatText(turnAngle[c], fYaw, sizeof(fYaw));
        FormatFloatText(len, fLen, sizeof(fLen));
        FormatFloatText(spd, fSpd, sizeof(fSpd));
        snprintf(line, sizeof(line), "%d|%s|%s|%s|%s|%s|%s|%s|%s|%s\n",
            c, nameAnsi.GetText(), tagAnsi.GetText(),
            loop ? "loop" : "once", turnLike ? "turn" : "-",
            fTravel, fNet, fYaw, fLen, fSpd);
        sb.Append(line);
    }
    String text;
    sb.ToString(text);
    const StringAnsi folderAnsi(cacheFolder);
    char path[1024];
    snprintf(path, sizeof(path), "%s/MMClipAudit.txt", folderAnsi.GetText());
    const StringAnsi bodyAnsi(text);
    FILE* file = fopen(path, "wb");
    if (file == nullptr)
    {
        error = String(TEXT("Clip audit stopped: cannot write MMClipAudit.txt."));
        return false;
    }
    fwrite(bodyAnsi.GetText(), 1, (size_t)bodyAnsi.Length(), file);
    fclose(file);
    return true;
}

void MotionMatchingRuntimePolicy::ResetOnTeleport()
{
    // Teleport rule: pose history + playback SURVIVE teleports. Bone poses are
    // root-relative (recomputed fresh against the current root every
    // tick), so a teleport changes neither positions nor velocities; zeroing them poisoned the first query (pose cost in the thousands on 44 dims) and stopping playback discarded continuity for nothing.
    // zeroing them poisoned the first query (pose cost in the thousands
    // on 44 dims) and stopping playback discarded continuity for
    // nothing. Only trajectory (world-frame) and chooser state reset;
    // the first post-teleport query still adopts tags immediately and
    // searches loops only (reseed rule below).
    _poseValid = false;
    _latestPose.Clear();
    _usedSearchSnapshot = false;
    _searchSnapshotLength = 0;
    _queryTimer = 0.0f;
    _timeSinceSwitch = 999.0f;
    _timeSinceInit = 0.0f;
    // _currentClip + playback survive (see above): continuity needs no
    // re-pick after a pure position jump.
    _hasLastActorPos = false;
    _chooserReseed = true;
    ActiveTag = String::Empty;
    // invalidate the change-only scope caches (both RunQuery's pair and
    // the fast-path-only pair) so the first post-reset query rebuilds the
    // splits AND the ActiveTag/ChooserStatus diagnostics even when the scope
    // string is unchanged across the reset. Keys AND caches are cleared
    // together: clearing the key alone would false-hit when the scope
    // legitimately equals the cleared value (empty scope "" would reuse the
    // stale pre-reset split and search the wrong scope instead of taking
    // the explicit-empty no-match path). Clearing both restores the
    // invariant key==scope ==> cache==Split(scope) with zero steady-state
    // cost (reset/teleport only; the next query rebuilds once).
    _queryTagsKey = String::Empty;
    _queryTagsCache.Clear();
    _queryExcludedKey = String::Empty;
    _queryExcludedCache.Clear();
    _queryExcludedJoin = String::Empty;
    _fastAllowedKey = String::Empty;
    _fastAllowedCache.Clear();
    _fastExcludedKey = String::Empty;
    _fastExcludedCache.Clear();
    _fastExcludedJoin = String::Empty;
    _reselectClip = -1;
    _reselectStartTime = -99.0f;
    _lastScanTotal = MAX_float;
    _lastFullScanTime = -99.0f;
    _lastFullScanFrame = -1;
    _consecutiveContinuationMisses = 0;
    _fastPathFrames = 0;
    _hasLastScanSignature = false;
    _fastPathReseed = true;
    ForcedReason = String::Empty;
    _hasSampleStarts = false;
    _tickTrajectory.Clear();
    FastPathHits = 0;
    LastFastPathAge = 0.0f;
    if (_playback != nullptr)
        _playback->SetPlayRate(1.0f);
    LastSwitchReason = String(TEXT("teleport-reset"));
    // teleport/reset is effective input (it reseeds the chooser to
    // LoopOnly on the next query). Record it as a structural command entry
    // so replay accounts for every discontinuity. Record-only: playback and
    // chooser state above are untouched.
    if (_trace != nullptr)
        RecordPolicyEvent("reset", _currentClip, String(TEXT("teleport-reset")));
    // Playback keeps playing (see above): same clip, same time, root
    // stripped — the picture never jumps. Only timers/locks release.
}

void MotionMatchingRuntimePolicy::CheckSchemaBinding()
{
    if (_bindingWarned || _database == nullptr || !_database->IsBaked())
        return;
    _bindingWarned = true;
    // Scene weights are live tuning (allowed to differ); skeleton names,
    // axes, sample rate and trajectory times must match the baked snapshot
    // or the query frame silently disagrees with the database.
    const PoseSearchSchema& live = _liveSchema;
    const PoseSearchSchema& baked = _database->GetSchema();
    const bool mismatch =
        live.Skeleton.Root != baked.Skeleton.Root ||
        live.Skeleton.Pelvis != baked.Skeleton.Pelvis ||
        live.Skeleton.LeftFoot != baked.Skeleton.LeftFoot ||
        live.Skeleton.RightFoot != baked.Skeleton.RightFoot ||
        live.SampleRate != baked.SampleRate ||
        !AxesNear(live.RootForwardAxis, baked.RootForwardAxis) ||
        !AxesNear(live.RootUpAxis, baked.RootUpAxis) ||
        !TimesEqual(live.TrajectorySampleTimes, baked.TrajectorySampleTimes);
    if (mismatch)
        LOG(Warning, "Motion matching schema differs from the baked database snapshot (skeleton/axes/rate/times). Query uses baked axes; rebake to reconcile. Scene weights remain live.");
}

bool MotionMatchingRuntimePolicy::AxesNear(const Float3& a, const Float3& b)
{
    const Float3 na = Float3::Normalize(a);
    const Float3 nb = Float3::Normalize(b);
    return (na - nb).Length() <= 0.001f;
}

bool MotionMatchingRuntimePolicy::TimesEqual(const Array<float>& a, const Array<float>& b)
{
    if (a.Count() != b.Count())
        return false;
    for (int32 i = 0; i < a.Count(); i++)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

String MotionMatchingRuntimePolicy::GetClipName(int32 clipIndex)
{
    Animation* anim = (_database != nullptr) ? _database->GetClipAnimation(clipIndex) : nullptr;
    if (anim != nullptr)
    {
        const StringAnsi pathAnsi(anim->GetPath());
        const char* path = pathAnsi.GetText();
        const int32 length = pathAnsi.Length();
        int32 start = 0;
        int32 dot = -1;
        for (int32 i = 0; i < length; i++)
        {
            if (path[i] == '/' || path[i] == '\\')
                start = i + 1;
            else if (path[i] == '.')
                dot = i;
        }
        const int32 end = (dot > start) ? dot : length;
        if (end > start)
            return String(path + start, end - start);
    }
    char fallback[32];
    snprintf(fallback, sizeof(fallback), "#%d", clipIndex);
    return String(fallback);
}

const Array<String>& MotionMatchingRuntimePolicy::ClipTags(int32 clipIndex)
{
    if (_clipTags.Count() > 0 && clipIndex >= 0 && clipIndex < _clipTags.Count())
        return _clipTags[clipIndex];
    if (_database != nullptr)
        return _database->GetClipTags(clipIndex);
    static const Array<String> empty;
    return empty;
}

void MotionMatchingRuntimePolicy::WarnOnce(const String& message)
{
    if (_warnedOnce)
        return;
    _warnedOnce = true;
    const StringAnsi ansi(message);
    LOG(Warning, "{}", message);
}

float MotionMatchingRuntimePolicy::SelectPlayRate(float characterPlanarSpeed)
{
    // Bend clip time to match world speed: characterPlanar / clipRootSpeed,
    // clamped to the playback [0.85, 1.15] window (SetPlayRate clamps again).
    // Clip speed reuses the audit path (planar travel / length from baked
    // root positions + clip lengths), cached once per database in
    // EnsureClipSpeedCache — never a per-tick DB scan. Character speed is
    // the Game-side planar speed (explicit Tick param, teleport-clean).
    // Missing metadata for a clip returns 1.0 for that tick.
    if (_database == nullptr || !_database->IsBaked() || _currentClip < 0)
        return 1.0f;
    if (characterPlanarSpeed <= 10.0f)
        return 1.0f;
    EnsureClipSpeedCache();
    if (_currentClip >= _clipSpeeds.Count())
        return 1.0f;
    const float clipSpeed = _clipSpeeds[_currentClip];
    if (clipSpeed <= 1.0f)
        return 1.0f;
    const float rate = characterPlanarSpeed / clipSpeed;
    if (rate < 0.85f)
        return 0.85f;
    if (rate > 1.15f)
        return 1.15f;
    return rate;
}

void MotionMatchingRuntimePolicy::EnsureClipSpeedCache()
{
    int32 dbId = 0;
    MotionMatchingDatabase* db = _database;
    int32 clipCount = 0;
    if (db != nullptr)
    {
        clipCount = db->GetClipCount();
        dbId = (int32)(((uintptr_t)db) & 0xFFFFFFFF) ^ (clipCount * 397) ^ (db->GetSampleCount() * 13);
    }
    if (dbId != 0 && dbId == _clipSpeedsDbId && _clipSpeeds.Count() == clipCount)
        return;
    _clipSpeeds.Clear();
    _clipSpeedsDbId = 0;
    if (db == nullptr || !db->IsBaked() || clipCount <= 0)
        return;
    const Array<float>& lengths = db->GetClipLengths();
    const Array<Float3>& positions = db->GetRootPositions();
    const Array<int32>& counts = db->GetClipSampleCounts();
    if (counts.Count() != clipCount || positions.Count() == 0)
        return;
    _clipSpeeds.Resize(clipCount);
    int32 cursor = 0;
    for (int32 c = 0; c < clipCount; c++)
    {
        float speed = 0.0f;
        const int32 n = c < counts.Count() ? counts[c] : 0;
        const float len = c < lengths.Count() ? lengths[c] : 0.0f;
        if (n > 1 && cursor + n <= positions.Count() && len > 0.01f)
        {
            float path = 0.0f;
            for (int32 i = 1; i < n; i++)
            {
                const Float3 dp = positions[cursor + i] - positions[cursor + i - 1];
                path += Math::Sqrt(dp.X * dp.X + dp.Z * dp.Z);
            }
            speed = path / len;
        }
        _clipSpeeds[c] = speed;
        cursor += n;
    }
    _clipSpeedsDbId = dbId;
}

void MotionMatchingRuntimePolicy::Tick(
    float deltaTime,
    const MotionMatchingPlayerSnapshot& state,
    const Array<TrajectoryPoint>& trajectoryPoints,
    const Matrix& actorWorld,
    const Array<float>& trajectorySampleTimes,
    const MotionMatchingPolicyTuning& tuning,
    int32 callerLoopFilter,
    int32 callerCandidateBand,
    float callerYawRate,
    float callerPlanarSpeed)
{
    if (!_samplingReady || _database == nullptr || _body == nullptr || _playback == nullptr || _trace == nullptr)
        return;

    // profiling (record-only): whole-Tick wall time. One chrono pair per
    // tick is nanoseconds; it never steers search or playback.
    const auto tickStart = std::chrono::high_resolution_clock::now();
    // field-wise change-only snapshot copy. MotionMatchingPlayerSnapshot
    // carries two native Strings; a blind struct copy reallocates both every
    // tick. Scope strings only change on transitions, so steady-state ticks
    // copy nothing (plain == compares, no allocation).
    if (state.AllowedTagSet != _snapshot.AllowedTagSet)
        _snapshot.AllowedTagSet = state.AllowedTagSet;
    if (state.ExcludedTagSet != _snapshot.ExcludedTagSet)
        _snapshot.ExcludedTagSet = state.ExcludedTagSet;
    _tuning = tuning;
    // caller filter (Game selection policy output): the ONLY LoopFilter /
    // CandidateBand / YawRate the core uses. Clamped to the valid range;
    _callerLoopFilter = callerLoopFilter >= 0 && callerLoopFilter <= 2 ? callerLoopFilter : 0;
    _callerCandidateBand = callerCandidateBand >= 0 && callerCandidateBand <= 3 ? callerCandidateBand : 0;
    _callerYawRate = callerYawRate >= 0.0f && callerYawRate < 1e29f ? callerYawRate : 0.0f;
    _callerPlanarSpeed = callerPlanarSpeed >= 0.0f && callerPlanarSpeed < 1e29f ? callerPlanarSpeed : 0.0f;
    // _trajectoryPoints/_sampleTimes dead copies deleted (were written
    // every tick but never read; RunQuery uses its params + _tickTrajectory).

    const float frameDt = MaxF(deltaTime, 0.000001f);
    _timeSinceSwitch += frameDt;
    _timeSinceInit += frameDt;
    _policyFrame++;

    // Teleport/discontinuity guard: an actor jump without Reset call would
    // otherwise leave stale previous positions and a locked continuity
    // winner. Threshold is far above any legitimate per-frame travel.
    const Float3 actorPos(actorWorld.M41, actorWorld.M42, actorWorld.M43);
    if (_hasLastActorPos && (actorPos - _lastActorPos).LengthSquared() > TeleportDistance * TeleportDistance)
    {
        ResetOnTeleport();
        // Re-accumulate this frame after the reset so warmup restarts cleanly.
        _timeSinceInit = frameDt;
        _timeSinceSwitch = 999.0f;
    }
    _lastActorPos = actorPos;
    _hasLastActorPos = true;

    CheckSchemaBinding();

    // Sample the live pose EVERY tick. Velocity = (pos - prev) / frameDt,
    // so the time basis always matches the sampling rate. Seeding happens
    // only after a successful sample (zero-velocity first frame).
    // Query-source rule: prefer the playback
    // snapshot over the displayed pose, so the layer and foot IK never feed
    // back into the next frame's candidate costs. Displayed-pose
    // sampling stays as the fallback (no snapshot before the first
    // playback tick). A source switch reseeds velocities (one
    // zero-velocity frame) because the two poses differ by layer/IK.
    bool justSeeded = false;
    // const reference (no Array-by-value copy; GetSearchLocals stays for
    // diagnostics/tools, Tick uses the ref).
    const Array<Matrix>& searchLocals = _playback->GetSearchLocalsRef();
    const bool haveSnapshot = searchLocals.Count() > 0;
    if (haveSnapshot != _usedSearchSnapshot ||
        (haveSnapshot && searchLocals.Count() != _searchSnapshotLength))
    {
        _hasPrevPositions = false;
        _usedSearchSnapshot = haveSnapshot;
        _searchSnapshotLength = haveSnapshot ? searchLocals.Count() : 0;
    }
    const float sampleDt = _hasPrevPositions ? frameDt : 0.0f;
    bool sampled = false;
    // member scratch instead of per-tick locals (Resize only when the
    // feature size changes; schema-fixed 44/prev sizes, so steady-state
    // reuses without native reallocation).
    Matrix rootWorld;
    String poseError;
    if (haveSnapshot)
    {
        sampled = MotionMatchingQuery::SampleSearchPose(_body, searchLocals, _nodes, sampleDt,
            _prevPositions, _database->GetSchema(), _poseScratch, _prevScratch, rootWorld, poseError);
    }
    else
    {
        sampled = MotionMatchingQuery::SampleLivePose(_body, _nodes, sampleDt, _prevPositions,
            _poseScratch, _prevScratch, rootWorld, poseError);
    }
    if (sampled)
    {
        if (!_hasPrevPositions)
        {
            _hasPrevPositions = true;
            justSeeded = true;
        }
        _prevPositions = _prevScratch;
        _latestPose = _poseScratch;
        _latestRootWorld = rootWorld;
        _poseValid = true;
    }
    else
    {
        // Keep the old previous positions and try again next tick; never
        // seed success on failure and never feed a stale pose to search.
        _poseValid = false;
        if (!_warnedOnce)
        {
            _warnedOnce = true;
            const StringAnsi errorAnsi(poseError);
            char message[512];
            snprintf(message, sizeof(message), "Motion matching query skipped: %s", errorAnsi.GetText());
            LOG(Warning, "{}", String(message));
        }
    }

    // Trajectory rule: trajectory is built EVERY tick (cheap), not just on query
    // ticks, so the continuing fast-path can score each frame. Failures
    // null the cache; RunQuery treats null like today's trajError path.
    // no intent signals — the Game owns onset/fair-fight and passes only
    // the resulting filter per tick (Tick params above). The fast-path yields
    // to a changed caller filter via the signature check inside TryFastPath.
    _tickTrajectory.Clear();
    if (_poseValid && trajectoryPoints.Count() == 6)
    {
        for (int32 i = 0; i < 6; i++)
        {
            _packedTrajectory[i * 6] = trajectoryPoints[i].Position.X;
            _packedTrajectory[i * 6 + 1] = trajectoryPoints[i].Position.Y;
            _packedTrajectory[i * 6 + 2] = trajectoryPoints[i].Position.Z;
            _packedTrajectory[i * 6 + 3] = trajectoryPoints[i].Facing.X;
            _packedTrajectory[i * 6 + 4] = trajectoryPoints[i].Facing.Y;
            _packedTrajectory[i * 6 + 5] = trajectoryPoints[i].Facing.Z;
        }
        // build into member scratch (no per-tick local Array).
        String trajError;
        if (!MotionMatchingQuery::BuildLiveTrajectory(_packedTrajectory, actorWorld, _latestRootWorld,
                _database->GetSchema(), _tickTrajScratch, trajError))
            _tickTrajectory.Clear();
        else
            _tickTrajectory = _tickTrajScratch;
    }


    _queryTimer += frameDt;
    const bool warmed = !justSeeded && _timeSinceInit >= WarmupTime;
    // Per-frame search: when enabled, every warmed tick may scan (the
    // budget/stride caps the cost, intent changes scan unstrided). The
    // cadence path below stays for SearchEveryFrame=false.
    const bool queryDue = warmed && (_tuning.SearchEveryFrame ||
        _queryTimer >= MaxF(_tuning.QueryInterval, 0.01f));
    // pure search dispatch (machine holds deleted).
    if (queryDue && warmed)
    {
        _queryTimer = 0.0f;
        // pure search dispatch (machine holds deleted). Cadence ticks
        // may run a full scan on continuation miss; off-cadence ticks below
        // only try the cheap continuation.
        _fastPathFrames++;
        if (!TryFastPath(frameDt))
            RunQuery(trajectoryPoints, actorWorld, frameDt);
    }
    else if (queryDue && (_timeSinceInit < WarmupTime || justSeeded))
    {
        // Warmup/seeding ticks consume the timer without searching so the
        // first real query sees settled pose velocity and trajectory past.
        _queryTimer = 0.0f;
    }
    else if (warmed && _poseValid)
    {
        // Off-cadence rule: cheap continuation only. A miss is
        // recorded (and forces the full scan at the next cadence tick), but
        // no scan runs here — a jittering frame time can no longer turn
        // every off-cadence frame into a full search (scan storm).
        TryFastPath(frameDt);
    }

    // Play-rate applies to the NEXT tick's advance: it is computed from this
    // tick's winning clip (_currentClip just decided above) against the
    // Game-side planar speed, so a start one-shot never slow-plays its first
    // frames on the stale rate of the previous clip (observed: idle rate
    // 1.0 -> start plays slow while char outruns it = start slide; the clip
    // then fast-forwards to catch up once its own speed is known).
    _playback->SetPlayRate(SelectPlayRate(_callerPlanarSpeed));
    // Tick-order rule: stamp this tick's search decision on playback before it
    // asserts intra-tick ordering (sample <= search <= save < layer < IK).
    NotifySearch();
    // playback never steers on gameplay state (grounded param unused
    // inside Tick); the Game owns airborne meaning and passes only the
    // trajectory + filter. Neutral true keeps the signature stable.
    _playback->Tick(frameDt, _body, true);
    // Keep the search stamp mirror in lockstep with playback's tick
    // counter: it advances only on a successful playback Tick, so early
    // outs never desync the stamp.
    if (_playback->SampleFrameId != _lastObservedSampleFrameId)
    {
        _lastObservedSampleFrameId = _playback->SampleFrameId;
        if (_playback->SampleFrameId >= 0)
            _nextSearchFrameId = _playback->SampleFrameId + 1;
    }

    // A finished one-shot releases the continuity lock so the next
    // query may leave immediately instead of holding the last frame.
    if (_playback->IsAtEnd)
        _timeSinceSwitch = 999.0f;

    // profiling tail: record whole-Tick time (see tickStart above).
    {
        const auto tickEnd = std::chrono::high_resolution_clock::now();
        const float tickMs = (float)std::chrono::duration<double, std::milli>(tickEnd - tickStart).count();
        LastTickMs = tickMs;
        TotalTickMs += tickMs;
        TickCount++;
    }
}

// gait-band helper + intent-signal updater deleted (were here). Gait
// bands, onset windows, fair-fight accumulators and all derived loop /
// one-shot candidate decisions live Game-side in LocomotionSelectionPolicy;
// the core uses only the per-tick caller filter (Tick params).

void MotionMatchingRuntimePolicy::RecordPolicyEvent(const char* kind, int32 clipIndex, const String& reason)
{
    if (_trace == nullptr)
        return;
    // policy events carry no movement diagnostics (the core never sees
    // velocity/yaw/grounded/sprint; the Game owns them). Speed/yaw/des
    // record 0, grounded true — keys stay for contract stability, values are
    // intentionally neutral. Yaw-rate-driven search uses _callerYawRate on
    // search/fastpath entries (RecordTrace), never here.
    MotionMatchingTraceEntry entry;
    entry.Kind = String(kind);
    entry.Time = Time::GetGameTime();
    entry.State = PolicyState;
    entry.Tags = ActiveTag;
    entry.Speed = 0.0f;
    entry.YawRate = 0.0f;
    entry.Grounded = true;
    entry.Sprinting = false;
    entry.DesiredSpeed = 0.0f;
    entry.Valid = false;
    entry.ClipIndex = clipIndex;
    entry.Reason = reason;
    entry.Chooser = ChooserStatusText;
    entry.QueryMs = 0.0f;
    // reset/command entries carry the reseed flag and the surviving
    // clip so replay can check them structurally. Vectors stay empty by
    // design (no search ran on these ticks).
    entry.TraceFlags = PackTraceFlags(false, _chooserReseed, false, false, false, false, false);
    entry.CommandClip = clipIndex;
    entry.DbClipCount = _database != nullptr ? _database->GetClipCount() : 0;
    entry.DbSampleCount = _database != nullptr ? _database->GetSampleCount() : 0;
    Array<float> empty;
    _trace->Record(entry, empty, empty);
}

void MotionMatchingRuntimePolicy::RunQuery(const Array<TrajectoryPoint>& trajectoryPoints, const Matrix& actorWorld, float frameDt)
{
    const auto watchStart = std::chrono::high_resolution_clock::now();

    if (!_poseValid || _latestPose.Count() == 0)
        return;

    // Tick-order rule: this tick's search decision is a real full scan.
    NotifySearch();

    // the core executes the caller filter verbatim. No airborne/policy
    // state, no speeds, no yaw gates, no intent windows here — the Game
    // selection policy owns all of that and passes LoopFilter/CandidateBand/
    // YawRate per tick (Tick params). Diagnostics that used movement record
    // neutral values (speed/des 0); yaw rides the caller value for the turn
    // prior + replay.
    // PolicyState is set once at Initialize ("Locomotion"); the per-query
    // re-assignment of the same literal is deleted (no String alloc per scan).
    const float yawRate = _callerYawRate;
    bool forced = false;

    // Reseed query (first query after a teleport runs neutral): the first query after a teleport/
    // discontinuity runs on a zero-velocity reseed pose that cannot honestly
    // match a transition (all bone velocities read zero). Loops only —
    // otherwise a planted one-shot (pivot) wins the poisoned query and the
    // playing pose then looks like itself forever. No gait read.
    const bool reseeded = _chooserReseed;
    // Keep the reseed LoopOnly guard armed until the Game-side planar speed
    // settles (teleport frames report ~0 while the spring spins up; the
    // first query after teleport otherwise plants a one-shot on a
    // zero-velocity poisoned pose and continuity holds it all phase).
    // _chooserReseed clears on the first settled scan below, not here.
    bool keepReseedArmed = _chooserReseed && _callerPlanarSpeed <= 5.0f;
    // match, no Split per query in steady-state). Bound: <=8 tags; rebuilt
    // only on scope change. Semantics unchanged: same tags array as Split.
    // twins: this RunQuery pair (_queryTagsKey/_queryTagsCache) is
    // SEPARATE from TryFastPath's pair (_fastAllowedKey/_fastAllowedCache,
    // see the header + GetFastAllowed) on purpose, so the incumbent check
    // never consumes RunQuery's scope-change signal. A const reference is
    // used (no Array<String> copy); a miss Splits once and caches (rare:
    // scope transitions only). tagsReused is captured BEFORE the cache call: the
    // call updates the key on a miss, so comparing after would always read
    // hit. A const reference is used (no Array<String> copy); tagsReused
    // also gates the ActiveTag/ChooserStatus rebuild below (on a hit the
    // values are already correct).
    const bool tagsReused = (_snapshot.AllowedTagSet == _queryTagsKey);
    const Array<String>& tags = GetCachedAllowed();
    if (!tagsReused)
    {
        ActiveTag = MotionMatchingDatabase::JoinTags(tags);
        ChooserStatusText = String(TEXT("ext:")) + ActiveTag;
    }
    if (tags.Count() == 0)
    {
        // the empty markers are set on EVERY empty tick (not only on a
        // scope change): an empty tick must report empty even when the scope
        // did not change since the last query. Empty scope never occurs in
        // steady play (suite scopes are always non-empty), so these small
        // String writes cost nothing steady-state and keep the trace reason
        // exact on every empty tick.
        ActiveTag = String::Empty;
        ChooserStatusText = String(TEXT("ext:empty (no-match)"));
        LastSwitchReason = String(TEXT("no-result:empty-scope"));
        // the explicit-empty short-circuit is effective input (a whole
        // tick decided with no search). Record it with the vectors on hand
        // so replay checks the no-match structurally. Record-only: the
        // early return below is unchanged. No retry may widen the scope:
        // empty stays empty (hold).
        {
            MotionMatchingSearchSettings emptySettings;
            emptySettings.PoseWeight = _tuning.PoseWeight;
            emptySettings.TrajectoryWeight = _tuning.TrajectoryWeight;
            emptySettings.ContinuityWeight = _tuning.ContinuityWeight;
            emptySettings.MinSwitchTime = _tuning.MinSwitchTime;
            emptySettings.LoopFilter = (MotionLoopFilter)_callerLoopFilter;
            emptySettings.OneShotBand = _callerCandidateBand;
            emptySettings.QueryYawRate = yawRate;
            emptySettings.TurnPenaltyWeight = _tuning.TurnPenaltyWeight;
            emptySettings.TurnPenaltyYawScale = _tuning.TurnPenaltyYawScale;
            // change-only excluded split (same shared cache as the main
            // path below; empty ExcludedTagSet disables with no Split).
            if (!_snapshot.ExcludedTagSet.IsEmpty())
                emptySettings.ExcludeTags = GetCachedExcluded();
            const float emptyRate = _playback != nullptr ? _playback->GetPlayRate() : 1.0f;
            MotionMatchingResult emptyResult;
            // LastSwitchReason already carries "no-result:empty-scope"
            // (set in the miss branch above); reuse it instead of building
            // the same literal again for the trace reason.
            RecordTrace(0.0f, yawRate, tags, _tickTrajectory, emptySettings,
                -1, 999.0f, 0, false, reseeded, false, false, frameDt, emptyRate,
                false, emptyResult, LastSwitchReason);
        }
        return;
    }

    if (trajectoryPoints.Count() != 6)
    {
        WarnOnce(String(TEXT("Motion matching query skipped: trajectory has 6 points.")));
        return;
    }

    // Trajectory rule: trajectory is built every tick in Tick (see _tickTrajectory);
    // a null cache means the build failed and matches today's trajError path.
    const Array<float>& trajectory = _tickTrajectory;
    if (trajectory.Count() == 0)
    {
        WarnOnce(String(TEXT("Motion matching query skipped: trajectory build failed this tick.")));
        return;
    }

    // Weights come from the live tuning, not the baked snapshot:
    // weights are never baked into features, so tuning them here applies
    // immediately with no rebake (edit them in the inspector, even mid-play).
    const float currentTime = (_currentClip >= 0 && _playback != nullptr) ? _playback->CurrentTime : -1.0f;
    // Play-rate rule: ExpectedTime = playback time + THIS frame
    // upcoming advance (frameDt * PlayRate). The decision takes effect in
    // this frame's _playback.Tick below, so the continuity anchor is the
    // post-advance position — never CurrentTime + the full inter-query
    // interval (that double-counts: CurrentTime already contains every
    // intermediate frame's advance). Matches TryFastPath's prediction.
    const float playRate = _playback != nullptr ? _playback->GetPlayRate() : 1.0f;
    float expectedTime = -1.0f;
    if (currentTime >= 0.0f)
    {
        const float advance = MaxF(frameDt, 0.000001f) * playRate;
        expectedTime = currentTime + advance;
        if (_database != nullptr && _currentClip >= 0 && _database->GetClipLoop(_currentClip))
        {
            const float len = _playback != nullptr ? _playback->GetCurrentClipLength() : 0.0f;
            if (len > 0.0f && expectedTime >= len)
                expectedTime = fmodf(expectedTime, len);
        }
    }
    // rest/settle/recovery intent branches deleted (were here). The Game
    // selection policy owns them and passes the resulting caller filter per
    // tick. The core applies the caller filter verbatim below — never
    // inferred from movement.
    // End-recovery RETRY (same scope, loop-only, below) is the only surviving
    // structural recovery: it reads playback IsAtEnd only, never gait, and
    // never widens the tag scope.
    const float minSwitch = _tuning.MinSwitchTime;
    MotionMatchingSearchSettings settings;
    settings.PoseWeight = _tuning.PoseWeight;
    settings.TrajectoryWeight = _tuning.TrajectoryWeight;
    settings.ContinuityWeight = _tuning.ContinuityWeight;
    settings.MinSwitchTime = minSwitch;
    settings.CurrentTime = currentTime;
    settings.SameClipExclusionWindow = _tuning.SameClipExclusion;
    settings.ExpectedTime = expectedTime;
    settings.ContinuingBias = _tuning.ContinuingBias;
    settings.ContinuingWindow = _tuning.ContinuingWindow;
    settings.SwitchMargin = _tuning.SwitchMargin;
    settings.LoopPreference = 0.0f;
    // Exclusion bans: reselect lives on a wall-clock lifetime owned here
    // (the search is stateless); twin + tail are always-on structural
    // scope cuts, replayed bit-exactly from the trace entry.
    {
        const bool banLive = EnableReselectBan && _reselectClip >= 0 &&
            Time::GetGameTime() - _reselectStartTime < ReselectHistoryDuration;
        settings.ReselectClip = banLive ? _reselectClip : -1;
        settings.ReselectSampleTime = _reselectSampleTime;
        settings.ReselectBanWindow = banLive ? ReselectSampleWindow : 0.0f;
    }
    settings.TwinBanWindow = SameSegmentTwinBan;
    settings.LoopTailExclusion = EnableLoopTailExclusion ? NonLoopTailExclusion : 0.0f;
    // Structural-validity rule: non-loop candidates whose
    // future horizon passes the clip end baked a clamped fake-stopped
    // future — never rank them (measured: start-tail samples winning
    // releases on exactly such fakes). C# bools default false, so this
    // must be set explicitly; the C++ default never reaches runtime.
    settings.ExcludeNonLoopTail = true;
    // Per-frame budgeted search: every query scores a rotating subset
    // (FrameSearchBudget samples). QueryFrame rotates the stride cursor
    // (policy frame, reset per session). No intent-change unstrided
    // bypass — the Game owns intent and expresses urgency through the
    // caller filter; the core budgets uniformly.
    settings.MaxCandidates = MaxI(_tuning.FrameSearchBudget, 0);
    settings.QueryFrame = _policyFrame;
    // Indexed search (default off; brute force reference).
    // Verify audit file-harness also forces both flags (no default
    // or scene change needed to measure divergence).
    settings.UseIndexedSearch = _tuning.UseIndexedSearch || _tuning.IndexedVerifyAuto;
    settings.TopKCandidates = MaxI(1, _tuning.TopKCandidates);
    settings.VerifyIndexedSearch = _tuning.VerifyIndexedSearch || _tuning.IndexedVerifyAuto;
    settings.QueryYawRate = yawRate;
    settings.TurnPenaltyWeight = _tuning.TurnPenaltyWeight;
    settings.TurnPenaltyYawScale = _tuning.TurnPenaltyYawScale;
    // foot-lock search prior deleted (was hard-zeroed here since the
    // plugin solver died; Game-side IK owns its own contact state). Contact
    // channels + feature extraction + DB stay untouched.
    // caller filter (Game selection policy output): reseed forces LoopOnly
    // (neutral structural rule, no gait read); otherwise the caller filter
    // rides verbatim. The core never derives LoopFilter/OneShotBand from
    // movement, grounded, sprint or yaw.
    settings.LoopFilter = (reseeded || keepReseedArmed) ? MotionLoopFilter::LoopOnly :
        (MotionLoopFilter)_callerLoopFilter;
    settings.OneShotBand = _callerCandidateBand;
    // Controller-decided exclusions ride the snapshot (plain-String
    // transport, split here). Independent of the include scope above.
    // change-only split via the shared cache (steady-state reuses).
    if (!_snapshot.ExcludedTagSet.IsEmpty())
        settings.ExcludeTags = GetCachedExcluded();
    settings.CollectClassDiagnostics = _tuning.EffectiveTelemetry;

    // Fast-path signature rule: capture the query signature this full scan anchors the fast
    // path against (clip/time cannot change until Play below).
    const FastPathSignature scanSignature = CaptureFastPathSignature();

    // no forced edge (take-off/landing bypass deleted with the gameplay
    // state). Continuity always anchors on the current clip; end-recovery
    // below bypasses via its own retry call (continuity -1 there).
    const int32 continuityClip = _currentClip;
    const float sinceSwitch = _timeSinceSwitch;

    MotionMatchingResult result;
    String error;
    MotionMatchingSearch::FindBest(_database, _latestPose, trajectory, settings,
        continuityClip, sinceSwitch, tags, result, error);

    // End recovery (KEPT, neutral): a finished one-shot with temporal
    // exclusion active can exclude every same-clip candidate and return
    // nothing (frozen last frame forever). Retry once without exclusion,
    // demanding LOOP clips structurally (LoopOnly) instead of naming a rest
    // gait: the finished one-shot cannot re-win on exact pose match,
    // whatever tag it has. Reads playback IsAtEnd only — never gait, speed,
    // yaw or intent — and never widens the tag scope (same tags).
    bool recovered = false;
    int32 retryKind = 0;
    if (!result.IsValid && _playback != nullptr && _playback->IsAtEnd)
    {
        MotionMatchingSearchSettings retry = settings;
        retry.CurrentTime = -1.0f;
        retry.SameClipExclusionWindow = 0.0f;
        // Recovery must always have somewhere to go: drop the reselect
        // ban (twin is already gated off by continuity -1, tail is moot
        // under LoopOnly).
        retry.ReselectClip = -1;
        retry.ReselectBanWindow = 0.0f;
        retry.LoopFilter = MotionLoopFilter::LoopOnly;
        MotionMatchingResult retryResult;
        String retryError;
        MotionMatchingSearch::FindBest(_database, _latestPose, trajectory, retry,
            -1, 999.0f, tags, retryResult, retryError);
        if (retryResult.IsValid)
        {
            result = retryResult;
            error = retryError;
            forced = true;
            recovered = true;
            retryKind = 1;
        }
    }

    const auto watchEnd = std::chrono::high_resolution_clock::now();
    LastQueryMs = (float)std::chrono::duration<double, std::milli>(watchEnd - watchStart).count();
    // profiling: LastQueryMs is the full-scan block (FindBest + retries).
    TotalFullSearchMs += LastQueryMs;
    FullSearchCount++;

    // onset-fallback retry DELETED (was here): it re-ran the search with
    // LoopFilter Any whenever the onset window found nothing, reading the
    // gameplay intent window to widen the caller filter. Empty stays empty;
    // no-match holds with no silent retry in another scope. Only the neutral
    // end-recovery retry above survives.

    // Fast-path signature rule: a completed full scan re-anchors the fast-path baseline and
    // clears reseed once planar settles (see keepReseedArmed above: clearing
    // on the poisoned zero-velocity query plants a one-shot all phase).
    // An invalid result invalidates the cost baseline so
    // the next continuation check misses instead of holding stale data.
    _lastFullScanTime = Time::GetGameTime();
    _lastFullScanFrame = _policyFrame;
    _fastPathFrames = 0;
    _consecutiveContinuationMisses = 0;
    _fastPathReseed = false;
    if (!keepReseedArmed)
        _chooserReseed = false;
    _lastScanSignature = scanSignature;
    _hasLastScanSignature = true;
    _lastScanTotal = result.IsValid ? result.Cost : MAX_float;

    if (error.IsEmpty() == false || !result.IsValid)
    {
        char message[512];
        const StringAnsi errorAnsi(error);
        snprintf(message, sizeof(message), "Motion matching search produced nothing: %s", errorAnsi.GetText());
        WarnOnce(String(message));
        LastSwitchReason = String(TEXT("no-result:hold"));
        // reuse the reason just stored (same literal, no second build).
        RecordTrace(0.0f, yawRate, tags, trajectory, settings,
            continuityClip, sinceSwitch, retryKind, forced, reseeded, false, false, frameDt, playRate,
            false, result, LastSwitchReason);
        return;
    }

    WinnerSample = result.SampleIndex;
    WinnerClip = result.ClipIndex;
    WinnerClipName = GetClipName(result.ClipIndex);
    WinnerCost = result.Cost;
    SecondBestCost = result.SecondBestCost;
    WinnerPoseCost = result.PoseCost;
    WinnerTrajectoryCost = result.TrajectoryCost;
    WinnerContinuityCost = result.ContinuityCost;
    WinnerLoopCost = result.LoopCost;
    WinnerIsLoop = _database != nullptr && result.ClipIndex >= 0 &&
        _database->GetClipLoop(result.ClipIndex);
    LoopPreferenceApplied = settings.LoopPreference;
    BestLoopCost = result.BestLoopCost;
    BestLoopClip = result.BestLoopClip;
    BestLoopClipName = result.BestLoopClip >= 0 ? GetClipName(result.BestLoopClip) : String::Empty;
    BestOneShotCost = result.BestOneShotCost;
    BestOneShotClip = result.BestOneShotClip;
    BestOneShotClipName = result.BestOneShotClip >= 0 ? GetClipName(result.BestOneShotClip) : String::Empty;
    LoopScoredCount = result.LoopScoredCount;
    OneShotScoredCount = result.OneShotScoredCount;
    LoopTotalCount = result.LoopTotalCount;
    OneShotTotalCount = result.OneShotTotalCount;
    // Play-rate rule: Compute timeError against ExpectedTime for continuity/telemetry trace
    float timeError = 0.0f;
    if (settings.ExpectedTime >= 0.0f && result.ClipIndex == continuityClip)
    {
        float diff = result.SampleTime - settings.ExpectedTime;
        if (WinnerIsLoop)
        {
            const float len = _playback != nullptr ? _playback->GetCurrentClipLength() : 0.0f;
            if (len > 0.0f)
            {
                diff = fmodf(diff, len);
                if (diff < 0.0f)
                    diff += len;
                if (diff > len * 0.5f)
                    diff -= len;
            }
        }
        timeError = diff;
    }
    LastTimeError = timeError;
    RecordTelemetry(yawRate, tags, result,
        Float3(actorWorld.M41, actorWorld.M42, actorWorld.M43), timeError);

    const bool didSwitch = result.ClipIndex != _currentClip;
    // Cost fuse: one absurd-cost winner never overturns steady tracking.
    // Corrupted query poses (~1-in-2000 frame glitches, e.g. cost 600+
    // mid-walk) self-heal next query; switching on them cascades. End
    // recovery always goes through, and with no current clip there is
    // nothing to hold. (no take-off/landing forced edges anymore.)
    const bool fused = didSwitch && !forced && !recovered && _currentClip >= 0 &&
        result.Cost > _tuning.CostFuseThreshold && result.Cost < 1e29f;
    if (fused)
    {
        LastSwitchReason = String(TEXT("held:cost-fuse"));
    }
    else if (didSwitch)
    {
        // Arm the reselect ban on the pose we just left (single slot):
        // it cannot re-win for ReselectHistoryDuration. Captured before
        // the switch from live playback state.
        const int32 leftClip = _currentClip;
        const float leftTime = _playback != nullptr ? _playback->CurrentTime : -1.0f;
        _currentClip = result.ClipIndex;
        _timeSinceSwitch = 0.0f;
        SwitchCount++;
        if (leftClip >= 0 && leftTime >= 0.0f)
        {
            _reselectClip = leftClip;
            _reselectSampleTime = leftTime;
            _reselectStartTime = Time::GetGameTime();
        }
        else
        {
            _reselectClip = -1;
        }
        _playback->Play(result.ClipIndex, result.SampleTime, _playback->IsAtEnd);
        if (recovered)
            LastSwitchReason = String(TEXT("end-recovery:looponly-retry"));
        else
            LastSwitchReason = String(TEXT("switched:new-clip"));
    }
    else
    {
        // Same-clip winner: continuation is the default (keep playing).
        // Only LOOP clips may seek a far phase: a finished one-shot must
        // never self-restart via seek:end-recovery, and a mid-play one-shot
        // must never jump phases via seek:same-clip-far (the search guard
        // already bars far same-clip one-shot wins; this is the playback
        // backstop for stale winners). A finished one-shot holds its end
        // pose here; the next tick's same-scope search (or its loop-only
        // end-recovery retry) selects the exit clip. Explicit ForcePlayClip
        // stays the only seek path for one-shots.
        const bool winnerIsLoop = _database != nullptr && result.ClipIndex >= 0 &&
            result.ClipIndex < _database->GetClipCount() &&
            _database->GetClipLoop(result.ClipIndex);
        const bool atEnd = _playback->IsAtEnd;
        const float jump = AbsF(result.SampleTime - _playback->CurrentTime);
        if (atEnd && winnerIsLoop)
        {
            _timeSinceSwitch = 0.0f;
            SeekCount++;
            _playback->Play(result.ClipIndex, result.SampleTime, true);
            LastSwitchReason = String(TEXT("seek:end-recovery"));
        }
        else if (atEnd)
        {
            // Finished one-shot: no self-restart. Hold the end pose; the
            // next tick's same-scope search (or its loop-only end-recovery
            // retry) selects the exit clip.
            LastSwitchReason = _timeSinceSwitch < _tuning.MinSwitchTime
                ? String(TEXT("held:continuity-lock")) : String(TEXT("continued"));
        }
        else if (winnerIsLoop && jump > _tuning.SameClipSeekThreshold && _timeSinceSwitch >= minSwitch)
        {
            _timeSinceSwitch = 0.0f;
            SeekCount++;
            _playback->Play(result.ClipIndex, result.SampleTime, true);
            LastSwitchReason = String(TEXT("seek:same-clip-far"));
        }
        else
        {
            LastSwitchReason = _timeSinceSwitch < _tuning.MinSwitchTime
                ? String(TEXT("held:continuity-lock")) : String(TEXT("continued"));
        }
    }
    // Anomaly watchdog (throttled): only a switched anomaly is a visible
    // pop — spikes the continuity lock absorbs are the system working.
    {
        const float now = Time::GetGameTime();
        if (didSwitch && !fused && result.Cost > 0.5f && result.Cost < 1e29f && now - _lastSpikeWarn > 5.0f)
        {
            _lastSpikeWarn = now;
            const StringAnsi clipAnsi(WinnerClipName);
            const StringAnsi tagAnsi(ActiveTag);
            char message[512];
            snprintf(message, sizeof(message),
                "Motion matching anomaly: cost spike %.1f (pose=%.1f traj=%.1f tags=%s clip=%s).",
                (double)result.Cost, (double)result.PoseCost, (double)result.TrajectoryCost,
                tagAnsi.GetText(), clipAnsi.GetText());
            LOG(Warning, "{}", String(message));
        }
    }
    RecordTrace(0.0f, yawRate, tags, trajectory, settings,
        continuityClip, sinceSwitch, retryKind, forced, reseeded, false, false, frameDt, playRate,
        true, result, LastSwitchReason);
}

bool MotionMatchingRuntimePolicy::TryFastPath(float frameDt)
{
    // Tick-order rule: evaluating the continuation is itself a search decision.
    NotifySearch();
    ForcedReason = String::Empty;

    // no intent bypass (onset/fair-fight deleted with the gameplay
    // state). Fresh Game intent arrives as a changed caller filter and is
    // caught by the signature check below on the same tick.
    // Reseed/teleport: the incumbent was measured against the old
    // baseline; the first continuation check after a reset must scan.
    // Also miss while the reseed LoopOnly guard is armed (keepReseedArmed):
    // holding a pre-teleport loop continuation through the poisoned window
    // re-plants the cascade just as a one-shot win would.
    if (_fastPathReseed || (_chooserReseed && _callerPlanarSpeed <= 5.0f))
        return MissFastPath("forced:reseed");
    // MaxValue as threshold would hold ANYTHING, freezing the pre-reset
    // clip forever (observed: whole phases stuck in idle). Force a scan.
    if (_lastScanTotal >= 1e29f)
        return MissFastPath("miss:no-baseline");
    if (_currentClip < 0 || _playback == nullptr || _database == nullptr ||
        !_poseValid || _latestPose.Count() == 0 ||
        _tickTrajectory.Count() == 0)
        return MissFastPath("miss:no-state");
    MotionMatchingDatabase* db = _database;
    if (!db->GetClipLoop(_currentClip))
        return MissFastPath("miss:non-loop");

    // Fast-path signature rule query signature: compare against the signature stored at the
    // last full scan. The signature carries the full candidate-set
    // inputs (allowed/excluded tags, loop filter, candidate band) plus
    // clip/time/database identity. A changed filter invalidates immediately
    // — the core can never hold a continuation from a previous filter after
    // the Game changed it. A changed database identity (rebake/reload) also
    // invalidates the baseline.
    const FastPathSignature sig = CaptureFastPathSignature();
    if (_hasLastScanSignature && sig.DatabaseId != _lastScanSignature.DatabaseId)
        return MissFastPath("miss:db-changed");
    if (_hasLastScanSignature && FastPathSignaturesDiffer(sig, _lastScanSignature))
        return MissFastPath("miss:filter-changed");

    // Incumbent must still be eligible under the CURRENT caller filter:
    // a scope change that keeps the same hash (impossible by construction,
    // but cheap to enforce) or a LoopOnly<->OneShotOnly flip must never
    // hold a clip the new filter forbids. Empty scope never holds.
    // change-only splits via the fast-path-only caches (separate keys
    // from RunQuery's pair, so this check never consumes the scope-change
    // signal); steady-state ticks reuse with no Split per tick. Semantics
    // unchanged: the same arrays Split would return.
    {
        const Array<String>& allowedNow = GetFastAllowed();
        if (allowedNow.Count() == 0)
            return MissFastPath("miss:empty-scope");
        const Array<String>& incumbentTags = ClipTags(_currentClip);
        bool overlaps = false;
        for (const String& t : incumbentTags)
        {
            if (allowedNow.Contains(t))
            {
                overlaps = true;
                break;
            }
        }
        if (!overlaps)
            return MissFastPath("miss:scope-changed");
        if (!_snapshot.ExcludedTagSet.IsEmpty())
        {
            const Array<String>& excludedNow = GetFastExcluded();
            for (const String& t : incumbentTags)
            {
                if (excludedNow.Contains(t))
                    return MissFastPath("miss:excluded");
            }
        }
        // Fast-path only holds loops (gated above): a caller OneShotOnly
        // filter forbids the incumbent by construction.
        if (_callerLoopFilter == (int32)MotionLoopFilter::OneShotOnly)
            return MissFastPath("miss:loopfilter");
    }

    // Finite incumbent lifetime: never hold past the age/frame caps even
    // when the continuation cost still looks good. _fastPathFrames counts
    // query opportunities; the frame cap scales with the query rate so it
    // still means ~MaxFastPathAge under per-frame search (at 10Hz cadence
    // this reduces to the original constant 5).
    if (Time::GetGameTime() - _lastFullScanTime > MaxFastPathAge)
        return MissFastPath("forced:age");
    const int32 frameCap = MaxI(MaxFastPathFrames,
        (int32)ceilf(MaxFastPathAge / MaxF(frameDt, 1e-4f)));
    if (_fastPathFrames > frameCap)
        return MissFastPath("forced:age-frames");
    // A streak of continuation misses means the incumbent keeps looking
    // worse; stop waiting out the caps and re-scan.
    if (_consecutiveContinuationMisses >= 3)
        return MissFastPath("miss:miss-streak");

    EnsureSampleStarts();
    if (!_hasSampleStarts)
        return MissFastPath("miss:no-offsets");

    const Array<int32>& counts = db->GetClipSampleCounts();
    if (counts.Count() == 0 || _currentClip >= counts.Count())
        return MissFastPath("miss:no-counts");
    const int32 count = counts[_currentClip];
    if (count <= 0)
        return MissFastPath("miss:empty-clip");
    const float rate = _tuning.SampleRate;
    if (rate <= 0.0f)
        return MissFastPath("miss:no-rate");
    // Loops only (gated above): wrap time into the cycle using playback's play rate.
    const float currentPlayRate = _playback != nullptr ? _playback->GetPlayRate() : 1.0f;
    const float effectiveDt = frameDt * currentPlayRate;
    const float t = _playback->CurrentTime + effectiveDt;
    const float len = (float)count / rate;
    if (len <= 0.0f)
        return MissFastPath("miss:no-length");
    float wrapped = fmodf(t, len);
    if (wrapped < 0.0f)
        wrapped += len;
    // Clip nearly at the loop seam: hand off with a real scan instead of
    // scoring a sample that is about to wrap.
    if (len - wrapped < effectiveDt * 1.5f)
        return MissFastPath("miss:clip-near-end");
    const int32 sample = _clipSampleStarts[_currentClip] + ((int32)roundf(wrapped * rate) % count);

    const float yawRate = _callerYawRate;
    float fast = 0.0f;
    String sampleError;
    // profiling (record-only): time the continuation SampleCost block.
    const auto fastStart = std::chrono::high_resolution_clock::now();
    const bool sampleOk = MotionMatchingSearch::SampleCost(db, _latestPose, _tickTrajectory, sample,
            _tuning.PoseWeight, _tuning.TrajectoryWeight,
            yawRate, true, _tuning.TurnPenaltyWeight, _tuning.TurnPenaltyYawScale,
            fast, sampleError);
    {
        const auto fastEnd = std::chrono::high_resolution_clock::now();
        const float fastMs = (float)std::chrono::duration<double, std::milli>(fastEnd - fastStart).count();
        LastFastPathMs = fastMs;
        TotalFastPathMs += fastMs;
    }
    if (!sampleOk)
        return MissFastPath("miss:sample-cost");

    const float threshold = _lastScanTotal + _tuning.SwitchMargin + _tuning.ContinuingBias;
    if (!(fast <= threshold))
        return MissFastPath("miss:cost");

    // Hit: hold the incumbent and record its age (telemetry/trace).
    _consecutiveContinuationMisses = 0;
    FastPathHits++;
    LastFastPathAge = Time::GetGameTime() - _lastFullScanTime;
    LastSwitchReason = String(TEXT("fast:continuing-good"));
    // build + record the hit entry only when the trace is enabled.
    // When disabled the hold above is unchanged and no entry Strings are
    // built (steady-state shipping ticks allocate nothing here). The
    // fast-path-only excluded join is used (never touches RunQuery's key).
    if (_trace != nullptr && _trace->Enabled)
    {
        MotionMatchingTraceEntry entry;
        // fast-path hits are first-class trace entries (Kind fastpath)
        // WITH the feature vectors they scored, so replay re-runs the same
        // SampleCost instead of skipping them. Record-only: the hold above
        // is unchanged.
        // no movement diagnostics (speed/des 0, grounded true); the
        // caller filter + turn-prior tuning ride explicitly for replay.
        entry.Kind = String(TEXT("fastpath"));
        entry.Time = Time::GetGameTime();
        entry.State = PolicyState;
        entry.Tags = ActiveTag;
        entry.ExcludedTags = GetFastExcludedJoin();
        entry.Speed = 0.0f;
        entry.YawRate = yawRate;
        entry.Grounded = true;
        entry.Sprinting = false;
        entry.DesiredSpeed = 0.0f;
        entry.PoseWeight = _tuning.PoseWeight;
        entry.TrajectoryWeight = _tuning.TrajectoryWeight;
        entry.ContinuityWeight = _tuning.ContinuityWeight;
        entry.MinSwitchTime = _tuning.MinSwitchTime;
        entry.ContinuingBias = _tuning.ContinuingBias;
        entry.ContinuingWindow = _tuning.ContinuingWindow;
        entry.SwitchMargin = _tuning.SwitchMargin;
        entry.LoopPreference = 0.0f;
        entry.LoopFilter = _callerLoopFilter;
        entry.OneShotBand = _callerCandidateBand;
        entry.MaxCandidates = MaxI(_tuning.FrameSearchBudget, 0);
        entry.QueryFrame = _policyFrame;
        entry.TraceFlags = PackTraceFlags(false, false, false, false,
            _tuning.UseIndexedSearch, _tuning.VerifyIndexedSearch, true);
        entry.TopKCandidates = MaxI(1, _tuning.TopKCandidates);
        entry.CostFuseThreshold = _tuning.CostFuseThreshold;
        entry.TurnPenaltyWeight = _tuning.TurnPenaltyWeight;
        entry.TurnPenaltyYawScale = _tuning.TurnPenaltyYawScale;
        entry.FrameDt = frameDt;
        entry.PlayRate = currentPlayRate;
        entry.DbClipCount = db->GetClipCount();
        entry.DbSampleCount = db->GetSampleCount();
        entry.Valid = true;
        entry.ClipIndex = _currentClip;
        entry.SampleIndex = sample;
        entry.SampleTime = t;
        entry.Cost = fast;
        entry.PoseCost = -1.0f;
        entry.TrajectoryCost = -1.0f;
        entry.SecondBestCost = threshold;
        entry.FastThreshold = threshold;
        entry.Reason = String(TEXT("fast:continuing-good"));
        entry.Chooser = ChooserStatusText;
        entry.QueryMs = 0.0f;
        _trace->Record(entry, _latestPose, _tickTrajectory);
    }
    return true;
}

bool MotionMatchingRuntimePolicy::MissFastPath(const char* reason)
{
    ForcedReason = String(reason);
    _consecutiveContinuationMisses++;
    return false;
}

int32 MotionMatchingRuntimePolicy::Quantize(float value, float step)
{
    if (step <= 0.0f)
        return 0;
    return (int32)roundf(value / step);
}

int32 MotionMatchingRuntimePolicy::HashTagSet(const String& tagSet)
{
    // Lowercased '+'-joined tag-set hash (djb2 over UTF-16 code units folded
    // to ASCII lowercase; empty stays 0). Fast-path-only: collisions only
    // cause an extra full scan, never a wrong hold (the eligibility check in
    // TryFastPath re-validates the incumbent tags explicitly).
    if (tagSet.IsEmpty())
        return 0;
    const Char* text = tagSet.GetText();
    const int32 length = tagSet.Length();
    uint32 hash = 5381u;
    for (int32 i = 0; i < length; i++)
    {
        Char c = text[i];
        if (c >= 'A' && c <= 'Z')
            c = (Char)(c + ('a' - 'A'));
        hash = ((hash << 5) + hash) + (uint32)c;
    }
    return (int32)(hash & 0x7FFFFFFFu);
}

MotionMatchingRuntimePolicy::FastPathSignature MotionMatchingRuntimePolicy::CaptureFastPathSignature() const
{
    // the signature carries every input that can change the valid
    // candidate set: allowed/excluded tags (hashed), loop filter, candidate
    // band, current clip/time and database identity. Movement-derived bits
    // (speed/intent/yaw/contacts) died with the gameplay state — trajectory
    // drift is caught by the continuation cost check itself.
    int32 dbId = 0;
    if (_database != nullptr)
        dbId = (int32)(((uintptr_t)_database) & 0xFFFFFFFF) ^ (_database->GetClipCount() * 397);
    FastPathSignature sig;
    sig.AllowedHash = HashTagSet(_snapshot.AllowedTagSet);
    sig.ExcludedHash = HashTagSet(_snapshot.ExcludedTagSet);
    sig.LoopFilter = _callerLoopFilter;
    sig.CandidateBand = _callerCandidateBand;
    sig.Clip = _currentClip;
    sig.ClipTime = Quantize(_playback != nullptr ? _playback->CurrentTime : 0.0f, 0.5f);
    sig.DatabaseId = dbId;
    return sig;
}

bool MotionMatchingRuntimePolicy::FastPathSignaturesDiffer(const FastPathSignature& a, const FastPathSignature& b)
{
    return a.AllowedHash != b.AllowedHash || a.ExcludedHash != b.ExcludedHash ||
        a.LoopFilter != b.LoopFilter || a.CandidateBand != b.CandidateBand ||
        a.Clip != b.Clip || a.ClipTime != b.ClipTime || a.DatabaseId != b.DatabaseId;
}

// shared change-only scope caches: Split/Join run only when the joined
// snapshot string differs from the cached key (exact match). Steady-state
// (same scope every tick) returns the stored arrays/strings with no
// allocation; a scope transition rebuilds once. Content-identical to
// SplitTagSet/JoinTags of the same input (same helpers, same input).
const Array<String>& MotionMatchingRuntimePolicy::GetCachedAllowed()
{
    if (_snapshot.AllowedTagSet != _queryTagsKey)
    {
        _queryTagsCache = MotionMatchingDatabase::SplitTagSet(_snapshot.AllowedTagSet);
        _queryTagsKey = _snapshot.AllowedTagSet;
    }
    return _queryTagsCache;
}

const Array<String>& MotionMatchingRuntimePolicy::GetCachedExcluded()
{
    if (_snapshot.ExcludedTagSet != _queryExcludedKey)
    {
        _queryExcludedCache = MotionMatchingDatabase::SplitTagSet(_snapshot.ExcludedTagSet);
        _queryExcludedKey = _snapshot.ExcludedTagSet;
        _queryExcludedJoin = MotionMatchingDatabase::JoinTags(_queryExcludedCache);
    }
    return _queryExcludedCache;
}

const String& MotionMatchingRuntimePolicy::GetCachedExcludedJoin()
{
    GetCachedExcluded();
    return _queryExcludedJoin;
}

// fast-path-only twins: identical change-only rule on separate keys so
// the incumbent check never consumes RunQuery's scope-change signal (see
// the header member comment). Content-identical to Split/Join.
const Array<String>& MotionMatchingRuntimePolicy::GetFastAllowed()
{
    if (_snapshot.AllowedTagSet != _fastAllowedKey)
    {
        _fastAllowedCache = MotionMatchingDatabase::SplitTagSet(_snapshot.AllowedTagSet);
        _fastAllowedKey = _snapshot.AllowedTagSet;
    }
    return _fastAllowedCache;
}

const Array<String>& MotionMatchingRuntimePolicy::GetFastExcluded()
{
    if (_snapshot.ExcludedTagSet != _fastExcludedKey)
    {
        _fastExcludedCache = MotionMatchingDatabase::SplitTagSet(_snapshot.ExcludedTagSet);
        _fastExcludedKey = _snapshot.ExcludedTagSet;
        _fastExcludedJoin = MotionMatchingDatabase::JoinTags(_fastExcludedCache);
    }
    return _fastExcludedCache;
}

const String& MotionMatchingRuntimePolicy::GetFastExcludedJoin()
{
    GetFastExcluded();
    return _fastExcludedJoin;
}

void MotionMatchingRuntimePolicy::NotifySearch()
{
    if (_playback != nullptr)
        _playback->NotifySearchExecuted(_nextSearchFrameId);
}

void MotionMatchingRuntimePolicy::EnsureSampleStarts()
{
    if (_hasSampleStarts)
        return;
    MotionMatchingDatabase* db = _database;
    if (db == nullptr)
        return;
    const int32 n = db->GetClipCount();
    const Array<int32>& counts = db->GetClipSampleCounts();
    if (n <= 0 || counts.Count() != n)
        return;
    _clipSampleStarts.Resize(n + 1);
    _clipSampleStarts[0] = 0;
    for (int32 i = 0; i < n; i++)
    {
        if (counts[i] < 0)
            return;
        _clipSampleStarts[i + 1] = _clipSampleStarts[i] + counts[i];
    }
    _hasSampleStarts = true;
}

void MotionMatchingRuntimePolicy::RecordTrace(float speed, float yawRate, const Array<String>& tags,
    const Array<float>& trajectory, const MotionMatchingSearchSettings& settings,
    int32 continuityClip, float sinceSwitch, int32 retryKind, bool forcedEdge, bool reseeded,
    bool fairFight, bool oneshotWindow, float frameDt, float playRate,
    bool valid, const MotionMatchingResult& result, const String& reason)
{
    // no record work (and no entry String builds) when the trace is
    // disabled. The hold/search decision above is unchanged; only the
    // diagnostics entry is skipped. _lastEntry then keeps the last
    // enabled-tick value (test sessions always enable the trace via
    // TraceOverride; shipping play never reads the last entry per tick).
    if (_trace == nullptr || !_trace->Enabled)
        return;
    if (_latestPose.Count() == 0 || trajectory.Count() == 0)
        return;
    MotionMatchingTraceEntry entry;
    entry.Kind = String(TEXT("search"));
    entry.Time = Time::GetGameTime();
    entry.State = PolicyState;
    for (int32 i = 0; i < tags.Count(); i++)
    {
        if (i > 0)
            entry.Tags += String(TEXT("+"));
        entry.Tags += tags[i];
    }
    entry.Speed = speed;
    entry.YawRate = yawRate;
    // the core never sees grounded/sprinting/desired velocity (Game
    // owns them). Keys stay for contract stability; values are neutral.
    entry.Grounded = true;
    entry.Sprinting = false;
    entry.DesiredSpeed = 0.0f;
    entry.PoseWeight = settings.PoseWeight;
    entry.TrajectoryWeight = settings.TrajectoryWeight;
    entry.ContinuityWeight = settings.ContinuityWeight;
    entry.MinSwitchTime = settings.MinSwitchTime;
    entry.CurrentTime = settings.CurrentTime;
    entry.ExclusionWindow = settings.SameClipExclusionWindow;
    entry.ExpectedTime = settings.ExpectedTime;
    entry.ContinuingBias = settings.ContinuingBias;
    entry.ContinuingWindow = settings.ContinuingWindow;
    entry.SwitchMargin = settings.SwitchMargin;
    entry.LoopPreference = settings.LoopPreference;
    entry.LoopFilter = (int32)settings.LoopFilter;
    entry.ReselectClip = settings.ReselectClip;
    entry.ReselectSampleTime = settings.ReselectSampleTime;
    entry.ReselectBanWindow = settings.ReselectBanWindow;
    entry.TwinBanWindow = settings.TwinBanWindow;
    entry.LoopTailExclusion = settings.LoopTailExclusion;
    entry.OneShotBand = settings.OneShotBand;
    entry.MaxCandidates = settings.MaxCandidates;
    entry.QueryFrame = settings.QueryFrame;
    entry.TraceFlags = PackTraceFlags(forcedEdge, reseeded, fairFight, oneshotWindow,
        settings.UseIndexedSearch, settings.VerifyIndexedSearch, settings.ExcludeNonLoopTail);
    entry.TopKCandidates = settings.TopKCandidates;
    // settings.ExcludeTags is always the shared change-only cache (all
    // three RunQuery call sites assign GetCachedExcluded()), so the joined
    // form is the cached join — no Join per query. Content-identical.
    entry.ExcludedTags = GetCachedExcludedJoin();
    entry.RetryKind = retryKind;
    entry.Recovered = retryKind == 1;
    entry.CostFuseThreshold = _tuning.CostFuseThreshold;
    entry.TurnPenaltyWeight = settings.TurnPenaltyWeight;
    entry.TurnPenaltyYawScale = settings.TurnPenaltyYawScale;
    entry.FrameDt = frameDt;
    entry.PlayRate = playRate;
    entry.DbClipCount = _database != nullptr ? _database->GetClipCount() : 0;
    entry.DbSampleCount = _database != nullptr ? _database->GetSampleCount() : 0;
    entry.ContinuityClip = continuityClip;
    entry.SinceSwitch = sinceSwitch;
    entry.Valid = valid;
    entry.ClipIndex = valid ? result.ClipIndex : -1;
    entry.SampleIndex = valid ? result.SampleIndex : -1;
    entry.SampleTime = valid ? result.SampleTime : 0.0f;
    entry.Cost = valid ? result.Cost : MAX_float;
    entry.PoseCost = valid ? result.PoseCost : MAX_float;
    entry.TrajectoryCost = valid ? result.TrajectoryCost : MAX_float;
    entry.ContinuityCost = valid ? result.ContinuityCost : 0.0f;
    entry.LoopCost = valid ? result.LoopCost : 0.0f;
    entry.SecondBestCost = valid ? result.SecondBestCost : MAX_float;
    entry.Reason = reason;
    entry.Chooser = ChooserStatusText;
    entry.QueryMs = LastQueryMs;
    _trace->Record(entry, _latestPose, trajectory);
    _lastEntry = entry;
}









float MotionMatchingRuntimePolicy::Wrap180(float a)
{
    while (a > 180.0f)
        a -= 360.0f;
    while (a < -180.0f)
        a += 360.0f;
    return a;
}













// Unconstrained fallback chooser deleted (was here).

void MotionMatchingRuntimePolicy::ForcePlayClip(int32 clipIndex, float startTime, const String& reason)
{
    if (_database == nullptr || _playback == nullptr || _trace == nullptr)
        return;
    if (clipIndex < 0 || clipIndex >= _database->GetClipCount())
        return;
    // Single authoritative hatch: playback advances the clip (arming
    // inertialization via Play), then every policy mirror follows in the
    // same call so playback and policy can never disagree for a frame.
    const int32 leftClip = _currentClip;
    _playback->Play(clipIndex, startTime, true);
    _currentClip = clipIndex;
    _timeSinceSwitch = 0.0f;
    // Continuity baseline: the forced clip is the new anchor. Invalidate the
    // fast-path cost baseline so the next continuation check scans instead
    // of holding stale data; clear reseed (force is not a teleport) and the
    // reselect ban (it must not immediately ban the forced clip).
    _lastScanTotal = MAX_float;
    _hasLastScanSignature = false;
    _fastPathReseed = false;
    _fastPathFrames = 0;
    _consecutiveContinuationMisses = 0;
    _reselectClip = -1;
    _reselectStartTime = -99.0f;
    // Winner mirrors follow the forced clip (costs unknown until the next
    // scan; reason carries the hatch label for telemetry/trace).
    WinnerClip = clipIndex;
    WinnerClipName = GetClipName(clipIndex);
    WinnerSample = -1;
    WinnerCost = MAX_float;
    SecondBestCost = MAX_float;
    LastSwitchReason = String(TEXT("force:")) + reason;
    (void)leftClip;
    // Structural command entry (no feature vectors by design): replay checks
    // the clip/time/reason structurally. Search resumes immediately on the
    // next tick with the caller filter (no hold).
    RecordPolicyEvent("command", clipIndex, LastSwitchReason);
    // Stamp the command clip/time on the last entry for the structural check
    // (RecordPolicyEvent stores clip in CommandClip; time follows).
    _lastEntry.CommandClip = clipIndex;
    _lastEntry.CommandTime = startTime;
}


void MotionMatchingRuntimePolicy::RecordTelemetry(float yawRate, const Array<String>& tags,
    const MotionMatchingResult& result, const Float3& actorPos, float timeError)
{
    // bounded enqueue only: no JSON formatting, no file I/O in Tick.
    // Gated by EffectiveTelemetry (caller checks before calling? No — check
    // here so off-ticks are a single branch with zero allocation).
    if (!_tuning.EffectiveTelemetry)
        return;
    // tags rides ActiveTag now (see below); the array param is kept for
    // signature stability (bridge/tools call sites + header decl).
    (void)tags;
    const auto ioStart = std::chrono::high_resolution_clock::now();
    TelemetryRecord rec;
    rec.Time = Time::GetGameTime();
    rec.YawRate = yawRate;
    rec.ClipIndex = result.ClipIndex;
    rec.SampleIndex = result.SampleIndex;
    rec.Cost = result.Cost;
    rec.PoseCost = result.PoseCost;
    rec.TrajectoryCost = result.TrajectoryCost;
    rec.ContinuityCost = result.ContinuityCost;
    rec.LoopCost = result.LoopCost;
    rec.SecondBestCost = result.SecondBestCost;
    rec.QueryMs = LastQueryMs;
    rec.SwitchCount = SwitchCount;
    rec.SeekCount = SeekCount;
    rec.CurrentClip = _currentClip;
    rec.PlaybackTime = _playback != nullptr ? _playback->CurrentTime : -1.0f;
    rec.PlayRate = _playback != nullptr ? _playback->GetPlayRateAfter() : 1.0f;
    rec.EffectiveDt = _playback != nullptr ? _playback->LastEffectiveDt : 0.0f;
    rec.Cycle = _playback != nullptr ? _playback->ClipCycle : 0;
    rec.TimeError = timeError;
    rec.BestLoopClip = result.BestLoopClip;
    rec.BestLoopCost = result.BestLoopCost;
    rec.BestOneShotClip = result.BestOneShotClip;
    rec.BestOneShotCost = result.BestOneShotCost;
    rec.LoopScored = result.LoopScoredCount;
    rec.OneShotScored = result.OneShotScoredCount;
    rec.LoopTotal = result.LoopTotalCount;
    rec.OneShotTotal = result.OneShotTotalCount;
    rec.LoopPreference = LoopPreferenceApplied;
    rec.ActorPos = actorPos;
    // single copy of the already-joined scope instead of a per-tag
    // concat loop. Content-identical: the only RecordTelemetry call site
    // passes the allowed-tags array whose Join is ActiveTag (non-empty
    // valid-result path; ActiveTag is rebuilt on every scope change).
    rec.Tags = ActiveTag;
    _telemetryRing[_telemetryHead] = rec;
    _telemetryHead = (_telemetryHead + 1) % TelemetryCapacity;
    if (_telemetryCount < TelemetryCapacity)
        _telemetryCount++;
    else
        _telemetryDropped++;
    {
        const auto ioEnd = std::chrono::high_resolution_clock::now();
        const float ioMs = (float)std::chrono::duration<double, std::milli>(ioEnd - ioStart).count();
        LastTelemetryMs = ioMs;
        TotalTelemetryMs += ioMs;
    }
    // Telemetry must never break gameplay: enqueue above cannot fail
    // (fixed ring, no file I/O). No silent file errors anymore — export
    // failures surface via FlushTelemetry (explicit tooling).
}

int32 MotionMatchingRuntimePolicy::GetTelemetryCount() const
{
    return _telemetryCount;
}

int32 MotionMatchingRuntimePolicy::GetTelemetryDropped() const
{
    return _telemetryDropped;
}

void MotionMatchingRuntimePolicy::ClearTelemetry()
{
    _telemetryHead = 0;
    _telemetryCount = 0;
    _telemetryDropped = 0;
    LastTelemetryMs = 0.0f;
}

bool MotionMatchingRuntimePolicy::FlushTelemetry(String& error)
{
    // explicit tooling flush (the ONLY writer of MMTelemetry.jsonl).
    // Formats the bounded ring as JSONL + writes the file. Never called
    // from Tick/RunQuery/TryFastPath/Initialize — only from the C#
    // controller FlushTelemetry at session end (or bridge/tooling).
    // path fix: CacheFolder already IS the Cache dir (project/Cache),
    // so the file is CacheFolder/MMTelemetry.jsonl (single). The old
    // per-tick writer used CacheFolder/Cache/... (double Cache bug).
    error = String::Empty;
    if (_telemetryCount == 0)
    {
        error = String(TEXT("Telemetry flush skipped: no records (telemetry was off or no queries ran)."));
        return false;
    }
    if (_cacheFolder.IsEmpty())
    {
        error = String(TEXT("Telemetry flush stopped: cache folder is not set."));
        return false;
    }
    const auto flushStart = std::chrono::high_resolution_clock::now();
    StringBuilder sb;
    char line[2048];
    char tagBuf[128];
    for (int32 n = 0; n < _telemetryCount; n++)
    {
        const TelemetryRecord& r = _telemetryRing[(_telemetryHead - _telemetryCount + n + TelemetryCapacity * 2) % TelemetryCapacity];
        tagBuf[0] = '\0';
        if (!r.Tags.IsEmpty())
        {
            const StringAnsi tagsAnsi(r.Tags);
            strncpy(tagBuf, tagsAnsi.GetText(), sizeof(tagBuf) - 1);
            tagBuf[sizeof(tagBuf) - 1] = '\0';
        }
        const String clipName = GetClipName(r.ClipIndex);
        const String bestLoopName = GetClipName(r.BestLoopClip);
        const String bestOneName = GetClipName(r.BestOneShotClip);
        const StringAnsi clipAnsi(clipName);
        const StringAnsi loopAnsi(bestLoopName);
        const StringAnsi oneAnsi(bestOneName);
        char fCost[32], fPose[32], fTraj[32], fCont[32], fLoop[32], fSecond[32];
        char fBl[32], fBo[32], fQms[32], fPt[32], fRate[32], fDt[32], fErr[32];
        char fYaw[32], fT[32], fPx[32], fPy[32], fPz[32], fPref[32];
        FormatFloatText(r.Cost, fCost, sizeof(fCost));
        FormatFloatText(r.PoseCost, fPose, sizeof(fPose));
        FormatFloatText(r.TrajectoryCost, fTraj, sizeof(fTraj));
        FormatFloatText(r.ContinuityCost, fCont, sizeof(fCont));
        FormatFloatText(r.LoopCost, fLoop, sizeof(fLoop));
        FormatFloatText(r.SecondBestCost, fSecond, sizeof(fSecond));
        FormatFloatText(r.BestLoopCost, fBl, sizeof(fBl));
        FormatFloatText(r.BestOneShotCost, fBo, sizeof(fBo));
        FormatFloatText(r.QueryMs, fQms, sizeof(fQms));
        FormatFloatText(r.PlaybackTime, fPt, sizeof(fPt));
        FormatFloatText(r.PlayRate, fRate, sizeof(fRate));
        FormatFloatText(r.EffectiveDt, fDt, sizeof(fDt));
        FormatFloatText(r.TimeError, fErr, sizeof(fErr));
        FormatFloatText(r.YawRate, fYaw, sizeof(fYaw));
        FormatFloatText(r.Time, fT, sizeof(fT));
        FormatFloatText(r.ActorPos.X, fPx, sizeof(fPx));
        FormatFloatText(r.ActorPos.Y, fPy, sizeof(fPy));
        FormatFloatText(r.ActorPos.Z, fPz, sizeof(fPz));
        FormatFloatText(r.LoopPreference, fPref, sizeof(fPref));
        snprintf(line, sizeof(line),
            "{\"t\":%s,\"state\":\"Locomotion\",\"tags\":\"%s\",\"speed\":0.0,\"yaw\":%s,\"grounded\":1,"
            "\"clip\":%d,\"name\":\"%s\",\"sample\":%d,\"cost\":%s,\"pose\":%s,\"traj\":%s,\"cont\":%s,\"loop\":%s,"
            "\"second\":%s,\"bestLoop\":\"%s:%s\",\"bestOne\":\"%s:%s\",\"loopN\":%d,\"oneN\":%d,"
            "\"loopT\":%d,\"oneT\":%d,"
            "\"pref\":%s,\"switches\":%d,\"seeks\":%d,\"cur\":%d,\"ptime\":%s,\"atEnd\":0,"
            "\"px\":%s,\"py\":%s,\"pz\":%s,\"des\":0.0,"
            "\"rateB\":1.000,\"rateA\":%s,\"effDt\":%s,\"cycle\":%d,\"tErr\":%s}\n",
            fT, tagBuf, fYaw,
            r.ClipIndex, clipAnsi.GetText(), r.SampleIndex, fCost, fPose, fTraj, fCont, fLoop,
            fSecond, loopAnsi.GetText(), fBl, oneAnsi.GetText(), fBo,
            r.LoopScored, r.OneShotScored,
            r.LoopTotal, r.OneShotTotal,
            fPref, r.SwitchCount, r.SeekCount, r.CurrentClip, fPt,
            fPx, fPy, fPz,
            fRate, fDt, r.Cycle, fErr);
        sb.Append(line);
    }
    String text;
    sb.ToString(text);
    const StringAnsi folderAnsi(_cacheFolder);
    char path[1024];
    snprintf(path, sizeof(path), "%s/MMTelemetry.jsonl", folderAnsi.GetText());
    const StringAnsi bodyAnsi(text);
    FILE* file = fopen(path, "wb");
    if (file == nullptr)
    {
        error = String(TEXT("Telemetry flush stopped: cannot write MMTelemetry.jsonl."));
        LOG(Warning, "Motion matching telemetry flush failed: cannot write MMTelemetry.jsonl ({} records, {} dropped).", _telemetryCount, _telemetryDropped);
        return false;
    }
    const size_t written = fwrite(bodyAnsi.GetText(), 1, (size_t)bodyAnsi.Length(), file);
    fclose(file);
    if ((int32)written != bodyAnsi.Length())
    {
        error = String(TEXT("Telemetry flush incomplete: file write truncated."));
        LOG(Warning, "Motion matching telemetry flush incomplete: {}/{} bytes ({} records, {} dropped).", (int32)written, bodyAnsi.Length(), _telemetryCount, _telemetryDropped);
        return false;
    }
    {
        const auto flushEnd = std::chrono::high_resolution_clock::now();
        const double flushMs = std::chrono::duration<double, std::milli>(flushEnd - flushStart).count();
        LOG(Info, "Motion matching telemetry flushed: {} records, {} dropped, {:.1f} ms -> MMTelemetry.jsonl.", _telemetryCount, _telemetryDropped, flushMs);
    }
    return true;
}
