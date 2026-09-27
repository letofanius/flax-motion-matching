// Contract test (editor tooling).
//
// Purpose: lock the plugin/host boundary (explicit-empty no-match, scope,
// reset soak, frame swap, gameplay-free input, pose-out, force-hatch,
// pure search, filter-change invalidation).
// Ownership: editor tooling driving the plugin runtime object.
// Tick/lifecycle role: PLAY-mode only; runs Update ticks on an ISOLATED
// controller clone so the live scene controller is never mutated.
// This file must never reference game/player/sample types.

using System.Collections.Generic;
using FlaxEngine;

namespace MotionMatching;

/// <summary>
/// Scripted input provider for the contract test below. Plain counter +
/// canned frame; no host logic. Plain class: the runtime owns no provider
/// slot — the host facade builds frames itself and passes them to Update.
/// </summary>
public class MMContractTestProvider : IMotionMatchingInputProvider
{
    public int BuildCalls;
    public bool ResetCalled;
    public MotionMatchingFrameInput FrameToReturn;

    public MotionMatchingFrameInput BuildInput(MotionMatchingInputContext ctx)
    {
        BuildCalls++;
        return FrameToReturn;
    }

    public void Reset()
    {
        ResetCalled = true;
    }
}

/// <summary>
/// Runtime-contract tests (plugin/host boundary locks): explicit-empty
/// no-match, provider scope respected, reset-then-soak stays usable,
/// per-frame source swap takes effect, input free of gameplay fields
/// (compile-level), pose-out equals the internal search locals, force-hatch
/// effect, pure search with no machine direct-plays, and
/// filter-change-mid-clip fast-path invalidation (+ LoopFilter flip).
/// Driver locks were deleted with the native machines.
///
/// All checks run through the plain runtime object (Update with canned
/// frames + an identity world transform), never through provider slots or
/// hand-built native structs: the generated snapshot/tuning layouts are owned
/// by the native build. Runs in PLAY mode (pose sampling and playback only
/// tick live there).
///
/// Deterministic harness: the runner pauses simulation (pause before this op,
/// resume after), so live host ticks stop. This test additionally runs on an
/// ISOLATED controller instance cloned from the live one (same
/// Database/CharacterBody/Schema/tuning, own native policy): the live
/// controller is never mutated, so test-1 explicit-empty mirrors cannot race
/// with the scene tick.
/// </summary>
public static class MMContractTest
{
    private const float Dt = 1.0f / 60.0f;

    // Field names that must NOT exist on the frame (gameplay lives
    // host-side; the core takes trajectory + scope + caller filter only).
    private static readonly string[] DeadFrameFields = new string[]
    {
        "HasScope",
        "Velocity", "DesiredVelocity", "Acceleration", "AngularVelocity",
        "TargetYaw", "CurrentYaw", "IsGrounded", "IsSprinting",
        "JumpTag", "TurnInPlaceTag", "TurnTravelTag", "RestTag",
        "RestSpeedMax", "JumpWalkSpeedMax", "JumpRunSpeedMax",
        "YawRateThreshold", "TagConfirmRepeats",
        "BackwardThreshold", "LateralThreshold", "LateralZRatio",
        "LayerAngleDeg", "GateBone",
        "EnableJumpDriver", "EnableTurnDrive",
    };

    private static readonly string[] MachineReasonPrefixes = new string[]
    {
        "jump:", "drive:", "held:landing", "held:turn", "replay:turn",
    };

    private static readonly Transform TestWorld = new Transform(Float3.Zero, Quaternion.Identity, Float3.One);

    // Isolated test controller: same assets/tuning as the live scene object,
    // but a fresh native policy (own _policy/playback/trace state). Live
    // mirrors are never written by the checks below. Telemetry/trace overrides
    // stay off: on/off must not change search decisions (telemetry is
    // record-only), and the isolated trace is never exported (only the
    // facade's session export runs).
    private static MotionMatchingController NewIsolated(MotionMatchingController live)
    {
        var iso = new MotionMatchingController
        {
            Schema = live.Schema,
            PlayerModel = live.PlayerModel,
            DatabasePath = live.DatabasePath,
            Database = live.Database,
            CharacterBody = live.CharacterBody,
            CacheFolder = live.CacheFolder,
            TelemetryOverride = false,
            IndexedVerifyAuto = live.IndexedVerifyAuto,
            TraceOverride = false,
            QueryInterval = live.QueryInterval,
            SearchEveryFrame = live.SearchEveryFrame,
            FrameSearchBudget = live.FrameSearchBudget,
            UseIndexedSearch = live.UseIndexedSearch,
            TopKCandidates = live.TopKCandidates,
            VerifyIndexedSearch = live.VerifyIndexedSearch,
            ContinuityWeight = live.ContinuityWeight,
            MinSwitchTime = live.MinSwitchTime,
            ContinuingBias = live.ContinuingBias,
            ContinuingWindow = live.ContinuingWindow,
            SwitchMargin = live.SwitchMargin,
            SameClipExclusion = live.SameClipExclusion,
            SameClipSeekThreshold = live.SameClipSeekThreshold,
            CostFuseThreshold = live.CostFuseThreshold,
            TurnPenaltyWeight = live.TurnPenaltyWeight,
            TurnPenaltyYawScale = live.TurnPenaltyYawScale,
            InertializationDuration = live.InertializationDuration,
            TelemetryLog = false,
        };
        iso.Initialize();
        return iso;
    }

    public static string RunAll(MotionMatchingController controller)
    {
        var lines = new List<string>();
        int passed = 0;
        int total = 0;

        void Check(string name, bool ok, string detail)
        {
            total++;
            if (ok)
                passed++;
            lines.Add((ok ? "PASS " : "FAIL ") + name + " — " + detail);
        }

        if (controller == null)
            return "CONTRACT FAIL: no controller.";
        if (controller.Database == null || !controller.Database.IsBaked)
            return "CONTRACT FAIL: database not baked.";
        if (controller.CharacterBody == null)
            return "CONTRACT FAIL: CharacterBody not assigned.";

        float[] sampleTimes = controller.TrajectorySampleTimes;
        if (sampleTimes == null || sampleTimes.Length != 6)
            return "CONTRACT FAIL: need 6 sample times, got " +
                (sampleTimes == null ? "null" : sampleTimes.Length.ToString()) + ".";

        // Isolated instance: all checks below run here; the live scene
        // controller is never written (see NewIsolated + class comment).
        var test = NewIsolated(controller);

        // 1: explicit-empty scope = no-match, never a silent pool fallback.
        {
            var prov = NewCanned(string.Empty, string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 100);
            bool statusOk = test.ChooserStatus.StartsWith("ext:empty");
            bool reasonOk = test.LastSwitchReason == "no-result:empty-scope";
            Check("explicit-empty", statusOk && reasonOk,
                "chooser='" + test.ChooserStatus + "' reason='" + test.LastSwitchReason + "'");
        }

        // 2: provider scope is respected (single-tag idle at rest).
        {
            var prov = NewCanned("idle", string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 120);
            bool ok = test.ActiveTag == "idle";
            Check("scope-respected", ok,
                "activeTag='" + test.ActiveTag + "' winner='" + test.WinnerClipName + "'");
        }

        // 3: reset leaves the runtime usable (the runtime owns no provider
        // slot, so there is no reset propagation to test here — the host
        // facade resets its provider separately. Reset then soak must still
        // settle on the scoped tag).
        {
            var prov = NewCanned("idle", string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 120);
            bool ok = test.ActiveTag == "idle";
            Check("reset-then-soak", ok,
                "activeTag='" + test.ActiveTag + "' after ResetPlayback");
        }

        // 4: per-frame source swap takes effect on the next update (the
        // host facade feeds frames directly, so swap = different frame
        // contents: idle soak, then empty scope must flip the chooser to
        // ext:empty, then idle must recover).
        {
            var idle = NewCanned("idle", string.Empty);
            var empty = NewCanned(string.Empty, string.Empty);
            test.ResetPlayback();
            TickProvider(test, idle, 60);
            TickProvider(test, empty, 60);
            bool emptyOk = test.ChooserStatus.StartsWith("ext:empty");
            TickProvider(test, idle, 120);
            bool recoverOk = test.ActiveTag == "idle";
            Check("frame-swap", emptyOk && recoverOk,
                "emptyChooser='" + test.ChooserStatus + "' recoveredTag='" + test.ActiveTag + "'");
            test.ResetPlayback();
        }

        // 5: input carries no gameplay fields (compile-level boundary).
        {
            var names = new HashSet<string>();
            foreach (var f in typeof(MotionMatchingFrameInput).GetFields())
                names.Add(f.Name);
            var leaked = new List<string>();
            foreach (string dead in DeadFrameFields)
            {
                if (names.Contains(dead))
                    leaked.Add(dead);
            }
            Check("input-no-gameplay-fields", leaked.Count == 0,
                leaked.Count == 0
                    ? "frame fields=[" + string.Join(",", names) + "]"
                    : "leaked=[" + string.Join(",", leaked) + "]");
        }

        // 6: pose-out equals the internal search locals.
        {
            var prov = NewCanned("idle", string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 30);
            bool ok = false;
            string detail = "TryGetBasePose=false";
            if (test.TryGetBasePose(out Matrix[] locals, out int clip, out float cost))
            {
                var playback = test.Playback;
                Matrix[] internals = playback != null ? playback.GetSearchLocals() : null;
                if (internals != null && locals.Length == internals.Length && locals.Length > 0)
                {
                    ok = true;
                    for (int i = 0; i < locals.Length; i++)
                    {
                        if (!locals[i].Equals(internals[i]))
                        {
                            ok = false;
                            break;
                        }
                    }
                    detail = "bones=" + locals.Length + " clip=" + clip;
                }
                else
                {
                    detail = "lens out=" + locals.Length + " internal=" + (internals == null ? "null" : internals.Length.ToString());
                }
            }
            Check("pose-out-is-search-locals", ok, detail);
        }

        // 7: force-hatch lands the requested clip (logs every call).
        {
            var prov = NewCanned("idle", string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 10);
            var playback = test.Playback;
            int clips = test.Database != null ? test.Database.ClipCount : 0;
            bool ok = false;
            string detail = "no playback/clips";
            if (playback != null && clips > 1)
            {
                int target = (playback.CurrentClip + 1) % clips;
                test.ForcePlayClip(target, 0.0f, "contract:force-hatch");
                ok = playback.CurrentClip == target;
                detail = "forced=" + target + " current=" + playback.CurrentClip;
            }
            Check("force-hatch", ok, detail);
        }

        // 8: pure search — a grounded fixed-scope soak never takes a
        // machine direct-play (no jump/drive/hold/replay reasons; only
        // the search-winner path and the ForcePlayClip hatch may Play).
        {
            var prov = NewCanned("idle", string.Empty);
            test.ResetPlayback();
            TickProvider(test, prov, 90);
            string reason = test.LastSwitchReason ?? string.Empty;
            bool machine = false;
            foreach (string prefix in MachineReasonPrefixes)
            {
                if (reason.StartsWith(prefix))
                {
                    machine = true;
                    break;
                }
            }
            Check("pure-search-no-machine-play", !machine,
                "reason='" + reason + "' chooser='" + test.ChooserStatus + "'");
        }

        // 9: filter-change-mid-clip invalidates the fast-path: soak idle
        // until the continuation holds, then change the scope to walk at
        // constant input. The core must NOT hold the stale idle clip — the
        // chooser must adopt the new filter within the soak window.
        {
            var idle = NewCanned("idle", string.Empty);
            var walk = NewCanned("walk", string.Empty);
            test.ResetPlayback();
            TickProvider(test, idle, 120);
            string before = test.ActiveTag ?? string.Empty;
            TickProvider(test, walk, 120);
            string after = test.ActiveTag ?? string.Empty;
            bool ok = before == "idle" && after == "walk";
            Check("filter-change-invalidates-fastpath", ok,
                "before='" + before + "' after='" + after + "'");
            // Same-scope LoopFilter flip must also take effect (Any -> LoopOnly
            // on the idle scope keeps idle loops; the recorded filter rides
            // the trace for replay).
            var idleLoopOnly = NewCannedFiltered("idle", string.Empty, 1, 0);
            TickProvider(test, idleLoopOnly, 60);
            bool loopOk = (test.ActiveTag ?? string.Empty) == "idle";
            Check("loopfilter-change-takes-effect", loopOk,
                "tag='" + test.ActiveTag + "' chooser='" + test.ChooserStatus + "'");
            test.ResetPlayback();
        }
        // 12: one-shot progression hold — force a long (>1s) non-loop clip at mid-length under its own tag scope; while the clip identity is unchanged, CurrentTime must never decrease and SeekCount must not grow (far self-seeks barred). A clip change at any tick is a legal exit and passes.
        {
            test.ResetPlayback();
            TickProvider(test, NewCanned("idle", string.Empty), 10);
            var playback = test.Playback;
            var db = test.Database;
            bool ok = false;
            string detail = "no playback/database";
            if (playback != null && db != null)
            {
                int target = -1;
                float targetLen = 0f;
                int clips = db.ClipCount;
                for (int c = 0; c < clips; c++)
                {
                    bool isLoop = false;
                    try { isLoop = db.GetClipLoop(c); } catch { continue; }
                    if (isLoop) continue;
                    float len = 0f;
                    try { var lens = db.ClipLengths; if (lens != null && c < lens.Length) len = lens[c]; } catch { len = 0f; }
                    if (len > 1.0f) { target = c; targetLen = len; break; }
                }
                if (target >= 0)
                {
                    string tag = "idle";
                    try { string t = db.GetClipTag(target); if (!string.IsNullOrEmpty(t)) tag = t; } catch { }
                    var prov = NewCanned(tag, string.Empty);
                    test.ForcePlayClip(target, targetLen * 0.5f, "contract:oneshot-progression");
                    int seeksBefore = test.SeekCount;
                    float lastTime = playback.CurrentTime;
                    bool regressed = false;
                    bool changed = false;
                    detail = "held";
                    for (int i = 0; i < 30; i++)
                    {
                        TickProvider(test, prov, 1);
                        if (playback.CurrentClip != target) { changed = true; break; }
                        float now = playback.CurrentTime;
                        if (now < lastTime - 0.0001f) { regressed = true; detail = "time regressed " + lastTime + " -> " + now; break; }
                        lastTime = now;
                    }
                    if (changed) { ok = true; detail = "clip changed (legal exit) target=" + target; }
                    else if (!regressed && test.SeekCount == seeksBefore) { ok = true; detail = "held progression target=" + target + " len=" + targetLen; }
                    else if (!regressed) { detail = "seeks grew " + seeksBefore + " -> " + test.SeekCount; }
                }
                else { detail = "no non-loop clip > 1s (fixture missing)"; }
            }
            Check("oneshot-progression-hold", ok, detail);
        }


        test.ResetPlayback();

        // 10: changing the structural schema binding releases the native
        // session before the next host tick, so no playback/trace handle from
        // the old rig can survive a host rebind.
        {
            PoseSearchSchema original = test.Schema;
            PoseSearchSchema rebound = original;
            rebound.SampleRate = original.SampleRate + 1;
            test.Schema = rebound;
            bool released = !test.IsReady && test.Playback == null && test.Trace == null;
            Check("schema-rebind-releases-session", released,
                "ready=" + test.IsReady + " playback=" + (test.Playback != null)
                + " trace=" + (test.Trace != null));
            test.Schema = original;
            test.Update(Dt, TestWorld, CannedFrame("idle", string.Empty, 0, 0));
        }

        // 11: host cleanup is public and idempotent; borrowed native handles
        // are gone immediately after the first call and remain gone after the
        // second call.
        test.Shutdown();
        test.Shutdown();
        Check("shutdown-idempotent", !test.IsReady && test.Playback == null && test.Trace == null,
            "ready=" + test.IsReady);

        string verdict = passed == total ? "CONTRACT PASS" : "CONTRACT FAIL";
        return verdict + ": " + passed + "/" + total + " checks\n" + string.Join("\n", lines);
    }

    private static MMContractTestProvider NewCanned(string allowed, string excluded)
    {
        return new MMContractTestProvider { FrameToReturn = CannedFrame(allowed, excluded, 0, 0) };
    }

    private static MMContractTestProvider NewCannedFiltered(string allowed, string excluded, int loop, int band)
    {
        return new MMContractTestProvider { FrameToReturn = CannedFrame(allowed, excluded, loop, band) };
    }

    private static void TickProvider(MotionMatchingController controller, MMContractTestProvider prov, int n)
    {
        for (int i = 0; i < n; i++)
        {
            var ctx = new MotionMatchingInputContext
            {
                ActorWorld = TestWorld,
                DeltaTime = Dt,
                FrameId = i,
                DatabaseTags = controller.AnimationTags,
                SampleTimes = controller.TrajectorySampleTimes,
            };
            MotionMatchingFrameInput frame = prov.BuildInput(ctx);
            controller.Update(Dt, TestWorld, frame);
        }
    }

    private static MotionMatchingFrameInput CannedFrame(string allowed, string excluded, int loop, int band)
    {
        // Slim frame: trajectory + scope + caller filter only.
        return new MotionMatchingFrameInput
        {
            Points = new TrajectoryPoint[6],
            AllowedTagSet = allowed ?? string.Empty,
            ExcludedTagSet = excluded ?? string.Empty,
            LoopFilter = loop,
            CandidateBand = band,
            QueryYawRate = 0.0f,
        };
    }
}
