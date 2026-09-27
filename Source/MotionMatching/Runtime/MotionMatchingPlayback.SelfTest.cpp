// MotionMatchingPlayback.SelfTest.cpp: root-yaw delta self-test (MotionMatchingPlayback::RunPlaybackSelfTest).
// Ownership: plugin runtime, no game/host references.
// Key invariants: five cases fail loudly on a missing fixture class (Play-then-tick delta, frame-by-frame delta, cross-clip Play reset, invalid/root-missing 0, loop-wrap seam suppression); the expected deltas mirror the live ConjugateQ(prev)*cur + TwistAbout chain without steering it.
#include "MotionMatchingPlayback.h"

#include "../Database/MotionMatchingDatabase.h"
#include "MotionMatchingSampler.h"
#include "Engine/Content/Assets/Animation.h"
#include "Engine/Content/Assets/SkinnedModel.h"
#include "Engine/Core/Log.h"
#include "Engine/Core/Math/Math.h"
#include "Engine/Level/Actors/AnimatedModel.h"

#include <cmath>

#if USE_EDITOR

// Turn-in-place root-yaw delta self-test, file-local — never touches
// MotionMatchingSearch.SelfTest.cpp. Each case FAILS LOUDLY when its fixture
// class is missing (no probe0, no silent pass on 0 probes):
//  1. current Play + one Tick returns the Play->tick delta,
//  2. the next Tick returns the frame-by-frame delta,
//  3. cross-clip Play resets the first frame,
//  4. invalid/root-missing returns 0,
//  5. loop wrap emits no seam-spanning delta.
// TurnPenalty/GetClipTurnAngle are untouched (net signed stays).

namespace
{
    constexpr float RootYawTol = 0.02f; // ~1.1deg; sampler math is exact, tolerance is float noise only.
    constexpr float RootYawMinMotion = 0.005f; // ~0.3deg: a fixture clip below this cannot prove a delta.

    float AbsF(float value)
    {
        return value < 0.0f ? -value : value;
    }

    int32 FindFirstLoopClip(MotionMatchingDatabase* database)
    {
        for (int32 i = 0; i < database->GetClipCount(); i++)
            if (database->GetClipLoop(i))
                return i;
        return -1;
    }

    int32 FindSecondClip(MotionMatchingDatabase* database, int32 other)
    {
        for (int32 i = 0; i < database->GetClipCount(); i++)
            if (i != other && database->GetClipAnimation(i) != nullptr)
                return i;
        return -1;
    }

    float ClipLength(MotionMatchingDatabase* database, int32 clip)
    {
        const Array<float>& lengths = database->GetClipLengths();
        return lengths.IsValidIndex(clip) ? lengths[clip] : 0.0f;
    }

    // Copy of the live measurement chain (ConjugateQ(prev)*cur + TwistAbout),
    // used ONLY to predict the expected delta for the same sampled pair —
    // the live path is MotionMatchingPlayback::Tick, this never steers it.
    float PredictRootTwist(Animation* clip, SkinnedModel* model, int32 rootIndex, const Float3& upAxis, float fromTime, float toTime)
    {
        Array<Matrix> aLocals;
        Array<Matrix> bLocals;
        String error;
        if (!MotionMatchingSampler::SampleClipLocalPose(clip, model, fromTime, aLocals, error))
            return 0.0f;
        if (!MotionMatchingSampler::SampleClipLocalPose(clip, model, toTime, bLocals, error))
            return 0.0f;
        if (rootIndex < 0 || rootIndex >= aLocals.Count() || rootIndex >= bLocals.Count())
            return 0.0f;
        Float3 aScl, aPos, bScl, bPos;
        Quaternion aRot, bRot;
        aLocals[rootIndex].Decompose(aScl, aRot, aPos);
        bLocals[rootIndex].Decompose(bScl, bRot, bPos);
        const float aLen = aRot.X * aRot.X + aRot.Y * aRot.Y + aRot.Z * aRot.Z + aRot.W * aRot.W;
        const float bLen = bRot.X * bRot.X + bRot.Y * bRot.Y + bRot.Z * bRot.Z + bRot.W * bRot.W;
        if (aLen <= 1e-12f || bLen <= 1e-12f)
            return 0.0f;
        const float aInv = 1.0f / (float)std::sqrt(aLen);
        const float bInv = 1.0f / (float)std::sqrt(bLen);
        const Quaternion aN(aRot.X * aInv, aRot.Y * aInv, aRot.Z * aInv, aRot.W * aInv);
        const Quaternion bN(bRot.X * bInv, bRot.Y * bInv, bRot.Z * bInv, bRot.W * bInv);
        const Quaternion delta = Quaternion(-aN.X, -aN.Y, -aN.Z, aN.W) * bN;
        const float dn = delta.X * delta.X + delta.Y * delta.Y + delta.Z * delta.Z + delta.W * delta.W;
        if (dn <= 1e-12f)
            return 0.0f;
        const float dInv = 1.0f / (float)std::sqrt(dn);
        float w = delta.W * dInv;
        w = w < -1.0f ? -1.0f : (w > 1.0f ? 1.0f : w);
        float ang = 2.0f * (float)std::acos((double)w);
        const float s = (float)std::sqrt(1.0f - w * w > 0.0f ? 1.0f - w * w : 0.0f);
        if (s <= 1e-6f)
            return 0.0f;
        const Float3 v(delta.X * dInv / s, delta.Y * dInv / s, delta.Z * dInv / s);
        const float sign = Float3::Dot(v, upAxis) >= 0.0f ? 1.0f : -1.0f;
        if (ang > 3.14159265358979323846f)
            ang = 2.0f * 3.14159265358979323846f - ang;
        return sign * ang;
    }
}

bool MotionMatchingPlayback::RunPlaybackSelfTest(MotionMatchingDatabase* database, AnimatedModel* body, String& log)
{
    log = String::Empty;
    if (database == nullptr || database->WaitForLoaded())
    {
        log = TEXT("Playback self-test failed: database is not loaded.");
        return false;
    }
    if (!database->IsBaked() || !database->ValidateData())
    {
        log = TEXT("Playback self-test failed: database is not baked or not valid.");
        return false;
    }
    if (body == nullptr)
    {
        log = TEXT("Playback self-test failed: body actor is missing (pass CharacterBody from the facade).");
        return false;
    }
    SkinnedModel* model = body->SkinnedModel.Get();
    if (model == nullptr || model->WaitForLoaded())
    {
        log = TEXT("Playback self-test failed: body has no loaded SkinnedModel.");
        return false;
    }

    const PoseSearchSchema& schema = database->GetSchema();
    Float3 upAxis = Float3::Zero;
    if (!schema.TryGetNormalizedUpAxis(upAxis))
    {
        log = TEXT("Playback self-test failed: baked schema root up axis is degenerate.");
        return false;
    }
    NodeIndices nodes;
    String resolveError;
    if (!MotionMatchingSkeleton::TryResolve(schema.Skeleton, model, nodes, resolveError))
    {
        log = String::Format(TEXT("Playback self-test failed: skeleton not resolved for this rig: {}."), resolveError);
        return false;
    }
    if (nodes.Root < 0)
    {
        log = TEXT("Playback self-test failed: skeleton profile resolved without a root node.");
        return false;
    }

    const int32 clipCount = database->GetClipCount();
    if (clipCount < 2)
    {
        log = String::Format(TEXT("Playback self-test failed: need >= 2 clips, database has {}."), clipCount);
        return false;
    }
    const float dt = schema.SampleRate > 0 ? 1.0f / (float)schema.SampleRate : 1.0f / 30.0f;

    int32 passed = 0;
    int32 tested = 0;
    auto check = [&](bool ok, const String& label, const String& detail) -> bool
    {
        tested++;
        if (!ok)
            log += String::Format(TEXT("{} FAIL: {}.\n"), label, detail);
        else
            passed++;
        return ok;
    };

    MotionMatchingPlayback playback;
    playback.Setup(database, model, schema.Skeleton, schema.RootUpAxis);

    // Case 1+2: Play->tick delta, then frame-by-frame delta on the next tick.
    // Fixture: first clip whose root actually yaws over one frame (fail loudly
    // when no clip in this database rotates its root).
    int32 yawClip = -1;
    float yawStart = 0.0f;
    float yawExpectFirst = 0.0f;
    float yawExpectSecond = 0.0f;
    for (int32 c = 0; c < clipCount && yawClip < 0; c++)
    {
        Animation* clip = database->GetClipAnimation(c);
        if (clip == nullptr)
            continue;
        const float len = ClipLength(database, c);
        if (len <= dt * 3.0f)
            continue;
        for (float t = 0.0f; t + dt * 2.0f <= len; t += dt)
        {
            const float d = PredictRootTwist(clip, model, nodes.Root, upAxis, t, t + dt);
            if (AbsF(d) >= RootYawMinMotion && std::isfinite(d))
            {
                yawClip = c;
                yawStart = t;
                yawExpectFirst = d;
                yawExpectSecond = PredictRootTwist(clip, model, nodes.Root, upAxis, t + dt, t + dt * 2.0f);
                break;
            }
        }
    }
    if (yawClip < 0)
    {
        log += TEXT("Playback self-test failed: no fixture clip with root yaw >= 0.005 rad/frame (need a turn clip with authored root yaw).");
        return false;
    }
    if (!std::isfinite(yawExpectSecond))
        yawExpectSecond = 0.0f;

    playback.SetPlayRate(1.0f);
    playback.Play(yawClip, yawStart, true);
    playback.Tick(dt, body, true);
    {
        const float got = playback.GetRootYawDelta();
        check(AbsF(got - yawExpectFirst) <= RootYawTol,
            TEXT("play-to-tick"),
            String::Format(TEXT("clip {} start {:.4f}: got {:.6f}, want {:.6f}"), yawClip, yawStart, got, yawExpectFirst));
    }
    playback.Tick(dt, body, true);
    {
        const float got = playback.GetRootYawDelta();
        check(AbsF(got - yawExpectSecond) <= RootYawTol,
            TEXT("frame-by-frame"),
            String::Format(TEXT("clip {}: got {:.6f}, want {:.6f}"), yawClip, got, yawExpectSecond));
    }

    // Case 3: cross-clip Play resets the first frame (delta measured from the
    // new Play start, never a blend of the old clip's tail).
    const int32 otherClip = FindSecondClip(database, yawClip);
    if (otherClip < 0)
    {
        log += TEXT("Playback self-test failed: need a second clip for the cross-clip reset case.");
        return false;
    }
    {
        Animation* clip = database->GetClipAnimation(otherClip);
        const float len = ClipLength(database, otherClip);
        if (clip == nullptr || len <= dt * 2.0f)
        {
            log += String::Format(TEXT("Playback self-test failed: cross-clip fixture {} is too short or missing."), otherClip);
            return false;
        }
        const float start = 0.0f;
        const float expect = PredictRootTwist(clip, model, nodes.Root, upAxis, start, start + dt);
        playback.SetPlayRate(1.0f);
        playback.Play(otherClip, start, true);
        playback.Tick(dt, body, true);
        const float got = playback.GetRootYawDelta();
        check(AbsF(got - expect) <= RootYawTol,
            TEXT("cross-clip-reset"),
            String::Format(TEXT("clip {}: got {:.6f}, want {:.6f}"), otherClip, got, expect));
    }

    // Case 4: invalid/root-missing returns 0 (never a stale delta).
    {
        MotionMatchingPlayback badAxis;
        badAxis.Setup(database, model, schema.Skeleton, Float3::Zero);
        badAxis.Play(yawClip, yawStart, true);
        badAxis.Tick(dt, body, true);
        check(badAxis.GetRootYawDelta() == 0.0f,
            TEXT("invalid-axis-zero"),
            String::Format(TEXT("got {:.6f}, want 0"), badAxis.GetRootYawDelta()));
        // Root-missing (empty skeleton profile: no root index): the sample
        // has no root rotation, so the same guard reports 0. One expected
        // skeleton-warning accompanies this Setup (fail-fast resolve path).
        MotionMatchingPlayback noRoot;
        const SkeletonProfile emptyProfile = SkeletonProfile();
        noRoot.Setup(database, model, emptyProfile, schema.RootUpAxis);
        noRoot.Play(yawClip, yawStart, true);
        noRoot.Tick(dt, body, true);
        check(noRoot.GetRootYawDelta() == 0.0f,
            TEXT("root-missing-zero"),
            String::Format(TEXT("got {:.6f}, want 0"), noRoot.GetRootYawDelta()));
    }

    // Case 5: loop wrap emits no seam-spanning delta (consume + reset baseline).
    {
        const int32 loopClip = database->GetClipLoop(yawClip) ? yawClip : FindFirstLoopClip(database);
        if (loopClip < 0)
        {
            log += TEXT("Playback self-test failed: no loop clip fixture for the wrap case.");
            return false;
        }
        const float len = ClipLength(database, loopClip);
        Animation* clip = database->GetClipAnimation(loopClip);
        if (clip == nullptr || len <= dt * 2.0f)
        {
            log += String::Format(TEXT("Playback self-test failed: loop fixture {} is too short or missing."), loopClip);
            return false;
        }
        playback.SetPlayRate(1.0f);
        playback.Play(loopClip, len - dt * 0.5f, true);
        playback.Tick(dt, body, true); // advances across the seam
        const float got = playback.GetRootYawDelta();
        check(got == 0.0f,
            TEXT("wrap-no-seam"),
            String::Format(TEXT("clip {}: got {:.6f}, want 0 (seam-spanning delta forbidden)"), loopClip, got));
        // Post-wrap tick resumes frame-by-frame measurement (baseline warm).
        playback.Tick(dt, body, true);
        check(std::isfinite(playback.GetRootYawDelta()) != 0,
            TEXT("post-wrap-finite"),
            TEXT("post-wrap delta is non-finite"));
    }

    log += String::Format(TEXT("Playback self-test: {}/{} passed (clips={}, yawClip={}, dt={:.5f}).\n"),
        passed, tested, clipCount, yawClip, dt);
    return passed == tested && tested == 7;
}

#endif // USE_EDITOR
