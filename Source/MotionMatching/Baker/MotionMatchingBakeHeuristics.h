// MotionMatchingBakeHeuristics.h: bake-time filename-to-metadata mapping (gait bands, turn-like flags).
// Ownership: plugin runtime, no game/host references.
// Key invariants: BandFromClipName/IsTurnLikePath run once at bake/registry time and the runtime core never matches names — the search only reads the stored band ints (GetClipSpeedBand) and turn-like flags (GetClipTurnLike), so renaming an asset without rebaking changes nothing at runtime. The C# auto-tag keyword set must stay identical to IsTurnLikePath: both feed the same baked turn-like flag from their respective import paths.

#pragma once

#include "Engine/Core/Types/String.h"


namespace MotionMatchingBakeHeuristics
{
    /// <summary>
    /// Stored gait speed band domain (ints in the database's per-clip band
    /// array, compared against the onset query's speed-derived band).
    /// 0 = Any (turns and unknowns stay eligible everywhere).
    /// </summary>
    enum class BakedClipSpeedBand : int32
    {
        Any = 0,
        Walk = 1,
        Run = 2,
        Sprint = 3,
    };

    /// <summary>
    /// Gait speed band of one clip, derived from the asset name AT BAKE TIME
    /// (filename-to-metadata, the sanctioned pattern: runtime never matches
    /// names). Lets onset queries restrict one-shots to the intended gait
    /// (a walk onset must not coin-flip a run start: at the onset instant the
    /// features carry almost no gait information). Loops are never band-cut.
    /// Walk_/Run_/Sprint_ prefix wins; Transition_X_to_Y takes the TARGET
    /// band; turns and anything else stay Any.
    /// </summary>
    inline BakedClipSpeedBand BandFromClipName(const String& clipName)
    {
        // Transition targets decide (Walk_to_Run accelerates into run, ...).
        // Checked before the plain gait tokens below.
        if (clipName.Contains(TEXT("to_Sprint")))
            return BakedClipSpeedBand::Sprint;
        if (clipName.Contains(TEXT("to_Run")))
            return BakedClipSpeedBand::Run;
        if (clipName.Contains(TEXT("to_Walk")))
            return BakedClipSpeedBand::Walk;
        if (clipName.Contains(TEXT("_Sprint_")))
            return BakedClipSpeedBand::Sprint;
        if (clipName.Contains(TEXT("_Run_")))
            return BakedClipSpeedBand::Run;
        if (clipName.Contains(TEXT("_Walk_")))
            return BakedClipSpeedBand::Walk;
        return BakedClipSpeedBand::Any;
    }

    /// <summary>
    /// Turn-like asset name keywords (turn/arc/reface/pivot/spin), evaluated
    /// AT BAKE TIME into the per-clip turn-like flag.
    /// </summary>
    inline bool IsTurnLikePath(const String& path)
    {
        if (path.IsEmpty())
            return false;
        // Note: String::ToLower returns a copy (does not modify in place) —
        // discarding the result silently disables every flag (seen once).
        String padded = (String(TEXT("_")) + path + String(TEXT("_"))).ToLower();
        return padded.Find(TEXT("_turn_")) >= 0 ||
               padded.Find(TEXT("_arc_")) >= 0 ||
               padded.Find(TEXT("_reface_")) >= 0 ||
               padded.Find(TEXT("_pivot_")) >= 0 ||
               padded.Find(TEXT("_spin_")) >= 0;
    }
}
