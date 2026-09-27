using FlaxEngine;

namespace MotionMatching;

// Documented removal point for the old controller inspector.
// Purpose: keep an explicit tombstone so the old
// [CustomEditor(typeof(MotionMatchingController))] binding has a named
// resting place instead of a silent disappearance.
// Ownership: editor tooling. The runtime (MotionMatchingController) is a
// plain C# class with zero scene presence, so there is no inspector
// CustomEditor for it anymore; the host facade carries the editor button
// through its own CustomEditor, which opens ManageMotionMatchingWindow on
// the facade's controller object.
// Tick/lifecycle role: none - this file owns no runtime state and runs no
// tick. This file must never reference game/player/sample types.
public static class MotionMatchingBakerEditorTombstone
{
    public static string Note => "P3 removed: use the MotionMatchingPlayer inspector button (game facade).";
}
