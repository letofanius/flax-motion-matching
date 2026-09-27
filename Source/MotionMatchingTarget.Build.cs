using Flax.Build;

public class MotionMatchingTarget : GameProjectTarget
{
    /// <inheritdoc />
    public override void Init()
    {
        base.Init();

        // Reference the modules for game: core runtime only.
        // Locomotion policy lives in the host (Game); the plugin keeps
        // no sample integration module. The editor module is editor-only
        // (see MotionMatchingEditorTarget).
        Modules.Add("MotionMatching");
    }
}
