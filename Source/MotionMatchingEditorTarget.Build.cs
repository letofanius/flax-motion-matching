using Flax.Build;

public class MotionMatchingEditorTarget : GameProjectEditorTarget
{
    /// <inheritdoc />
    public override void Init()
    {
        base.Init();

        // Reference the modules for editor: core + editor services.
        // No sample module: locomotion policy lives in the host.
        Modules.Add("MotionMatching");
        Modules.Add("MotionMatchingEditor");
    }
}
