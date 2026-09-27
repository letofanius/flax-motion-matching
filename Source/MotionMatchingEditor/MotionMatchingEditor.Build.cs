using Flax.Build;
using Flax.Build.NativeCpp;

public class MotionMatchingEditor : GameEditorModule
{
    /// <inheritdoc />
    public override void Setup(BuildOptions options)
    {
        base.Setup(options);

        options.ScriptingAPI.IgnoreMissingDocumentationWarnings = true;

        // Editor services for the motion-matching plugin: baker UI, tuning
        // and registry windows, import presets and contract tests. Depends
        // on the runtime module (public: editor code names runtime types
        // and native linking needs the ordered dependency); the runtime
        // module must never depend back.
        options.PublicDependencies.Add("MotionMatching");
    }
}
