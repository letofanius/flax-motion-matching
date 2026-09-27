using Flax.Build;
using Flax.Build.NativeCpp;

public class MotionMatching : GameModule
{
    /// <inheritdoc />
    public override void Init()
    {
        base.Init();

        // Native binding is used by the motion-matching runtime
        // (search/query/policy/playback/database/baker). This module must
        // never reference game, player, automation or editor-sample code:
        // it only depends on the engine.
        BuildNativeCode = true;
    }

    /// <inheritdoc />
    public override void Setup(BuildOptions options)
    {
        base.Setup(options);

        options.ScriptingAPI.IgnoreMissingDocumentationWarnings = true;
    }
}
