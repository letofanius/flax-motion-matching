"""Deterministic, offline contract gate for the MotionMatching plugin.

The gate intentionally reads only plugin-local source/docs/project files. It
does not require Flax, generated bindings, a database, an editor, or a bridge.
Exit 0 means the documented public surface and the source-level invariants
checked here are present.
"""
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_REFERENCES = {
    "Source/MotionMatching/MotionMatchingController.cs": (
        "class MotionMatchingController", "public void Initialize()",
        "public void Shutdown()", "public void ResetPlayback()",
        "public void Update(float deltaTime, Transform actorWorld, MotionMatchingFrameInput input)",
    ),
    "Source/MotionMatching/MotionMatchingInputContract.cs": (
        "interface IMotionMatchingInputProvider", "struct MotionMatchingInputContext",
        "struct MotionMatchingFrameInput", "public int LoopFilter;",
        "public int CandidateBand;", "public float QueryYawRate;",
    ),
    "Source/MotionMatching/Runtime/MotionMatchingRuntimePolicy.h": (
        "API_FUNCTION() void Initialize(", "API_FUNCTION() void Shutdown()",
        "API_FUNCTION() void ResetOnTeleport()", "API_FUNCTION() void Tick(",
        "API_FUNCTION() void ForcePlayClip(",
    ),
    "Source/MotionMatching/Runtime/MotionMatchingPlayback.h": (
        "API_FUNCTION() void Setup(", "API_FUNCTION() void Stop()",
        "API_FUNCTION() void Play(", "API_FUNCTION() Array<Matrix> GetSearchLocals() const",
    ),
    "Source/MotionMatching/Schema/PoseSearchSchema.h": (
        "struct MOTIONMATCHING_API PoseSearchSchema", "API_FIELD() SkeletonProfile Skeleton;",
        "TrajectorySampleTimes", "bool TryValidate(String& error) const",
    ),
    "Source/MotionMatching/Database/MotionMatchingDatabase.h": (
        "class MOTIONMATCHING_API MotionMatchingDatabase", "GetClipTags",
        "GetClipLoop", "ValidateSavedDatabase",
    ),
    "Source/MotionMatchingEditor/MotionMatchingEditorService.cs": (
        "class ClipRegistryImportOptions", "ExportClipRegistry",
        "ImportClipRegistry", "BakeDatabase", "GenerateClipAudit",
    ),
}

DOC_FILES = (
    "docs/INTEGRATION.md", "docs/PLUGIN_CONTRACT.md", "docs/LIFECYCLE.md",
    "docs/AUTHORING.md", "docs/MIGRATION.md",
)


def _text(root, rel, overrides):
    if rel in overrides:
        return overrides[rel]
    return (root / rel).read_text(encoding="utf-8")


def _code_without_comments_or_strings(text):
    # Enough for deterministic boundary scanning of C#/C++ identifiers; this
    # is deliberately not a compiler/parser.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"(?s)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')", "", text)
    return text


def audit(root=ROOT, overrides=None):
    root = Path(root)
    overrides = overrides or {}
    failures = []

    for rel, markers in PUBLIC_REFERENCES.items():
        try:
            text = _text(root, rel, overrides)
        except OSError as ex:
            failures.append("missing public source %s: %s" % (rel, ex))
            continue
        for marker in markers:
            if marker not in text:
                failures.append("public reference missing: %s :: %s" % (rel, marker))

    for rel in DOC_FILES:
        try:
            _text(root, rel, overrides)
        except OSError as ex:
            failures.append("missing contract document %s: %s" % (rel, ex))

    readme = _text(root, "README.md", overrides)
    for rel in DOC_FILES:
        if "(%s)" % rel not in readme:
            failures.append("README does not link %s" % rel)
    for stale in ("Tools/mm_gates.py", "docs/DONE.md", "docs/PLUGIN_CONTRACT.md#old"):
        if stale in readme:
            failures.append("stale README reference: %s" % stale)

    # Filter enum and runtime semantics.
    query = _text(root, "Source/MotionMatching/Search/MotionMatchingQuery.h", overrides)
    search = _text(root, "Source/MotionMatching/Search/MotionMatchingSearch.cpp", overrides)
    policy = _text(root, "Source/MotionMatching/Runtime/MotionMatchingRuntimePolicy.cpp", overrides)
    input_contract = _text(root, "Source/MotionMatching/MotionMatchingInputContract.cs", overrides)
    required_filter = ("Any = 0", "LoopOnly = 1", "OneShotOnly = 2")
    for marker in required_filter:
        if marker not in query and marker not in input_contract:
            failures.append("filter enum marker missing: %s" % marker)
    for marker in (
        "if (allowedTags.IsEmpty())\n            return true;",
        "if (excludeTags.IsEmpty())\n            return false;",
        "TagAllowed(allowedTags, clipTags) && !TagExcluded(excludeTags, clipTags)",
    ):
        if marker not in search:
            failures.append("filter implementation marker missing: %s" % marker.replace("\n", " "))
    for marker in (
        'ChooserStatusText = String(TEXT("ext:empty (no-match)"))',
        'LastSwitchReason = String(TEXT("no-result:empty-scope"))',
        "settings.LoopFilter = (reseeded || keepReseedArmed) ? MotionLoopFilter::LoopOnly",
        "retry.LoopFilter = MotionLoopFilter::LoopOnly",
        "if (!result.IsValid && _playback != nullptr && _playback->IsAtEnd)",
        "retry.ReselectClip = -1",
    ):
        if marker not in policy:
            failures.append("runtime recovery marker missing: %s" % marker)
    for marker in ("same tag", "LoopOnly", "OneShotOnly", "no-match"):
        if marker not in _text(root, "docs/PLUGIN_CONTRACT.md", overrides):
            failures.append("contract docs omit recovery/filter term: %s" % marker)

    # Asset identity is intentionally a raw-source check: the string must stay
    # available to the factory even though it is not a scripting namespace.
    database_cpp = _text(root, "Source/MotionMatching/Database/MotionMatchingDatabase.cpp", overrides)
    if 'MotionMatchingDatabase::TypeName = TEXT("Game.MotionMatchingDatabase")' not in database_cpp:
        failures.append("legacy database TypeName policy changed")
    migration = _text(root, "docs/MIGRATION.md", overrides)
    for marker in ("Game.MotionMatchingDatabase", "frozen", "Do not rewrite"):
        if marker not in migration:
            failures.append("migration docs omit TypeName policy: %s" % marker)

    # Coupling boundary: comments and literals do not count as code references.
    source_root = root / "Source"
    forbidden = re.compile(r"\b(?:PlayerState|PlayerCoordinator|MotionMatchingPlayer|MotionMatchingPlayerProvider)\b|\bSource/Game/\b")
    sample_rel = "MotionMatchingEditor/Samples/"
    for path in source_root.rglob("*"):
        if path.suffix not in (".cs", ".h", ".cpp"):
            continue
        rel = path.relative_to(source_root).as_posix()
        if rel.startswith(sample_rel):
            continue
        key = "Source/" + rel
        try:
            text = _text(root, key, overrides)
        except OSError:
            continue
        match = forbidden.search(_code_without_comments_or_strings(text))
        if match:
            failures.append("forbidden host coupling in %s: %s" % (key, match.group(0)))

    plugin_project = _text(root, "MotionMatching.flaxproj", overrides)
    if "$(EnginePath)/Flax.flaxproj" not in plugin_project:
        failures.append("plugin project lost its engine-only reference")
    if "$(ProjectPath)" in plugin_project or "Source/Game" in plugin_project:
        failures.append("plugin project contains host-project coupling")

    runtime_build = _text(root, "Source/MotionMatching/MotionMatching.Build.cs", overrides)
    editor_build = _text(root, "Source/MotionMatchingEditor/MotionMatchingEditor.Build.cs", overrides)
    for marker in ("MotionMatchingEditor", "MotionMatchingLocomotion"):
        if marker in runtime_build:
            failures.append("runtime build depends on optional/host module: %s" % marker)
    if (root / "Source/MotionMatchingLocomotion").exists():
        failures.append("deleted sample module still present: Source/MotionMatchingLocomotion")
    if 'options.PublicDependencies.Add("MotionMatching")' not in editor_build:
        failures.append("editor module does not declare runtime dependency")

    sample_rel_path = "Source/MotionMatchingEditor/Samples/MotionMatchingAutoSetup.cs"
    sample = root / sample_rel_path
    if sample_rel_path in overrides or sample.exists():
        failures.append("sample auto-setup still present: Source/MotionMatchingEditor/Samples/MotionMatchingAutoSetup.cs")
    if any(key == "Source/MotionMatchingEditor/Samples" or key.startswith("Source/MotionMatchingEditor/Samples/") for key in overrides) or (root / "Source/MotionMatchingEditor/Samples").exists():
        failures.append("sample directory still present: Source/MotionMatchingEditor/Samples")
    if (root / "Source/MotionMatchingEditor/MotionMatchingAutoSetup.cs").exists():
        failures.append("sample auto-setup leaked into generic editor source")
    if (root / "Samples").exists():
        failures.append("sample directory still present: Samples")
    for path in (root / "Source").rglob("*AutoSetup*"):
        failures.append("sample auto-setup still present: %s" % path.relative_to(root).as_posix())
        break

    # Verify every relative Markdown link in plugin docs points at a file.
    for path in (root / "docs").glob("*.md"):
        text = _text(root, path.relative_to(root).as_posix(), overrides)
        for target in re.findall(r"\[[^]]+\]\(([^)#]+)", text):
            if "://" in target:
                continue
            if not (path.parent / target).exists():
                failures.append("broken documentation link: %s -> %s" % (path.name, target))
    return failures


def main():
    failures = audit()
    print("plugin contract audit: %s" % ("PASS" if not failures else "FAIL"))
    for failure in failures:
        print("  VIOL " + failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
