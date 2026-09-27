"""Focused mutation checks for check_contract.py (stdlib only)."""
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_contract", HERE / "check_contract.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def expect_failure(label, rel, mutate):
    original = (gate.ROOT / rel).read_text(encoding="utf-8")
    mutated = mutate(original)
    failures = gate.audit(overrides={rel: mutated})
    if not failures:
        raise AssertionError("mutation was not rejected: " + label)


def main():
    expect_failure(
        "caller loop filter recovery assignment",
        "Source/MotionMatching/Runtime/MotionMatchingRuntimePolicy.cpp",
        lambda text: text.replace(
            "settings.LoopFilter = (reseeded || keepReseedArmed) ? MotionLoopFilter::LoopOnly",
            "settings.LoopFilter = (reseeded || keepReseedArmed) ? MotionLoopFilter::Any",
            1),
    )
    expect_failure(
        "end recovery loop-only retry",
        "Source/MotionMatching/Runtime/MotionMatchingRuntimePolicy.cpp",
        lambda text: text.replace(
            "retry.LoopFilter = MotionLoopFilter::LoopOnly;",
            "retry.LoopFilter = MotionLoopFilter::Any;",
            1),
    )
    expect_failure(
        "legacy asset TypeName",
        "Source/MotionMatching/Database/MotionMatchingDatabase.cpp",
        lambda text: text.replace(
            'MotionMatchingDatabase::TypeName = TEXT("Game.MotionMatchingDatabase")',
            'MotionMatchingDatabase::TypeName = TEXT("MotionMatching.MotionMatchingDatabase")',
            1),
    )
    expect_failure(
        "forbidden game coupling",
        "Source/MotionMatching/MotionMatchingController.cs",
        lambda text: text + "\nPlayerCoordinator hostLeak;\n",
    )
    expect_failure(
        "public documentation link",
        "README.md",
        lambda text: text.replace("docs/LIFECYCLE.md", "docs/REMOVED.md"),
    )
    failures = gate.audit(overrides={"Source/MotionMatchingEditor/Samples/MotionMatchingAutoSetup.cs": "placeholder"})
    if not any("sample" in f.lower() for f in failures):
        raise AssertionError("sample absence policy was not enforced")
    print("contract audit mutation tests: PASS (6/6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
