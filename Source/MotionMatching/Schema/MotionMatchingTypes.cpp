// MotionMatchingTypes.cpp: skeleton profile helpers (MotionMatchingSkeleton::CreateDefault, MotionMatchingSkeleton::TryResolve).
// Ownership: plugin runtime, no game/host references.
// Key invariants: CreateDefault returns an empty profile with no built-in rig preset; TryResolve fails fast with the scene configuration path when a required name is empty or a required node is missing from the model, and never falls back to built-in bone names.
#include "MotionMatchingTypes.h"

SkeletonProfile MotionMatchingSkeleton::CreateDefault()
{
    return SkeletonProfile();
}

bool MotionMatchingSkeleton::TryResolve(
    const SkeletonProfile& profile,
    SkinnedModel* model,
    NodeIndices& result,
    String& error)
{
    result = NodeIndices();
    error = String::Empty;

    if (model == nullptr || model->WaitForLoaded())
    {
        error = TEXT("PlayerModel is not loaded.");
        return false;
    }

    // Fail fast on an unconfigured profile: empty required names mean the
    // host never set the skeleton for its rig (the struct ships empty on
    // purpose, with no mannequin fallback). Name the exact scene path.
    if (profile.Root.IsEmpty() || profile.Pelvis.IsEmpty() ||
        profile.LeftFoot.IsEmpty() || profile.RightFoot.IsEmpty())
    {
        error = TEXT("SkeletonProfile is not configured for this rig: "
            "Root, Pelvis, LeftFoot and RightFoot names are empty. "
            "Configure them for your rig (scene → MotionMatchingController → Schema → Skeleton).");
        return false;
    }

    result.Root = model->FindNode(profile.Root);
    result.Pelvis = model->FindNode(profile.Pelvis);
    result.Spine = model->FindNode(profile.Spine);
    result.Head = model->FindNode(profile.Head);
    result.LeftFoot = model->FindNode(profile.LeftFoot);
    result.RightFoot = model->FindNode(profile.RightFoot);
    result.LeftHand = model->FindNode(profile.LeftHand);
    result.RightHand = model->FindNode(profile.RightHand);

    if (!result.IsValid())
    {
        String missing;
        auto append = [&missing](const String& name)
        {
            if (missing.HasChars())
                missing += TEXT(", ");
            missing += name;
        };
        if (result.Root < 0)
            append(profile.Root);
        if (result.Pelvis < 0)
            append(profile.Pelvis);
        if (result.LeftFoot < 0)
            append(profile.LeftFoot);
        if (result.RightFoot < 0)
            append(profile.RightFoot);
        error = String::Format(
            TEXT("Missing required skeleton nodes in model: {}. "
                "Configure the SkeletonProfile for your rig "
                "(scene → MotionMatchingController → Schema → Skeleton)."),
            missing);
        return false;
    }

    return true;
}
