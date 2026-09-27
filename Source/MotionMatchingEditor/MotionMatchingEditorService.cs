using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using FlaxEditor;
using FlaxEngine;
using FlaxEngine.Json;

namespace MotionMatching;

/// <summary>
/// One registry entry in the portable clips JSON. Paths are normally
/// Content-relative, but absolute paths are also accepted by the importer.
/// Tags and loop are authored data; optional host callbacks can supply
/// inference for registries that intentionally omit either field.
/// </summary>
public class RegistryClipEntry
{
    public string path;
    public string[] tags;
    public bool? loop;
}

/// <summary>
/// Portable clip registry root object.
/// </summary>
public class ClipRegistryFile
{
    public int version = 1;
    public RegistryClipEntry[] clips = Array.Empty<RegistryClipEntry>();
}

/// <summary>
/// Host-provided behavior for importing a registry. The editor module has no
/// built-in filename, folder, skeleton, or rig convention. A host may
/// opt into its own inference and asset path resolver explicitly.
/// </summary>
public sealed class ClipRegistryImportOptions
{
    public Func<string, string[]> InferTags;
    public Func<string, bool> InferLoop;
    public Func<string, string, string> ResolvePath;
}

/// <summary>
/// Editor operations for <see cref="MotionMatchingController"/>.
///
/// Purpose: registry, bake, validate, self-test, audit, preset, top-N, and
/// trace helpers that edit-mode tooling runs against a controller.
/// Ownership: editor tooling. The host facade owns the controller object and
/// hands it over (host -> plugin direction); this service never names host
/// types.
/// Tick/lifecycle role: edit-mode only, no per-tick work; bake/audit/self-test
/// run on demand and reload the database afterwards. Scene-dirty marking
/// arrives as an optional callback because the controller has no Actor.
/// </summary>
public static class MotionMatchingEditorService
{
    public static void AddClips(MotionMatchingController controller, string[] tags, Animation[] clips, bool loop = true, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (tags == null || tags.Length == 0 || clips == null || clips.Length == 0)
            return;

        if (MotionMatchingDatabase.AddClips(controller.DatabaseFile, tags, clips, loop))
            ReloadDatabase(controller, true, onEdited);
    }

    public static void AddClipFiles(MotionMatchingController controller, string[] tags, string[] paths, bool loop = true, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (tags == null || tags.Length == 0 || paths == null || paths.Length == 0)
            return;

        if (MotionMatchingDatabase.AddClipFiles(
                controller.DatabaseFile, tags, paths, loop, out int addedCount))
        {
            Debug.Log($"Registered {addedCount} motion matching clips for '{string.Join("+", tags)}'.");
            ReloadDatabase(controller, true, onEdited);
        }
    }

    public static void ClearAllAnimationClips(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (MotionMatchingDatabase.ClearClips(controller.DatabaseFile))
            ReloadDatabase(controller, true, onEdited);
    }

    /// <summary>
    /// Neutral default registry location for editor export. Hosts can pass
    /// any path to export/import and are not required to use this location.
    /// </summary>
    public static string DefaultRegistryPath()
    {
        return Path.Combine(Globals.ProjectContentFolder, "MotionMatchingRegistry.json");
    }

    private static string ToContentRelative(string path)
    {
        if (string.IsNullOrEmpty(path))
            return path;
        string full = path.Replace('/', Path.DirectorySeparatorChar);
        string root = (Globals.ProjectContentFolder ?? string.Empty)
            .Replace('/', Path.DirectorySeparatorChar)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!string.IsNullOrEmpty(root) && full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
            full = full.Substring(root.Length);
        return full.TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            .Replace(Path.DirectorySeparatorChar, '/');
    }

    private static string ResolveRegistryPath(string assetPath, string registryPath, ClipRegistryImportOptions options)
    {
        if (string.IsNullOrEmpty(assetPath))
            return assetPath;
        if (options?.ResolvePath != null)
            return options.ResolvePath(assetPath, registryPath);

        string rawPath = assetPath.Replace('/', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(rawPath))
            return rawPath;
        string normalized = rawPath.TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        string contentCandidate = Path.Combine(Globals.ProjectContentFolder, normalized);
        if (File.Exists(contentCandidate))
            return contentCandidate;
        string registryFolder = Path.GetDirectoryName(registryPath);
        if (!string.IsNullOrEmpty(registryFolder))
        {
            string registryCandidate = Path.GetFullPath(Path.Combine(registryFolder, normalized));
            if (File.Exists(registryCandidate))
                return registryCandidate;
        }
        return contentCandidate;
    }

    /// <summary>
    /// Exports the current registry to JSON. Returns a summary or an error.
    /// </summary>
    public static string ExportClipRegistry(MotionMatchingController controller, string path)
    {
        try
        {
            if (controller == null)
                return "Export failed: no controller.";
            var clips = GetRegisteredClips(controller) ?? Array.Empty<RegisteredClip>();
            var entries = new List<RegistryClipEntry>();
            foreach (var clip in clips.OrderBy(c => GetRegisteredClipName(c)))
            {
                string assetPath = clip.Clip != null ? clip.Clip.Path : null;
                if (string.IsNullOrEmpty(assetPath))
                    continue;
                entries.Add(new RegistryClipEntry
                {
                    path = ToContentRelative(assetPath),
                    tags = (clip.Tag ?? string.Empty).Split('+'),
                    loop = clip.Loop,
                });
            }
            var root = new ClipRegistryFile { version = 1, clips = entries.ToArray() };
            File.WriteAllText(path, JsonSerializer.Serialize(root, true));
            return $"Exported {entries.Count} clips to '{path}'.";
        }
        catch (Exception ex)
        {
            return "Export failed: " + ex.Message;
        }
    }

    /// <summary>
    /// Imports a registry JSON, REPLACING the whole registry (clear first —
    /// replace-all has fewer failure modes than merge). Entries missing tags
    /// or loop are skipped/defaulted unless the caller explicitly supplies
    /// inference callbacks in <see cref="ClipRegistryImportOptions"/>.
    /// </summary>
    public static string ImportClipRegistry(MotionMatchingController controller, string path)
    {
        return ImportClipRegistry(controller, path, null);
    }

    /// <summary>
    /// Imports a registry using only generic behavior by default. Host
    /// conventions must be passed explicitly through <paramref name="options"/>.
    /// </summary>
    public static string ImportClipRegistry(MotionMatchingController controller, string path,
        ClipRegistryImportOptions options)
    {
        try
        {
            if (controller == null)
                return "Import failed: no controller.";
            if (!File.Exists(path))
                return $"Import failed: file not found '{path}'.";
            var root = JsonSerializer.Deserialize<ClipRegistryFile>(File.ReadAllText(path));
            if (root == null || root.clips == null || root.clips.Length == 0)
                return $"Import failed: no clips in '{path}'.";
            if (!EnsureDatabaseLoaded(controller) || !controller.Database)
                return "Import failed: database is not loaded.";

            var groups = new Dictionary<string, List<string>>();
            int skipped = 0;
            foreach (var entry in root.clips)
            {
                if (entry == null || string.IsNullOrWhiteSpace(entry.path))
                {
                    skipped++;
                    continue;
                }
                string full = ResolveRegistryPath(entry.path.Trim(), path, options);
                if (!File.Exists(full))
                {
                    skipped++;
                    continue;
                }
                string[] tags = entry.tags;
                if ((tags == null || tags.Length == 0) && options?.InferTags != null)
                    tags = options.InferTags(full);
                var clean = new List<string>();
                foreach (string tag in tags ?? Array.Empty<string>())
                {
                    if (!string.IsNullOrWhiteSpace(tag))
                        clean.Add(tag.Trim().ToLowerInvariant());
                }
                if (clean.Count == 0)
                {
                    skipped++;
                    continue;
                }
                bool loop = entry.loop ?? options?.InferLoop?.Invoke(full) ?? false;
                string key = string.Join("+", clean.OrderBy(t => t, StringComparer.Ordinal)) +
                    "|" + (loop ? "loop" : "once");
                if (!groups.TryGetValue(key, out var list))
                {
                    list = new List<string>();
                    groups[key] = list;
                }
                list.Add(full);
            }

            ClearAllAnimationClips(controller);
            int added = 0;
            foreach (var pair in groups.OrderBy(p => p.Key, StringComparer.Ordinal))
            {
                string[] parts = pair.Key.Split('|');
                string[] files = pair.Value.ToArray();
                // AddClipFiles resolves content-relative or absolute paths.
                AddClipFiles(controller, parts[0].Split('+'), files, parts[1] == "loop");
                added += files.Length;
            }
            return $"Imported {added} clips in {groups.Count} groups from '{path}' (skipped {skipped}). Bake required.";
        }
        catch (Exception ex)
        {
            return "Import failed: " + ex.Message;
        }
    }

    /// <summary>
    /// Explicit audit command: scans the baked database and writes
    /// Cache/MMClipAudit.txt on demand. Runtime init/play never produces
    /// this file — tools must call this first (the clip_audit tooling op or
    /// the tuning window) instead of assuming a play session wrote it.
    /// Returns a one-line summary or error.
    /// </summary>
    public static string GenerateClipAudit(MotionMatchingController controller)
    {
        if (controller == null)
            return "Clip audit failed: no controller.";
        if (!EnsureDatabaseLoaded(controller) || !controller.Database)
            return "Clip audit failed: database is not loaded.";
        if (controller.GenerateClipAudit(out string error))
            return "Clip audit written: Cache/MMClipAudit.txt.";
        return "Clip audit failed: " + error;
    }

    /// <summary>
    /// Bake, then reload + validate, then self-test — one pipeline for the
    /// management window. Results surface via the editor log; the return
    /// value is a one-line summary.
    /// </summary>
    public static string BakeValidateAndTest(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return "No controller.";
        BakeDatabase(controller, onEdited);
        if (!controller.PersistenceValid)
            return "Bake failed; validate and self-test skipped. See log.";
        ValidatePersistence(controller);
        if (!controller.PersistenceValid)
            return "Bake ok; persistence validation failed. See log.";
        RunSearchSelfTest(controller);
        return $"Bake + validate ok: clips={controller.Database.AnimationClipsCount}, samples={controller.Database.SampleCount}. Self-test in log.";
    }

    /// <summary>
    /// One registered source clip for the review UI.
    /// </summary>
    public struct RegisteredClip
    {
        public int SourceIndex;
        public Animation Clip;
        public string Tag;
        public bool Loop;
    }

    public static RegisteredClip[] GetRegisteredClips(MotionMatchingController controller)
    {
        if (controller == null)
            return Array.Empty<RegisteredClip>();
        if (!EnsureDatabaseLoaded(controller) || !controller.Database)
            return Array.Empty<RegisteredClip>();

        int count = controller.Database.GetSourceClipCount();
        var result = new RegisteredClip[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = new RegisteredClip
            {
                SourceIndex = i,
                Clip = controller.Database.GetSourceClipAnimation(i),
                Tag = controller.Database.GetSourceClipTag(i),
                Loop = controller.Database.GetSourceClipLoop(i),
            };
        }
        return result;
    }

    public static string GetRegisteredClipName(RegisteredClip clip)
    {
        string path = clip.Clip?.Path;
        if (!string.IsNullOrEmpty(path))
            return Path.GetFileNameWithoutExtension(path);
        return $"#{clip.SourceIndex}";
    }

    /// <summary>
    /// tagSet is one '+'-joined tag set ("walk+turn_travel"), split here.
    /// </summary>
    public static void UpdateRegisteredClip(MotionMatchingController controller, int sourceIndex, string tagSet, bool loop, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (string.IsNullOrWhiteSpace(tagSet))
            return;
        string[] tags = tagSet.Split('+');

        if (MotionMatchingDatabase.UpdateSourceClip(controller.DatabaseFile, sourceIndex, tags, loop))
            ReloadDatabase(controller, true, onEdited);
    }

    public static void RemoveRegisteredClip(MotionMatchingController controller, int sourceIndex, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (MotionMatchingDatabase.RemoveSourceClip(controller.DatabaseFile, sourceIndex))
            ReloadDatabase(controller, true, onEdited);
    }

    /// <summary>
    /// Commits staged table edits in a single database load/save round-trip.
    /// </summary>
    public static void ApplyRegisteredClipEdits(MotionMatchingController controller,
        int[] updateIndices,
        string[] updateTags,
        bool[] updateLoops,
        int[] removeIndices,
        Action onEdited = null)
    {
        if (controller == null)
            return;
        updateIndices = updateIndices ?? Array.Empty<int>();
        updateTags = updateTags ?? Array.Empty<string>();
        removeIndices = removeIndices ?? Array.Empty<int>();
        if (updateIndices.Length == 0 && removeIndices.Length == 0)
            return;

        var loops = new int[updateLoops?.Length ?? 0];
        for (int i = 0; i < loops.Length; i++)
            loops[i] = updateLoops[i] ? 1 : 0;

        if (MotionMatchingDatabase.ApplyClipEdits(controller.DatabaseFile, updateIndices, updateTags, loops, removeIndices))
            ReloadDatabase(controller, true, onEdited);
    }

    public static void BakeDatabase(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return;
        if (!EnsureDatabaseLoaded(controller) || controller.Database.AnimationClipsCount == 0)
        {
            Debug.LogWarning("Motion matching bake stopped: no animation clips.");
            return;
        }

        if (!controller.PlayerModel)
        {
            Debug.LogWarning("Motion matching bake stopped: PlayerModel is not set.");
            return;
        }

        controller.PersistenceValid = MotionMatchingBaker.Bake(
            controller.PlayerModel,
            controller.Schema,
            controller.Database,
            out int clips,
            out int samples);

        if (!controller.PersistenceValid)
        {
            ReloadDatabase(controller, false);
            return;
        }

        ReloadDatabase(controller, true, onEdited);
        Debug.Log($"Motion matching database ready: clips={clips}, samples={samples}.");
    }

    public static void ValidatePersistence(MotionMatchingController controller)
    {
        if (controller == null)
            return;
        controller.PersistenceValid = MotionMatchingDatabase.ValidateSavedDatabase(
            controller.DatabaseFile, out _, out _, out _, out _, out _);
        ReloadDatabase(controller, false);
    }

    /// <summary>
    /// Runs the native search self-test (exact queries must return their own
    /// sample with ~zero cost) and prints the per-probe report.
    /// Calls the editor-only native RunSelfTest (USE_EDITOR-guarded C++,
    /// FLAX_EDITOR-guarded binding). This service lives in the editor module,
    /// so it only ever compiles in editor builds where the native exists.
    /// </summary>
    public static void RunSearchSelfTest(MotionMatchingController controller)
    {
        if (controller == null)
            return;
        if (!EnsureDatabaseLoaded(controller) || !controller.Database)
        {
            Debug.LogWarning("Motion matching self-test stopped: database is not loaded.");
            return;
        }

#if FLAX_EDITOR
        bool passed = MotionMatchingSearch.RunSelfTest(controller.Database, out string log);
        Debug.Log($"Motion matching search self-test {(passed ? "PASSED" : "FAILED")}:\n{log}");
        bool pbPassed = MotionMatchingPlayback.RunPlaybackSelfTest(controller.Database, controller.CharacterBody, out string pbLog);
        Debug.Log($"Motion matching playback self-test {(pbPassed ? "PASSED" : "FAILED")}:\n{pbLog}");
        if (!pbPassed)
            passed = false;
#else
        Debug.LogWarning("Motion matching self-test stopped: editor-only test is not available in game builds.");
#endif
    }

    // Weight presets: scene Schema weights are live tuning — never baked
    // into features — so presets apply immediately with no rebake, even
    // mid-play. Rebake is only needed for feature changes.
    public static void ApplyPresetBaseline(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return;
        controller.Schema.PoseWeight = 1.0f;
        controller.Schema.TrajectoryWeight = 2.0f;
        MarkTuningEdited(controller, "baseline (trajectory-heavy)", onEdited);
    }

    public static void ApplyPresetPosePrecision(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return;
        controller.Schema.PoseWeight = 2.5f;
        controller.Schema.TrajectoryWeight = 1.0f;
        MarkTuningEdited(controller, "pose-heavy precision", onEdited);
    }

    public static void ApplyPresetTurnHeavy(MotionMatchingController controller, Action onEdited = null)
    {
        if (controller == null)
            return;
        controller.Schema.PoseWeight = 1.0f;
        controller.Schema.TrajectoryWeight = 3.0f;
        MarkTuningEdited(controller, "turn-heavy", onEdited);
    }

    private static void MarkTuningEdited(MotionMatchingController controller, string preset, Action onEdited = null)
    {
        Debug.Log($"Motion matching weights preset applied: {preset} " +
            $"(pose={controller.Schema.PoseWeight}, trajectory={controller.Schema.TrajectoryWeight}). No rebake needed.");
        try
        {
            onEdited?.Invoke();
        }
        catch
        {
        }
    }

    /// <summary>
    /// One unpacked top-N row for the tuning window.
    /// </summary>
    public struct MotionMatchingCandidate
    {
        public int SampleIndex;
        public int ClipIndex;
        public string ClipName;
        public float SampleTime;
        public float Cost;
        public float PoseCost;
        public float TrajectoryCost;
        public float ContinuityCost;
        public float LoopCost;
        public bool IsContinuation;
        public bool IsLoop;
        public bool IsRecordedWinner;
    }

    // Top-N: re-runs the native ranking on the last recorded query's
    // inputs (works paused in the editor — no play input needed).
    public static MotionMatchingCandidate[] QueryTopCandidates(MotionMatchingController controller, int topN)
    {
        if (controller == null)
            return Array.Empty<MotionMatchingCandidate>();
        return QueryTopCandidatesFor(controller, controller.LastTraceEntry, topN);
    }

    // Same ranking for an entry loaded from a trace file (bridge/agent path):
    // the in-memory policy trace is gone after play stops, the file remains.
    public static MotionMatchingCandidate[] QueryTopCandidatesFor(MotionMatchingController controller, MotionMatchingTraceEntry entry, int topN)
    {
        var empty = Array.Empty<MotionMatchingCandidate>();
        if (controller == null)
            return empty;
        if (entry.Pose == null || entry.Trajectory == null || entry.Pose.Length == 0 || entry.Trajectory.Length == 0)
            return empty;
        if (!EnsureDatabaseLoaded(controller) || !controller.Database)
            return empty;

        string[] tags = string.IsNullOrEmpty(entry.Tags)
            ? Array.Empty<string>()
            : entry.Tags.Split('+');
        var settings = new MotionMatchingSearchSettings
        {
            PoseWeight = entry.PoseWeight,
            TrajectoryWeight = entry.TrajectoryWeight,
            ContinuityWeight = entry.ContinuityWeight,
            MinSwitchTime = entry.MinSwitchTime,
            CurrentTime = entry.CurrentTime,
            SameClipExclusionWindow = entry.ExclusionWindow,
            ExpectedTime = entry.ExpectedTime,
            ContinuingBias = entry.ContinuingBias,
            ContinuingWindow = entry.ContinuingWindow,
            SwitchMargin = entry.SwitchMargin,
            LoopPreference = entry.LoopPreference,
            LoopFilter = (MotionLoopFilter)entry.LoopFilter,
            ReselectClip = entry.ReselectClip,
            ReselectSampleTime = entry.ReselectSampleTime,
            ReselectBanWindow = entry.ReselectBanWindow,
            TwinBanWindow = entry.TwinBanWindow,
            LoopTailExclusion = entry.LoopTailExclusion,
        };
        MotionMatchingSearch.FindTopCandidates(controller.Database, entry.Pose, entry.Trajectory,
            settings, entry.ContinuityClip, tags, topN, out float[] packed, out string error);
        if (!string.IsNullOrEmpty(error) && (packed == null || packed.Length == 0))
        {
            Debug.LogWarning($"Motion matching top-N: {error}");
            return empty;
        }
        if (packed == null || packed.Length == 0)
            return empty;
        const int stride = 10;
        int rows = packed.Length / stride;
        var result = new MotionMatchingCandidate[rows];
        for (int i = 0; i < rows; i++)
        {
            int b = i * stride;
            int clip = (int)packed[b + 1];
            int sample = (int)packed[b];
            result[i] = new MotionMatchingCandidate
            {
                SampleIndex = sample,
                ClipIndex = clip,
                ClipName = ClipDisplayName(controller, clip),
                SampleTime = packed[b + 2],
                Cost = packed[b + 3],
                PoseCost = packed[b + 4],
                TrajectoryCost = packed[b + 5],
                ContinuityCost = packed[b + 6],
                LoopCost = packed[b + 7],
                IsContinuation = packed[b + 8] > 0.5f,
                IsLoop = packed[b + 9] > 0.5f,
                IsRecordedWinner = entry.Valid && sample == entry.SampleIndex,
            };
        }
        return result;
    }

    private static string ClipDisplayName(MotionMatchingController controller, int clipIndex)
    {
        try
        {
            string path = controller.Database?.GetClipAnimation(clipIndex)?.Path;
            if (!string.IsNullOrEmpty(path))
                return Path.GetFileNameWithoutExtension(path);
        }
        catch
        {
        }
        return $"#{clipIndex}";
    }

    // Trace file helpers for the tuning window (default folder: Cache/).
    public static string TraceDefaultFolder()
    {
        try
        {
            string folder = Globals.ProjectFolder;
            if (string.IsNullOrEmpty(folder))
                folder = Path.GetDirectoryName(Globals.ProjectContentFolder);
            string cache = Path.Combine(folder, "Cache");
            if (!Directory.Exists(cache))
                Directory.CreateDirectory(cache);
            return cache;
        }
        catch
        {
            return Globals.ProjectContentFolder;
        }
    }

    private static bool EnsureDatabaseLoaded(MotionMatchingController controller)
    {
        // Timeout 0: probe only, never block the editor thread (default is 30s).
        return controller.Database && !controller.Database.WaitForLoaded(0) || ReloadDatabase(controller, false);
    }

    private static bool ReloadDatabase(MotionMatchingController controller, bool markSceneEdited, Action onEdited = null)
    {
        var loaded = Content.Load<MotionMatchingDatabase>(controller.DatabaseFile);
        if (!loaded)
        {
            controller.ClearDatabaseStatus();
            return false;
        }

        controller.Database = loaded;
        controller.RefreshDatabaseStatus();
        if (markSceneEdited)
        {
            try
            {
                onEdited?.Invoke();
            }
            catch
            {
            }
        }
        return true;
    }
}
