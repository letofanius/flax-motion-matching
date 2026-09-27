using System;
using System.IO;
using FlaxEngine;
using FlaxEditor;
using FlaxEditor.CustomEditors;
using FlaxEditor.CustomEditors.Elements;

namespace MotionMatching;

/// <summary>
/// Tuning window (editor tooling).
///
/// Purpose: database validity, live query telemetry, top-N cost ranking for
/// the last recorded query, trace export/import/replay, and weight presets.
/// Top-N and replay run on recorded inputs, so they work paused in the
/// editor with no play input. Text rows (no owner-drawn table): 8 fixed
/// labels, updated on demand.
/// Ownership: editor tooling operating on the host-owned controller object.
/// Tick/lifecycle role: edit-mode only; reads controller mirrors and the
/// recorded trace on demand, applies weight presets live (no rebake).
/// This window must never reference game/player/sample types.
/// </summary>
public class MotionMatchingTuningWindow : CustomEditorWindow
{
    private const int TopRows = 8;

    private readonly MotionMatchingController _controller;
    private readonly System.Action _onEdited;
    private LabelElement _validity;
    private LabelElement _live;
    private readonly LabelElement[] _topLabels = new LabelElement[TopRows];
    private LabelElement _status;

    public MotionMatchingTuningWindow(MotionMatchingController controller, System.Action onEdited = null)
    {
        _controller = controller;
        _onEdited = onEdited;
    }

    public override void Initialize(LayoutElementsContainer layout)
    {
        layout.Label("Motion Matching Tuning (M6)", TextAlignment.Center);
        layout.Space(4);

        layout.Label("Database", TextAlignment.Near);
        _validity = layout.Label("-");
        layout.Space(4);

        layout.Label("Live Query (inspector mirrors these at runtime)", TextAlignment.Near);
        _live = layout.Label("-");
        var refresh = layout.Button("Refresh").Button;
        refresh.Clicked += RefreshAll;
        layout.Space(4);

        layout.Label("Top-N (last recorded query inputs, raw cost rank)", TextAlignment.Near);
        layout.Label("Raw search candidates: no lock, margin, hold, replay or LoopOnly recovery. '*' = recorded policy winner.");
        var topPanel = layout.VerticalPanel();
        for (int i = 0; i < TopRows; i++)
            _topLabels[i] = topPanel.Label("-");
        var topButton = layout.Button("Query Top-8 From Last Entry").Button;
        topButton.Clicked += QueryTop;
        layout.Space(4);

        layout.Label("Weight Presets (live, no rebake)", TextAlignment.Near);
        var presets = layout.HorizontalPanel();
        var baseline = presets.Button("Baseline").Button;
        baseline.Clicked += () => { if (_controller != null) { MotionMatchingEditorService.ApplyPresetBaseline(_controller, _onEdited); ApplyPresetDone("baseline"); } };
        var precision = presets.Button("Pose Precision").Button;
        precision.Clicked += () => { if (_controller != null) { MotionMatchingEditorService.ApplyPresetPosePrecision(_controller, _onEdited); ApplyPresetDone("pose precision"); } };
        var turn = presets.Button("Turn Heavy").Button;
        turn.Clicked += () => { if (_controller != null) { MotionMatchingEditorService.ApplyPresetTurnHeavy(_controller, _onEdited); ApplyPresetDone("turn heavy"); } };
        layout.Space(4);

        layout.Label("Trace (Cache/MMTrace.jsonl, Cache/MMTrace.csv)", TextAlignment.Near);
        var traceRow = layout.HorizontalPanel();
        var exportJson = traceRow.Button("Export JSON").Button;
        exportJson.Clicked += ExportJson;
        var exportCsv = traceRow.Button("Export CSV").Button;
        exportCsv.Clicked += ExportCsv;
        var traceRow2 = layout.HorizontalPanel();
        var importJson = traceRow2.Button("Import JSON").Button;
        importJson.Clicked += ImportJson;
        var replay = traceRow2.Button("Verify Replay x200").Button;
        replay.Clicked += VerifyReplay;
        var clear = layout.Button("Clear Trace").Button;
        clear.Clicked += ClearTrace;
        _status = layout.Label("trace: -");

        RefreshAll();
    }

    private void ApplyPresetDone(string name)
    {
        _status.Label.Text = "preset: " + name + " applied (live, no rebake).";
        RefreshAll();
    }

    private void RefreshAll()
    {
        if (_controller == null)
            return;
        int tagCount = _controller.AnimationTags != null ? _controller.AnimationTags.Length : 0;
        float margin = _controller.SecondBestCost - _controller.WinnerCost;
        _validity.Label.Text =
            $"clips={_controller.AnimationClipsCount} samples={_controller.SampleCount} " +
            $"tags={tagCount} valid={_controller.PersistenceValid}\n" +
            $"live weights: pose={_controller.Schema.PoseWeight} traj={_controller.Schema.TrajectoryWeight}";
        _live.Label.Text =
            $"state={_controller.PolicyState} tags={_controller.ActiveTag}\n" +
            $"winner={_controller.WinnerClipName} t={_controller.WinnerSample} " +
            $"cost={_controller.WinnerCost:F3} (pose={_controller.WinnerPoseCost:F3} " +
            $"traj={_controller.WinnerTrajectoryCost:F3} cont={_controller.WinnerContinuityCost:F3} " +
            $"loop={_controller.WinnerLoopCost:F3}) margin={margin:F3}\n" +
            $"reason={_controller.LastSwitchReason} chooser={_controller.ChooserStatus}\n" +
            $"switches={_controller.SwitchCount} seeks={_controller.SeekCount} " +
            $"qms={_controller.LastQueryMs:F2} trace={_controller.TraceEntries}";
    }

    private void QueryTop()
    {
        if (_controller == null)
            return;
        var rows = MotionMatchingEditorService.QueryTopCandidates(_controller, TopRows);
        if (rows.Length == 0)
        {
            for (int i = 0; i < TopRows; i++)
                _topLabels[i].Label.Text = "-";
            _status.Label.Text = "top-N: no recorded query yet (play first).";
            return;
        }
        for (int i = 0; i < TopRows; i++)
        {
            if (i >= rows.Length)
            {
                _topLabels[i].Label.Text = "-";
                continue;
            }
            var r = rows[i];
            string mark = r.IsRecordedWinner ? "*" : " ";
            string cont = r.IsContinuation ? " cont" : string.Empty;
            string loop = r.IsLoop ? " loop" : " 1shot";
            _topLabels[i].Label.Text =
                $"{mark}{i}: {r.ClipName} t={r.SampleTime:F2} cost={r.Cost:F3} " +
                $"(p={r.PoseCost:F3} t={r.TrajectoryCost:F3} c={r.ContinuityCost:F3} l={r.LoopCost:F3})" +
                $"{loop}{cont}";
        }
        var trace = _controller != null ? _controller.Trace : null;
        var last = _controller != null ? _controller.LastTraceEntry : new MotionMatchingTraceEntry();
        _status.Label.Text = trace != null && trace.Count > 0
            ? $"top-N: t={last.Time:F2} tags={last.Tags} reason={last.Reason}"
            : "top-N: done.";
    }

    private string JsonPath() => Path.Combine(MotionMatchingEditorService.TraceDefaultFolder(), "MMTrace.jsonl");
    private string CsvPath() => Path.Combine(MotionMatchingEditorService.TraceDefaultFolder(), "MMTrace.csv");

    private void ExportJson()
    {
        var trace = _controller != null ? _controller.Trace : null;
        if (trace == null || trace.Count == 0)
        {
            _status.Label.Text = "export: trace is empty (play first).";
            return;
        }
        if (trace.ExportJson(JsonPath(), out string error))
            _status.Label.Text = $"export: {trace.Count} entries -> MMTrace.jsonl";
        else
            _status.Label.Text = "export failed: " + error;
    }

    private void ExportCsv()
    {
        var trace = _controller != null ? _controller.Trace : null;
        if (trace == null || trace.Count == 0)
        {
            _status.Label.Text = "export: trace is empty (play first).";
            return;
        }
        if (trace.ExportCsv(CsvPath(), out string error, out string summary))
            _status.Label.Text = "csv: " + summary;
        else
            _status.Label.Text = "export failed: " + error;
    }

    private void ImportJson()
    {
        var trace = _controller != null ? _controller.Trace : null;
        if (trace == null)
            return;
        if (trace.ImportJson(JsonPath(), out string error))
        {
            _status.Label.Text = $"import: {trace.Count} entries from MMTrace.jsonl";
            RefreshAll();
        }
        else
        {
            _status.Label.Text = "import failed: " + error;
        }
    }

    private void VerifyReplay()
    {
        var trace = _controller != null ? _controller.Trace : null;
        if (trace == null || trace.Count == 0)
        {
            _status.Label.Text = "replay: trace is empty (play or import first).";
            return;
        }
        bool ok = trace.VerifyReplay(_controller.Database, 200, out string report);
        _status.Label.Text = (ok ? "Search replay OK: " : "Search replay MISMATCH: ") + report;
        Debug.Log("Motion matching search replay: " + report);
    }

    private void ClearTrace()
    {
        _controller?.Trace?.Clear();
        _status.Label.Text = "trace cleared.";
        RefreshAll();
    }
}
