using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using FlaxEngine;
using FlaxEditor;
using FlaxEditor.CustomEditors;
using FlaxEditor.CustomEditors.Elements;

namespace MotionMatching;

/// <summary>
/// Clip registration window (editor tooling).
///
/// Purpose: register animation assets into the clip registry and review the
/// registered set before bake. Top: single-path register (folder scans
/// recursively, file dialog multi-selects, ';' separates manual paths).
/// Bottom: owner-drawn table of every registered clip (one control total)
/// with a shared detail bar; edits stage in RAM until Save All.
/// Ownership: editor tooling operating on the host-owned controller object.
/// Tick/lifecycle role: edit-mode only; staged edits commit through
/// MotionMatchingEditorService and take effect after bake + reload.
/// This window must never reference game/player/sample types.
/// </summary>
public class RegisterClipsWindow : CustomEditorWindow
{
    private class StagedRow
    {
        public int SourceIndex;
        public string Name;
        public string BaseTag;
        public bool BaseLoop;
        public string Tag;
        public bool Loop;
        public bool Removed;
        public bool Dirty;
    }

    private readonly MotionMatchingController _controller;
    private TextBoxElement _source;
    private TextBoxElement _tag;
    private CheckBoxElement _loop;

    private LabelElement _count;
    private TextBoxElement _filter;
    private ClipTableControl _table;
    private LabelElement _clipName;
    private TextBoxElement _editTag;
    private CheckBoxElement _editLoop;

    private readonly List<StagedRow> _all = new();
    private readonly List<StagedRow> _visible = new();
    private bool _refreshing;

    public RegisterClipsWindow(MotionMatchingController controller)
    {
        _controller = controller;
    }

    public override void Initialize(LayoutElementsContainer layout)
    {
        layout.Label("Register Imported Animations", TextAlignment.Center);
        layout.Space(4);
        _source = layout.AddPropertyItem("Source", "Folder (scanned recursively), .flax file, or ';'-separated paths.").TextBox();
        _source.TextBox.Text = string.Empty;
        _tag = layout.AddPropertyItem("Tags", "Motion groups, '+'-separated. Tags are supplied by the host.").TextBox();
        _tag.TextBox.Text = string.Empty;
        _loop = layout.Checkbox("Loop Clips", "Wrap samples at clip boundaries.");
        _loop.CheckBox.Checked = false;
        var buttons = layout.HorizontalPanel();
        var browseFolder = buttons.Button("Browse Folder").Button;
        browseFolder.Clicked += BrowseFolder;
        var browseFiles = buttons.Button("Browse Files").Button;
        browseFiles.Clicked += BrowseFiles;
        var register = buttons.Button("Register").Button;
        register.Clicked += Register;
        var buttonPanel = buttons.ContainerControl;
        bool heightsLogged = false;
        void FitButtons()
        {
            // Note: HorizontalPanel forces children height = panel height on
            // every layout pass, so the panel carries the height, not the buttons.
            buttonPanel.Height = 16.0f;
            float width = Math.Max(60.0f, (buttonPanel.Width - 8.0f) / 3.0f);
            browseFolder.Width = width;
            browseFiles.Width = width;
            register.Width = width;
            // One-shot layout diagnostic: log the settled sizes once, so a
            // mis-sized button row can be traced to the panel vs the buttons.
            if (!heightsLogged && buttonPanel.Width > 200.0f)
            {
                heightsLogged = true;
                Debug.Log($"[RegisterClips] panel={buttonPanel.Size} browse={browseFolder.Size} register={register.Size}");
            }
        }
        buttonPanel.SizeChanged += _ => FitButtons();
        FitButtons();

        layout.Space(12);
        layout.Label("Registered Clips", TextAlignment.Center);
        _count = layout.Label("0 clips");
        _filter = layout.AddPropertyItem("Filter", "Show clips whose name or tag contains this text.").TextBox();
        _filter.TextBox.TextChanged += ApplyFilter;
        _table = layout.Custom<ClipTableControl>().CustomControl;
        _table.SelectionChanged += ShowSelection;
        _table.LoopToggled += ToggleLoop;
        _table.RemoveClicked += StageRemove;

        layout.Space(4);
        _clipName = layout.Label("-");
        _editTag = layout.AddPropertyItem("Tags", "Tags for the selected clip, '+'-separated.").TextBox();
        _editLoop = layout.Checkbox("Loop", "Wrap samples at clip boundaries.");
        layout.Button("Set Selected (stage)").Button.Clicked += StageSelected;
        layout.Space(4);
        layout.Button("Tick Shown Loops").Button.Clicked += () => SetShownLoops(true);
        layout.Button("Untick Shown Loops").Button.Clicked += () => SetShownLoops(false);
        layout.Space(4);
        layout.Button("Save All Changes").Button.Clicked += SaveAll;
        layout.Button("Revert").Button.Clicked += RebuildList;
        layout.Button("Refresh List").Button.Clicked += RebuildList;

        RebuildList();
    }

    private void BrowseFolder()
    {
        if (FileSystem.ShowBrowseFolderDialog(
                Editor.Instance.Windows.MainWindow,
                Globals.ProjectContentFolder,
                "Select Animation Folder",
                out string folder) || !Directory.Exists(folder))
            return;
        _source.TextBox.Text = folder;
    }

    private void BrowseFiles()
    {
        const string filter = "Flax Animation\0*.flax\0";
        if (!FileSystem.ShowOpenFileDialog(
                Editor.Instance.Windows.MainWindow,
                Globals.ProjectContentFolder,
                filter,
                true,
                "Select Animations",
                out string[] files) && files?.Length > 0)
        {
            _source.TextBox.Text = string.Join(";", files);
        }
    }

    private void Register()
    {
        string tagText = _tag?.TextBox?.Text?.Trim();
        if (string.IsNullOrEmpty(tagText))
            return;
        string[] tags = tagText.Split('+');

        var files = new List<string>();
        foreach (string raw in (_source?.TextBox?.Text ?? string.Empty).Split(';'))
        {
            string path = raw.Trim().Trim('"');
            if (string.IsNullOrEmpty(path))
                continue;
            if (Directory.Exists(path))
            {
                files.AddRange(Directory.GetFiles(path, "*.flax", SearchOption.AllDirectories));
            }
            else if (File.Exists(path) && path.EndsWith(".flax", StringComparison.OrdinalIgnoreCase))
            {
                files.Add(path);
            }
            else
            {
                Debug.LogWarning($"Motion matching register skipped unknown path: '{path}'.");
            }
        }

        if (files.Count > 0 && _controller != null)
        {
            MotionMatchingEditorService.AddClipFiles(_controller, tags, files.ToArray(), _loop?.CheckBox.Checked ?? true);
            RebuildList();
        }
    }

    private void RebuildList()
    {
        int keepSource = SelectedSourceIndex();

        _all.Clear();
        var clips = _controller != null ? MotionMatchingEditorService.GetRegisteredClips(_controller) : null;
        if (clips != null)
        {
            foreach (var clip in clips.OrderBy(c => MotionMatchingEditorService.GetRegisteredClipName(c)))
            {
                _all.Add(new StagedRow
                {
                    SourceIndex = clip.SourceIndex,
                    Name = MotionMatchingEditorService.GetRegisteredClipName(clip),
                    BaseTag = clip.Tag,
                    BaseLoop = clip.Loop,
                    Tag = clip.Tag,
                    Loop = clip.Loop,
                });
            }
        }

        ApplyFilter(keepSource);
    }

    private void ApplyFilter()
    {
        ApplyFilter(SelectedSourceIndex());
    }

    private void ApplyFilter(int keepSource)
    {
        if (_refreshing)
            return;

        string filter = _filter?.TextBox?.Text?.Trim().ToLowerInvariant() ?? string.Empty;
        _visible.Clear();
        foreach (var row in _all)
        {
            if (row.Removed)
                continue;
            if (string.IsNullOrEmpty(filter) ||
                row.Name.ToLowerInvariant().Contains(filter) ||
                (row.Tag ?? string.Empty).ToLowerInvariant().Contains(filter))
            {
                _visible.Add(row);
            }
        }

        _refreshing = true;
        _table.Rows.Clear();
        foreach (var row in _visible)
        {
            _table.Rows.Add(new ClipTableRow
            {
                SourceIndex = row.SourceIndex,
                Name = row.Name,
                Tag = row.Tag,
                Loop = row.Loop,
                Dirty = row.Dirty,
            });
        }
        int selected = keepSource >= 0 ? _visible.FindIndex(r => r.SourceIndex == keepSource) : -1;
        _table.SelectedRow = selected;
        _table.EnsureVisible(Math.Max(0, selected));
        _refreshing = false;

        UpdateCount();
        ShowSelection();
    }

    private void ShowSelection()
    {
        var row = SelectedRow();
        _refreshing = true;
        if (row != null)
        {
            _clipName.Label.Text = row.Name;
            _editTag.TextBox.Text = row.Tag;
            _editLoop.CheckBox.Checked = row.Loop;
        }
        else
        {
            _clipName.Label.Text = "-";
            _editTag.TextBox.Text = string.Empty;
            _editLoop.CheckBox.Checked = false;
        }
        _refreshing = false;
    }

    private void UpdateCount()
    {
        int edited = _all.Count(r => r.Dirty && !r.Removed);
        int removed = _all.Count(r => r.Removed);
        _count.Label.Text = $"{_all.Count} clips ({_visible.Count} shown, {edited} edited, {removed} removed)";
    }

    private StagedRow SelectedRow()
    {
        int selected = _table.SelectedRow;
        return selected >= 0 && selected < _visible.Count ? _visible[selected] : null;
    }

    private int SelectedSourceIndex()
    {
        return SelectedRow()?.SourceIndex ?? -1;
    }

    private void ToggleLoop(int tableRow)
    {
        if (tableRow < 0 || tableRow >= _visible.Count)
            return;
        var row = _visible[tableRow];
        row.Loop = !row.Loop;
        row.Dirty = row.Tag != row.BaseTag || row.Loop != row.BaseLoop;
        SyncTableRow(tableRow, row);
        if (_table.SelectedRow == tableRow)
            ShowSelection();
        UpdateCount();
    }

    private void StageRemove(int tableRow)
    {
        if (tableRow < 0 || tableRow >= _visible.Count)
            return;
        _visible[tableRow].Removed = true;
        ApplyFilter(-1);
    }

    private void StageSelected()
    {
        var row = SelectedRow();
        if (row == null)
            return;
        string tag = _editTag.TextBox.Text?.Trim();
        if (string.IsNullOrEmpty(tag))
            return;
        row.Tag = tag;
        row.Loop = _editLoop.CheckBox.Checked;
        row.Dirty = row.Tag != row.BaseTag || row.Loop != row.BaseLoop;
        SyncTableRow(_table.SelectedRow, row);
        UpdateCount();
    }

    private void SyncTableRow(int tableRow, StagedRow row)
    {
        if (tableRow < 0 || tableRow >= _table.Rows.Count)
            return;
        _table.Rows[tableRow].Tag = row.Tag;
        _table.Rows[tableRow].Loop = row.Loop;
        _table.Rows[tableRow].Dirty = row.Dirty;
    }

    private void SetShownLoops(bool loop)
    {
        for (int i = 0; i < _visible.Count; i++)
        {
            var row = _visible[i];
            row.Loop = loop;
            row.Dirty = row.Tag != row.BaseTag || row.Loop != row.BaseLoop;
            SyncTableRow(i, row);
        }
        ShowSelection();
        UpdateCount();
    }

    private void SaveAll()
    {
        var updateIndices = new List<int>();
        var updateTags = new List<string>();
        var updateLoops = new List<bool>();
        var removeIndices = new List<int>();
        foreach (var row in _all)
        {
            if (row.Removed)
            {
                removeIndices.Add(row.SourceIndex);
            }
            else if (row.Dirty)
            {
                updateIndices.Add(row.SourceIndex);
                updateTags.Add(row.Tag);
                updateLoops.Add(row.Loop);
            }
        }

        if (updateIndices.Count == 0 && removeIndices.Count == 0)
            return;

        if (_controller == null)
            return;
        MotionMatchingEditorService.ApplyRegisteredClipEdits(_controller,
            updateIndices.ToArray(),
            updateTags.ToArray(),
            updateLoops.ToArray(),
            removeIndices.ToArray());
        RebuildList();
    }
}
