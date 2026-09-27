// Clip table control (editor tooling).
//
// Purpose: owner-drawn, single-control clip list for the registration window.
// Ownership: editor tooling, hosted by RegisterClipsWindow.
// Tick/lifecycle role: edit-mode only; paints visible rows on demand and
// reports selection/loop/remove clicks through events; holds no database state.
// This file must never reference game/player/sample types.

using System;
using System.Collections.Generic;
using FlaxEngine;
using FlaxEngine.GUI;

namespace MotionMatching;

/// <summary>
/// One row of staged clip data drawn by <see cref="ClipTableControl"/>.
/// </summary>
public class ClipTableRow
{
    public int SourceIndex;
    public string Name;
    public string Tag;
    public bool Loop;
    public bool Dirty;
}

/// <summary>
/// Owner-drawn clip table: a single control no matter how many rows, so a full
/// scrollable list stays cheap. Only the visible rows are painted; scrolling is
/// internal (mouse wheel), selection and cell clicks are hit-tested by hand.
/// </summary>
public class ClipTableControl : Control
{
    public const float RowHeight = 22.0f;
    public const float HeaderHeight = 22.0f;

    // Column edges as width fractions: name | tag | loop | remove.
    private const float TagEdge = 0.48f;
    private const float LoopEdge = 0.74f;
    private const float RemoveEdge = 0.86f;

    public readonly List<ClipTableRow> Rows = new();
    public int SelectedRow = -1;
    public int FirstRow;

    public event Action SelectionChanged;
    public event Action<int> LoopToggled;
    public event Action<int> RemoveClicked;

    public ClipTableControl()
    {
        Size = new Float2(540.0f, 360.0f);
    }

    public int VisibleRowCount => Math.Max(0, (int)((Height - HeaderHeight) / RowHeight) - 1);

    public int MaxFirstRow => Math.Max(0, Rows.Count - VisibleRowCount);

    public void EnsureVisible(int row)
    {
        if (row < FirstRow)
            FirstRow = row;
        else if (row >= FirstRow + VisibleRowCount)
            FirstRow = row - VisibleRowCount + 1;
        FirstRow = Mathf.Clamp(FirstRow, 0, MaxFirstRow);
    }

    public override void Draw()
    {
        base.Draw();

        var style = Style.Current;
        var font = style?.FontSmall;
        if (font == null || Width <= 0.0f || Height <= 0.0f)
            return;

        Color fg = style.Foreground;
        Render2D.FillRectangle(new Rectangle(Float2.Zero, Size), style.Background);
        Render2D.FillRectangle(new Rectangle(0.0f, 0.0f, Width, HeaderHeight), style.LightBackground);

        DrawCell(font, "Name", 0.0f, TagEdge, 0, fg, true);
        DrawCell(font, "Tag", TagEdge, LoopEdge, 0, fg, true);
        DrawCell(font, "Loop", LoopEdge, RemoveEdge, 0, fg, true);

        int last = Math.Min(Rows.Count, FirstRow + VisibleRowCount + 1);
        for (int r = FirstRow; r < last; r++)
        {
            float y = HeaderHeight + (r - FirstRow) * RowHeight;
            if (r == SelectedRow)
                Render2D.FillRectangle(new Rectangle(0.0f, y, Width, RowHeight), style.LightBackground);

            var row = Rows[r];
            string name = row.Dirty ? "* " + row.Name : row.Name;
            DrawCell(font, name, 0.0f, TagEdge, y, fg, false);
            DrawCell(font, row.Tag, TagEdge, LoopEdge, y, fg, false);
            DrawCell(font, row.Loop ? "[x]" : "[ ]", LoopEdge, RemoveEdge, y, fg, false);
            DrawCell(font, "x", RemoveEdge, 1.0f, y, fg, false);
        }

        Color grid = style.ForegroundGrey;
        foreach (float edge in new[] { TagEdge, LoopEdge, RemoveEdge })
            Render2D.DrawLine(new Float2(Width * edge, 0.0f), new Float2(Width * edge, Height), grid, 1.0f);
        Render2D.DrawLine(new Float2(0.0f, HeaderHeight), new Float2(Width, HeaderHeight), grid, 1.0f);
        Render2D.DrawRectangle(new Rectangle(Float2.Zero, Size), grid, 1.0f);
    }

    private void DrawCell(Font font, string text, float x0, float x1, float y, Color color, bool header)
    {
        float left = Width * x0 + 4.0f;
        float right = Width * x1 - 4.0f;
        if (right <= left)
            return;
        var clip = new Rectangle(left, y, right - left, header ? HeaderHeight : RowHeight);
        Render2D.PushClip(clip);
        Render2D.DrawText(font, text ?? string.Empty, color, new Float2(left, y + 4.0f));
        Render2D.PopClip();
    }

    public override bool OnMouseDown(Float2 location, MouseButton button)
    {
        if (button != MouseButton.Left)
            return false;

        int row = HitRow(location.Y);
        if (row < 0 || row >= Rows.Count)
            return false;

        float x = location.X / Math.Max(1.0f, Width);
        if (x >= RemoveEdge)
        {
            RemoveClicked?.Invoke(row);
            return true;
        }
        if (x >= LoopEdge)
        {
            LoopToggled?.Invoke(row);
            return true;
        }
        SelectedRow = row;
        EnsureVisible(row);
        SelectionChanged?.Invoke();
        return true;
    }

    public override bool OnMouseWheel(Float2 location, float delta)
    {
        FirstRow = Mathf.Clamp(FirstRow - Math.Sign(delta) * 3, 0, MaxFirstRow);
        return true;
    }

    private int HitRow(float y)
    {
        if (y < HeaderHeight)
            return -1;
        return FirstRow + (int)((y - HeaderHeight) / RowHeight);
    }
}
