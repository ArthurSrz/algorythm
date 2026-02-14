"""
Manim animation: Understanding A* Through the 8-Puzzle
======================================================
Render:  cd 8Puzzle && manim -pql astar_animation.py AStarExplained
Final:   cd 8Puzzle && manim -pqh astar_animation.py AStarExplained
"""
from manim import *
import numpy as np

# ─── Color Palette ──────────────────────────────────────────────
TILE_BLUE = "#58C4DD"
BLANK_GREY = "#333333"
GOAL_GREEN = "#83C167"
REJECT_RED = "#FC6255"


# ─── Helpers ────────────────────────────────────────────────────

def create_grid(tiles, cell_size=0.8, highlight=None, highlight_color=YELLOW):
    """Create a 3x3 grid of the 8-puzzle as a VGroup."""
    gap = cell_size * 0.08
    stride = cell_size + gap
    grid = VGroup()
    for i in range(9):
        row, col = i // 3, i % 3
        is_blank = tiles[i] == 0
        is_highlighted = i == highlight

        fill = highlight_color if is_highlighted else (BLANK_GREY if is_blank else TILE_BLUE)
        opacity = 0.9 if not is_blank else 0.4
        stroke = highlight_color if is_highlighted else WHITE
        sq = Square(
            side_length=cell_size,
            fill_color=fill, fill_opacity=opacity,
            stroke_color=stroke,
            stroke_width=3 if is_highlighted else 1.5,
        )
        sq.move_to(np.array([col * stride, -row * stride, 0]))

        if is_blank:
            cell = VGroup(sq)
        else:
            txt = Text(str(tiles[i]), font_size=int(cell_size * 30), color=WHITE)
            txt.move_to(sq.get_center())
            cell = VGroup(sq, txt)
        grid.add(cell)
    grid.center()
    return grid


def create_queue_visual(items, max_shown=5):
    """Priority queue as stacked colored cards."""
    group = VGroup()
    shown = items[:max_shown]
    for i, (f, label) in enumerate(shown):
        t = i / max(len(shown) - 1, 1)
        color = interpolate_color(GREEN, RED, t)
        card = Rectangle(
            width=3.2, height=0.45,
            fill_color=color, fill_opacity=0.25, stroke_color=color,
        )
        text = Text(f"f={f}  {label}", font_size=18, color=WHITE)
        text.move_to(card.get_center())
        pair = VGroup(card, text)
        pair.shift(DOWN * i * 0.52)
        group.add(pair)
    if len(items) > max_shown:
        dots = Text("...", font_size=24, color=GREY)
        dots.shift(DOWN * max_shown * 0.52)
        group.add(dots)
    group.center()
    return group


def question_banner(text):
    """User question displayed as italic quote at top of scene."""
    q = Text(f'"{text}"', font_size=26, color=BLUE, slant=ITALIC)
    q.to_edge(UP, buff=0.4)
    return q


# ─── Main Scene ─────────────────────────────────────────────────

class AStarExplained(Scene):
    """Four-part A* explanation driven by learner questions."""

    def construct(self):
        self._scene_1_heuristic()
        self._scene_2_lower_bound()
        self._scene_3_moves()
        self._scene_4_astar_loop()

    def _clear(self):
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=0.5)

    # ────────────────────────────────────────────────────────────
    # Scene 1 — "What does the heuristic compute?"
    # ────────────────────────────────────────────────────────────

    def _scene_1_heuristic(self):
        self._clear()
        q = question_banner("What does the heuristic compute?")
        self.play(Write(q))
        self.wait(0.8)

        # Two grids: current (shuffled) vs goal (solved)
        current_tiles = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal_tiles = [1, 2, 3, 4, 5, 6, 7, 8, 0]

        cur = create_grid(current_tiles)
        goal = create_grid(goal_tiles)
        cur.shift(LEFT * 3 + UP * 0.2)
        goal.shift(RIGHT * 3 + UP * 0.2)

        cl = Text("Current state", font_size=20, color=WHITE).next_to(cur, UP, buff=0.25)
        gl = Text("Goal state", font_size=20, color=GOAL_GREEN).next_to(goal, UP, buff=0.25)

        self.play(FadeIn(cur), FadeIn(goal), Write(cl), Write(gl))
        self.wait(0.8)

        # ── Tile-by-tile Manhattan distance ──
        # We'll highlight three tiles one at a time and show their distance.
        # tile value → (index in current, index in goal)
        demos = [
            # (tile_val, cur_idx, goal_idx, cur_row, cur_col, goal_row, goal_col, manhattan)
            (8, 1, 7, 0, 1, 2, 1, 2),
            (6, 4, 5, 1, 1, 1, 2, 1),
            (5, 8, 4, 2, 2, 1, 1, 2),
        ]
        running_sum = 0
        for tile_val, ci, gi, cr, cc, gr, gc, md in demos:
            cur_h = create_grid(current_tiles, highlight=ci, highlight_color=YELLOW)
            cur_h.shift(LEFT * 3 + UP * 0.2)
            goal_h = create_grid(goal_tiles, highlight=gi, highlight_color=YELLOW)
            goal_h.shift(RIGHT * 3 + UP * 0.2)
            self.play(Transform(cur, cur_h), Transform(goal, goal_h), run_time=0.5)

            formula = MathTex(
                rf"\text{{Tile {tile_val}: }} |{cr}-{gr}| + |{cc}-{gc}| = {md}",
                font_size=26, color=YELLOW,
            ).shift(DOWN * 2.0)
            self.play(Write(formula), run_time=0.6)
            self.wait(0.8)
            running_sum += md
            self.play(FadeOut(formula), run_time=0.3)

        # Restore un-highlighted grids
        cur_plain = create_grid(current_tiles)
        cur_plain.shift(LEFT * 3 + UP * 0.2)
        goal_plain = create_grid(goal_tiles)
        goal_plain.shift(RIGHT * 3 + UP * 0.2)
        self.play(Transform(cur, cur_plain), Transform(goal, goal_plain), run_time=0.4)

        # Total: sum across ALL tiles = 9
        total_formula = MathTex(
            r"h(\text{state}) = \sum_{i=1}^{8} \text{Manhattan}_i = 9",
            font_size=28, color=GOAL_GREEN,
        ).shift(DOWN * 2.2)

        answer = Text(
            "It takes each tile's position, compares to goal,\n"
            "and sums the row + column differences.",
            font_size=18, color=WHITE, line_spacing=1.3,
        ).shift(DOWN * 3.2)

        self.play(Write(total_formula))
        self.play(FadeIn(answer))
        self.wait(2.5)

    # ────────────────────────────────────────────────────────────
    # Scene 2 — "Why is it a lower bound?"
    # ────────────────────────────────────────────────────────────

    def _scene_2_lower_bound(self):
        self._clear()
        q = question_banner("Why is it a lower bound?")
        self.play(Write(q))
        self.wait(0.8)

        # 3x3 background grid
        cell = 1.0
        grid_bg = VGroup()
        for r in range(3):
            for c in range(3):
                sq = Square(side_length=cell, stroke_color=GREY, stroke_width=1)
                sq.move_to(np.array([c * cell, -r * cell, 0]))
                grid_bg.add(sq)
        grid_bg.center().shift(UP * 0.3)

        # Tile at (0,0), goal at (1,2)  →  Manhattan = 3
        tile = Square(side_length=cell * 0.8, fill_color=TILE_BLUE,
                      fill_opacity=0.8, stroke_color=WHITE)
        tile_txt = Text("T", font_size=28, color=WHITE)
        tile_grp = VGroup(tile, tile_txt)

        target = Square(side_length=cell * 0.8, stroke_color=GOAL_GREEN,
                        stroke_width=3, fill_opacity=0)
        target_txt = Text("Goal", font_size=13, color=GOAL_GREEN)
        target_grp = VGroup(target, target_txt)

        start_pos = grid_bg[0].get_center()  # row0 col0
        goal_pos = grid_bg[5].get_center()   # row1 col2

        tile_grp.move_to(start_pos)
        target_grp.move_to(goal_pos)

        self.play(FadeIn(grid_bg), FadeIn(target_grp), FadeIn(tile_grp))
        self.wait(0.5)

        # Manhattan label
        md_label = MathTex(r"|0\!-\!1| + |0\!-\!2| = 3", font_size=24, color=YELLOW)
        md_label.shift(DOWN * 2.2)
        self.play(Write(md_label))
        self.wait(0.5)

        # Path 1: right → right → down  (3 moves, the minimum)
        p1 = Text("Path 1:  right → right → down  =  3 moves", font_size=18, color=YELLOW)
        p1.shift(DOWN * 2.8)
        self.play(Write(p1))

        pos_01 = grid_bg[1].get_center()
        pos_02 = grid_bg[2].get_center()

        self.play(tile_grp.animate.move_to(pos_01), run_time=0.35)
        self.play(tile_grp.animate.move_to(pos_02), run_time=0.35)
        self.play(tile_grp.animate.move_to(goal_pos), run_time=0.35)
        self.wait(0.6)

        # Path 2: right → down → right  (still 3 — can't do fewer)
        tile_grp.move_to(start_pos)
        p2 = Text("Path 2:  right → down → right  =  3 moves", font_size=18, color=TILE_BLUE)
        p2.shift(DOWN * 3.4)
        self.play(Write(p2))

        self.play(tile_grp.animate.move_to(pos_01), run_time=0.35)
        self.play(tile_grp.animate.move_to(grid_bg[4].get_center()), run_time=0.35)
        self.play(tile_grp.animate.move_to(goal_pos), run_time=0.35)
        self.wait(0.6)

        # Diagonal impossibility
        tile_grp.move_to(start_pos)
        diag = DashedLine(start_pos, goal_pos, color=REJECT_RED, stroke_width=3)
        x1 = Line(diag.get_center() + UL * 0.2, diag.get_center() + DR * 0.2,
                  color=REJECT_RED, stroke_width=4)
        x2 = Line(diag.get_center() + UR * 0.2, diag.get_center() + DL * 0.2,
                  color=REJECT_RED, stroke_width=4)
        self.play(FadeOut(p1), FadeOut(p2))
        self.play(Create(diag), FadeIn(VGroup(x1, x2)))

        no_diag = Text("No diagonal jumps — only slide.", font_size=18, color=REJECT_RED)
        no_diag.shift(DOWN * 2.8)
        self.play(Write(no_diag))
        self.wait(0.8)

        # Conclusion: admissibility in one sentence
        self.play(FadeOut(no_diag), FadeOut(md_label))
        conclusion = VGroup(
            Text("Manhattan distance = minimum moves for one tile.",
                 font_size=20, color=GOAL_GREEN),
            Text("Summing over all tiles still ≤ true cost (tiles don't block each other",
                 font_size=17, color=WHITE),
            Text("in the heuristic's model).  Never overestimates → admissible → A* is optimal.",
                 font_size=17, color=WHITE),
        ).arrange(DOWN, buff=0.15).shift(DOWN * 2.8)
        self.play(FadeIn(conclusion))
        self.wait(3)

    # ────────────────────────────────────────────────────────────
    # Scene 3 — "How does it compute the possible moves?"
    # ────────────────────────────────────────────────────────────

    def _scene_3_moves(self):
        self._clear()
        q = question_banner("How does it compute the possible moves?")
        self.play(Write(q))
        self.wait(0.8)

        # Main grid: blank in center (index 4)
        tiles = [1, 2, 3, 4, 0, 6, 7, 8, 5]
        main = create_grid(tiles, cell_size=0.85)
        main.shift(UP * 0.8)

        blank_label = Text("blank", font_size=14, color=GREY)
        blank_label.move_to(main[4].get_center())

        self.play(FadeIn(main), FadeIn(blank_label))
        self.wait(0.6)

        # Four directional arrows from the blank
        center = main[4].get_center()
        dirs = [UP, DOWN, LEFT, RIGHT]
        dir_names = ["up", "down", "left", "right"]
        swap_tiles = [
            [1, 0, 3, 4, 2, 6, 7, 8, 5],  # swap blank ↔ tile 2
            [1, 2, 3, 4, 8, 6, 7, 0, 5],  # swap blank ↔ tile 8
            [1, 2, 3, 0, 4, 6, 7, 8, 5],  # swap blank ↔ tile 4
            [1, 2, 3, 4, 6, 0, 7, 8, 5],  # swap blank ↔ tile 6
        ]
        swapped_with = [2, 8, 4, 6]

        arrows = VGroup()
        for d in dirs:
            a = Arrow(center, center + d * 0.55, color=YELLOW,
                      stroke_width=2.5, buff=0.12,
                      max_tip_length_to_length_ratio=0.35)
            arrows.add(a)
        self.play(*[GrowArrow(a) for a in arrows])
        self.play(FadeOut(blank_label))
        self.wait(0.3)

        # Show each resulting board
        child_positions = [
            LEFT * 4.5 + DOWN * 1.8,
            LEFT * 1.5 + DOWN * 1.8,
            RIGHT * 1.5 + DOWN * 1.8,
            RIGHT * 4.5 + DOWN * 1.8,
        ]
        for st, name, sw, pos in zip(swap_tiles, dir_names, swapped_with, child_positions):
            child = create_grid(st, cell_size=0.4)
            child.move_to(pos)
            lbl = Text(f"swap with {sw} ({name})", font_size=12, color=GREY)
            lbl.next_to(child, UP, buff=0.1)
            self.play(FadeIn(VGroup(child, lbl)), run_time=0.45)

        self.wait(0.5)

        # Edge case: blank at corner
        edge_note = Text(
            "At an edge or corner, some directions go off-grid → fewer moves.",
            font_size=17, color=WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(edge_note))
        self.wait(1)

        expl = Text(
            "Each valid swap produces a new board state = one neighbor in the search graph.",
            font_size=18, color=GOAL_GREEN,
        ).next_to(edge_note, UP, buff=0.2)
        self.play(Write(expl))
        self.wait(2.5)

    # ────────────────────────────────────────────────────────────
    # Scene 4 — "How does A* actually search?"
    # ────────────────────────────────────────────────────────────

    def _scene_4_astar_loop(self):
        self._clear()
        q = question_banner("How does A* actually search?")
        self.play(Write(q))
        self.wait(0.8)

        # Layout: queue on left, current board on right
        qtitle = Text("Priority Queue (min f first)", font_size=18, color=BLUE)
        qtitle.shift(LEFT * 4.5 + UP * 2.8)
        btitle = Text("Current Board", font_size=18, color=WHITE)
        btitle.shift(RIGHT * 2 + UP * 2.8)
        self.play(Write(qtitle), Write(btitle))

        # ── Iteration 1 ──────────────────────────────────────

        iter_label = Text("Iteration 1", font_size=16, color=GREY)
        iter_label.next_to(q, DOWN, buff=0.15)
        self.play(FadeIn(iter_label))

        # Queue has start state
        qvis = create_queue_visual([(9, "start  g=0")])
        qvis.next_to(qtitle, DOWN, buff=0.3)
        self.play(FadeIn(qvis))
        self.wait(0.3)

        # Step 1: POP
        step = Text("1. Pop lowest f from queue", font_size=18, color=YELLOW)
        step.to_edge(DOWN, buff=0.3)
        self.play(Write(step))

        board = create_grid([2, 8, 3, 1, 6, 4, 7, 0, 5], cell_size=0.6)
        board.shift(RIGHT * 2 + UP * 0.5)
        fl = MathTex(r"f = g + h = 0 + 9 = 9", font_size=22, color=YELLOW)
        fl.next_to(board, DOWN, buff=0.2)
        self.play(FadeIn(board), Write(fl))
        self.wait(0.8)

        # Step 2: GOAL CHECK
        step2 = Text("2. Is this the goal?  No.", font_size=18, color=RED)
        step2.to_edge(DOWN, buff=0.3)
        self.play(Transform(step, step2))
        self.wait(0.5)

        # Step 3: EXPAND
        step3 = Text("3. Generate neighbors (swap blank with each adjacent tile)",
                      font_size=17, color=TILE_BLUE)
        step3.to_edge(DOWN, buff=0.3)
        self.play(Transform(step, step3))

        # Two neighbors shown
        n1 = create_grid([2, 8, 3, 1, 0, 4, 7, 6, 5], cell_size=0.35)
        n2 = create_grid([2, 8, 3, 1, 6, 4, 0, 7, 5], cell_size=0.35)
        n3 = create_grid([2, 8, 3, 1, 6, 4, 7, 5, 0], cell_size=0.35)
        n1.shift(RIGHT * 0 + DOWN * 0.8)
        n2.shift(RIGHT * 2 + DOWN * 0.8)
        n3.shift(RIGHT * 4 + DOWN * 0.8)

        n1f = MathTex(r"f\!=\!1\!+\!7\!=\!8", font_size=14, color=GREEN
                      ).next_to(n1, DOWN, buff=0.08)
        n2f = MathTex(r"f\!=\!1\!+\!10\!=\!11", font_size=14, color=ORANGE
                      ).next_to(n2, DOWN, buff=0.08)
        n3f = MathTex(r"f\!=\!1\!+\!10\!=\!11", font_size=14, color=ORANGE
                      ).next_to(n3, DOWN, buff=0.08)

        self.play(FadeIn(n1), FadeIn(n2), FadeIn(n3),
                  Write(n1f), Write(n2f), Write(n3f), run_time=0.8)
        self.wait(0.8)

        # Step 4: COMPUTE f = g + h, INSERT INTO QUEUE
        step4 = Text("4. Compute f = g+h for each, insert into queue",
                      font_size=17, color=GOAL_GREEN)
        step4.to_edge(DOWN, buff=0.3)
        self.play(Transform(step, step4))

        new_q = create_queue_visual([
            (8, "A  g=1"), (11, "B  g=1"), (11, "C  g=1"),
        ])
        new_q.next_to(qtitle, DOWN, buff=0.3)
        self.play(Transform(qvis, new_q))
        self.wait(1)

        # ── Iteration 2 (brief) ──────────────────────────────

        iter2 = Text("Iteration 2", font_size=16, color=GREY)
        iter2.next_to(q, DOWN, buff=0.15)

        self.play(
            FadeOut(n1), FadeOut(n2), FadeOut(n3),
            FadeOut(n1f), FadeOut(n2f), FadeOut(n3f),
            FadeOut(board), FadeOut(fl),
            Transform(iter_label, iter2),
            Transform(step, Text("1. Pop f=8 (lowest)", font_size=18, color=YELLOW
                                  ).to_edge(DOWN, buff=0.3)),
        )

        board2 = create_grid([2, 8, 3, 1, 0, 4, 7, 6, 5], cell_size=0.6)
        board2.shift(RIGHT * 2 + UP * 0.5)
        fl2 = MathTex(r"f = 1 + 7 = 8", font_size=22, color=YELLOW)
        fl2.next_to(board2, DOWN, buff=0.2)
        self.play(FadeIn(board2), Write(fl2))
        self.wait(0.6)

        step_g = Text("2. Goal? No.  3. Expand.  4. Insert new neighbors.",
                       font_size=17, color=WHITE)
        step_g.to_edge(DOWN, buff=0.3)
        self.play(Transform(step, step_g))
        self.wait(0.8)

        # g-score check
        gcheck = Text(
            "g_scores check: if a neighbor was already reached\n"
            "with a lower g, skip it — we already know a better path.",
            font_size=16, color=GREY, line_spacing=1.3,
        ).shift(LEFT * 4.3 + DOWN * 1.2)
        self.play(Write(gcheck), run_time=1)
        self.wait(1.5)

        # ── Fast-forward to solution ──────────────────────────

        self.play(
            FadeOut(board2), FadeOut(fl2), FadeOut(gcheck),
            FadeOut(step), FadeOut(iter_label),
        )

        ff = Text("… the loop repeats: pop → check → expand → insert …",
                   font_size=20, color=BLUE)
        self.play(Write(ff))
        self.wait(1.2)

        ff2 = Text("until the popped state IS the goal.", font_size=20, color=BLUE)
        ff2.next_to(ff, DOWN, buff=0.2)
        self.play(Write(ff2))
        self.wait(1.5)

        # Show solved board
        self.play(FadeOut(ff), FadeOut(ff2), FadeOut(qvis), FadeOut(qtitle), FadeOut(btitle))

        solved = create_grid([1, 2, 3, 4, 5, 6, 7, 8, 0], cell_size=0.9)
        solved_lbl = Text("Goal reached — optimal path guaranteed.", font_size=22, color=GOAL_GREEN)
        solved_lbl.next_to(solved, DOWN, buff=0.5)
        self.play(FadeIn(solved), Write(solved_lbl))
        self.wait(1.5)

        # Final summary
        summary = VGroup(
            MathTex(r"f(n) = g(n) + h(n)", font_size=30, color=WHITE),
            Text("g = moves so far     h = Manhattan distance to goal",
                 font_size=18, color=GREY),
            Text("A* always expands the most promising state first.",
                 font_size=18, color=TILE_BLUE),
        ).arrange(DOWN, buff=0.2).shift(DOWN * 2.8)
        self.play(FadeIn(summary))
        self.wait(3)
