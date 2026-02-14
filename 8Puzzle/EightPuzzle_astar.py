import os
import pygame
import heapq
import time
import sys

from Game_UI import SlidePuzzle

FPS = 60


def main():
    """
    The main function to run the game.
    """
    pygame.init()
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.display.set_caption("8-Puzzle game")
    screen = pygame.display.set_mode((800, 500))
    fpsclock = pygame.time.Clock()
    while True:
        puzzle = SlidePuzzle((3, 3), 160, 5, screen)
        choice = puzzle.selectPlayerMenu("8 Puzzle using A* search")
        if choice == "AI":
            puzzle.shuffle()
            playAIGame(puzzle, fpsclock)
        else:
            puzzle.playHumanGame(fpsclock)


def playAIGame(puzzle, fpsclock):
    """
    Play the game with AI.

    :param puzzle: The puzzle instance
    :param fpsclock: Track time.
    """
    finished = False

    # Save initial conf of game
    conf_init = puzzle.tiles[:]

    # Solve the game with A*
    start = time.time()
    path = iter(solveAI(puzzle))
    print("Exec time", time.time() - start)
    if path is None:
        print("Error, the AI did not find any solution.")
        pygame.quit()
        sys.exit()

    # reset state of puzzle
    puzzle.tiles = conf_init
    for h in range(len(puzzle.tiles)):
        puzzle.tilepos[h] = puzzle.tilePOS[puzzle.tiles[h]]

    while not finished and not puzzle.want_to_quit:
        dt = fpsclock.tick(FPS)
        puzzle.screen.fill((0, 0, 0))
        puzzle.draw()
        puzzle.drawShortcuts(False, None)
        pygame.display.flip()
        puzzle.catchGameEvents(False, lambda: puzzle.switch(next(path), True))
        puzzle.update(dt)

        finished = puzzle.checkGameState(True)


def solveAI(puzzle):
    """
    Implementation of the A* algorithm to solve the 8-puzzle game.

    :param puzzle: The puzzle instance.
                   puzzle.tiles can be read/written (list of 9 (x,y) tuples).
                   puzzle.isWin() returns True when puzzle.tiles matches the goal.
    :return:       A list of blank-tile positions tracing the solution path.
                   e.g. [(2,2), (1,2), (1,1), ...] — each entry is where the blank
                   moves to at each step.

    Available helpers:
        - heapq.heappush(queue, (priority, state, path))
        - heapq.heappop(queue) -> (priority, state, path)
        - moves(puzzle) -> list of neighbor configurations (set puzzle.tiles first!)
        - heuristic(puzzle, config) -> int (Manhattan distance for a configuration)
    """
    start_tiles = puzzle.tiles[:]
    queue = []
    heapq.heappush(queue, (heuristic(puzzle, start_tiles), start_tiles, [start_tiles[-1]]))
    g_scores = {str(start_tiles): 0}

    while queue:
        f_score, current_config, path = heapq.heappop(queue)
        puzzle.tiles = current_config

        if puzzle.isWin():
            return path

        for next_config in moves(puzzle):
            g = g_scores[str(current_config)] + 1
            if str(next_config) not in g_scores or g < g_scores[str(next_config)]:
                g_scores[str(next_config)] = g
                f = g + heuristic(puzzle, next_config)
                heapq.heappush(queue, (f, next_config, path + [next_config[-1]]))


def moves(puzzle):
    """
    Compute the accessible configurations from the current one with the possible moves.

    :param puzzle: The puzzle instance.
                   puzzle.tiles is the current configuration (list of 9 (x,y) tuples).
                   puzzle.tiles[-1] is the blank tile position.
                   puzzle.adjacent() returns 4 neighbor positions (up/down/left/right of blank).
                   puzzle.inGrid(pos) checks if a position is within the 3x3 grid.
                   puzzle.getBlank() returns the current blank position.
    :return:       A list of new configurations (each is a list of 9 (x,y) tuples),
                   one per valid move.
    """
    adjacent_positions = puzzle.adjacent()
    for pos in adjacent_positions:
        if puzzle.inGrid(pos):
            new_config = puzzle.tiles[:]
            blank_pos = puzzle.getBlank()
            blank_index = new_config.index(blank_pos)
            tile_index = new_config.index(pos)
            new_config[blank_index], new_config[tile_index] = new_config[tile_index], new_config[blank_index]
            yield new_config


def heuristic(puzzle, n):
    """
    Compute the Manhattan distance of all tiles corresponding to the heuristic used in the A* algorithm.

    :param puzzle: The puzzle instance.
                   puzzle.winCdt is the goal configuration: a list of 9 (x, y) tuples
                   representing where each tile SHOULD be.
    :param n:      A candidate configuration: a list of 9 (x, y) tuples
                   representing where each tile currently IS.
    :return:       Total Manhattan distance (int) summed across all 9 tiles.
    """
    
    dist = 0
    for i in range(9):
        current_x, current_y = n[i]
        goal_x, goal_y = puzzle.winCdt[i]
        dist += abs(current_x - goal_x) + abs(current_y - goal_y)
    return dist


if __name__ == "__main__":
    main()
