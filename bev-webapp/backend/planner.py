import heapq
import numpy as np


def astar(grid, cost_map, start, goal):
    h, w = grid.shape

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    directions = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy

            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue

            if grid[nx, ny] == 1:
                continue

            forward_bonus = (current[0] - nx) * 0.4
            turn_penalty = 0.2 if (dx, dy) not in [(0, -1), (-1, 0)] else 0
            move_cost = cost_map[nx, ny] - forward_bonus + turn_penalty

            temp_g = g_score[current] + move_cost

            if (nx, ny) not in g_score or temp_g < g_score[(nx, ny)]:
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = temp_g

                f_score = temp_g + heuristic((nx, ny), goal)
                heapq.heappush(open_set, (f_score, (nx, ny)))

    return []


def smooth_path(path):
    smoothed = []
    for i in range(len(path)):
        if i == 0 or i == len(path) - 1:
            smoothed.append(path[i])
        else:
            x = int((path[i-1][0] + path[i][0] + path[i+1][0]) / 3)
            y = int((path[i-1][1] + path[i][1] + path[i+1][1]) / 3)
            smoothed.append((x, y))
    return smoothed


def destination_point(destination, h, w):
    mapping = {
        "forward": (0, w // 2),
        "left": (h // 2, 0),
        "right": (h // 2, w - 1),
        "top centre": (0, w // 2),
        "bottom centre": (h - 1, w // 2),
        "left centre": (h // 2, 0),
        "right centre": (h // 2, w - 1),
        "top left corner": (0, 0),
        "top right corner": (0, w - 1),
        "bottom left corner": (h - 1, 0),
        "bottom right corner": (h - 1, w - 1),
    }
    return mapping.get(destination.lower(), (0, w // 2))


def plan_path(binary, risk_map, destination="top centre"):
    h, w = binary.shape

    start = (h // 2, w // 2)
    primary_goal = destination_point(destination, h, w)

    goal_offsets = [
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    best_path = []
    best_cost = float("inf")

    for dx, dy in goal_offsets:
        goal = (primary_goal[0] + dx, primary_goal[1] + dy)
        if goal[0] < 0 or goal[1] < 0 or goal[0] >= h or goal[1] >= w:
            continue
        if binary[goal[0], goal[1]] == 1:
            continue

        path = astar(binary, risk_map, start, goal)
        if not path:
            continue

        cost = sum(risk_map[x, y] for (x, y) in path)
        if cost < best_cost:
            best_cost = cost
            best_path = path

    if not best_path:
        return []

    return smooth_path(best_path)