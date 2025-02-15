from typing import List, Tuple
from math import comb

def find_paths(grid_size: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int], max_step: int = 11) -> List[List[Tuple[int, int]]]:
    n, m = grid_size
    paths = []
    
    # Ajusta o ponto final caso esteja fora do alcance máximo
    dist_x = abs(end[0] - start[0])
    dist_y = abs(end[1] - start[1])
    new_end_x = start[0] + min(dist_x, max_step) * (1 if end[0] >= start[0] else -1)
    new_end_y = start[1] + min(dist_y, max_step) * (1 if end[1] >= start[1] else -1)
    end = (new_end_x, new_end_y)
    
    def get_directions(start, end):
        dx = 1 if start[0] < end[0] else -1 if start[0] > end[0] else 0
        dy = 1 if start[1] < end[1] else -1 if start[1] > end[1] else 0
        return [(dx, 0), (0, dy)]
    
    def is_valid(x, y, visited):
        return 0 <= x < n and 0 <= y < m and (x, y) not in visited
    
    def dfs(current_path):
        x, y = current_path[-1]
        
        if (x, y) == end:
            paths.append(current_path[:])
            return
        
        for dx, dy in get_directions(start, end):
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, set(current_path)):
                current_path.append((nx, ny))
                dfs(current_path)
                current_path.pop()
    
    dfs([start])
    return paths

def count_paths(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    return comb(dx + dy, dx)  # Número de caminhos possíveis usando combinatória

# Exemplo de uso
grid_size = (13, 26)  # Grid 3x3
start = (10, 17)      # Ponto inicial
end = (12, 0)        # Ponto final

start1 = (9, 8)      # Ponto inicial
end1 = (6, 22)        # Ponto final

num_paths = count_paths(start, end)
print(f"Número total de caminhos possíveis: {num_paths}")
paths = find_paths(grid_size, start, end)
print(paths)
for i, path in enumerate(paths):
    print(f"Caminho {i + 1}: {path}")