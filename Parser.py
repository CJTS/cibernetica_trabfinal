from Game import GRID_HEIGHT, GRID_WIDTH, Tiles, Point

def file_to_grid(name):
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    player = None
    exit = None

    with open(name, 'r') as f:
        l = [[str(num) for num in line.replace("\n","")] for line in f]

    for idx, line in enumerate(l):
        for idy, char in enumerate(line):
            if(char == 'w'):
                grid[idx][idy] = Tiles.WALL
            elif(char == '.'):
                grid[idx][idy] = Tiles.DIRT
            elif(char == 'o'):
                grid[idx][idy] = Tiles.BOULDER
            elif(char == 'x'):
                grid[idx][idy] = Tiles.GEM
            elif(char == '-' or char == 'c' or char == 'b'):
                grid[idx][idy] = Tiles.EMPTY
            elif(char == 'e'):
                grid[idx][idy] = Tiles.EXIT
                exit = Point(idx, idy)
            elif(char == 'A'):
                grid[idx][idy] = Tiles.PLAYER
                player = Point(idx, idy)
    return grid, player, exit

if __name__ == '__main__':
    file_to_grid('levels/lvl1.txt')