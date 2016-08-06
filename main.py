from fishy import *
import yaml

config = yaml.safe_load(open("./config.yml"))

OCEANSIZE = config["oceansize"]
START = (config["startx"],config["starty"])
END = (OCEANSIZE-1, OCEANSIZE-1)
POP_SIZE    = config["popsize"]
GENERATIONS = config["generations"]
MUTATION_CHANCE = config["mutation_chance"]
DATA_PATH = config["data_path"]

if __name__ == "__main__":
    fittest_fish = evolve()
    ocean = fittest_fish.ocean
    openwater = []
    seaweed = []
    for row in range(OCEANSIZE):
        for col in range(OCEANSIZE):
            if ocean[row][col] == 1:
                seaweed.append((col, row, 200))
            else:
                openwater.append((col, row, 200))
    # x,y,s = zip(*openwater)
    # plt.scatter(x,y, color='red', s=s)
    x,y,s = zip(*seaweed)
    plt.scatter(x,y, color='green', s=s)
    x,y = zip(*fittest_fish.path)
    plt.plot(x,y)
    plt.title("Initial Pop: %d Ocean Size: %d Gens: %d Path Length: %d Collisions: %d"
        % (POP_SIZE, OCEANSIZE, GENERATIONS, fittest_fish.path_length(),  fittest_fish.seaweed()))
    plt.savefig(DATA_PATH+str(datetime.now().isoformat())+".png")
    exit(0)
