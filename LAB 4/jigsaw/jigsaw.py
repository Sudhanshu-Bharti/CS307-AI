import numpy as np
import matplotlib.pyplot as plt

# Load the scrambled image from the .mat file
f = open('scrambled.mat')

# Skip the first 5 lines
for i in range(5):
    f.readline()

m = []
line = f.readline()
# Read the values from the file until an empty line is reached
while line[1:] != '':
    val = int(line[1:])
    m.append(val)
    line = f.readline()

# Convert the list to a NumPy array
x = np.array(m)

# Ensure we reshape according to the actual size of the input data
size = int(np.sqrt(x.size))  # Calculate size based on the number of elements
mat = x.reshape(size, size).T  # Reshape and transpose the array

# Class definition for Energy calculation
class Energy:
    def __init__(self, image):
        self.image = image
        self.height = 4
        self.width = 4

    def getLeftRightEnergy(self, tile):
        try:
            i, j = tile
            x1 = 128 * i
            x2 = 128 * (i + 1)
            y = 128 * (j + 1) - 1
            diff = self.image[x1:x2, y] - self.image[x1:x2, y + 1]
            return np.sqrt((diff ** 2).mean())
        except IndexError:
            return 0

    def getUpDownEnergy(self, tile):
        try:
            i, j = tile
            y1 = 128 * j
            y2 = 128 * (j + 1)
            x = 128 * (i + 1) - 1
            diff = self.image[x, y1:y2] - self.image[x + 1, y1:y2]
            return np.sqrt((diff ** 2).mean())
        except IndexError:
            return 0

    def getEnergyAround(self, tile):
        i, j = tile
        e = np.zeros(4)
        e[0] = self.getLeftRightEnergy((i, j - 1))
        e[1] = self.getLeftRightEnergy((i, j))
        e[2] = self.getUpDownEnergy((i - 1, j))
        e[3] = self.getUpDownEnergy((i, j))
        return e.sum()

    def getEnergyAround2Tiles(self, t1, t2):
        return self.getEnergyAround(t1) + self.getEnergyAround(t2)

    def energy(self):
        energy = 0
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                energy += self.getEnergyAround((i, j))
        return energy

# Function to plot and save the current state of the puzzle
def plot_puzzle(image, iteration, energy):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Puzzle State at Iteration {iteration}\nEnergy: {energy:.2f}")
    plt.axis('off')
    plt.savefig(f'puzzle_state_iter_{iteration}.png')
    plt.close()

# Initialize Energy object
e = Energy(image=mat)
initial_energy = e.energy()

# Simulated Annealing parameters
max_iter = 100000
temp = 1000
stop_temp = 0.00005
decay = 0.9995
x = np.arange(0, 4)
y = np.arange(0, 4)
curr_iter = 0
best_cost = initial_energy
best = mat.copy()
cost_list = [best_cost]

# Plot interval (e.g., every 10000 iterations)
plot_interval = 10000

# Plot initial state
plot_puzzle(best, 0, best_cost)

# Simulated Annealing loop
while curr_iter < max_iter and temp > stop_temp:
    new = best.copy()
    np.random.shuffle(x)
    np.random.shuffle(y)
    
    cost_old = Energy(image=new).getEnergyAround2Tiles((x[0], y[0]), (x[1], y[1]))
    new[128*x[0]:128*x[0]+128, 128*y[0]:128*y[0]+128], new[128*x[1]:128*x[1]+128, 128*y[1]:128*y[1]+128] = \
        new[128*x[1]:128*x[1]+128, 128*y[1]:128*y[1]+128].copy(), new[128*x[0]:128*x[0]+128, 128*y[0]:128*y[0]+128].copy()
    
    cost_new = Energy(image=new).getEnergyAround2Tiles((x[0], y[0]), (x[1], y[1]))
    
    if cost_new < cost_old or np.random.rand() < np.exp(-abs(cost_old - cost_new) / temp):
        best = new.copy()
        best_cost = Energy(image=best).energy()
    
    temp *= decay
    curr_iter += 1
    cost_list.append(best_cost)
    
    # Print progress and plot puzzle state at fixed intervals
    if curr_iter % plot_interval == 0:
        print(f"Iteration: {curr_iter}, Best Cost: {best_cost}, Temperature: {temp:.6f}")
        plot_puzzle(best, curr_iter, best_cost)

# Plot the final solved image
plot_puzzle(best, curr_iter, best_cost)

# Plot the cost vs iteration graph
plt.figure(figsize=(10, 6))
plt.plot(cost_list)
plt.xlabel('Iterations')
plt.ylabel('Energy Cost')
plt.title('Energy Cost vs. Iterations')
plt.savefig('cost_vs_iterations.png')
plt.close()

print(f"Initial energy: {initial_energy}")
print(f"Final energy: {best_cost}")
print(f"Total iterations: {curr_iter}")
print("Solved jigsaw puzzle saved as 'puzzle_state_iter_{curr_iter}.png'")
print("Intermediate puzzle states saved as 'puzzle_state_iter_X.png'")
print("Cost vs. iterations graph saved as 'cost_vs_iterations.png'")