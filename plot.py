import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(title, attr, episode_durations):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(attr)
    plt.plot(durations_t.numpy())
    if is_ipython:
        display.display(plt.gcf())
        
data = [0.9499549865722656, 0.9449548721313477, 0.9561209678649902, 0.9379630088806152, 0.9558289051055908, 0.9735307693481445, 0.977928876876831, 0.9522120952606201, 0.9602160453796387, 0.9660170078277588, 0.9577510356903076, 0.9367270469665527, 0.9619998931884766, 0.9367830753326416, 0.95359206199646, 0.9165539741516113, 0.8913679122924805, 0.8926630020141602, 0.9088408946990967, 0.9291770458221436, 0.9046030044555664, 0.9251708984375, 0.9096362590789795, 0.8976719379425049, 0.886275053024292, 0.8905239105224609, 0.8903870582580566, 0.8915560245513916, 0.8867628574371338, 0.894428014755249]

plot_durations(f"Plan length of Random mode in level 1", "Plan length", data)
plt.ioff()
plt.show()
