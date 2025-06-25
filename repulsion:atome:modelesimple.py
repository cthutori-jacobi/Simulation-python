import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramètres de la simulation
L = 20.0         # Taille du carré 
num_particles = 50
dt = 0.01        
T = 15.0         # Temps total de simulation
min_distance = 3.0  # Distance minimale 

# Paramètres physiques
mass = 1.0       
k_rep = 100      # Constante de force répulsive
speed = 5.0     

# Positions et vitesses initiales
positions = np.zeros((num_particles, 2))
velocities = np.zeros((num_particles, 2))
for i in range(num_particles):
    # Sélection d'un bord au hasard
    side = np.random.choice(['left', 'right', 'top', 'bottom'])
    if side == 'left':
        positions[i, 0] = 0.5
        positions[i, 1] = np.random.uniform(0.5, L - 0.5)
        direction = np.array([1.0, np.random.uniform(-1, 1)])
    elif side == 'right':
        positions[i, 0] = L - 0.5
        positions[i, 1] = np.random.uniform(0.5, L - 0.5)
        direction = np.array([-1.0, np.random.uniform(-1, 1)])
    elif side == 'top':
        positions[i, 1] = L - 0.5
        positions[i, 0] = np.random.uniform(0.5, L - 0.5)
        direction = np.array([np.random.uniform(-1, 1), -1.0])
    else:  # bottom
        positions[i, 1] = 0.5
        positions[i, 0] = np.random.uniform(0.5, L - 0.5)
        direction = np.array([np.random.uniform(-1, 1), 1.0])
    # Normalisation de la direction
    velocities[i] = speed * direction / np.linalg.norm(direction)

# Stockage des positions pour tracer les trajectoires
positions_over_time = []

def compute_forces(pos):
    forces = np.zeros_like(pos)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if 0 < dist < min_distance:
                # Force de répulsion type ressort linéaire
                force_mag = k_rep * (min_distance - dist)
                force_dir = diff / dist
                force_vec = force_mag * force_dir
                forces[i] -= force_vec
                forces[j] += force_vec
    return forces

# Boucle de simulation
num_steps = int(T / dt)
for step in range(num_steps):
    forces = compute_forces(positions)
    accelerations = forces / mass
    velocities += accelerations * dt
    positions += velocities * dt

    
    for i in range(num_particles):
        if positions[i, 0] <= 0 or positions[i, 0] >= L:
            velocities[i, 0] *= -1
            positions[i, 0] = np.clip(positions[i, 0], 0, L)
        if positions[i, 1] <= 0 or positions[i, 1] >= L:
            velocities[i, 1] *= -1
            positions[i, 1] = np.clip(positions[i, 1], 0, L)
    
    positions_over_time.append(positions.copy())

# Génération de couleurs pour chaque particule avec une colormap
colors = plt.cm.jet(np.linspace(0, 1, num_particles))

#l'animation
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_title("Animation de 50 billes avec force répulsive")

# Points et traces pour chaque particule
scatters = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(num_particles)]
paths = [ax.plot([], [], '-', color=colors[i], alpha=0.5)[0] for i in range(num_particles)]

def init():
    for scatter in scatters:
        scatter.set_data([], [])
    for path in paths:
        path.set_data([], [])
    return scatters + paths

def animate(frame):
    current_positions = positions_over_time[frame]
    for i in range(num_particles):
        scatters[i].set_data(current_positions[i, 0], current_positions[i, 1])
        # Reconstruction de la trajectoire jusqu'à la frame courante
        traj = np.array([pos[i] for pos in positions_over_time[:frame+1]])
        paths[i].set_data(traj[:, 0], traj[:, 1])
    return scatters + paths

ani = animation.FuncAnimation(fig, animate,
                              frames=len(positions_over_time),
                              interval=dt*1000,
                              blit=False,
                              init_func=init)

plt.show()

# Affichage trajectoires finales
fig2, ax2 = plt.subplots()
ax2.set_xlim(0, L)
ax2.set_ylim(0, L)
ax2.set_title("Trajectoires finales de 50 billes")

for i in range(num_particles):
    traj = np.array([pos[i] for pos in positions_over_time])
    ax2.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.5)
    ax2.plot(traj[-1, 0], traj[-1, 1], 'o', color=colors[i])

plt.show()