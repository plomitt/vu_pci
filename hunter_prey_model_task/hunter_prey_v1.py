import pygame as pg
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type # Import List and Type for type hinting
import matplotlib.pyplot as plt # Import matplotlib
import numpy as np # Import numpy for array operations
import sys # Import sys for console output

from vi import Agent, Config, Simulation
from pygame.math import Vector2

# --- Configuration for Lotka-Volterra Simulation ---
@dataclass
class LVConfig(Config):
    """
    Configuration settings for the Lotka-Volterra Predator-Prey simulation.
    This version focuses solely on the core Lotka-Volterra mechanics without energy.
    """
    # General Simulation Parameters
    movement_speed: float = 2.0
    radius: int = 50 # Perception radius for agents
    image_rotation: bool = True
    fps_limit: int = 60 # Frames per second limit
    duration: int = 2000 # Simulation duration in ticks, capped at 2000

    # --- Lotka-Volterra Specific Parameters ---
    # Rabbit parameters
    rabbit_reproduce_chance: float = 0.005  # Probability per tick for a rabbit to reproduce

    # Fox parameters
    fox_spontaneous_death_chance: float = 0.005 # Probability per tick for a fox to die spontaneously


# --- Agent Definitions ---

class Rabbit(Agent[LVConfig]):
    """
    Represents a Rabbit agent (Prey).
    Behaves passively, reproduces spontaneously, dies only if eaten by a fox.
    """
    def on_spawn(self):
        """Initializes rabbit movement."""
        self.move = Vector2(self.config.movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

    def change_position(self):
        """
        Rabbit movement and reproduction logic.
        """
        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # Random walk for rabbits: occasionally change direction slightly
        if self.shared.prng_move.random() < 0.1:
            self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
            self.move.normalize_ip()
            self.move *= self.config.movement_speed

        self.pos += self.move

        # --- Boundary Bouncing Logic for Rabbit ---
        if self.pos.x - agent_half_width < 0: # Hit left border
            self.pos.x = agent_half_width
            self.move.x *= -1
        elif self.pos.x + agent_half_width > window_width: # Hit right border
            self.pos.x = window_width - agent_half_width
            self.move.x *= -1

        if self.pos.y - agent_half_height < 0: # Hit top border
            self.pos.y = agent_half_height
            self.move.y *= -1
        elif self.pos.y + agent_half_height > window_height: # Hit bottom border
            self.pos.y = window_height - agent_half_height
            self.move.y *= -1

        # Spontaneous asexual reproduction for rabbits
        if self.shared.prng_move.random() < self.config.rabbit_reproduce_chance:
            new_rabbit_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
            Rabbit(images=self._images, simulation=self._Agent__simulation, pos=new_rabbit_pos)


class Fox(Agent[LVConfig]):
    """
    Represents a Fox agent (Predator).
    Hunts rabbits, reproduces by eating, dies spontaneously.
    """
    def on_spawn(self):
        """Initializes fox movement."""
        self.move = Vector2(self.config.movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

    def change_position(self):
        """
        Fox movement, hunting, reproduction (by eating), and death logic.
        """
        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        target_rabbit = None
        min_dist = float('inf')

        # Find the closest rabbit in proximity
        for neighbor, distance in self.in_proximity_accuracy():
            if isinstance(neighbor, Rabbit) and neighbor.is_alive():
                if distance < min_dist:
                    min_dist = distance
                    target_rabbit = neighbor

        if target_rabbit:
            direction_to_rabbit = (target_rabbit.pos - self.pos).normalize()
            self.move = direction_to_rabbit * self.config.movement_speed

            if self.pos.distance_to(target_rabbit.pos) < 10: # Close enough to eat
                target_rabbit.kill()
                
                new_fox_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
                Fox(images=self._images, simulation=self._Agent__simulation, pos=new_fox_pos)
        else:
            if self.shared.prng_move.random() < 0.1:
                self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
                self.move.normalize_ip()
                self.move *= self.config.movement_speed

        self.pos += self.move

        # --- Boundary Bouncing Logic for Fox ---
        if self.pos.x - agent_half_width < 0: # Hit left border
            self.pos.x = agent_half_width
            self.move.x *= -1
        elif self.pos.x + agent_half_width > window_width: # Hit right border
            self.pos.x = window_width - agent_half_width
            self.move.x *= -1

        if self.pos.y - agent_half_height < 0: # Hit top border
            self.pos.y = agent_half_height
            self.move.y *= -1
        elif self.pos.y + agent_half_height > window_height: # Hit bottom border
            self.pos.y = window_height - agent_half_height
            self.move.y *= -1

        # --- Spontaneous Death Logic for Foxes ---
        if self.shared.prng_move.random() < self.config.fox_spontaneous_death_chance:
            self.kill()


# --- Simulation Setup ---
pg.init() # Initialize Pygame

sim_config = LVConfig()

sim = Simulation(sim_config)

window_width, window_height = sim.config.window.as_tuple()

# Load agent images
rabbit_image_path = "images/rabbit.png" # Placeholder image for rabbit
fox_image_path = "images/fox.png"       # Placeholder image for fox

try:
    loaded_rabbit_image = pg.image.load(rabbit_image_path).convert_alpha()
    rabbit_images_list = [loaded_rabbit_image]
    loaded_fox_image = pg.image.load(fox_image_path).convert_alpha()
    fox_images_list = [loaded_fox_image]
except pg.error as e:
    print(f"Error loading images: {e}. Please ensure 'images/rabbit.png' and 'images/fox.png' exist.")
    exit()

# Initial population counts
initial_rabbits = 100
initial_foxes = 10

# Helper function for spawning agents in non-colliding positions
def spawn_agent_safely(agent_class: Type[Agent], images_list: List[pg.Surface], num_to_spawn: int, simulation_instance: Simulation):
    count = 0
    max_spawn_attempts_per_agent = 500
    
    dummy_agent = agent_class(images=images_list, simulation=simulation_instance, pos=Vector2(-100, -100))

    while count < num_to_spawn:
        attempts = 0
        is_valid_pos = False
        while attempts < max_spawn_attempts_per_agent:
            rand_x = random.randint(0, window_width)
            rand_y = random.randint(0, window_height)
            proposed_pos = Vector2(rand_x, rand_y)

            dummy_agent.pos = proposed_pos
            dummy_agent.rect.center = proposed_pos.xy

            is_colliding = False
            for obstacle in simulation_instance._obstacles:
                if pg.sprite.collide_mask(dummy_agent, obstacle):
                    is_colliding = True
                    break
            
            if not is_colliding:
                is_valid_pos = True
                break
            attempts += 1
        
        if is_valid_pos:
            agent_class(images=images_list, simulation=simulation_instance, pos=proposed_pos)
            count += 1
        else:
            print(f"Warning: Could not find non-colliding spawn position for a {agent_class.__name__}.")
            break
    
    dummy_agent.kill()
    return count

# Spawn initial rabbits
spawn_agent_safely(Rabbit, rabbit_images_list, initial_rabbits, sim)
# Spawn initial foxes
spawn_agent_safely(Fox, fox_images_list, initial_foxes, sim)


# --- Data Collection for Plots ---
rabbit_population_history: List[int] = []
fox_population_history: List[int] = []

# Store original tick method
original_sim_tick = sim.tick

# Patch the sim.tick method for data collection, termination, and console tick output
def custom_tick(self): # 'self' refers to the Simulation instance
    # 1. Event handling (from original tick method) - copied directly
    for event in pg.event.get():
        if event.type == pg.QUIT:
            self._running = False
            return # Exit tick method if user quits

    # 2. Collect current population data *before* any updates for this tick
    # This ensures we capture the state at the beginning of the frame,
    # and crucial for checking termination conditions before internal VI processing.
    current_rabbits = len([agent for agent in self._agents if isinstance(agent, Rabbit)])
    current_foxes = len([agent for agent in self._agents if isinstance(agent, Fox)])
    
    rabbit_population_history.append(current_rabbits)
    fox_population_history.append(current_foxes)

    # 3. Check for termination conditions based on population (after recording this frame's data).
    # If a termination condition is met, set self._running to False and return immediately.
    # This PREVENTS the original_sim_tick() (and thus its internal _metrics._merge())
    # from being called when populations are zero, which is the cause of the ShapeError.
    if current_foxes == 0:
        print(f"Simulation ended at tick {len(rabbit_population_history) -1}: All foxes died.")
        self._running = False # Stop the simulation loop
        return # Exit this custom tick method. The main sim.run() loop will terminate next.

    # 4. Print Current Tick Number to Console (replacing previous line)
    # Use len(rabbit_population_history) as the current tick count
    sys.stdout.write(f"\rCurrent Tick: {len(rabbit_population_history)}   ") # Spaces to clear previous line
    sys.stdout.flush()

    # 5. Call the original sim.tick() method to execute VI's standard per-tick logic.
    # This will handle clock updates, position updates, metrics updates, and drawing.
    # It will only be called if the termination condition above was NOT met.
    original_sim_tick()

# Apply the custom tick method by binding it correctly to the sim instance
sim.tick = custom_tick.__get__(sim, type(sim))


# Run the simulation
# sim.run() will now call our custom_tick method at each step, ensuring robust termination.
sim.run()

# --- Final Console Output Cleanup ---
sys.stdout.write("\n") # Move to a new line after the simulation finishes

# --- Cleanup after loop finishes ---
if pg.display.get_init():
    pg.display.flip() # Ensure final frame is shown if the loop exited prematurely
pg.quit() # Always quit pygame to free up resources.


# --- Plotting Collected Data at the End ---
# Adjust time array to match the actual number of collected history points
time = np.arange(len(rabbit_population_history))

plt.figure(figsize=(12, 7))

plt.plot(time, rabbit_population_history, label='Rabbits (Prey)', color='green', linewidth=2)
plt.plot(time, fox_population_history, label='Foxes (Predator)', color='red', linewidth=2, linestyle='--')

plt.title('Lotka-Volterra Population Dynamics Over Time', fontsize=16)
plt.xlabel('Time (Simulation Ticks)', fontsize=12)
plt.ylabel('Population Count', fontsize=12)

plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

if rabbit_population_history or fox_population_history:
    max_rabbit_pop = max(rabbit_population_history) if rabbit_population_history else 0
    max_fox_pop = max(fox_population_history) if fox_population_history else 0
    max_pop = max(max_rabbit_pop, max_fox_pop)
    plt.ylim(0, max_pop * 1.1 + 1)
else:
    plt.ylim(0, 10)

plt.tight_layout()

plt.savefig('population_dynamics.png')
plt.close()

print("\n--- Plot saved as 'population_dynamics.png' ---")
