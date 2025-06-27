import pygame as pg
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type # Import List and Type for type hinting
import matplotlib.pyplot as plt # Import matplotlib
import numpy as np # Import numpy for array operations
import sys # Import sys for console output
import os # Import os for file system operations
import pandas as pd # Import pandas for CSV saving
import json # Import json for saving metrics
from itertools import product # Import product for grid search combinations
from matplotlib.gridspec import GridSpec

from vi import Agent, Config, HeadlessSimulation, Simulation
from pygame.math import Vector2

# --- ANSI Escape Codes for Pretty Printing ---
# These codes allow you to change text color and style in the terminal.
# They might not work in all environments (e.g., some IDE output windows),
# but are standard for most modern terminals.
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
CYAN = "\033[96m"

# --- Scenario Name Constants ---
ENERGY_FREE_SCENARIO_NAME = "energy_free"
ENERGY_ENABLED_SCENARIO_NAME = "energy_enabled"

# --- Configuration for Lotka-Volterra Simulation ---
@dataclass
class LVConfig(Config):
    """
    Configuration settings for the Lotka-Volterra Predator-Prey simulation.
    """
    # General Simulation Parameters
    # The 'radius' attribute on the base Config is assumed to be the default/global one
    # and will be dynamically set by agents when they operate.
    image_rotation: bool = True
    fps_limit: int = 60 # Frames per second limit
    duration: int = 2000 # Simulation duration in ticks, capped at 2000

    # --- Lotka-Volterra Specific Parameters ---
    # Movement speeds for rabbits and foxes
    rabbit_movement_speed: float = 2.0  # Separate movement speed for rabbits
    fox_movement_speed: float = 3.0     # Separate movement speed for foxes

    # NEW: Separate perception radii for rabbits and foxes
    rabbit_perception_radius: int = 50
    fox_perception_radius: int = 70 # Foxes can see a bit further
    grass_perception_radius: int = 30 # Rabbits can see grass a bit closer

    rabbit_reproduce_chance: float = 0.005  # Probability per tick for a rabbit to reproduce

    # Fox parameters
    fox_spontaneous_death_chance: float = 0.005 # Probability per tick for a fox to die spontaneously
    
    # --- Energy System Parameters (Toggleable) ---
    energy_enabled: bool = False # This will be set by the experiment runner

    # Fox Energy Parameters
    fox_initial_energy: float = 100.0     # Starting energy for foxes
    fox_energy_decay_rate: float = 0.5    # Energy lost per tick by a fox
    rabbit_energy_gain_for_fox: float = 10.0 # Energy gained by fox when eating a rabbit
    fox_starvation_threshold: float = 0.0 # Energy level at which a fox dies

    # NEW: Rabbit Energy Parameters
    rabbit_initial_energy: float = 50.0   # Starting energy for rabbits
    rabbit_energy_decay_rate: float = 0.2 # Energy lost per tick by a rabbit
    grass_energy_gain_for_rabbit: float = 5.0 # Energy gained by rabbit when eating grass
    rabbit_starvation_threshold: float = 0.0 # Energy level at which a rabbit dies

    # NEW: Grass Parameters
    initial_grass: int = 200 # Initial number of grass patches
    grass_regrowth_time: int = 100 # Ticks for grass to respawn after one is eaten


# --- Agent Definitions ---

class Grass(Agent[LVConfig]):
    """
    Represents a Grass agent (Food for Rabbits).
    Is static, can be eaten, and is removed when eaten.
    The simulation handles respawning new grass randomly.
    """
    def on_spawn(self):
        """Initializes grass state. Grass does not move."""
        self.move = Vector2(0, 0) # Grass agents are static.
        # Grass has no perception radius as it doesn't interact proactively.
        # It's only detected by rabbits.
        self.config.radius = 0 

    def change_position(self):
        """
        Grass agents are static and do not change their own position or state.
        Their lifecycle (being eaten and replaced) is managed by Rabbit agents
        and the main simulation loop.
        """
        pass # Grass does not move or have internal logic for regrowth


class Rabbit(Agent[LVConfig]):
    """
    Represents a Rabbit agent (Prey).
    Behaves passively, reproduces spontaneously, dies if eaten by a fox or starves.
    """
    energy: float = 0.0 # Rabbit's energy level (only used if energy_enabled is True)

    def on_spawn(self):
        """Initializes rabbit movement, energy, and sets its specific perception radius."""
        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        if self.config.energy_enabled:
            self.energy = self.config.rabbit_initial_energy
        # Set the radius attribute of the config object specific to this agent type
        # This relies on sequential processing of agents by vi
        self.config.radius = self.config.rabbit_perception_radius

    def change_position(self):
        """
        Rabbit movement, eating grass (if energy enabled), reproduction logic, and starvation.
        """
        # Ensure the correct radius is active for this agent's proximity checks
        self.config.radius = self.config.rabbit_perception_radius

        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # --- Energy Decay for Rabbit (only if energy system is enabled) ---
        if self.config.energy_enabled:
            self.energy -= self.config.rabbit_energy_decay_rate # Energy decays
            if self.energy <= self.config.rabbit_starvation_threshold:
                self.kill() # Rabbit dies of starvation
                return # Stop further actions if dead

        # --- Find and Eat Grass (only if energy system is enabled) ---
        target_grass = None
        if self.config.energy_enabled: # Only look for grass if energy system is active
            min_dist_grass = float('inf')

            # Temporarily set radius to grass_perception_radius for finding grass
            original_radius = self.config.radius
            self.config.radius = self.config.grass_perception_radius

            for neighbor, distance in self.in_proximity_accuracy():
                # Check if it's a Grass agent and if it's still "alive" (not killed yet)
                if isinstance(neighbor, Grass) and neighbor.is_alive():
                    if distance < min_dist_grass:
                        min_dist_grass = distance
                        target_grass = neighbor
            
            # Revert radius back to rabbit_perception_radius for other interactions
            self.config.radius = original_radius

            if target_grass:
                # Move towards grass
                direction_to_grass = (target_grass.pos - self.pos).normalize()
                self.move = direction_to_grass * self.config.rabbit_movement_speed

                # Check if close enough to eat
                if self.pos.distance_to(target_grass.pos) < 10: 
                    # Grass is eaten, so kill it. A new one will be spawned by the simulation later.
                    target_grass.kill()
                    
                    self.energy += self.config.grass_energy_gain_for_rabbit
                    # Cap energy at a maximum to prevent infinite growth
                    self.energy = min(self.energy, self.config.rabbit_initial_energy * 2) # Example cap at 2x initial
        
        # If no grass is found (or energy system is disabled), do random walk
        if not target_grass:
            if self.shared.prng_move.random() < 0.1:
                self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
                self.move.normalize_ip()
                self.move *= self.config.rabbit_movement_speed # Use rabbit-specific speed

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

        # Spontaneous asexual reproduction for rabbits (can be tied to energy/health later)
        if self.shared.prng_move.random() < self.config.rabbit_reproduce_chance:
            new_rabbit_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
            Rabbit(images=self._images, simulation=self._Agent__simulation, pos=new_rabbit_pos)


class Fox(Agent[LVConfig]):
    """
    Represents a Fox agent (Predator).
    Hunts rabbits, reproduces by eating, dies spontaneously or from starvation.
    """
    energy: float = 0.0 # Fox's energy level (only used if energy_enabled is True)

    def on_spawn(self):
        """Initializes fox movement, energy, and sets its specific perception radius."""
        self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        if self.config.energy_enabled:
            self.energy = self.config.fox_initial_energy
        # Set the radius attribute of the config object specific to this agent type
        # This relies on sequential processing of agents by vi
        self.config.radius = self.config.fox_perception_radius

    def change_position(self):
        """
        Fox movement, hunting, reproduction (by eating), and death logic.
        """
        # Ensure the correct radius is active for this agent's proximity checks
        self.config.radius = self.config.fox_perception_radius

        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # --- Energy Decay for Fox (only if energy system is enabled) ---
        if self.config.energy_enabled:
            self.energy -= self.config.fox_energy_decay_rate # Energy decays
            if self.energy <= self.config.fox_starvation_threshold:
                self.kill() # Fox dies of starvation
                return # Stop further actions if dead

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
            self.move = direction_to_rabbit * self.config.fox_movement_speed # Use fox-specific speed

            if self.pos.distance_to(target_rabbit.pos) < 10: # Close enough to eat
                target_rabbit.kill()
                
                # Fox reproduces directly after eating a rabbit
                new_fox_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
                Fox(images=self._images, simulation=self._Agent__simulation, pos=new_fox_pos)

                # Replenish energy if energy system is enabled
                if self.config.energy_enabled:
                    self.energy += self.config.rabbit_energy_gain_for_fox
                    # Cap energy at a maximum to prevent infinite growth
                    self.energy = min(self.energy, self.config.fox_initial_energy * 2) # Example cap at 2x initial
        else:
            if self.shared.prng_move.random() < 0.1:
                self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
                self.move.normalize_ip()
                self.move *= self.config.fox_movement_speed # Use fox-specific speed

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

        # --- Spontaneous Death Logic (only if energy system is NOT enabled) ---
        # Starvation death is handled at the beginning of change_position if energy is enabled.
        if not self.config.energy_enabled:
            if self.shared.prng_move.random() < self.config.fox_spontaneous_death_chance:
                self.kill() # Fox dies spontaneously

# --- Helper function for spawning agents in non-colliding positions ---
def spawn_agent_safely(agent_class: Type[Agent], images_list: List[pg.Surface], num_to_spawn: int, simulation_instance: Simulation):
    count = 0
    max_spawn_attempts_per_agent = 500
    
    # Get window dimensions here, as it's needed for random positions
    window_width, window_height = simulation_instance.config.window.as_tuple()

    # Create a dummy agent just to get its size for collision checking
    dummy_agent = agent_class(images=images_list, simulation=simulation_instance, pos=Vector2(-100, -100))
    dummy_agent_half_width = dummy_agent.rect.width / 2
    dummy_agent_half_height = dummy_agent.rect.height / 2

    while count < num_to_spawn:
        attempts = 0
        is_valid_pos = False
        while attempts < max_spawn_attempts_per_agent:
            # Generate random position, ensuring the whole agent fits within bounds
            rand_x = random.randint(int(dummy_agent_half_width), int(window_width - dummy_agent_half_width))
            rand_y = random.randint(int(dummy_agent_half_height), int(window_height - dummy_agent_half_height))
            proposed_pos = Vector2(rand_x, rand_y)

            dummy_agent.pos = proposed_pos
            dummy_agent.rect.center = proposed_pos.xy

            is_colliding = False
            # Check collision with existing agents and obstacles
            for existing_agent in simulation_instance._agents:
                if existing_agent is not dummy_agent and pg.sprite.collide_mask(dummy_agent, existing_agent):
                    is_colliding = True
                    break
            
            if not is_colliding:
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
            print(f"Warning: Could not find non-colliding spawn position for a {agent_class.__name__} after {max_spawn_attempts_per_agent} attempts. Spawning fewer agents than requested.")
            break
    
    dummy_agent.kill() # Remove dummy agent
    return count

# --- Experiment Runner Function ---
def run_experiment(
    run_name: str, 
    base_output_dir: str,
    initial_rabbits: int, 
    initial_foxes: int, 
    initial_grass: int, 
    config_overrides: dict,
    current_sim_index: int, 
    total_sims_in_scenario: int 
):
    """
    Runs a single simulation experiment with specified parameters and saves results.

    Args:
        run_name (str): A unique name for this specific run, used for folder creation.
        base_output_dir (str): The base directory to save results (e.g., "simulation_results_energy/baseline_energy_free").
        initial_rabbits (int): Initial number of rabbit agents.
        initial_foxes (int): Initial number of fox agents.
        initial_grass (int): Initial number of grass agents (only used if energy_enabled is True).
        config_overrides (dict): A dictionary of LVConfig parameters to override defaults.
        current_sim_index (int): The current index of the simulation in the grid search (1-based).
        total_sims_in_scenario (int): The total number of simulations in the current grid search scenario.
    """
    # Print simulation progress with styling
    print(f"\n{GREEN}{BOLD}--- Running simulation {current_sim_index}/{total_sims_in_scenario}: {run_name} ---{RESET}")
    print(f"{CYAN}Parameters:{RESET}")
    # Pretty print parameters with only numbers highlighted
    for key, value in config_overrides.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {BOLD}{CYAN}{value}{RESET}")
        else:
            print(f"  {key}: {CYAN}{value}{RESET}")

    # Create the specific output directory for this run
    run_output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    

    # Reset population history for each run
    rabbit_population_history = []
    fox_population_history = []
    grass_population_history = [] 
    fox_energy_history = [] 
    rabbit_energy_history = [] 

    # Apply config overrides
    current_config = LVConfig()
    for key, value in config_overrides.items():
        setattr(current_config, key, value)

    sim = HeadlessSimulation(current_config)

    # Load agent images
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    rabbit_image_path = os.path.join(script_dir, "images", "rabbit.png")
    fox_image_path = os.path.join(script_dir, "images", "fox.png")
    grass_image_path = os.path.join(script_dir, "images", "grass.png") 
    
    try:
        loaded_rabbit_image = pg.image.load(rabbit_image_path).convert_alpha()
        rabbit_images_list = [loaded_rabbit_image]
        loaded_fox_image = pg.image.load(fox_image_path).convert_alpha()
        fox_images_list = [loaded_fox_image]
        loaded_grass_image = pg.image.load(grass_image_path).convert_alpha() 
        grass_images_list = [loaded_grass_image] 
    except pg.error as e:
        print(f"{YELLOW}Error loading images: {e}. Please ensure 'images/rabbit.png', 'images/fox.png', and 'images/grass.png' exist in the script directory.{RESET}")
        print(f"{YELLOW}Creating dummy images for agents to continue simulation.{RESET}")
        dummy_rabbit_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_rabbit_surf, (0, 200, 0), (10, 10), 10) # Green circle
        rabbit_images_list = [dummy_rabbit_surf]

        dummy_fox_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_fox_surf, (200, 0, 0), (10, 10), 10) # Red circle
        fox_images_list = [dummy_fox_surf]

        dummy_grass_surf = pg.Surface((15, 15), pg.SRCALPHA) 
        pg.draw.rect(dummy_grass_surf, (0, 150, 0), (0,0,15,15)) # Dark green square 
        grass_images_list = [dummy_grass_surf] 


    # Spawn initial populations
    spawn_agent_safely(Rabbit, rabbit_images_list, initial_rabbits, sim)
    spawn_agent_safely(Fox, fox_images_list, initial_foxes, sim)
    
    # NEW: Conditionally spawn grass
    if current_config.energy_enabled:
        spawn_agent_safely(Grass, grass_images_list, initial_grass, sim) 
        sim._grass_images_list = grass_images_list # Attach for respawn logic
        sim._grass_respawn_counter = 0 # Initialize respawn counter for this simulation
    else:
        # Ensure these are not set if energy_enabled is False to prevent issues
        sim._grass_images_list = [] # Empty list if no grass
        sim._grass_respawn_counter = 0


    # Store original tick method
    original_sim_tick = sim.tick

    # Patch the sim.tick method for data collection, termination, and console tick output
    def custom_tick_for_run(self): # 'self' refers to the Simulation instance
        # 1. Event handling (from original tick method)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self._running = False
                return # Exit tick method if user quits

        # 2. Collect current population data *before* any updates for this tick
        current_rabbits = len([agent for agent in self._agents if isinstance(agent, Rabbit)])
        current_foxes = len([agent for agent in self._agents if isinstance(agent, Fox)])
        
        # Only count grass if energy_enabled is True, otherwise it's effectively zero.
        current_grass = len([agent for agent in self._agents if isinstance(agent, Grass)]) if self.config.energy_enabled else 0
        
        # Always append current population to history
        rabbit_population_history.append(current_rabbits)
        fox_population_history.append(current_foxes)
        grass_population_history.append(current_grass) 

        # Collect energy history only if enabled
        if self.config.energy_enabled:
            live_foxes = [agent for agent in self._agents if isinstance(agent, Fox) and agent.is_alive()]
            avg_fox_energy = sum(fox.energy for fox in live_foxes) / len(live_foxes) if live_foxes else 0.0
            fox_energy_history.append(avg_fox_energy)

            live_rabbits = [agent for agent in self._agents if isinstance(agent, Rabbit) and agent.is_alive()]
            avg_rabbit_energy = sum(rabbit.energy for rabbit in live_rabbits) / len(live_rabbits) if live_rabbits else 0.0
            rabbit_energy_history.append(avg_rabbit_energy)
        else:
            # If energy is not enabled, append 0 to maintain consistent list length for plotting
            fox_energy_history.append(0.0)
            rabbit_energy_history.append(0.0)


        # 3. Check for termination conditions based on population (after recording this frame's data).
        if current_foxes == 0 or current_rabbits == 0:
            extinction_type = 'foxes' if current_foxes == 0 else 'rabbits'
            print(f"\n{RED}Simulation ended at tick {BOLD}{len(rabbit_population_history) - 1}{RESET}{RED}: All {extinction_type} died.{RESET}")
            self._running = False 
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return 
        
        # Also check if max duration is reached.
        if len(rabbit_population_history) > self.config.duration:
            print(f"\n{YELLOW}Simulation ended at tick {BOLD}{len(rabbit_population_history) - 1}{RESET}{YELLOW}: Max duration reached.{RESET}")
            self._running = False
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return

        # 4. Print Current Tick Number to Console (replacing previous line)
        sys.stdout.write(f"\rCurrent Tick: {BOLD}{len(rabbit_population_history)}{RESET}   ") # Spaces to clear previous line
        sys.stdout.flush()

        # 5. Call the original sim.tick() method to execute VI's standard per-tick logic.
        # This will process agent movements and apply 'kill()' actions from the previous tick.
        original_sim_tick()

        # 6. NEW: Grass Respawn Logic (only if energy system is enabled)
        if self.config.energy_enabled:
            current_grass_after_tick = len([agent for agent in self._agents if isinstance(agent, Grass)])
            if current_grass_after_tick < self.config.initial_grass:
                self._grass_respawn_counter += 1

                if self._grass_respawn_counter >= self.config.grass_regrowth_time:
                    # Spawn one grass agent if we are below the initial_grass limit
                    spawn_agent_safely(Grass, self._grass_images_list, 1, self)
                    self._grass_respawn_counter = 0 # Reset counter after spawning
            else:
                # If grass is at or above initial_grass, reset counter to avoid premature respawn
                self._grass_respawn_counter = 0


    # Apply the custom tick method by binding it correctly to the sim instance
    sim.tick = custom_tick_for_run.__get__(sim, type(sim))

    # Run the simulation
    sim.run()

    # --- Final Console Output Cleanup for this run ---
    sys.stdout.write("\n") # Move to a new line after the simulation finishes

    # --- Saving Results ---
    # Determine extinction ticks and peak populations
    ticks_completed = len(rabbit_population_history) - 1
    
    # Corrected extinction tick logic: If population is not 0 at the end, it did not go extinct.
    extinction_tick_rabbits = ticks_completed if rabbit_population_history and rabbit_population_history[-1] == 0 else "N/A"
    extinction_tick_foxes = ticks_completed if fox_population_history and fox_population_history[-1] == 0 else "N/A"
    
    peak_rabbits = max(rabbit_population_history) if rabbit_population_history else 0
    peak_foxes = max(fox_population_history) if fox_population_history else 0
    # Peak grass should be 0 if energy_enabled is false
    peak_grass = max(grass_population_history) if grass_population_history and current_config.energy_enabled else 0 


    # Save general metrics to JSON
    run_metrics = {
        'run_name': run_name,
        'parameters': config_overrides,
        'ticks_completed': ticks_completed,
        'final_rabbit_population': rabbit_population_history[-1] if rabbit_population_history else 0,
        'final_fox_population': fox_population_history[-1] if fox_population_history else 0,
        'final_grass_population': grass_population_history[-1] if grass_population_history and current_config.energy_enabled else 0, 
        'extinction_tick_rabbits': extinction_tick_rabbits,
        'extinction_tick_foxes': extinction_tick_foxes,
        'peak_rabbits': peak_rabbits,
        'peak_foxes': peak_foxes,
        'peak_grass': peak_grass 
    }
    metrics_json_path = os.path.join(run_output_dir, f"{run_name}_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(run_metrics, f, indent=4)
    print(f"Run metrics saved to: {metrics_json_path}")


    # Save population data to CSV
    df_populations = pd.DataFrame({
        'Tick': np.arange(len(rabbit_population_history)),
        'Rabbits': rabbit_population_history,
        'Foxes': fox_population_history,
        'Grass': grass_population_history 
    })
    csv_path = os.path.join(run_output_dir, f"{run_name}_population_data.csv")
    df_populations.to_csv(csv_path, index=False)
    print(f"Population data saved to: {csv_path}")

    # Save population plot to PNG
    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(len(rabbit_population_history)), rabbit_population_history, label='Rabbits (Prey)', color='green', linewidth=2)
    plt.plot(np.arange(len(fox_population_history)), fox_population_history, label='Foxes (Predator)', color='red', linewidth=2, linestyle='--')
    # Only plot grass if energy is enabled
    if current_config.energy_enabled:
        plt.plot(np.arange(len(grass_population_history)), grass_population_history, label='Grass (Food)', color='darkgreen', linewidth=2, linestyle=':') 
    plt.title(f'Population Dynamics for {run_name}', fontsize=16)
    plt.xlabel('Time (Simulation Ticks)', fontsize=12)
    plt.ylabel('Population Count', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust ylim based on relevant populations
    max_pop = 0
    if rabbit_population_history:
        max_pop = max(max_pop, max(rabbit_population_history))
    if fox_population_history:
        max_pop = max(max_pop, max(fox_population_history))
    if current_config.energy_enabled and grass_population_history: # Only consider grass max if enabled
        max_pop = max(max_pop, max(grass_population_history))
    
    if max_pop > 0:
        plt.ylim(0, max_pop * 1.1 + 1)
    else:
        plt.ylim(0, 10) # Set a default small range if no data

    plt.tight_layout()
    plot_path = os.path.join(run_output_dir, f"{run_name}_population_dynamics.png")
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory
    print(f"Population plot saved to: {plot_path}")

    # Save Average Fox Energy data and plot (if energy is enabled)
    if current_config.energy_enabled: # Only save/plot if energy is enabled
        df_fox_energy = pd.DataFrame({
            'Tick': np.arange(len(fox_energy_history)),
            'Average_Fox_Energy': fox_energy_history
        })
        energy_csv_path = os.path.join(run_output_dir, f"{run_name}_avg_fox_energy_data.csv")
        df_fox_energy.to_csv(energy_csv_path, index=False)
        print(f"Average fox energy data saved to: {energy_csv_path}")

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(fox_energy_history)), fox_energy_history, label='Average Fox Energy', color='orange', linewidth=2)
        plt.title(f'Average Fox Energy Over Time for {run_name}', fontsize=16)
        plt.xlabel('Time (Simulation Ticks)', fontsize=12)
        plt.ylabel('Average Energy', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        energy_plot_path = os.path.join(run_output_dir, f"{run_name}_avg_fox_energy_dynamics.png")
        plt.savefig(energy_plot_path)
        plt.close()
        print(f"Average fox energy plot saved to: {energy_plot_path}")

        # Save Average Rabbit Energy data and plot (if energy is enabled)
        df_rabbit_energy = pd.DataFrame({
            'Tick': np.arange(len(rabbit_energy_history)),
            'Average_Rabbit_Energy': rabbit_energy_history
        })
        rabbit_energy_csv_path = os.path.join(run_output_dir, f"{run_name}_avg_rabbit_energy_data.csv")
        df_rabbit_energy.to_csv(rabbit_energy_csv_path, index=False)
        print(f"Average rabbit energy data saved to: {rabbit_energy_csv_path}")

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(rabbit_energy_history)), rabbit_energy_history, label='Average Rabbit Energy', color='purple', linewidth=2)
        plt.title(f'Average Rabbit Energy Over Time for {run_name}', fontsize=16)
        plt.xlabel('Time (Simulation Ticks)', fontsize=12)
        plt.ylabel('Average Energy', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        rabbit_energy_plot_path = os.path.join(run_output_dir, f"{run_name}_avg_rabbit_energy_dynamics.png")
        plt.savefig(rabbit_energy_plot_path)
        plt.close()
        print(f"Average rabbit energy plot saved to: {rabbit_energy_plot_path}")


    # Return all collected data for category-level aggregation, including histories
    return {
        'run_name': run_name,
        'config_overrides': config_overrides,
        'ticks_completed': ticks_completed,
        'extinction_tick_rabbits': extinction_tick_rabbits,
        'extinction_tick_foxes': extinction_tick_foxes,
        'peak_rabbits': peak_rabbits,
        'peak_foxes': peak_foxes,
        'peak_grass': peak_grass, # Will be 0 if energy_enabled is false
        'fox_energy_history': fox_energy_history, # Will be 0s if energy_enabled is false
        'rabbit_energy_history': rabbit_energy_history, # Will be 0s if energy_enabled is false
        'rabbit_population_history': rabbit_population_history, 
        'fox_population_history': fox_population_history, 
        'grass_population_history': grass_population_history # Will be 0s if energy_enabled is false
    }


# --- Function for Category-Level Analysis and Plotting ---
def analyze_category_results(results_list: List[dict], category_output_dir: str, category_name: str):
    """
    Analyzes and saves aggregated metrics for a given category of simulation runs.
    """
    print(f"\n{BLUE}{BOLD}--- Analyzing results for category: {category_name} ---{RESET}")

    # Initialize lists to collect data for averaging
    all_extinction_ticks_rabbits = []
    all_extinction_ticks_foxes = []
    all_peak_rabbits = []
    all_peak_foxes = []
    all_peak_grass = [] 
    all_fox_energy_histories = [] 
    all_rabbit_energy_histories = [] 
    all_rabbit_population_histories = [] 
    all_fox_population_histories = [] 
    all_grass_population_histories = [] 

    # Determine if this category has energy enabled based on the first result
    # Assuming all runs in a scenario have the same energy_enabled setting
    energy_enabled_for_category = results_list[0]['config_overrides'].get('energy_enabled', False) if results_list else False


    for result in results_list:
        # Collect scalar metrics
        if isinstance(result['extinction_tick_rabbits'], int):
            all_extinction_ticks_rabbits.append(result['extinction_tick_rabbits'])
        if isinstance(result['extinction_tick_foxes'], int):
            all_extinction_ticks_foxes.append(result['extinction_tick_foxes'])
        
        all_peak_rabbits.append(result['peak_rabbits'])
        all_peak_foxes.append(result['peak_foxes'])
        
        # Only collect peak grass if energy is enabled for the category
        if energy_enabled_for_category:
            all_peak_grass.append(result['peak_grass']) 

        # Collect time series histories only if energy is enabled for the category
        if energy_enabled_for_category:
            if result['fox_energy_history']:
                all_fox_energy_histories.append(result['fox_energy_history'])
            if result['rabbit_energy_history']: 
                all_rabbit_energy_histories.append(result['rabbit_energy_history']) 
        
        all_rabbit_population_histories.append(result['rabbit_population_history'])
        all_fox_population_histories.append(result['fox_population_history'])
        
        # Only collect grass population history if energy is enabled for the category
        if energy_enabled_for_category:
            all_grass_population_histories.append(result['grass_population_history']) 


    # Calculate Averages for scalar metrics
    avg_extinction_rabbits = np.mean(all_extinction_ticks_rabbits) if all_extinction_ticks_rabbits else "N/A"
    avg_extinction_foxes = np.mean(all_extinction_ticks_foxes) if all_extinction_ticks_foxes else "N/A"
    avg_peak_rabbits = np.mean(all_peak_rabbits) if all_peak_rabbits else 0
    avg_peak_foxes = np.mean(all_peak_foxes) if all_peak_foxes else 0
    avg_peak_grass = np.mean(all_peak_grass) if all_peak_grass else 0 

    # Save aggregated metrics to CSV
    aggregated_metrics = {
        'Category': category_name,
        'Avg_Extinction_Tick_Rabbits': avg_extinction_rabbits,
        'Avg_Extinction_Tick_Foxes': avg_extinction_foxes,
        'Avg_Peak_Rabbits': avg_peak_rabbits,
        'Avg_Peak_Foxes': avg_peak_foxes,
    }
    # Only add Avg_Peak_Grass if energy is enabled for the category
    if energy_enabled_for_category:
        aggregated_metrics['Avg_Peak_Grass'] = avg_peak_grass

    agg_metrics_df = pd.DataFrame([aggregated_metrics])
    agg_metrics_csv_path = os.path.join(category_output_dir, f"{category_name}_aggregated_metrics.csv")
    agg_metrics_df.to_csv(agg_metrics_csv_path, index=False)
    print(f"Aggregated metrics saved to: {agg_metrics_csv_path}")

    # Plot Average Time to Extinction (Bar Chart)
    if all_extinction_ticks_rabbits or all_extinction_ticks_foxes:
        extinction_data = {
            'Species': [],
            'Average Extinction Tick': []
        }
        if isinstance(avg_extinction_rabbits, (int, float)):
            extinction_data['Species'].append('Rabbits')
            extinction_data['Average Extinction Tick'].append(avg_extinction_rabbits)
        if isinstance(avg_extinction_foxes, (int, float)):
            extinction_data['Species'].append('Foxes')
            extinction_data['Average Extinction Tick'].append(avg_extinction_foxes)

        if extinction_data['Species']: 
            df_extinction = pd.DataFrame(extinction_data)
            plt.figure(figsize=(8, 6))
            plt.bar(df_extinction['Species'], df_extinction['Average Extinction Tick'], color=['green', 'red'])
            plt.title(f'Average Time to Extinction for {category_name}', fontsize=14)
            plt.xlabel('Species', fontsize=12)
            plt.ylabel('Average Extinction Tick', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            ext_plot_path = os.path.join(category_output_dir, f"{category_name}_avg_extinction_time.png")
            plt.savefig(ext_plot_path)
            plt.close()
            print(f"Average extinction time plot saved to: {ext_plot_path}")

    # Plot Average Peak Populations (Bar Chart)
    peak_pop_species = ['Rabbits', 'Foxes']
    peak_pop_values = [avg_peak_rabbits, avg_peak_foxes]
    peak_pop_colors = ['lightgreen', 'salmon']

    if energy_enabled_for_category:
        peak_pop_species.append('Grass')
        peak_pop_values.append(avg_peak_grass)
        peak_pop_colors.append('darkseagreen')

    peak_pop_data = {
        'Species': peak_pop_species, 
        'Average Peak Population': peak_pop_values 
    }
    df_peak_pop = pd.DataFrame(peak_pop_data)
    plt.figure(figsize=(8, 6))
    plt.bar(df_peak_pop['Species'], df_peak_pop['Average Peak Population'], color=peak_pop_colors) 
    plt.title(f'Average Peak Populations for {category_name}', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Average Peak Population', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    peak_pop_plot_path = os.path.join(category_output_dir, f"{category_name}_avg_peak_populations.png")
    plt.savefig(peak_pop_plot_path)
    plt.close()
    print(f"Average peak populations plot saved to: {peak_pop_plot_path}")

    # Plot Averaged Fox Energy Over Time (if applicable)
    if energy_enabled_for_category and all_fox_energy_histories: # Only plot if energy is enabled AND data exists
        max_len_energy = max(len(h) for h in all_fox_energy_histories)
        
        padded_energy_histories = []
        for history in all_fox_energy_histories:
            if history:
                padded_energy_histories.append(np.pad(history, (0, max_len_energy - len(history)), 'edge')) 
            else:
                padded_energy_histories.append(np.pad([], (0, max_len_energy), 'constant', constant_values=0))

        avg_fox_energy_time_series = np.mean(padded_energy_histories, axis=0)

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(avg_fox_energy_time_series)), avg_fox_energy_time_series, label='Average Fox Energy (Averaged Across Runs)', color='orange', linewidth=2)
        plt.title(f'Averaged Fox Energy Over Time for {category_name}', fontsize=16)
        plt.xlabel('Time (Simulation Ticks)', fontsize=12)
        plt.ylabel('Average Energy', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        avg_energy_plot_path = os.path.join(category_output_dir, f"{category_name}_averaged_fox_energy_dynamics.png")
        plt.savefig(avg_energy_plot_path)
        plt.close()
        print(f"Averaged fox energy time series plot saved to: {avg_energy_plot_path}")

        # Save averaged fox energy time series to CSV
        df_avg_fox_energy_ts = pd.DataFrame({
            'Tick': np.arange(len(avg_fox_energy_time_series)),
            'Average_Fox_Energy': avg_fox_energy_time_series
        })
        avg_energy_csv_path = os.path.join(category_output_dir, f"{category_name}_averaged_fox_energy_time_series.csv")
        df_avg_fox_energy_ts.to_csv(avg_energy_csv_path, index=False)
        print(f"Averaged fox energy time series data saved to: {avg_energy_csv_path}")

    # Plot Averaged Rabbit Energy Over Time (if applicable)
    if energy_enabled_for_category and all_rabbit_energy_histories: # Only plot if energy is enabled AND data exists
        max_len_energy_r = max(len(h) for h in all_rabbit_energy_histories)
        
        padded_rabbit_energy_histories = []
        for history in all_rabbit_energy_histories:
            if history:
                padded_rabbit_energy_histories.append(np.pad(history, (0, max_len_energy_r - len(history)), 'edge')) 
            else:
                padded_rabbit_energy_histories.append(np.pad([], (0, max_len_energy_r), 'constant', constant_values=0))

        avg_rabbit_energy_time_series = np.mean(padded_rabbit_energy_histories, axis=0)

        plt.figure(figsize=(12, 7))
        plt.plot(np.arange(len(avg_rabbit_energy_time_series)), avg_rabbit_energy_time_series, label='Average Rabbit Energy (Averaged Across Runs)', color='purple', linewidth=2)
        plt.title(f'Averaged Rabbit Energy Over Time for {category_name}', fontsize=16)
        plt.xlabel('Time (Simulation Ticks)', fontsize=12)
        plt.ylabel('Average Energy', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        avg_rabbit_energy_plot_path = os.path.join(category_output_dir, f"{category_name}_averaged_rabbit_energy_dynamics.png")
        plt.savefig(avg_rabbit_energy_plot_path)
        plt.close()
        print(f"Averaged rabbit energy time series plot saved to: {avg_rabbit_energy_plot_path}")

        # Save averaged rabbit energy time series to CSV
        df_avg_rabbit_energy_ts = pd.DataFrame({
            'Tick': np.arange(len(avg_rabbit_energy_time_series)),
            'Average_Rabbit_Energy': avg_rabbit_energy_time_series
        })
        avg_rabbit_energy_csv_path = os.path.join(category_output_dir, f"{category_name}_averaged_rabbit_energy_time_series.csv")
        df_avg_rabbit_energy_ts.to_csv(avg_rabbit_energy_csv_path, index=False)
        print(f"Averaged rabbit energy time series data saved to: {avg_rabbit_energy_csv_path}")


    # Plot Averaged Population Dynamics Over Time
    if all_rabbit_population_histories or all_fox_population_histories or (energy_enabled_for_category and all_grass_population_histories): 
        # Determine the maximum length among all population histories for proper padding
        max_len_pop_r = max(len(h) for h in all_rabbit_population_histories) if all_rabbit_population_histories else 0
        max_len_pop_f = max(len(h) for h in all_fox_population_histories) if all_fox_population_histories else 0
        max_len_pop_g = max(len(h) for h in all_grass_population_histories) if energy_enabled_for_category and all_grass_population_histories else 0 
        max_len_pop = max(max_len_pop_r, max_len_pop_f, max_len_pop_g) 

        if max_len_pop > 0: 
            padded_rabbit_histories = []
            for history in all_rabbit_population_histories:
                if history:
                    padded_rabbit_histories.append(np.pad(history, (0, max_len_pop - len(history)), 'edge'))
                else:
                    padded_rabbit_histories.append(np.pad([], (0, max_len_pop), 'constant', constant_values=0))
            
            padded_fox_histories = []
            for history in all_fox_population_histories:
                if history:
                    padded_fox_histories.append(np.pad(history, (0, max_len_pop - len(history)), 'edge'))
                else:
                    padded_fox_histories.append(np.pad([], (0, max_len_pop), 'constant', constant_values=0))

            # Only pad grass histories if energy is enabled for the category
            padded_grass_histories = [] 
            if energy_enabled_for_category:
                for history in all_grass_population_histories: 
                    if history: 
                        padded_grass_histories.append(np.pad(history, (0, max_len_pop - len(history)), 'edge')) 
                    else: 
                        padded_grass_histories.append(np.pad([], (0, max_len_pop), 'constant', constant_values=0)) 
            else: # If energy is not enabled, create a list of empty arrays for grass
                 padded_grass_histories = [np.zeros(max_len_pop)] * len(all_rabbit_population_histories) # Match number of runs


            avg_rabbit_pop_time_series = np.mean(padded_rabbit_histories, axis=0)
            avg_fox_pop_time_series = np.mean(padded_fox_histories, axis=0)
            avg_grass_pop_time_series = np.mean(padded_grass_histories, axis=0) 

            plt.figure(figsize=(12, 7))
            plt.plot(np.arange(len(avg_rabbit_pop_time_series)), avg_rabbit_pop_time_series, label='Average Rabbits (Prey)', color='darkgreen', linewidth=2)
            plt.plot(np.arange(len(avg_fox_pop_time_series)), avg_fox_pop_time_series, label='Average Foxes (Predator)', color='darkred', linewidth=2, linestyle='--')
            # Only plot grass if energy is enabled for the category
            if energy_enabled_for_category:
                plt.plot(np.arange(len(avg_grass_pop_time_series)), avg_grass_pop_time_series, label='Average Grass (Food)', color='olivedrab', linewidth=2, linestyle=':') 
            plt.title(f'Averaged Population Dynamics Over Time for {category_name}', fontsize=16)
            plt.xlabel('Time (Simulation Ticks)', fontsize=12)
            plt.ylabel('Average Population Count', fontsize=12)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)

            # Adjust ylim based on calculated average max population
            max_avg_pop = 0
            if avg_rabbit_pop_time_series.size > 0:
                max_avg_pop = max(max_avg_pop, avg_rabbit_pop_time_series.max())
            if avg_fox_pop_time_series.size > 0:
                max_avg_pop = max(max_avg_pop, avg_fox_pop_time_series.max())
            if energy_enabled_for_category and avg_grass_pop_time_series.size > 0: # Only consider grass max if enabled
                max_avg_pop = max(max_avg_pop, avg_grass_pop_time_series.max()) 
            plt.ylim(0, max_avg_pop * 1.1 + 1)

            plt.tight_layout()
            avg_pop_plot_path = os.path.join(category_output_dir, f"{category_name}_averaged_population_dynamics.png")
            plt.savefig(avg_pop_plot_path)
            plt.close()
            print(f"Averaged population dynamics plot saved to: {avg_pop_plot_path}")

            # Save averaged population time series to CSV
            df_avg_pop_ts_data = {
                'Tick': np.arange(len(avg_rabbit_pop_time_series)),
                'Average_Rabbits': avg_rabbit_pop_time_series,
                'Average_Foxes': avg_fox_pop_time_series,
            }
            if energy_enabled_for_category: # Only add grass column if energy is enabled
                df_avg_pop_ts_data['Average_Grass'] = avg_grass_pop_time_series
            
            df_avg_pop_ts = pd.DataFrame(df_avg_pop_ts_data)
            avg_pop_csv_path = os.path.join(category_output_dir, f"{category_name}_averaged_population_time_series.csv")
            df_avg_pop_ts.to_csv(avg_pop_csv_path, index=False)
            print(f"Averaged population time series data saved to: {avg_pop_csv_path}")

# --- Grid Search Runner Function ---
# Define parameter abbreviations
param_abbreviations = {
    'rabbit_movement_speed': 'rms',
    'fox_movement_speed': 'fms',
    'rabbit_perception_radius': 'rpr',
    'fox_perception_radius': 'fpr',
    'grass_perception_radius': 'gpr', 
    'rabbit_reproduce_chance': 'rrc',
    'fox_spontaneous_death_chance': 'fsdc',
    'fox_initial_energy': 'fie',
    'fox_energy_decay_rate': 'fedr',
    'rabbit_energy_gain_for_fox': 'regff', 
    'fox_starvation_threshold': 'fst',
    'rabbit_initial_energy': 'rie', 
    'rabbit_energy_decay_rate': 'redr', 
    'grass_energy_gain_for_rabbit': 'gegr', 
    'rabbit_starvation_threshold': 'rst', 
    'grass_regrowth_time': 'grt' 
}

def run_grid_search_for_scenario(
    scenario_name: str,
    initial_rabbits: int,
    initial_foxes: int,
    initial_grass: int, 
    fixed_config_params: dict, 
    varying_params_grid: dict 
):
    """
    Performs a grid search over specified parameters for a given scenario.
    """
    category_output_dir = os.path.join("simulation_results_energy", scenario_name)
    os.makedirs(category_output_dir, exist_ok=True)
    
    scenario_run_results = []
    
    # Generate all combinations of varying parameters
    keys = sorted(varying_params_grid.keys())
    values = [varying_params_grid[key] for key in keys]
    combinations = list(product(*values))
    total_combinations = len(combinations)

    print(f"\n{BLUE}{BOLD}====================================================={RESET}")
    print(f"{BLUE}{BOLD}  Running Grid Search for: {scenario_name.replace('_', ' ').title()}  {RESET}")
    print(f"{BLUE}{BOLD}====================================================={RESET}")
    print(f"Fixed Parameters: {fixed_config_params}")
    print(f"Varying Parameters: {varying_params_grid}")
    print(f"{BOLD}Total combinations for {scenario_name}: {BOLD}{total_combinations}{RESET}")

    for i, combo_values in enumerate(combinations):
        current_config_overrides = {**fixed_config_params} 
        run_name_parts = []
        for j, key in enumerate(keys):
            current_config_overrides[key] = combo_values[j]
            abbreviated_key = param_abbreviations.get(key, key) 
            # Format value: replace '.' with 'p' for floats, convert to string
            formatted_value = str(combo_values[j]).replace('.', 'p')
            run_name_parts.append(f"{abbreviated_key}{formatted_value}")
        
        # Ensure duration is always set from the config's default
        current_config_overrides['duration'] = LVConfig().duration 

        run_name = f"combo_{i+1}_" + "_".join(run_name_parts)
        
        result = run_experiment(
            run_name=run_name,
            base_output_dir=category_output_dir,
            initial_rabbits=initial_rabbits,
            initial_foxes=initial_foxes,
            initial_grass=initial_grass, 
            config_overrides=current_config_overrides,
            current_sim_index=i+1, 
            total_sims_in_scenario=total_combinations 
        )
        scenario_run_results.append(result)
    
    analyze_category_results(scenario_run_results, category_output_dir, scenario_name)


# --- New Function for Cross-Scenario Comparison ---
def compare_scenarios(base_results_dir: str, scenario_names: List[str]):
    """
    Compares aggregated results from different scenarios and plots them in the main results directory.
    Generates a combined plot of key comparison metrics with an improved layout,
    and also saves each of the three component plots separately.

    Args:
        base_results_dir (str): The main directory where simulation results are stored.
        scenario_names (List[str]): List of names of the scenario subdirectories to compare.
    """
    print(f"\n{BLUE}{BOLD}--- Generating Cross-Scenario Comparison Plots ---{RESET}")

    # Load aggregated metrics for each scenario
    scenario_metrics = {}
    for name in scenario_names:
        metrics_path = os.path.join(base_results_dir, name, f"{name}_aggregated_metrics.csv")
        if os.path.exists(metrics_path):
            scenario_metrics[name] = pd.read_csv(metrics_path).iloc[0]
        else:
            print(f"{YELLOW}Warning: Aggregated metrics not found for {name} at {metrics_path}{RESET}")
            scenario_metrics[name] = None

    # Load averaged population time series for each scenario
    scenario_pop_data = {}
    for name in scenario_names:
        pop_path = os.path.join(base_results_dir, name, f"{name}_averaged_population_time_series.csv")
        if os.path.exists(pop_path):
            scenario_pop_data[name] = pd.read_csv(pop_path)
        else:
            print(f"{YELLOW}Warning: Averaged population data not found for {name} at {pop_path}{RESET}")
            scenario_pop_data[name] = None

    # Load averaged fox energy time series for each scenario (only applicable if energy_enabled)
    scenario_fox_energy_data = {}
    for name in scenario_names:
        energy_path = os.path.join(base_results_dir, name, f"{name}_averaged_fox_energy_time_series.csv")
        if os.path.exists(energy_path):
            scenario_fox_energy_data[name] = pd.read_csv(energy_path)
        else:
            scenario_fox_energy_data[name] = None

    # Load averaged rabbit energy time series for each scenario
    scenario_rabbit_energy_data = {}
    for name in scenario_names:
        rabbit_energy_path = os.path.join(base_results_dir, name, f"{name}_averaged_rabbit_energy_time_series.csv")
        if os.path.exists(rabbit_energy_path):
            scenario_rabbit_energy_data[name] = pd.read_csv(rabbit_energy_path)
        else:
            scenario_rabbit_energy_data[name] = None

    # Determine if any population data exists to plot
    has_population_data = False
    max_pop_overall = 0
    for df in scenario_pop_data.values():
        if df is not None and not df.empty:
            has_population_data = True
            if 'Average_Rabbits' in df.columns:
                max_pop_overall = max(max_pop_overall, df['Average_Rabbits'].max())
            if 'Average_Foxes' in df.columns:
                max_pop_overall = max(max_pop_overall, df['Average_Foxes'].max())
            if 'Average_Grass' in df.columns:
                max_pop_overall = max(max_pop_overall, df['Average_Grass'].max())

    # --- Prepare data for Extinction and Peak Population Plots ---
    ext_labels = []
    ext_rabbits = []
    ext_foxes = []
    peak_labels = []
    peak_rabbits = []
    peak_foxes = []
    peak_grass_data = []

    for name in scenario_names:
        metrics = scenario_metrics.get(name)
        if metrics is not None and not metrics.empty:
            # Extinction data
            ext_labels.append(name.replace("_", " ").title())
            duration_for_na = LVConfig().duration # Get max duration from LVConfig
            ext_rabbits.append(metrics['Avg_Extinction_Tick_Rabbits'] if metrics['Avg_Extinction_Tick_Rabbits'] != 'N/A' else duration_for_na)
            ext_foxes.append(metrics['Avg_Extinction_Tick_Foxes'] if metrics['Avg_Extinction_Tick_Foxes'] != 'N/A' else duration_for_na)

            # Peak population data
            peak_labels.append(name.replace("_", " ").title())
            peak_rabbits.append(metrics['Avg_Peak_Rabbits'])
            peak_foxes.append(metrics['Avg_Peak_Foxes'])
            if 'Avg_Peak_Grass' in metrics:
                peak_grass_data.append(metrics['Avg_Peak_Grass'])
            else:
                peak_grass_data.append(0) # Ensure list has same length even if grass data is missing


    # --- Combined Comparison Plot ---
    fig = plt.figure(figsize=(28, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[16, 9], hspace=0.3, wspace=0.2)

    ax_pop = fig.add_subplot(gs[:, 0])
    ax_ext = fig.add_subplot(gs[0, 1])
    ax_peak = fig.add_subplot(gs[1, 1])

    # Plot 1: Averaged Population Dynamics (Left, wider plot)
    if has_population_data:
        for name in scenario_names:
            df = scenario_pop_data.get(name)
            if df is not None and not df.empty:
                ax_pop.plot(df['Tick'], df['Average_Rabbits'], label=f'{name.replace("_", " ").title()} - Rabbits', linestyle='-', linewidth=2)
                ax_pop.plot(df['Tick'], df['Average_Foxes'], label=f'{name.replace("_", " ").title()} - Foxes', linestyle='--', linewidth=2)
                if 'Average_Grass' in df.columns:
                    ax_pop.plot(df['Tick'], df['Average_Grass'], label=f'{name.replace("_", " ").title()} - Grass', linestyle=':', linewidth=2)
        ax_pop.set_title('Averaged Population Dynamics', fontsize=16)
        ax_pop.set_xlabel('Time (Simulation Ticks)', fontsize=12)
        ax_pop.set_ylabel('Average Population Count', fontsize=12)
        ax_pop.legend(loc='upper right', fontsize=10)
        ax_pop.grid(True, linestyle='--', alpha=0.6)
        if max_pop_overall > 0:
            ax_pop.set_ylim(0, max_pop_overall * 1.1 + 1)
        else:
            ax_pop.set_ylim(0, 10)
    else:
        ax_pop.text(0.5, 0.5, 'No Population Data Available', horizontalalignment='center', verticalalignment='center', transform=ax_pop.transAxes, fontsize=16, color='gray')
        ax_pop.set_title('Averaged Population Dynamics', fontsize=16)
        ax_pop.set_xlabel('Time (Simulation Ticks)', fontsize=12)
        ax_pop.set_ylabel('Average Population Count', fontsize=12)

    # Plot 2: Compared Average Extinction Time (Top-Right)
    if ext_labels:
        x = np.arange(len(ext_labels))
        width = 0.35
        ax_ext.bar(x - width/2, ext_rabbits, width, label='Rabbits', color='green')
        ax_ext.bar(x + width/2, ext_foxes, width, label='Foxes', color='red')
        ax_ext.set_ylabel('Avg. Extinction Tick (or Max Duration)', fontsize=10)
        ax_ext.set_title('Average Time to Extinction', fontsize=14)
        ax_ext.set_xticks(x)
        ax_ext.set_xticklabels(ext_labels, rotation=45, ha='right', fontsize=9)
        ax_ext.legend(fontsize=9)
        ax_ext.grid(axis='y', linestyle='--', alpha=0.7)
        ax_ext.set_ylim(0, LVConfig().duration * 1.1)
    else:
        ax_ext.text(0.5, 0.5, 'No Extinction Data', horizontalalignment='center', verticalalignment='center', transform=ax_ext.transAxes, fontsize=14, color='gray')
        ax_ext.set_title('Average Time to Extinction', fontsize=14)
        ax_ext.set_ylabel('Avg. Extinction Tick', fontsize=10)

    # Plot 3: Compared Average Peak Populations (Bottom-Right)
    if peak_labels:
        x = np.arange(len(peak_labels))
        width = 0.25
        ax_peak.bar(x - width, peak_rabbits, width, label='Rabbits', color='lightgreen')
        ax_peak.bar(x, peak_foxes, width, label='Foxes', color='salmon')
        if any(pg > 0 for pg in peak_grass_data):
            ax_peak.bar(x + width, peak_grass_data, width, label='Grass', color='darkseagreen')
        ax_peak.set_ylabel('Average Peak Population', fontsize=10)
        ax_peak.set_title('Average Peak Populations', fontsize=14)
        ax_peak.set_xticks(x)
        ax_peak.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=9)
        ax_peak.legend(fontsize=9)
        ax_peak.grid(axis='y', linestyle='--', alpha=0.7)
        max_all_peak_pop = max(max(peak_rabbits), max(peak_foxes), max(peak_grass_data) if peak_grass_data else 0)
        ax_peak.set_ylim(0, max_all_peak_pop * 1.2 + 1)
    else:
        ax_peak.text(0.5, 0.5, 'No Peak Population Data', horizontalalignment='center', verticalalignment='center', transform=ax_peak.transAxes, fontsize=14, color='gray')
        ax_peak.set_title('Average Peak Populations', fontsize=14)
        ax_peak.set_ylabel('Average Peak Population', fontsize=10)

    # Add a main title for the entire figure
    fig.suptitle('Cross-Scenario Comparison of Predator-Prey Dynamics', fontsize=20, y=0.98)

    # Adjust overall layout and save the combined figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_plot_path = os.path.join(base_results_dir, "combined_scenario_comparison.png")
    plt.savefig(combined_plot_path)
    print(f"Combined scenario comparison plot saved to: {combined_plot_path}")

    # --- Save Separate Plots from the existing Axes objects ---
    # To save individual subplots, you need to create a new figure for each
    # and copy the content from the combined figure's axes.
    # This involves setting up new axes and transferring elements.

    # 1. Population Dynamics Plot
    if has_population_data:
        fig_pop, ax_pop_single = plt.subplots(figsize=(14, 8))
        # Copy elements from the combined plot's ax_pop
        for line in ax_pop.get_lines():
            ax_pop_single.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                               linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                               label=line.get_label())
        ax_pop_single.set_title(ax_pop.get_title(), fontsize=16)
        ax_pop_single.set_xlabel(ax_pop.get_xlabel(), fontsize=12)
        ax_pop_single.set_ylabel(ax_pop.get_ylabel(), fontsize=12)
        ax_pop_single.legend(loc='upper right', fontsize=10)
        ax_pop_single.grid(True, linestyle='--', alpha=0.6)
        ax_pop_single.set_ylim(ax_pop.get_ylim()) # Reuse the calculated y-limit
        plt.tight_layout()
        plot_path_pop = os.path.join(base_results_dir, "compared_average_population_dynamics.png")
        plt.savefig(plot_path_pop)
        plt.close(fig_pop) # Close the single figure to free memory
        print(f"Separate average population dynamics plot saved to: {plot_path_pop}")
    else:
        print(f"{YELLOW}No population data available for separate population dynamics plot.{RESET}")

    # 2. Extinction Time Plot
    if ext_labels:
        fig_ext, ax_ext_single = plt.subplots(figsize=(10, 7))
        # Copy elements from the combined plot's ax_ext
        for bar_container in ax_ext.containers:
            for bar in bar_container:
                ax_ext_single.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   width=bar.get_width(), color=bar.get_facecolor(),
                                   label=bar_container.get_label()) # This label might need adjustment if bars have distinct labels
        # Fix for legend from bars
        handles, labels = [], []
        for container in ax_ext.containers:
            if container.get_label(): # Check if container has a label
                handles.append(container.patches[0]) # Get the first patch (bar) as handle
                labels.append(container.get_label())
        ax_ext_single.legend(handles, labels, fontsize=9)


        ax_ext_single.set_ylabel(ax_ext.get_ylabel(), fontsize=10)
        ax_ext_single.set_title(ax_ext.get_title(), fontsize=16)
        ax_ext_single.set_xticks(ax_ext.get_xticks())
        ax_ext_single.set_xticklabels(ax_ext.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax_ext_single.grid(axis='y', linestyle='--', alpha=0.7)
        ax_ext_single.set_ylim(ax_ext.get_ylim())
        plt.tight_layout()
        plot_path_ext = os.path.join(base_results_dir, "compared_average_extinction_time.png")
        plt.savefig(plot_path_ext)
        plt.close(fig_ext)
        print(f"Separate average extinction time plot saved to: {plot_path_ext}")
    else:
        print(f"{YELLOW}No extinction time data available for separate extinction time plot.{RESET}")

    # 3. Peak Populations Plot
    if peak_labels:
        fig_peak, ax_peak_single = plt.subplots(figsize=(10, 7))
        # Copy elements from the combined plot's ax_peak
        for bar_container in ax_peak.containers:
            for bar in bar_container:
                ax_peak_single.bar(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   width=bar.get_width(), color=bar.get_facecolor(),
                                   label=bar_container.get_label())
        # Fix for legend from bars
        handles, labels = [], []
        for container in ax_peak.containers:
            if container.get_label():
                handles.append(container.patches[0])
                labels.append(container.get_label())
        ax_peak_single.legend(handles, labels, fontsize=9)


        ax_peak_single.set_ylabel(ax_peak.get_ylabel(), fontsize=10)
        ax_peak_single.set_title(ax_peak.get_title(), fontsize=16)
        ax_peak_single.set_xticks(ax_peak.get_xticks())
        ax_peak_single.set_xticklabels(ax_peak.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax_peak_single.grid(axis='y', linestyle='--', alpha=0.7)
        ax_peak_single.set_ylim(ax_peak.get_ylim())
        plt.tight_layout()
        plot_path_peak = os.path.join(base_results_dir, "compared_average_peak_populations.png")
        plt.savefig(plot_path_peak)
        plt.close(fig_peak)
        print(f"Separate average peak populations plot saved to: {plot_path_peak}")
    else:
        print(f"{YELLOW}No peak population data available for separate peak population plot.{RESET}")

    # It's crucial to close the combined figure after saving all plots to free memory.
    plt.close(fig) # Close the original combined figure

    # --- Save combined data for comparison plots into CSVs (unchanged logic) ---
    # For population dynamics
    combined_pop_df = pd.DataFrame()
    for name in scenario_names:
        df = scenario_pop_data.get(name)
        if df is not None and not df.empty:
            if combined_pop_df.empty:
                combined_pop_df['Tick'] = df['Tick']
            combined_pop_df[f'{name}_Average_Rabbits'] = df['Average_Rabbits']
            combined_pop_df[f'{name}_Average_Foxes'] = df['Average_Foxes']
            if 'Average_Grass' in df.columns:
                combined_pop_df[f'{name}_Average_Grass'] = df['Average_Grass']
    if not combined_pop_df.empty:
        combined_pop_csv_path = os.path.join(base_results_dir, "compared_average_population_time_series.csv")
        combined_pop_df.to_csv(combined_pop_csv_path, index=False)
        print(f"Combined average population time series data saved to: {combined_pop_csv_path}")

    # For scalar metrics (extinction time and peak populations)
    if scenario_metrics:
        combined_metrics_df = pd.DataFrame(columns=['Metric'] + [s.replace("_", " ").title() for s in scenario_names])

        ext_row_rabbits = {'Metric': 'Avg_Extinction_Tick_Rabbits'}
        ext_row_foxes = {'Metric': 'Avg_Extinction_Tick_Foxes'}
        for name in scenario_names:
            metrics = scenario_metrics.get(name)
            ext_row_rabbits[name.replace("_", " ").title()] = metrics['Avg_Extinction_Tick_Rabbits'] if metrics is not None else "N/A"
            ext_row_foxes[name.replace("_", " ").title()] = metrics['Avg_Extinction_Tick_Foxes'] if metrics is not None else "N/A"
        combined_metrics_df = pd.concat([combined_metrics_df, pd.DataFrame([ext_row_rabbits, ext_row_foxes])], ignore_index=True)

        peak_row_rabbits = {'Metric': 'Avg_Peak_Rabbits'}
        peak_row_foxes = {'Metric': 'Avg_Peak_Foxes'}
        peak_row_grass = {'Metric': 'Avg_Peak_Grass'}

        for name in scenario_names:
            metrics = scenario_metrics.get(name)
            peak_row_rabbits[name.replace("_", " ").title()] = metrics['Avg_Peak_Rabbits'] if metrics is not None else 0
            peak_row_foxes[name.replace("_", " ").title()] = metrics['Avg_Peak_Foxes'] if metrics is not None else 0
            if metrics is not None and 'Avg_Peak_Grass' in metrics:
                peak_row_grass[name.replace("_", " ").title()] = metrics['Avg_Peak_Grass']
            else:
                peak_row_grass[name.replace("_", " ").title()] = 0

        combined_metrics_df = pd.concat([combined_metrics_df, pd.DataFrame([peak_row_rabbits, peak_row_foxes, peak_row_grass])], ignore_index=True)

        combined_metrics_csv_path = os.path.join(base_results_dir, "compared_aggregated_metrics.csv")
        combined_metrics_df.to_csv(combined_metrics_csv_path, index=False)
        print(f"Combined aggregated metrics data saved to: {combined_metrics_csv_path}")

# --- Main Experiment Execution Block ---
if __name__ == "__main__":
    # Ensure Pygame is initialized only once for all runs
    pg.init() 

    base_results_dir = "simulation_results_energy"
    os.makedirs(base_results_dir, exist_ok=True)

    # Common initial populations
    initial_rabbits = 100
    initial_foxes = 10
    initial_grass = 200 
    
    # Get the fixed duration from LVConfig
    fixed_duration = LVConfig().duration

    # --- Energy-Free Scenario Grid Search ---
    energy_free_params_to_vary = {
        'rabbit_movement_speed': [2.0],
        'fox_movement_speed': [3.0],
        'rabbit_perception_radius': [50],
        'fox_perception_radius': [70],
        'rabbit_reproduce_chance': [0.005],
        'fox_spontaneous_death_chance': [0.005]
    }
    fixed_energy_free_params = {
        'energy_enabled': False,
    }
    
    # Calculate and print total combinations for energy-free scenario
    energy_free_keys = sorted(energy_free_params_to_vary.keys())
    energy_free_values = [energy_free_params_to_vary[key] for key in energy_free_keys]
    total_energy_free_combinations = len(list(product(*energy_free_values)))
    print(f"\n{BOLD}Total combinations for Energy-Free Scenario: {BOLD}{total_energy_free_combinations}{RESET}")

    run_grid_search_for_scenario(
        scenario_name=ENERGY_FREE_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        initial_grass=initial_grass, # Passed, but will be ignored due to energy_enabled: False
        fixed_config_params=fixed_energy_free_params,
        varying_params_grid=energy_free_params_to_vary
    )

    # --- Energy-Enabled Scenario Grid Search ---
    energy_enabled_params_to_vary = {
        'fox_initial_energy': [100.0], 
        'fox_energy_decay_rate': [0.5], 
        'rabbit_energy_gain_for_fox': [10.0], 
        'fox_starvation_threshold': [0.0],
        'rabbit_initial_energy': [50.0], 
        'rabbit_energy_decay_rate': [0.2], 
        'grass_energy_gain_for_rabbit': [5.0], 
        'rabbit_starvation_threshold': [0.0], 
        'grass_regrowth_time': [100], 
        'rabbit_movement_speed': [2.0],
        'fox_movement_speed': [3.0],
        'rabbit_perception_radius': [50],
        'fox_perception_radius': [70],
        'grass_perception_radius': [30] 
    }
    fixed_energy_enabled_params = {
        'energy_enabled': True,
    }

    # Calculate and print total combinations for energy-enabled scenario
    energy_enabled_keys = sorted(energy_enabled_params_to_vary.keys())
    energy_enabled_values = [energy_enabled_params_to_vary[key] for key in energy_enabled_keys]
    total_energy_enabled_combinations = len(list(product(*energy_enabled_values)))
    print(f"\n{BOLD}Total combinations for Energy-Enabled Scenario: {BOLD}{total_energy_enabled_combinations}{RESET}")

    run_grid_search_for_scenario(
        scenario_name=ENERGY_ENABLED_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        initial_grass=initial_grass, 
        fixed_config_params=fixed_energy_enabled_params,
        varying_params_grid=energy_enabled_params_to_vary
    )
    
    # --- Cross-Scenario Comparison ---
    # Call the new function to generate comparison plots and data
    compare_scenarios(base_results_dir, [ENERGY_FREE_SCENARIO_NAME, ENERGY_ENABLED_SCENARIO_NAME])


    # Ensure Pygame is quit only once after all runs
    pg.quit()
    print(f"\n{CYAN}{BOLD}--- All simulations concluded. ---{RESET}")
    print(f"{CYAN}Results saved in the '{base_results_dir}' directory.{RESET}")