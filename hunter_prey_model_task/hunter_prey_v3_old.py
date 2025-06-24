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

from vi import Agent, Config, Simulation
from pygame.math import Vector2

# --- Configuration for Lotka-Volterra Simulation ---
@dataclass
class LVConfig(Config):
    """
    Configuration settings for the Lotka-Volterra Predator-Prey simulation.
    """
    # General Simulation Parameters
    radius: int = 50 # Perception radius for agents
    image_rotation: bool = True
    fps_limit: int = 60 # Frames per second limit
    duration: int = 2000 # Simulation duration in ticks, capped at 2000

    # --- Lotka-Volterra Specific Parameters ---
    # Movement speeds for rabbits and foxes
    rabbit_movement_speed: float = 2.0  # Separate movement speed for rabbits
    fox_movement_speed: float = 3.0     # Separate movement speed for foxes
    
    rabbit_reproduce_chance: float = 0.005  # Probability per tick for a rabbit to reproduce

    # Fox parameters
    fox_spontaneous_death_chance: float = 0.005 # Probability per tick for a fox to die spontaneously
    
    # --- Energy System Parameters (Toggleable) ---
    energy_enabled: bool = False # This will be set by the experiment runner
    fox_initial_energy: float = 100.0     # Starting energy for foxes
    fox_energy_decay_rate: float = 0.5    # Energy lost per tick by a fox
    rabbit_energy_gain: float = 10.0       # Energy gained by fox when eating a rabbit
    fox_starvation_threshold: float = 0.0 # Energy level at which a fox dies


# --- Agent Definitions ---

class Rabbit(Agent[LVConfig]):
    """
    Represents a Rabbit agent (Prey).
    Behaves passively, reproduces spontaneously, dies only if eaten by a fox.
    """
    def on_spawn(self):
        """Initializes rabbit movement."""
        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

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

        # Spontaneous asexual reproduction for rabbits
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
        """Initializes fox movement and energy."""
        # Use fox_movement_speed from config
        self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        if self.config.energy_enabled:
            self.energy = self.config.fox_initial_energy

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
            self.move = direction_to_rabbit * self.config.fox_movement_speed # Use fox-specific speed

            if self.pos.distance_to(target_rabbit.pos) < 10: # Close enough to eat
                target_rabbit.kill()
                
                # Fox reproduces directly after eating a rabbit
                new_fox_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
                Fox(images=self._images, simulation=self._Agent__simulation, pos=new_fox_pos)

                # Replenish energy if energy system is enabled
                if self.config.energy_enabled:
                    self.energy += self.config.rabbit_energy_gain
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

        # --- Death Logic: Starvation (if energy enabled) or Spontaneous (if energy disabled) ---
        if self.config.energy_enabled:
            self.energy -= self.config.fox_energy_decay_rate # Energy decays
            if self.energy <= self.config.fox_starvation_threshold:
                self.kill() # Fox dies of starvation
        else:
            # Spontaneous death only if energy system is NOT enabled
            if self.shared.prng_move.random() < self.config.fox_spontaneous_death_chance:
                self.kill() # Fox dies spontaneously

# --- Helper function for spawning agents in non-colliding positions ---
def spawn_agent_safely(agent_class: Type[Agent], images_list: List[pg.Surface], num_to_spawn: int, simulation_instance: Simulation):
    count = 0
    max_spawn_attempts_per_agent = 500
    
    # Get window dimensions here, as it's needed for random positions
    window_width, window_height = simulation_instance.config.window.as_tuple()

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
    config_overrides: dict
):
    """
    Runs a single simulation experiment with specified parameters and saves results.

    Args:
        run_name (str): A unique name for this specific run, used for folder creation.
        base_output_dir (str): The base directory to save results (e.g., "simulation_results/baseline_energy_free").
        initial_rabbits (int): Initial number of rabbit agents.
        initial_foxes (int): Initial number of fox agents.
        config_overrides (dict): A dictionary of LVConfig parameters to override defaults.
    """
    # Create the specific output directory for this run
    run_output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"\n--- Starting run: {run_name} ---")
    print(f"Parameters: {config_overrides}")

    # Reset population history for each run
    rabbit_population_history = []
    fox_population_history = []

    # Apply config overrides
    current_config = LVConfig()
    for key, value in config_overrides.items():
        setattr(current_config, key, value)

    sim = Simulation(current_config)

    # Load agent images
    # Assuming images are in a subdirectory "images" relative to where the script is run
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    rabbit_image_path = os.path.join(script_dir, "images", "rabbit.png")
    fox_image_path = os.path.join(script_dir, "images", "fox.png")
    
    try:
        loaded_rabbit_image = pg.image.load(rabbit_image_path).convert_alpha()
        rabbit_images_list = [loaded_rabbit_image]
        loaded_fox_image = pg.image.load(fox_image_path).convert_alpha()
        fox_images_list = [loaded_fox_image]
    except pg.error as e:
        print(f"Error loading images: {e}. Please ensure 'images/rabbit.png' and 'images/fox.png' exist in the script directory.")
        # Attempt to create dummy images if loading fails, to allow simulation to proceed
        print("Creating dummy images for agents to continue simulation.")
        dummy_rabbit_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_rabbit_surf, (0, 200, 0), (10, 10), 10) # Green circle
        rabbit_images_list = [dummy_rabbit_surf]

        dummy_fox_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_fox_surf, (200, 0, 0), (10, 10), 10) # Red circle
        fox_images_list = [dummy_fox_surf]


    # Spawn initial populations
    spawn_agent_safely(Rabbit, rabbit_images_list, initial_rabbits, sim)
    spawn_agent_safely(Fox, fox_images_list, initial_foxes, sim)

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
        
        # Always append current population to history
        rabbit_population_history.append(current_rabbits)
        fox_population_history.append(current_foxes)

        # 3. Check for termination conditions based on population (after recording this frame's data).
        if current_foxes == 0 or current_rabbits == 0:
            print(f"Simulation ended at tick {len(rabbit_population_history) - 1}: All {'foxes' if current_foxes == 0 else 'rabbits'} died.")
            self._running = False # Stop the simulation loop
            # Crucial: Explicitly set _record_snapshots to False to prevent further Polars errors
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return # Exit this custom tick method. The main sim.run() will terminate next.
        
        # Also check if max duration is reached.
        if len(rabbit_population_history) > self.config.duration:
            print(f"Simulation ended at tick {len(rabbit_population_history) - 1}: Max duration reached.")
            self._running = False
            # Crucial: Explicitly set _record_snapshots to False to prevent further Polars errors
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return

        # 4. Print Current Tick Number to Console (replacing previous line)
        sys.stdout.write(f"\rCurrent Tick: {len(rabbit_population_history)}   ") # Spaces to clear previous line
        sys.stdout.flush()

        # 5. Call the original sim.tick() method to execute VI's standard per-tick logic.
        original_sim_tick()

    # Apply the custom tick method by binding it correctly to the sim instance
    sim.tick = custom_tick_for_run.__get__(sim, type(sim))

    # Run the simulation
    sim.run()

    # --- Final Console Output Cleanup for this run ---
    sys.stdout.write("\n") # Move to a new line after the simulation finishes

    # --- Saving Results ---
    # Save population data to CSV
    df_populations = pd.DataFrame({
        'Tick': np.arange(len(rabbit_population_history)),
        'Rabbits': rabbit_population_history,
        'Foxes': fox_population_history
    })
    csv_path = os.path.join(run_output_dir, f"{run_name}_population_data.csv")
    df_populations.to_csv(csv_path, index=False)
    print(f"Population data saved to: {csv_path}")

    # Save population plot to PNG
    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(len(rabbit_population_history)), rabbit_population_history, label='Rabbits (Prey)', color='green', linewidth=2)
    plt.plot(np.arange(len(fox_population_history)), fox_population_history, label='Foxes (Predator)', color='red', linewidth=2, linestyle='--')
    plt.title(f'Population Dynamics for {run_name}', fontsize=16)
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
    plot_path = os.path.join(run_output_dir, f"{run_name}_population_dynamics.png")
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory
    print(f"Population plot saved to: {plot_path}")

    # Return some key metrics if needed for higher-level analysis
    return {
        'final_rabbits': rabbit_population_history[-1] if rabbit_population_history else 0,
        'final_foxes': fox_population_history[-1] if fox_population_history else 0,
        'ticks_completed': len(rabbit_population_history) - 1,
        'parameters': config_overrides
    }


# --- Main Experiment Execution Block ---
if __name__ == "__main__":
    # Ensure Pygame is initialized only once for all runs
    pg.init() 

    # Define base output directory
    base_results_dir = "simulation_results"
    os.makedirs(base_results_dir, exist_ok=True)

    # --- Scenario 1: Baseline Runs (Energy-Free) ---
    print("--- Running Baseline Energy-Free Scenarios ---")
    energy_free_dir = os.path.join(base_results_dir, "baseline_energy_free")
    os.makedirs(energy_free_dir, exist_ok=True)

    for i in range(3): # Example: 3 runs for baseline
        run_experiment(
            run_name=f"run_{i+1}",
            base_output_dir=energy_free_dir,
            initial_rabbits=100,
            initial_foxes=10,
            config_overrides={
                'energy_enabled': False,
                'rabbit_movement_speed': 2.0,
                'fox_movement_speed': 3.0,
                'duration': 2000 # Ensure consistent duration
            }
        )

    # --- Scenario 2: Baseline Runs (Energy-Enabled) ---
    print("\n--- Running Baseline Energy-Enabled Scenarios ---")
    energy_enabled_dir = os.path.join(base_results_dir, "baseline_energy_enabled")
    os.makedirs(energy_enabled_dir, exist_ok=True)

    for i in range(3): # Example: 3 runs for baseline
        run_experiment(
            run_name=f"run_{i+1}",
            base_output_dir=energy_enabled_dir,
            initial_rabbits=100,
            initial_foxes=10,
            config_overrides={
                'energy_enabled': True,
                'fox_initial_energy': 100.0,
                'fox_energy_decay_rate': 0.5,
                'rabbit_energy_gain': 10.0,
                'fox_starvation_threshold': 0.0,
                'rabbit_movement_speed': 2.0,
                'fox_movement_speed': 3.0,
                'duration': 2000
            }
        )

    # --- Scenario 3: Parameter Variation (Energy-Enabled) ---
    print("\n--- Running Parameter Variation Scenarios (Energy-Enabled) ---")
    param_variation_dir = os.path.join(base_results_dir, "parameter_variations")
    os.makedirs(param_variation_dir, exist_ok=True)

    # Example: Vary fox_energy_decay_rate
    decay_rates = [0.1, 0.25, 0.75, 1.0]
    for i, rate in enumerate(decay_rates):
        run_experiment(
            run_name=f"decay_rate_{rate}",
            base_output_dir=param_variation_dir,
            initial_rabbits=100,
            initial_foxes=10,
            config_overrides={
                'energy_enabled': True,
                'fox_initial_energy': 100.0,
                'fox_energy_decay_rate': rate, # VARYING THIS
                'rabbit_energy_gain': 10.0,
                'fox_starvation_threshold': 0.0,
                'rabbit_movement_speed': 2.0,
                'fox_movement_speed': 3.0,
                'duration': 2000
            }
        )
    
    # Example: Vary rabbit_energy_gain
    energy_gains = [5.0, 15.0, 25.0]
    for i, gain in enumerate(energy_gains):
        run_experiment(
            run_name=f"energy_gain_{gain}",
            base_output_dir=param_variation_dir,
            initial_rabbits=100,
            initial_foxes=10,
            config_overrides={
                'energy_enabled': True,
                'fox_initial_energy': 100.0,
                'fox_energy_decay_rate': 0.5,
                'rabbit_energy_gain': gain, # VARYING THIS
                'fox_starvation_threshold': 0.0,
                'rabbit_movement_speed': 2.0,
                'fox_movement_speed': 3.0,
                'duration': 2000
            }
        )

    # Ensure Pygame is quit only once after all runs
    pg.quit()
    print("\n--- All simulations concluded. ---")
    print(f"Results saved in the '{base_results_dir}' directory.")
