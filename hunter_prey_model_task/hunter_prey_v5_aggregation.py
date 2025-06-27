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
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
CYAN = "\033[96m"

# --- Scenario Name Constants ---
NO_AGGREGATION_SCENARIO_NAME = "no_aggregation"
AGGREGATION_ENABLED_SCENARIO_NAME = "with_aggregation"

# Define the states for the Rabbit agent's finite state machine for aggregation
class RabbitState(Enum):
    WANDERING = 1
    JOINING = 2
    STILL = 3
    LEAVING = 4

# --- Configuration for Lotka-Volterra Simulation with Aggregation ---
@dataclass
class LVConfig(Config):
    """
    Configuration settings for the Lotka-Volterra Predator-Prey simulation
    with optional Rabbit aggregation.
    """
    # General Simulation Parameters
    image_rotation: bool = True
    fps_limit: int = 60 # Frames per second limit
    duration: int = 2000 # Simulation duration in ticks, capped at 2000

    # --- Lotka-Volterra Specific Parameters ---
    rabbit_movement_speed: float = 3.0  # Separate movement speed for rabbits
    fox_movement_speed: float = 4.0     # Separate movement speed for foxes

    rabbit_perception_radius: int = 50
    fox_perception_radius: int = 70

    rabbit_reproduce_chance: float = 0.005  # Probability per tick for a rabbit to reproduce

    # Fox parameters
    fox_spontaneous_death_chance: float = 0.005 # Probability per tick for a fox to die spontaneously
    
    # --- Aggregation System Parameters (Toggleable) ---
    aggregation_enabled: bool = False # This will be set by the experiment runner

    # Parameters for the P_join probability function: P_join(n) = 1 - (1 / (1 + n * p_join_factor))
    p_join_factor: float = 10.0 

    # Parameters for the P_leave probability function: P_leave(n) = 1 / (1 + n * p_leave_factor)
    p_leave_factor: float = 50.0 

    # Timers for state transitions (in simulation ticks)
    t_join_min: int = 10   
    t_join_max: int = 20   
    t_leave_min: int = 180 
    t_leave_max: int = 360 

    # Frequency (in ticks) for checking P_leave in the Still state
    d_check_frequency: int = 30

    # Number of aggregation sites to spawn (fixed for simplicity in this setup)
    num_aggregation_sites: int = 2


# --- Agent Definitions ---

class Rabbit(Agent[LVConfig]):
    """
    Represents a Rabbit agent (Prey).
    Behaves passively, reproduces spontaneously, dies if eaten by a fox.
    Can aggregate around sites if aggregation_enabled is True.
    """
    _state: RabbitState = RabbitState.WANDERING
    _timer: int = 0         # General timer for JOINING and LEAVING states
    _check_timer: int = 0   # Timer for checking P_leave in STILL state

    def on_spawn(self):
        """Initializes rabbit movement, and sets its specific perception radius."""
        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        # Set the radius attribute of the config object specific to this agent type
        self.config.radius = self.config.rabbit_perception_radius
        self._state = RabbitState.WANDERING # Default state

    def _calculate_p_join(self, n_neighbors: int) -> float:
        """
        Calculates the probability of transitioning from Wandering to Joining a site.
        This probability increases with the number of perceived neighbors.
        """
        return 1.0 - (1.0 / (1.0 + n_neighbors * self.config.p_join_factor))

    def _calculate_p_leave(self, n_neighbors: int) -> float:
        """
        Calculates the probability of transitioning from Still to Leaving a site.
        This probability decreases as the number of perceived neighbors increases,
        encouraging agents to stay in denser aggregates.
        """
        return 1.0 / (1.0 + n_neighbors * self.config.p_leave_factor)


    def change_position(self):
        """
        Rabbit movement, reproduction logic, and state transitions for aggregation.
        """
        # Ensure the correct radius is active for this agent's proximity checks
        self.config.radius = self.config.rabbit_perception_radius

        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # --- Aggregation Logic (only if aggregation_enabled is True) ---
        if self.config.aggregation_enabled:
            if self._state == RabbitState.WANDERING:
                # Random walk: occasionally change direction slightly
                if self.shared.prng_move.random() < 0.1: # 10% chance to slightly change direction
                    self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
                    self.move.normalize_ip()
                    self.move *= self.config.rabbit_movement_speed

                # Check if the agent has entered a site and might transition to JOINING
                if self.on_site(): # This already checks if the agent is inside a site
                    # Count only other agents in proximity, not the agent itself
                    n_neighbors = len([agent for agent, dist in self.in_proximity_accuracy() if isinstance(agent, Rabbit) and agent.id != self.id])
                    p_join = self._calculate_p_join(n_neighbors)
                    
                    if self.shared.prng_move.random() < p_join:
                        self._state = RabbitState.JOINING
                        # Set a random timer for the JOINING state duration
                        self._timer = self.shared.prng_move.randint(self.config.t_join_min, self.config.t_join_max)
                        # Agent continues moving in the JOINING state
                        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

            elif self._state == RabbitState.JOINING:
                # Agent moves for a set duration
                self._timer -= 1
                if self._timer <= 0:
                    # IMPORTANT: Only transition to STILL if still inside a site
                    if self.on_site():
                        self._state = RabbitState.STILL
                        self.freeze_movement() # Stop moving when STILL
                        self._check_timer = self.config.d_check_frequency # Initialize check timer for leaving decisions
                    else:
                        # If agent left the site while JOINING, revert to WANDERING
                        self._state = RabbitState.WANDERING
                        self.continue_movement()
                        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

            elif self._state == RabbitState.STILL:
                # Agent is not moving, but periodically checks if it should LEAVE
                self.freeze_movement() # Ensure it remains still
                
                # Agents should only remain STILL if they are actually inside a site.
                if not self.on_site():
                    self._state = RabbitState.WANDERING
                    self.continue_movement()
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
                    self._timer = 0 # Reset timer, as it's no longer relevant for STILL state
                    self._check_timer = 0 # Reset check timer
                else: # Only proceed with leave check if still on site
                    self._check_timer -= 1
                    if self._check_timer <= 0:
                        self._check_timer = self.config.d_check_frequency # Reset check timer for next interval

                        # Count neighbors to decide on leaving
                        n_neighbors = len([agent for agent, dist in self.in_proximity_accuracy() if isinstance(agent, Rabbit) and agent.id != self.id])
                        p_leave = self._calculate_p_leave(n_neighbors)
                        
                        if self.shared.prng_move.random() < p_leave:
                            self._state = RabbitState.LEAVING
                            # Set a random timer for the LEAVING state duration
                            self._timer = self.shared.prng_move.randint(self.config.t_leave_min, self.config.t_leave_max)
                            self.continue_movement() # Start moving again
                            # Give a random direction to leave
                            self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

            elif self._state == RabbitState.LEAVING:
                # Agent moves for a set duration, then returns to wandering
                self._timer -= 1
                if self._timer <= 0:
                    self._state = RabbitState.WANDERING
                    # Give a random direction for wandering
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
            
            # Apply movement if the agent is not in the STILL state for aggregation scenario
            if self._state != RabbitState.STILL:
                self.pos += self.move
        else: # If aggregation is NOT enabled, rabbits just do random walk
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
    Hunts rabbits, reproduces by eating, dies spontaneously.
    """
    def on_spawn(self):
        """Initializes fox movement and sets its specific perception radius."""
        self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
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

        # --- Spontaneous Death Logic ---
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
            
            # For aggregation scenario, also check collision with sites during initial spawn
            if simulation_instance.config.aggregation_enabled and not is_colliding:
                 for site in simulation_instance._sites:
                    if pg.sprite.collide_mask(dummy_agent, site):
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
    config_overrides: dict,
    current_sim_index: int, 
    total_sims_in_scenario: int 
):
    """
    Runs a single simulation experiment with specified parameters and saves results.

    Args:
        run_name (str): A unique name for this specific run, used for folder creation.
        base_output_dir (str): The base directory to save results (e.g., "simulation_results/baseline_no_aggregation").
        initial_rabbits (int): Initial number of rabbit agents.
        initial_foxes (int): Initial number of fox agents.
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
    
    # Apply config overrides
    current_config = LVConfig()
    for key, value in config_overrides.items():
        setattr(current_config, key, value)

    sim = HeadlessSimulation(current_config)

    # Define paths for original images
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    rabbit_image_path = os.path.join(script_dir, "images", "rabbit.png")
    fox_image_path = os.path.join(script_dir, "images", "fox.png")
    site_small_image_path = os.path.join(script_dir, "images", "bubble-small-miku.png") 
    site_large_image_path = os.path.join(script_dir, "images", "bubble-full-miku.png") 

    # Variables to store the actual Pygame Surface objects for agents
    rabbit_images_list = []
    fox_images_list = []

    # Variables to store the FINAL paths (string) to be used for sites
    final_site_small_image_path = site_small_image_path
    final_site_large_image_path = site_large_image_path

    # Try loading real images, if not, create and save dummy images
    try:
        loaded_rabbit_image = pg.image.load(rabbit_image_path).convert_alpha()
        rabbit_images_list.append(loaded_rabbit_image)
        loaded_fox_image = pg.image.load(fox_image_path).convert_alpha()
        fox_images_list.append(loaded_fox_image)
        
        # Check if site images can be loaded; if not, they will be handled in the except block
        # We don't need to assign them here, just check if loading them causes an error.
        pg.image.load(site_small_image_path)
        pg.image.load(site_large_image_path)

    except pg.error as e:
        print(f"{YELLOW}Error loading images: {e}. Please ensure 'images/rabbit.png', 'images/fox.png' and site images exist.{RESET}")
        print(f"{YELLOW}Creating and saving dummy images to proceed.{RESET}")

        # Create dummy surfaces for agents
        dummy_rabbit_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_rabbit_surf, (0, 200, 0), (10, 10), 10) # Green circle
        rabbit_images_list.clear() # Clear any partially loaded images
        rabbit_images_list.append(dummy_rabbit_surf)

        dummy_fox_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_fox_surf, (200, 0, 0), (10, 10), 10) # Red circle
        fox_images_list.clear() # Clear any partially loaded images
        fox_images_list.append(dummy_fox_surf)

        # Create dummy surfaces for sites
        dummy_site_small_surf = pg.Surface((50, 50), pg.SRCALPHA)
        pg.draw.circle(dummy_site_small_surf, (100, 100, 255, 100), (25, 25), 25) # Light blue transparent circle
        dummy_site_large_surf = pg.Surface((80, 80), pg.SRCALPHA)
        pg.draw.circle(dummy_site_large_surf, (100, 100, 255, 100), (40, 40), 40) # Light blue transparent circle
        
        # Define a temporary directory to save dummy images within the run_output_dir
        # This ensures unique temporary files for each run and easier cleanup
        temp_dir_for_run = os.path.join(run_output_dir, "temp_site_images")
        os.makedirs(temp_dir_for_run, exist_ok=True) # Ensure temp directory exists

        # Generate unique filenames for dummy images within the run's temp directory
        dummy_site_small_filepath = os.path.join(temp_dir_for_run, f"dummy_site_small.png")
        dummy_site_large_filepath = os.path.join(temp_dir_for_run, f"dummy_site_large.png")

        try:
            # Save dummy site surfaces to files
            pg.image.save(dummy_site_small_surf, dummy_site_small_filepath)
            pg.image.save(dummy_site_large_surf, dummy_site_large_filepath)

            # Update the final paths to point to the newly created dummy files
            final_site_small_image_path = dummy_site_small_filepath
            final_site_large_image_path = dummy_site_large_filepath
        except pg.error as save_e:
            print(f"{RED}Critical Error: Could not save dummy site images to temp directory: {save_e}. Sites may not be displayed.{RESET}")
            # If saving fails, it's safer to ensure that spawn_site won't try to load a non-existent path.
            # While vi's spawn_site might draw a default if image is None, here we just ensure a string is passed.
            # If saving fails, the paths will remain as None, which *will* cause the TypeError.
            # For now, we'll keep it as it is, as the primary fix is the path consistency.
            # If this error persists, the vi library's spawn_site might need to be patched directly
            # to accept pygame.Surface objects directly or handle None image_path more gracefully.
            pass


    # Spawn initial populations using the *loaded Surface objects* for agents
    spawn_agent_safely(Rabbit, rabbit_images_list, initial_rabbits, sim)
    spawn_agent_safely(Fox, fox_images_list, initial_foxes, sim)
    
    # Conditionally spawn aggregation sites
    if current_config.aggregation_enabled:
        window_width, window_height = sim.config.window.as_tuple()
        site1_x = window_width // 4
        site2_x = 3 * window_width // 4
        site_y = window_height // 2

        # Spawn sites using the final image paths (string paths)
        # These paths will either be to the original images or the newly saved dummy images.
        if final_site_small_image_path:
            sim.spawn_site(final_site_small_image_path, x=site1_x, y=site_y) # Small site
        if final_site_large_image_path:
            sim.spawn_site(final_site_large_image_path, x=site2_x, y=site_y) # Large site

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
        original_sim_tick()

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

    # Save general metrics to JSON
    run_metrics = {
        'run_name': run_name,
        'parameters': config_overrides,
        'ticks_completed': ticks_completed,
        'final_rabbit_population': rabbit_population_history[-1] if rabbit_population_history else 0,
        'final_fox_population': fox_population_history[-1] if fox_population_history else 0,
        'extinction_tick_rabbits': extinction_tick_rabbits,
        'extinction_tick_foxes': extinction_tick_foxes,
        'peak_rabbits': peak_rabbits,
        'peak_foxes': peak_foxes,
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
    
    # Adjust ylim based on relevant populations
    max_pop = 0
    if rabbit_population_history:
        max_pop = max(max_pop, max(rabbit_population_history))
    if fox_population_history:
        max_pop = max(max_pop, max(fox_population_history))
    
    if max_pop > 0:
        plt.ylim(0, max_pop * 1.1 + 1)
    else:
        plt.ylim(0, 10) # Set a default small range if no data

    plt.tight_layout()
    plot_path = os.path.join(run_output_dir, f"{run_name}_population_dynamics.png")
    plt.savefig(plot_path)
    plt.close() # Close the plot to free memory
    print(f"Population plot saved to: {plot_path}")

    # Return all collected data for category-level aggregation, including histories
    return {
        'run_name': run_name,
        'config_overrides': config_overrides,
        'ticks_completed': ticks_completed,
        'extinction_tick_rabbits': extinction_tick_rabbits,
        'extinction_tick_foxes': extinction_tick_foxes,
        'peak_rabbits': peak_rabbits,
        'peak_foxes': peak_foxes,
        'rabbit_population_history': rabbit_population_history, 
        'fox_population_history': fox_population_history, 
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
    all_rabbit_population_histories = [] 
    all_fox_population_histories = [] 

    for result in results_list:
        # Collect scalar metrics
        if isinstance(result['extinction_tick_rabbits'], int):
            all_extinction_ticks_rabbits.append(result['extinction_tick_rabbits'])
        if isinstance(result['extinction_tick_foxes'], int):
            all_extinction_ticks_foxes.append(result['extinction_tick_foxes'])
        
        all_peak_rabbits.append(result['peak_rabbits'])
        all_peak_foxes.append(result['peak_foxes'])
        
        all_rabbit_population_histories.append(result['rabbit_population_history'])
        all_fox_population_histories.append(result['fox_population_history'])
        

    # Calculate Averages for scalar metrics
    avg_extinction_rabbits = np.mean(all_extinction_ticks_rabbits) if all_extinction_ticks_rabbits else "N/A"
    avg_extinction_foxes = np.mean(all_extinction_ticks_foxes) if all_extinction_ticks_foxes else "N/A"
    avg_peak_rabbits = np.mean(all_peak_rabbits) if all_peak_rabbits else 0
    avg_peak_foxes = np.mean(all_peak_foxes) if all_peak_foxes else 0

    # Save aggregated metrics to CSV
    aggregated_metrics = {
        'Category': category_name,
        'Avg_Extinction_Tick_Rabbits': avg_extinction_rabbits,
        'Avg_Extinction_Tick_Foxes': avg_extinction_foxes,
        'Avg_Peak_Rabbits': avg_peak_rabbits,
        'Avg_Peak_Foxes': avg_peak_foxes,
    }

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

    # Plot Averaged Population Dynamics Over Time
    if all_rabbit_population_histories or all_fox_population_histories: 
        # Determine the maximum length among all population histories for proper padding
        max_len_pop_r = max(len(h) for h in all_rabbit_population_histories) if all_rabbit_population_histories else 0
        max_len_pop_f = max(len(h) for h in all_fox_population_histories) if all_fox_population_histories else 0
        max_len_pop = max(max_len_pop_r, max_len_pop_f) 

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

            avg_rabbit_pop_time_series = np.mean(padded_rabbit_histories, axis=0)
            avg_fox_pop_time_series = np.mean(padded_fox_histories, axis=0)

            plt.figure(figsize=(12, 7))
            plt.plot(np.arange(len(avg_rabbit_pop_time_series)), avg_rabbit_pop_time_series, label='Average Rabbits (Prey)', color='darkgreen', linewidth=2)
            plt.plot(np.arange(len(avg_fox_pop_time_series)), avg_fox_pop_time_series, label='Average Foxes (Predator)', color='darkred', linewidth=2, linestyle='--')
            
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
    'rabbit_reproduce_chance': 'rrc',
    'fox_spontaneous_death_chance': 'fsdc',
    'p_join_factor': 'pjf',
    'p_leave_factor': 'plf',
    't_join_min': 'tjm',
    't_join_max': 'tjx',
    't_leave_min': 'tlm',
    't_leave_max': 'tlx',
    'd_check_frequency': 'dcf',
    'num_aggregation_sites': 'nas'
}

def run_grid_search_for_scenario(
    scenario_name: str,
    initial_rabbits: int,
    initial_foxes: int,
    fixed_config_params: dict, 
    varying_params_grid: dict 
):
    """
    Performs a grid search over specified parameters for a given scenario.
    """
    category_output_dir = os.path.join("simulation_results_aggregation", scenario_name)
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

    # --- Prepare data for Extinction and Peak Population Plots ---
    ext_labels = []
    ext_rabbits = []
    ext_foxes = []
    peak_labels = []
    peak_rabbits = []
    peak_foxes = []

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
        ax_peak.bar(x - width/2, peak_rabbits, width, label='Rabbits', color='lightgreen')
        ax_peak.bar(x + width/2, peak_foxes, width, label='Foxes', color='salmon')
        ax_peak.set_ylabel('Average Peak Population', fontsize=10)
        ax_peak.set_title('Average Peak Populations', fontsize=14)
        ax_peak.set_xticks(x)
        ax_peak.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=9)
        ax_peak.legend(fontsize=9)
        ax_peak.grid(axis='y', linestyle='--', alpha=0.7)
        max_all_peak_pop = max(max(peak_rabbits), max(peak_foxes))
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
                                   label=bar_container.get_label())
        # Fix for legend from bars
        handles, labels = [], []
        for container in ax_ext.containers:
            if container.get_label(): 
                handles.append(container.patches[0]) 
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
        
        for name in scenario_names:
            metrics = scenario_metrics.get(name)
            peak_row_rabbits[name.replace("_", " ").title()] = metrics['Avg_Peak_Rabbits'] if metrics is not None else 0
            peak_row_foxes[name.replace("_", " ").title()] = metrics['Avg_Peak_Foxes'] if metrics is not None else 0

        combined_metrics_df = pd.concat([combined_metrics_df, pd.DataFrame([peak_row_rabbits, peak_row_foxes])], ignore_index=True)

        combined_metrics_csv_path = os.path.join(base_results_dir, "compared_aggregated_metrics.csv")
        combined_metrics_df.to_csv(combined_metrics_csv_path, index=False)
        print(f"Combined aggregated metrics data saved to: {combined_metrics_csv_path}")

# --- Main Experiment Execution Block ---
if __name__ == "__main__":
    # Ensure Pygame is initialized only once for all runs
    pg.init() 

    # Corrected: Use "simulation_results" to match where run_grid_search_for_scenario saves results
    base_results_dir = "simulation_results_aggregation" 
    os.makedirs(base_results_dir, exist_ok=True)

    # Common initial populations
    initial_rabbits = 100
    initial_foxes = 10
    
    # Get the fixed duration from LVConfig
    fixed_duration = LVConfig().duration

    # --- No-Aggregation Scenario Grid Search ---
    no_aggregation_params_to_vary = {
        'rabbit_movement_speed': [2.0, 2.0],
        'fox_movement_speed': [3.0, 3.0],
        'rabbit_perception_radius': [50, 50],
        'fox_perception_radius': [70, 70],
        'rabbit_reproduce_chance': [0.005, 0.005],
        'fox_spontaneous_death_chance': [0.005, 0.005]
    }
    fixed_no_aggregation_params = {
        'aggregation_enabled': False,
    }
    
    # Calculate and print total combinations for no-aggregation scenario
    no_aggregation_keys = sorted(no_aggregation_params_to_vary.keys())
    no_aggregation_values = [no_aggregation_params_to_vary[key] for key in no_aggregation_keys]
    total_no_aggregation_combinations = len(list(product(*no_aggregation_values)))
    print(f"\n{BOLD}Total combinations for No-Aggregation Scenario: {BOLD}{total_no_aggregation_combinations}{RESET}")

    run_grid_search_for_scenario(
        scenario_name=NO_AGGREGATION_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        fixed_config_params=fixed_no_aggregation_params,
        varying_params_grid=no_aggregation_params_to_vary
    )

    # --- Aggregation-Enabled Scenario Grid Search ---
    aggregation_enabled_params_to_vary = {
        'p_join_factor': [10.0, 10.0], 
        'p_leave_factor': [50.0, 50.0], 
        't_join_min': [10, 10],   
        't_join_max': [20, 20],   
        't_leave_min': [180, 180], 
        't_leave_max': [360, 360], 
        'd_check_frequency': [30],
        'num_aggregation_sites': [2], # Fixed number of sites for now
        'rabbit_movement_speed': [2.0],
        'fox_movement_speed': [3.0],
        'rabbit_perception_radius': [50],
        'fox_perception_radius': [70],
        'rabbit_reproduce_chance': [0.005], # Keep consistent with no-agg for comparison
        'fox_spontaneous_death_chance': [0.005] # Keep consistent with no-agg for comparison
    }
    fixed_aggregation_enabled_params = {
        'aggregation_enabled': True,
    }

    # Calculate and print total combinations for aggregation-enabled scenario
    aggregation_enabled_keys = sorted(aggregation_enabled_params_to_vary.keys())
    aggregation_enabled_values = [aggregation_enabled_params_to_vary[key] for key in aggregation_enabled_keys]
    total_aggregation_enabled_combinations = len(list(product(*aggregation_enabled_values)))
    print(f"\n{BOLD}Total combinations for Aggregation-Enabled Scenario: {BOLD}{total_aggregation_enabled_combinations}{RESET}")

    run_grid_search_for_scenario(
        scenario_name=AGGREGATION_ENABLED_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        fixed_config_params=fixed_aggregation_enabled_params,
        varying_params_grid=aggregation_enabled_params_to_vary
    )
    
    # --- Cross-Scenario Comparison ---
    # Call the new function to generate comparison plots and data
    compare_scenarios(base_results_dir, [NO_AGGREGATION_SCENARIO_NAME, AGGREGATION_ENABLED_SCENARIO_NAME])

    # Ensure Pygame is quit only once after all runs
    pg.quit()
    print(f"\n{CYAN}{BOLD}--- All simulations concluded. ---{RESET}")
    print(f"{CYAN}Results saved in the '{base_results_dir}' directory.{RESET}")
