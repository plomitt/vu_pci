import pygame as pg
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import json
from itertools import product
from matplotlib.gridspec import GridSpec

# Removed direct import of Obstacle as it's not exposed for top-level import
from vi import Agent, Config, HeadlessSimulation, Simulation
from pygame.math import Vector2

# --- ANSI Escape Codes for Pretty Printing ---
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
CYAN = "\033[96m"

# --- Scenario Name Constants ---
PROBABILISTIC_SCENARIO_NAME = "no_flocking"
FLOCKING_SCENARIO_NAME = "with_flocking"

# --- Configuration for Lotka-Volterra Simulation with Flocking ---
@dataclass
class LVConfig(Config):
    """
    Configuration settings for the Lotka-Volterra Predator-Prey simulation
    with optional Flocking behavior.
    """
    # General Simulation Parameters
    image_rotation: bool = True
    fps_limit: int = 60
    duration: int = 500

    # --- Lotka-Volterra Specific Parameters ---
    rabbit_movement_speed: float = 2.0
    fox_movement_speed: float = 3.0

    # Perception radii for interaction (predation/prey) and flocking
    rabbit_perception_radius: int = 70 # Rabbits need to see other rabbits for flocking
    fox_perception_radius: int = 100 # Foxes need to see other foxes for flocking

    rabbit_reproduce_chance: float = 0.005
    fox_spontaneous_death_chance: float = 0.005
    
    # --- Flocking System Parameters (Toggleable) ---
    flocking_enabled: bool = False # This will be set by the experiment runner
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_weight: float = 1.0

# --- Agent Definitions ---

class Rabbit(Agent[LVConfig]):
    """
    Represents a Rabbit agent (Prey).
    Behaves passively, reproduces spontaneously, dies if eaten by a fox.
    Can exhibit flocking behavior if enabled.
    """
    def on_spawn(self):
        """Initializes rabbit movement and sets its specific perception radius."""
        self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        self.config.radius = self.config.rabbit_perception_radius

    def change_position(self):
        """
        Rabbit movement, reproduction logic, and potential flocking.
        """
        self.config.radius = self.config.rabbit_perception_radius

        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # Check for nearby foxes (predators)
        flee_vector = Vector2(0, 0)
        fox_nearby = False
        for neighbor, distance in self.in_proximity_accuracy():
            if isinstance(neighbor, Fox) and neighbor.is_alive():
                # Flee from foxes - prioritize fleeing over flocking if a fox is very close
                # Inverse square law for stronger repulsion when closer
                if distance > 0: # Ensure distance is greater than 0 before normalizing
                    flee_vector += (self.pos - neighbor.pos).normalize() / distance
                    fox_nearby = True
                
        if fox_nearby:
            # If a fox is nearby, prioritize fleeing
            if flee_vector.length_squared() > 1e-6: # Use length_squared for robustness
                self.move += flee_vector.normalize() * self.config.rabbit_movement_speed
            else:
                # If flee_vector is effectively zero, assign a random movement to avoid standing still
                self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

            if self.move.length_squared() > self.config.rabbit_movement_speed**2: # Compare squared lengths
                if self.move.length_squared() > 1e-6: # Ensure self.move has a non-zero length before normalizing
                    self.move.normalize_ip()
                    self.move *= self.config.rabbit_movement_speed
                else:
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360)) # Re-initialize if zero
        elif self.config.flocking_enabled:
            # Apply flocking behavior if no immediate threat and flocking is enabled
            alignment_vector = Vector2(0, 0)
            cohesion_vector = Vector2(0, 0)
            separation_vector = Vector2(0, 0)
            
            neighbor_count = 0
            for neighbor, distance in self.in_proximity_accuracy():
                # Only consider other rabbits for flocking
                if isinstance(neighbor, Rabbit) and neighbor is not self:
                    neighbor_count += 1
                    
                    # Alignment: Steer towards the average direction of local boids
                    alignment_vector += neighbor.move.normalize() if neighbor.move.length_squared() > 1e-6 else Vector2(0,0)

                    # Cohesion: Steer towards the average position (center of mass) of local boids
                    cohesion_vector += neighbor.pos

                    # Separation: Steer away to avoid being in too close proximity to local boids
                    if distance > 0: # Check distance > 0 to prevent ValueError on normalize() if agents are at same spot
                        separation_vector += (self.pos - neighbor.pos).normalize() / distance
            
            if neighbor_count > 0:
                alignment_vector = (alignment_vector / neighbor_count)
                if alignment_vector.length_squared() > 1e-6:
                    alignment_vector.normalize_ip()
                else:
                    alignment_vector = Vector2(0,0)

                cohesion_center = cohesion_vector / neighbor_count
                cohesion_vector = (cohesion_center - self.pos)
                if cohesion_vector.length_squared() > 1e-6:
                    cohesion_vector.normalize_ip()
                else:
                    cohesion_vector = Vector2(0,0)
                
                separation_vector = separation_vector
                if separation_vector.length_squared() > 1e-6:
                    separation_vector.normalize_ip()
                else:
                    separation_vector = Vector2(0,0)
            else:
                # If no neighbors, no flocking forces, revert to random walk tendencies
                alignment_vector = Vector2(0,0)
                cohesion_vector = Vector2(0,0)
                separation_vector = Vector2(0,0)

            steering_force = (
                self.config.separation_weight * separation_vector +
                self.config.cohesion_weight * cohesion_vector +
                self.config.alignment_weight * alignment_vector
            )
            
            self.move += steering_force
            # Limit the speed of the agent using the `movement_speed` from the config
            if self.move.length_squared() > self.config.rabbit_movement_speed**2:
                if self.move.length_squared() > 1e-6: # Ensure self.move has a non-zero length before normalizing
                    self.move.normalize_ip()
                    self.move *= self.config.rabbit_movement_speed
                else:
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360)) # Re-initialize if zero
        else:
            # Default random walk if no flocking and no fox nearby
            if self.shared.prng_move.random() < 0.1:
                # If current move vector is zero, re-initialize it to a random direction
                if self.move.length_squared() < 1e-6:
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
                else:
                    # Otherwise, slightly rotate the existing non-zero move vector
                    self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))

                # After potential rotation/initialization, ensure it's normalized to speed if not zero
                if self.move.length_squared() > 1e-6: # Essential check before normalize_ip()
                    self.move.normalize_ip()
                    self.move *= self.config.rabbit_movement_speed
                else:
                    # Fallback for extreme edge cases where rotation somehow yields a zero vector
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

        self.pos += self.move

        # Obstacle collision handling: Reflect movement vector for a bounce effect
        # Iterate through all known obstacles in the simulation
        for obstacle_sprite in self._Agent__simulation._obstacles:
            # Check if this rabbit is colliding with the current obstacle
            if pg.sprite.collide_mask(self, obstacle_sprite):
                # Calculate vector from obstacle center to rabbit center (collision normal)
                # This vector points from the obstacle OUTWARDS towards the agent
                # Use obstacle_sprite.rect.center as the obstacle's position
                vec_from_obstacle = self.pos - Vector2(obstacle_sprite.rect.center)
                if vec_from_obstacle.length_squared() > 1e-6:
                    collision_normal = vec_from_obstacle.normalize()
                else:
                    # If at the exact center of an obstacle, push in a random direction
                    collision_normal = Vector2(1, 0).rotate(self.shared.prng_move.uniform(0, 360))
                
                # Move agent slightly out of collision along the normal
                # This step is crucial to prevent the agent from getting stuck inside the obstacle
                self.pos += collision_normal * 5  # Nudge by 5 pixels along the normal

                # Reflect the movement vector off the collision normal
                # Formula: R = V - 2 * (V . N) * N, where V is current move, N is collision_normal
                dot_product = self.move.dot(collision_normal)
                reflected_move = self.move - 2 * dot_product * collision_normal
                
                # Update movement vector with the reflected direction, maintaining the agent's speed
                if reflected_move.length_squared() > 1e-6: # Ensure reflected move is not zero
                    self.move = reflected_move.normalize() * self.config.rabbit_movement_speed
                else:
                    self.move = Vector2(self.config.rabbit_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360)) # Assign random if reflected is zero
                
                # Break after handling the first collision to prevent multiple reflections
                # and potential erratic behavior if colliding with multiple obstacles simultaneously.
                break 

        # --- Boundary Bouncing Logic for Rabbit ---
        if self.pos.x - agent_half_width < 0:
            self.pos.x = agent_half_width
            self.move.x *= -1
        elif self.pos.x + agent_half_width > window_width:
            self.pos.x = window_width - agent_half_width
            self.move.x *= -1

        if self.pos.y - agent_half_height < 0:
            self.pos.y = agent_half_height
            self.move.y *= -1
        elif self.pos.y + agent_half_height > window_height:
            self.pos.y = window_height - agent_half_height
            self.move.y *= -1

        # Spontaneous asexual reproduction for rabbits
        if self.shared.prng_move.random() < self.config.rabbit_reproduce_chance:
            new_rabbit_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
            Rabbit(images=self._images, simulation=self._Agent__simulation, pos=new_rabbit_pos)


class Fox(Agent[LVConfig]):
    """
    Represents a Fox agent (Predator).
    Hunts rabbits, reproduces by eating.
    Can exhibit flocking behavior if enabled.
    """
    def on_spawn(self):
        """Initializes fox movement and sets its specific perception radius."""
        self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
        self.config.radius = self.config.fox_perception_radius

    def change_position(self):
        """
        Fox movement, hunting, reproduction (by eating), and death logic.
        """
        self.config.radius = self.config.fox_perception_radius

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
            # Prioritize hunting if a rabbit is detected
            vec_to_rabbit = target_rabbit.pos - self.pos
            if vec_to_rabbit.length_squared() > 1e-6: # Use length_squared for robustness
                direction_to_rabbit = vec_to_rabbit.normalize()
                self.move = direction_to_rabbit * self.config.fox_movement_speed
            else:
                self.move = Vector2(0,0) # Fox is on top of rabbit, no movement needed for hunting

            if self.pos.distance_to(target_rabbit.pos) < 10: # Close enough to eat
                target_rabbit.kill()
                
                # Fox reproduces directly after eating a rabbit
                new_fox_pos = self.pos + Vector2(self.shared.prng_move.uniform(-10, 10), self.shared.prng_move.uniform(-10, 10))
                Fox(images=self._images, simulation=self._Agent__simulation, pos=new_fox_pos)
        elif self.config.flocking_enabled:
            # Apply flocking behavior if no immediate prey and flocking is enabled
            alignment_vector = Vector2(0, 0)
            cohesion_vector = Vector2(0, 0)
            separation_vector = Vector2(0, 0)
            
            neighbor_count = 0
            for neighbor, distance in self.in_proximity_accuracy():
                # Only consider other foxes for flocking
                if isinstance(neighbor, Fox) and neighbor is not self:
                    neighbor_count += 1
                    
                    alignment_vector += neighbor.move.normalize() if neighbor.move.length_squared() > 1e-6 else Vector2(0,0)
                    cohesion_vector += neighbor.pos
                    # Added check for distance > 0 to prevent ValueError on normalize() if agents are at same spot
                    if distance > 0:
                        separation_vector += (self.pos - neighbor.pos).normalize() / distance
            
            if neighbor_count > 0:
                alignment_vector = (alignment_vector / neighbor_count)
                if alignment_vector.length_squared() > 1e-6:
                    alignment_vector.normalize_ip()
                else:
                    alignment_vector = Vector2(0,0)

                cohesion_center = cohesion_vector / neighbor_count
                cohesion_vector = (cohesion_center - self.pos)
                if cohesion_vector.length_squared() > 1e-6:
                    cohesion_vector.normalize_ip()
                else:
                    cohesion_vector = Vector2(0,0)
                
                separation_vector = separation_vector
                if separation_vector.length_squared() > 1e-6:
                    separation_vector.normalize_ip()
                else:
                    separation_vector = Vector2(0,0)
            else:
                # If no neighbors, no flocking forces, revert to random walk tendencies
                alignment_vector = Vector2(0,0)
                cohesion_vector = Vector2(0,0)
                separation_vector = Vector2(0,0)

            steering_force = (
                self.config.separation_weight * separation_vector +
                self.config.cohesion_weight * cohesion_vector +
                self.config.alignment_weight * alignment_vector
            )
            
            self.move += steering_force
            # Ensure self.move has a valid length before normalizing/limiting speed
            if self.move.length_squared() > self.config.fox_movement_speed**2:
                if self.move.length_squared() > 1e-6: # Only normalize if it has a non-zero length
                    self.move.normalize_ip()
                    self.move *= self.config.fox_movement_speed
                else:
                    # If self.move became zero due to steering forces cancelling out, give it a random direction
                    self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360)) # Assign a new random direction
        else:
            # Default random walk if no flocking and no rabbit nearby
            if self.shared.prng_move.random() < 0.1:
                # If current move vector is zero, re-initialize it to a random direction
                # This handles cases where .move might have become zero from previous logic (e.g., hunting)
                if self.move.length_squared() < 1e-6:
                    self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))
                else:
                    # Otherwise, slightly rotate the existing non-zero move vector
                    self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))

                # After potential rotation/initialization, ensure it's normalized to speed if not zero
                if self.move.length_squared() > 1e-6: # Essential check before normalize_ip()
                    self.move.normalize_ip()
                    self.move *= self.config.fox_movement_speed
                else:
                    # Fallback for extreme edge cases where rotation somehow yields a zero vector
                    self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

        self.pos += self.move

        # Obstacle collision handling: Reflect movement vector for a bounce effect
        # Iterate through all known obstacles in the simulation
        for obstacle_sprite in self._Agent__simulation._obstacles:
            # Check if this fox is colliding with the current obstacle
            if pg.sprite.collide_mask(self, obstacle_sprite):
                # Calculate vector from obstacle center to fox center (collision normal)
                # Use obstacle_sprite.rect.center as the obstacle's position
                vec_from_obstacle = self.pos - Vector2(obstacle_sprite.rect.center)
                if vec_from_obstacle.length_squared() > 1e-6:
                    collision_normal = vec_from_obstacle.normalize()
                else:
                    # If at the exact center of an obstacle, push in a random direction
                    collision_normal = Vector2(1, 0).rotate(self.shared.prng_move.uniform(0, 360))

                # Move agent slightly out of collision along the normal
                self.pos += collision_normal * 5  # Nudge by 5 pixels along the normal

                # Reflect the movement vector
                dot_product = self.move.dot(collision_normal)
                reflected_move = self.move - 2 * dot_product * collision_normal
                
                # Update movement and maintain speed
                if reflected_move.length_squared() > 1e-6: # Ensure reflected move is not zero
                    self.move = reflected_move.normalize() * self.config.fox_movement_speed
                else:
                    self.move = Vector2(self.config.fox_movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360)) # Assign random if reflected is zero
                
                # Break after handling the first collision
                break

        # --- Boundary Bouncing Logic for Fox ---
        if self.pos.x - agent_half_width < 0:
            self.pos.x = agent_half_width
            self.move.x *= -1
        elif self.pos.x + agent_half_width > window_width:
            self.pos.x = window_width - agent_half_width
            self.move.x *= -1

        if self.pos.y - agent_half_height < 0:
            self.pos.y = agent_half_height
            self.move.y *= -1
        elif self.pos.y + agent_half_height > window_height:
            self.pos.y = window_height - agent_half_height
            self.move.y *= -1

        # Spontaneous Death Logic for foxes (always active in this modified version)
        if self.shared.prng_move.random() < self.config.fox_spontaneous_death_chance:
            self.kill()

# --- Helper function for spawning agents in non-colliding positions ---
def spawn_agent_safely(agent_class: Type[Agent], images_list: List[pg.Surface], num_to_spawn: int, simulation_instance: Simulation):
    count = 0
    max_spawn_attempts_per_agent = 500
    
    window_width, window_height = simulation_instance.config.window.as_tuple()

    dummy_agent = agent_class(images=images_list, simulation=simulation_instance, pos=Vector2(-100, -100))
    dummy_agent_half_width = dummy_agent.rect.width / 2
    dummy_agent_half_height = dummy_agent.rect.height / 2

    while count < num_to_spawn:
        attempts = 0
        is_valid_pos = False
        while attempts < max_spawn_attempts_per_agent:
            rand_x = random.randint(int(dummy_agent_half_width), int(window_width - dummy_agent_half_width))
            rand_y = random.randint(int(dummy_agent_half_height), int(window_height - dummy_agent_half_height))
            proposed_pos = Vector2(rand_x, rand_y)

            dummy_agent.pos = proposed_pos
            dummy_agent.rect.center = proposed_pos.xy

            is_colliding = False
            for existing_agent in simulation_instance._agents:
                if existing_agent is not dummy_agent and pg.sprite.collide_mask(dummy_agent, existing_agent):
                    is_colliding = True
                    break
            
            # Check for collision with obstacles by checking if it's not an Agent instance
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
    
    dummy_agent.kill()
    return count

# --- Experiment Runner Function ---
def run_experiment(
    run_name: str, 
    base_output_dir: str,
    initial_rabbits: int, 
    initial_foxes: int, 
    config_overrides: dict,
    current_sim_index: int, 
    total_sims_in_scenario: int,
    spawn_obstacles: bool = False # New parameter to control obstacle spawning
):
    """
    Runs a single simulation experiment with specified parameters and saves results.

    Args:
        run_name (str): A unique name for this specific run, used for folder creation.
        base_output_dir (str): The base directory to save results.
        initial_rabbits (int): Initial number of rabbit agents.
        initial_foxes (int): Initial number of fox agents.
        config_overrides (dict): A dictionary of LVConfig parameters to override defaults.
        current_sim_index (int): The current index of the simulation in the grid search (1-based).
        total_sims_in_scenario (int): The total number of simulations in the current grid search scenario.
        spawn_obstacles (bool): If True, obstacles will be spawned in the simulation.
    """
    print(f"\n{GREEN}{BOLD}--- Running simulation {current_sim_index}/{total_sims_in_scenario}: {run_name} ---{RESET}")
    print(f"{CYAN}Parameters:{RESET}")
    for key, value in config_overrides.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {BOLD}{CYAN}{value}{RESET}")
        else:
            print(f"  {key}: {CYAN}{value}{RESET}")

    run_output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    rabbit_population_history = []
    fox_population_history = []
    
    current_config = LVConfig()
    for key, value in config_overrides.items():
        setattr(current_config, key, value)

    sim = HeadlessSimulation(current_config)

    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    rabbit_image_path = os.path.join(script_dir, "images", "rabbit.png")
    fox_image_path = os.path.join(script_dir, "images", "fox.png")
    
    try:
        loaded_rabbit_image = pg.image.load(rabbit_image_path).convert_alpha()
        rabbit_images_list = [loaded_rabbit_image]
        loaded_fox_image = pg.image.load(fox_image_path).convert_alpha()
        fox_images_list = [loaded_fox_image]
    except pg.error as e:
        print(f"{YELLOW}Error loading images: {e}. Please ensure 'images/rabbit.png' and 'images/fox.png' exist in the script directory.{RESET}")
        print(f"{YELLOW}Creating dummy images for agents to continue simulation.{RESET}")
        dummy_rabbit_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_rabbit_surf, (0, 200, 0), (10, 10), 10)
        rabbit_images_list = [dummy_rabbit_surf]

        dummy_fox_surf = pg.Surface((20, 20), pg.SRCALPHA)
        pg.draw.circle(dummy_fox_surf, (200, 0, 0), (10, 10), 10)
        fox_images_list = [dummy_fox_surf]

    # Spawn initial populations
    spawn_agent_safely(Rabbit, rabbit_images_list, initial_rabbits, sim)
    spawn_agent_safely(Fox, fox_images_list, initial_foxes, sim)

    # Conditionally spawn obstacles for the flocking scenario
    if spawn_obstacles:
        obstacle_image_path = os.path.join(script_dir, "images", "bubble-small.png")
        try:
            loaded_obstacle_image = pg.image.load(obstacle_image_path).convert_alpha()
            sim.spawn_obstacle(obstacle_image_path, x=300, y=300) 
            sim.spawn_obstacle(obstacle_image_path, x=700, y=200) 
            sim.spawn_obstacle(obstacle_image_path, x=500, y=500)
        except pg.error as e:
            print(f"{YELLOW}Warning: Could not load obstacle image '{obstacle_image_path}'. Obstacles will not be spawned. Error: {e}{RESET}")

    original_sim_tick = sim.tick

    def custom_tick_for_run(self):
        # The following event handling loop is not needed for headless simulations
        # and can cause errors if a display system is not initialized.
        # for event in pg.event.get():
        #     if event.type == pg.QUIT:
        #         self._running = False
        #         return

        current_rabbits = len([agent for agent in self._agents if isinstance(agent, Rabbit)])
        current_foxes = len([agent for agent in self._agents if isinstance(agent, Fox)])
        
        rabbit_population_history.append(current_rabbits)
        fox_population_history.append(current_foxes)

        if current_foxes == 0 or current_rabbits == 0:
            extinction_type = 'foxes' if current_foxes == 0 else 'rabbits'
            print(f"\n{RED}Simulation ended at tick {BOLD}{len(rabbit_population_history) - 1}{RESET}{RED}: All {extinction_type} died.{RESET}")
            self._running = False
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return
        
        if len(rabbit_population_history) > self.config.duration:
            print(f"\n{YELLOW}Simulation ended at tick {BOLD}{len(rabbit_population_history) - 1}{RESET}{YELLOW}: Max duration reached.{RESET}")
            self._running = False
            if hasattr(self, '_metrics') and hasattr(self._metrics, '_record_snapshots'):
                self._metrics._record_snapshots = False
            return

        sys.stdout.write(f"\rCurrent Tick: {BOLD}{len(rabbit_population_history)}{RESET}   ")
        sys.stdout.flush()

        original_sim_tick()

    sim.tick = custom_tick_for_run.__get__(sim, type(sim))

    sim.run()

    sys.stdout.write("\n")

    ticks_completed = len(rabbit_population_history) - 1
    
    extinction_tick_rabbits = ticks_completed if rabbit_population_history and rabbit_population_history[-1] == 0 else "N/A"
    extinction_tick_foxes = ticks_completed if fox_population_history and fox_population_history[-1] == 0 else "N/A"
    
    peak_rabbits = max(rabbit_population_history) if rabbit_population_history else 0
    peak_foxes = max(fox_population_history) if fox_population_history else 0

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

    df_populations = pd.DataFrame({
        'Tick': np.arange(len(rabbit_population_history)),
        'Rabbits': rabbit_population_history,
        'Foxes': fox_population_history,
    })
    csv_path = os.path.join(run_output_dir, f"{run_name}_population_data.csv")
    df_populations.to_csv(csv_path, index=False)
    print(f"Population data saved to: {csv_path}")

    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(len(rabbit_population_history)), rabbit_population_history, label='Rabbits (Prey)', color='green', linewidth=2)
    plt.plot(np.arange(len(fox_population_history)), fox_population_history, label='Foxes (Predator)', color='red', linewidth=2, linestyle='--')
    plt.title(f'Population Dynamics for {run_name}', fontsize=16)
    plt.xlabel('Time (Simulation Ticks)', fontsize=12)
    plt.ylabel('Population Count', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    max_pop = 0
    if rabbit_population_history:
        max_pop = max(max_pop, max(rabbit_population_history))
    if fox_population_history:
        max_pop = max(max_pop, max(fox_population_history))
    
    if max_pop > 0:
        plt.ylim(0, max_pop * 1.1 + 1)
    else:
        plt.ylim(0, 10)

    plt.tight_layout()
    plot_path = os.path.join(run_output_dir, f"{run_name}_population_dynamics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Population plot saved to: {plot_path}")

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

    all_extinction_ticks_rabbits = []
    all_extinction_ticks_foxes = []
    all_peak_rabbits = []
    all_peak_foxes = []
    all_rabbit_population_histories = [] # Histories for *this* scenario, not all overall
    all_fox_population_histories = []     # Histories for *this* scenario, not all overall

    for result in results_list:
        if isinstance(result['extinction_tick_rabbits'], int):
            all_extinction_ticks_rabbits.append(result['extinction_tick_rabbits'])
        if isinstance(result['extinction_tick_foxes'], int):
            all_extinction_ticks_foxes.append(result['extinction_tick_foxes'])
        all_peak_rabbits.append(result['peak_rabbits'])
        all_peak_foxes.append(result['peak_foxes'])
        all_rabbit_population_histories.append(result['rabbit_population_history'])
        all_fox_population_histories.append(result['fox_population_history'])

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
    df_aggregated = pd.DataFrame([aggregated_metrics])
    aggregated_csv_path = os.path.join(category_output_dir, f"{category_name}_aggregated_metrics.csv")
    df_aggregated.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated metrics saved to: {aggregated_csv_path}")

    # Plot average population dynamics
    max_len = max(len(hist) for hist in all_rabbit_population_histories + all_fox_population_histories) if all_rabbit_population_histories or all_fox_population_histories else 0
    
    # Pad shorter histories with their last value to match max_len
    padded_rabbit_histories = [np.pad(hist, (0, max_len - len(hist)), 'edge') for hist in all_rabbit_population_histories]
    padded_fox_histories = [np.pad(hist, (0, max_len - len(hist)), 'edge') for hist in all_fox_population_histories]

    avg_rabbit_pop = np.mean(padded_rabbit_histories, axis=0) if padded_rabbit_histories else np.array([])
    avg_fox_pop = np.mean(padded_fox_histories, axis=0) if padded_fox_histories else np.array([])

    # Save average population histories as JSON for cross-scenario comparison
    with open(os.path.join(category_output_dir, f"{category_name}_average_rabbit_pop.json"), 'w') as f:
        json.dump(avg_rabbit_pop.tolist(), f)
    with open(os.path.join(category_output_dir, f"{category_name}_average_fox_pop.json"), 'w') as f:
        json.dump(avg_fox_pop.tolist(), f)

    plt.figure(figsize=(12, 7))
    if avg_rabbit_pop.size > 0:
        plt.plot(np.arange(len(avg_rabbit_pop)), avg_rabbit_pop, label='Average Rabbits', color='darkgreen', linewidth=2)
    if avg_fox_pop.size > 0:
        plt.plot(np.arange(len(avg_fox_pop)), avg_fox_pop, label='Average Foxes', color='darkred', linewidth=2, linestyle='--')
    
    plt.title(f'Average Population Dynamics for {category_name}', fontsize=16)
    plt.xlabel('Time (Simulation Ticks)', fontsize=12)
    plt.ylabel('Average Population Count', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    max_avg_pop = 0
    if avg_rabbit_pop.size > 0:
        max_avg_pop = max(max_avg_pop, np.max(avg_rabbit_pop))
    if avg_fox_pop.size > 0:
        max_avg_pop = max(max_avg_pop, np.max(avg_fox_pop))
    
    if max_avg_pop > 0:
        plt.ylim(0, max_avg_pop * 1.1 + 1)
    else:
        plt.ylim(0, 10)

    plt.tight_layout()
    avg_plot_path = os.path.join(category_output_dir, f"{category_name}_average_population_dynamics.png")
    plt.savefig(avg_plot_path)
    plt.close()
    print(f"Average population plot saved to: {avg_plot_path}")


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

# Define parameter abbreviations for consistent file naming
param_abbreviations = {
    'rabbit_movement_speed': 'rms',
    'fox_movement_speed': 'fms',
    'rabbit_perception_radius': 'rpr',
    'fox_perception_radius': 'fpr',
    'rabbit_reproduce_chance': 'rrc',
    'fox_spontaneous_death_chance': 'fsdc',
    'alignment_weight': 'aw',
    'cohesion_weight': 'cw',
    'separation_weight': 'sw'
}

# --- Grid Search Runner Function ---
def run_grid_search_for_scenario(
    scenario_name: str,
    initial_rabbits: int,
    initial_foxes: int,
    fixed_config_params: dict,
    varying_params_grid: dict,
    base_results_dir: str,
    spawn_obstacles_for_scenario: bool = False # New parameter
):
    """
    Executes a grid search for a given scenario, running multiple simulations
    and collecting their results.
    """
    scenario_results_dir = os.path.join(base_results_dir, scenario_name)
    os.makedirs(scenario_results_dir, exist_ok=True)
    
    keys = sorted(varying_params_grid.keys())
    values = [varying_params_grid[key] for key in keys]
    
    all_combinations = list(product(*values))
    total_combinations = len(all_combinations)

    scenario_all_run_results = []

    for i, combination in enumerate(all_combinations):
        current_overrides = dict(fixed_config_params)
        run_name_parts = []
        for j, key in enumerate(keys):
            current_overrides[key] = combination[j]
            abbreviated_key = param_abbreviations.get(key, key)
            # Format value: replace '.' with 'p' for floats, convert to string
            formatted_value = str(combination[j]).replace('.', 'p')
            run_name_parts.append(f"{abbreviated_key}{formatted_value}")
        
        # Ensure duration is always set from the config's default
        current_overrides['duration'] = LVConfig().duration 

        # Changed run_name format to start with 'combo_' for statistical_analysis compatibility
        run_name = f"combo_{i+1:03d}_" + "_".join(run_name_parts)
        
        # Include obstacle spawning in run_experiment call
        run_result = run_experiment(
            run_name=run_name,
            base_output_dir=scenario_results_dir,
            initial_rabbits=initial_rabbits,
            initial_foxes=initial_foxes,
            config_overrides=current_overrides,
            current_sim_index=i + 1,
            total_sims_in_scenario=total_combinations,
            spawn_obstacles=spawn_obstacles_for_scenario
        )
        scenario_all_run_results.append(run_result)

    # Analyze and plot category-level results
    analyze_category_results(scenario_all_run_results, scenario_results_dir, scenario_name)

    return scenario_all_run_results

# --- Cross-Scenario Comparison Function ---
def compare_scenarios(base_results_dir: str, scenario_names: List[str]):
    """
    Compares aggregated results from multiple scenarios and generates comparison plots.
    """
    print(f"\n{BLUE}{BOLD}--- Comparing Scenarios: {', '.join(scenario_names)} ---{RESET}")

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
        width = 0.35 # Adjusted width for two bars
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
    # 1. Population Dynamics Plot
    if has_population_data:
        fig_pop, ax_pop_single = plt.subplots(figsize=(14, 8))
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
        plt.close(fig_pop)
        print(f"Separate average population dynamics plot saved to: {plot_path_pop}")
    else:
        print(f"{YELLOW}No population data available for separate population dynamics plot.{RESET}")

    # 2. Extinction Time Plot
    if ext_labels:
        fig_ext, ax_ext_single = plt.subplots(figsize=(10, 7))
        # Create bars directly on the new single figure's axes
        x = np.arange(len(ext_labels))
        width = 0.35
        ax_ext_single.bar(x - width/2, ext_rabbits, width, label='Rabbits', color='green')
        ax_ext_single.bar(x + width/2, ext_foxes, width, label='Foxes', color='red')

        ax_ext_single.set_ylabel('Avg. Extinction Tick (or Max Duration)', fontsize=10)
        ax_ext_single.set_title('Average Time to Extinction', fontsize=16)
        ax_ext_single.set_xticks(x)
        ax_ext_single.set_xticklabels(ext_labels, rotation=45, ha='right', fontsize=9)
        ax_ext_single.legend(fontsize=9)
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
        # Create bars directly on the new single figure's axes
        x = np.arange(len(peak_labels))
        width = 0.35
        ax_peak_single.bar(x - width/2, peak_rabbits, width, label='Rabbits', color='lightgreen')
        ax_peak_single.bar(x + width/2, peak_foxes, width, label='Foxes', color='salmon')

        ax_peak_single.set_ylabel('Average Peak Population', fontsize=10)
        ax_peak_single.set_title('Average Peak Populations', fontsize=16)
        ax_peak_single.set_xticks(x)
        ax_peak_single.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=9)
        ax_peak_single.legend(fontsize=9)
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
            # Ensure the 'Tick' column is correctly handled for merging
            if combined_pop_df.empty:
                combined_pop_df['Tick'] = df['Tick']
            # Merge on 'Tick' to align data correctly, especially if durations differ
            combined_pop_df = pd.merge(combined_pop_df, df[['Tick', 'Average_Rabbits', 'Average_Foxes']], on='Tick', how='outer', suffixes=('', f'_{name}'))
            combined_pop_df.rename(columns={'Average_Rabbits': f'{name}_Average_Rabbits', 'Average_Foxes': f'{name}_Average_Foxes'}, inplace=True)

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


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure Pygame is initialized only once for all runs
    pg.init() 

    base_results_dir = "simulation_results_flocking"
    os.makedirs(base_results_dir, exist_ok=True)

    # Initial populations (can be fixed or varied in a higher-level grid search)
    initial_rabbits = 100
    initial_foxes = 20

    # --- Scenario 1: Probabilistic (Energy Free, No Flocking, No Grass) ---
    print(f"\n{BOLD}{BLUE}--- Starting Probabilistic Scenario Grid Search ---{RESET}")
    probabilistic_fixed_params = {
        'flocking_enabled': False,
    }
    probabilistic_varying_params = {
        'rabbit_reproduce_chance': [0.005, 0.005],
        'fox_spontaneous_death_chance': [0.005, 0.005],
        'rabbit_movement_speed': [2.0, 2.0],
        'fox_movement_speed': [3.0, 3.0],
        'rabbit_perception_radius': [50, 50],
        'fox_perception_radius': [70, 70],
    }

    run_grid_search_for_scenario(
        scenario_name=PROBABILISTIC_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        fixed_config_params=probabilistic_fixed_params,
        varying_params_grid=probabilistic_varying_params,
        base_results_dir=base_results_dir,
        spawn_obstacles_for_scenario=False
    )

    # --- Scenario 2: Flocking Enabled (No Grass, With Obstacles) ---
    print(f"\n{BOLD}{BLUE}--- Starting Flocking Scenario Grid Search ---{RESET}")
    flocking_fixed_params = {
        'flocking_enabled': True,
    }
    flocking_varying_params = {
        'alignment_weight': [1.0],
        'cohesion_weight': [1.0],
        'separation_weight': [1.0],
        'rabbit_reproduce_chance': [0.005, 0.005],
        'fox_spontaneous_death_chance': [0.005, 0.005],
        'rabbit_movement_speed': [2.0, 2.0],
        'fox_movement_speed': [3.0, 3.0],
        'rabbit_perception_radius': [50, 50],
        'fox_perception_radius': [70, 70],
    }

    run_grid_search_for_scenario(
        scenario_name=FLOCKING_SCENARIO_NAME,
        initial_rabbits=initial_rabbits,
        initial_foxes=initial_foxes,
        fixed_config_params=flocking_fixed_params,
        varying_params_grid=flocking_varying_params,
        base_results_dir=base_results_dir,
        spawn_obstacles_for_scenario=True # Obstacles for flocking scenario
    )
    
    # --- Cross-Scenario Comparison ---
    compare_scenarios(base_results_dir, [PROBABILISTIC_SCENARIO_NAME, FLOCKING_SCENARIO_NAME])

    # Ensure Pygame is quit only once after all runs
    pg.quit()
    print(f"\n{BOLD}{GREEN}All simulations and analyses complete. Results are in '{base_results_dir}' directory.{RESET}")
