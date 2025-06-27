import pygame as pg
import random
from dataclasses import dataclass, field
from enum import Enum

from vi import Agent, Config, Simulation
from pygame.math import Vector2

# Define the states for the Cockroach agent's finite state machine
class CockroachState(Enum):
    WANDERING = 1
    JOINING = 2
    STILL = 3
    LEAVING = 4

@dataclass
class CockroachConfig(Config):
    """
    Configuration settings for the Cockroach aggregation simulation.
    Inherits from vi.Config to allow for easy parameter tuning.
    """
    # Parameters for the P_join probability function: P_join(n) = 1 - (1 / (1 + n * p_join_factor))
    p_join_factor: float = 20.0 # Increased significantly for very high probability of joining

    # Parameters for the P_leave probability function: P_leave(n) = 1 / (1 + n * p_leave_factor)
    p_leave_factor: float = 50.0 # Increased significantly for very low probability of leaving dense aggregates

    # Timers for state transitions (in simulation ticks)
    t_join_min: int = 10   # Very short minimum ticks for fast transition to Still
    t_join_max: int = 20   # Very short maximum ticks for fast transition to Still
    t_leave_min: int = 180 # Longer minimum ticks to ensure agent leaves aggregate effectively
    t_leave_max: int = 360 # Longer maximum ticks to ensure agent leaves aggregate effectively

    # Frequency (in ticks) for checking P_leave in the Still state
    d_check_frequency: int = 30

    # Movement speed for agents when not in the Still state
    movement_speed: float = 5.0 # Adjusted to a more reasonable speed for aggregation

    # Perception radius for agents to detect neighbors
    radius: int = 70

    # Flag for image rotation, useful for triangular agents to show direction
    image_rotation: bool = True

    # Controls the maximum frames-per-second, which dictates the simulation speed.
    fps_limit: int = 120 # Increased for faster simulation speed


class CockroachAgent(Agent[CockroachConfig]):
    """
    Represents a single cockroach agent with a probabilistic finite state machine.
    Its behavior (movement and state transitions) is defined by its current state.
    """
    _state: CockroachState = CockroachState.WANDERING
    _timer: int = 0         # General timer for JOINING and LEAVING states
    _check_timer: int = 0   # Timer for checking P_leave in STILL state

    def on_spawn(self):
        """
        Initializes the agent's state and movement when it is first spawned.
        This method is called automatically by the simulation.
        """
        self._state = CockroachState.WANDERING
        self._timer = 0
        self._check_timer = 0
        # Ensure agent starts moving in a random direction
        self.move = Vector2(self.config.movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

    def _calculate_p_join(self, n_neighbors: int) -> float:
        """
        Calculates the probability of transitioning from Wandering to Joining.
        This probability increases with the number of perceived neighbors.
        """
        # Using a logistic-like function for P_join, now very aggressive
        return 1.0 - (1.0 / (1.0 + n_neighbors * self.config.p_join_factor))

    def _calculate_p_leave(self, n_neighbors: int) -> float:
        """
        Calculates the probability of transitioning from Still to Leaving.
        This probability decreases as the number of perceived neighbors increases,
        encouraging agents to stay in denser aggregates.
        """
        # Using an inverse-logistic-like function for P_leave, now very aggressive
        return 1.0 / (1.0 + n_neighbors * self.config.p_leave_factor)

    def change_position(self):
        """
        Updates the agent's position and manages its state transitions.
        This method is called at every tick of the simulation.
        """
        # Store the current position to revert if a collision occurs
        old_pos = self.pos.copy()

        # Get simulation boundaries for bouncing
        window_width, window_height = self._Agent__simulation.config.window.as_tuple()
        agent_half_width = self.rect.width / 2
        agent_half_height = self.rect.height / 2

        # --- State-dependent behavior logic for spontaneous aggregation (no sites) ---
        if self._state == CockroachState.WANDERING:
            # Random walk: occasionally change direction slightly
            if self.shared.prng_move.random() < 0.1: # 10% chance to slightly change direction
                self.move = self.move.rotate(self.shared.prng_move.uniform(-10, 10))
                self.move.normalize_ip()
                self.move *= self.config.movement_speed

            # Transition to JOINING based on neighbor count
            n_neighbors = len([agent for agent, dist in self.in_proximity_accuracy() if agent.id != self.id])
            p_join = self._calculate_p_join(n_neighbors)
            
            if self.shared.prng_move.random() < p_join:
                self._state = CockroachState.JOINING
                # Set a random timer for the JOINING state duration
                self._timer = self.shared.prng_move.randint(self.config.t_join_min, self.config.t_join_max)
                self.freeze_movement() # Agent freezes immediately upon deciding to join

        elif self._state == CockroachState.JOINING:
            # Agent remains frozen (or moves minimally if desired) for a set duration
            self._timer -= 1
            if self._timer <= 0:
                self._state = CockroachState.STILL
                self.freeze_movement() # Ensure it is still for the STILL state
                self._check_timer = self.config.d_check_frequency # Initialize check timer for leaving decisions

        elif self._state == CockroachState.STILL:
            # Agent is not moving, but periodically checks if it should LEAVE
            self.freeze_movement() # Ensure it remains still
            
            self._check_timer -= 1
            if self._check_timer <= 0:
                self._check_timer = self.config.d_check_frequency # Reset check timer for next interval

                # Count neighbors to decide on leaving
                n_neighbors = len([agent for agent, dist in self.in_proximity_accuracy() if agent.id != self.id])
                p_leave = self._calculate_p_leave(n_neighbors)
                
                if self.shared.prng_move.random() < p_leave:
                    self._state = CockroachState.LEAVING
                    # Set a random timer for the LEAVING state duration
                    self._timer = self.shared.prng_move.randint(self.config.t_leave_min, self.config.t_leave_max)
                    self.continue_movement() # Start moving again
                    # Give a random direction to leave
                    self.move = Vector2(self.config.movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

        elif self._state == CockroachState.LEAVING:
            # Agent moves for a set duration, then returns to wandering
            self._timer -= 1
            if self._timer <= 0:
                self._state = CockroachState.WANDERING
                # Give a random direction for wandering
                self.move = Vector2(self.config.movement_speed, 0).rotate(self.shared.prng_move.uniform(0, 360))

        # Apply movement if the agent is not in the STILL state
        if self._state != CockroachState.STILL:
            self.pos += self.move

        # --- Boundary Bouncing Logic ---
        # Check if agent hits horizontal boundaries
        if self.pos.x - agent_half_width < 0: # Hit left border
            self.pos.x = agent_half_width # Reposition at border
            self.move.x *= -1 # Reverse horizontal movement
        elif self.pos.x + agent_half_width > window_width: # Hit right border
            self.pos.x = window_width - agent_half_width # Reposition at border
            self.move.x *= -1 # Reverse horizontal movement

        # Check if agent hits vertical boundaries
        if self.pos.y - agent_half_height < 0: # Hit top border
            self.pos.y = agent_half_height # Reposition at border
            self.move.y *= -1 # Reverse vertical movement
        elif self.pos.y + agent_half_height > window_height: # Hit bottom border
            self.pos.y = window_height - agent_half_height # Reposition at border
            self.move.y *= -1 # Reverse vertical movement


        # --- Obstacle collision detection and response ---
        # This part ensures agents do not pass through obstacles
        for obstacle in self._Agent__simulation._obstacles:
            if pg.sprite.collide_mask(self, obstacle):
                self.pos = old_pos # Revert to previous position
                self.move = -self.move # Reverse direction
                # Add a slight random perturbation to avoid getting stuck or oscillating
                self.move = self.move.rotate(self._Agent__simulation.shared.prng_move.uniform(-10, 10))
                break # Handle only one collision per tick to prevent multiple reversals

# Initialize Pygame once (required for image loading)
pg.init()

# Define simulation configuration using the CockroachConfig
sim_config = CockroachConfig()

# Create the simulation instance
sim = Simulation(sim_config)

# Get window dimensions for placing agents
window_width, window_height = sim.config.window.as_tuple()

# --- No Sites / Spontaneous Aggregation Experiment ---
# Remove all site spawning logic.
# The simulation will now rely on agents aggregating based purely on their interactions.

# Load agent image once for efficiency (e.g., a simple circle or custom cockroach image)
agent_image_path = "images/pearto.png" # Assuming a simple circular agent image
try:
    loaded_agent_image = pg.image.load(agent_image_path).convert_alpha()
    agent_images_list = [loaded_agent_image]
except pg.error as e:
    print(f"Error loading agent image: {e}. Please ensure '{agent_image_path}' exists.")
    exit() # Exit if critical image cannot be loaded

num_agents_to_spawn = 50 # Number of agents to simulate

spawned_agents_count = 0
max_spawn_attempts_per_agent = 500 # Max attempts to find a non-colliding spawn position for each agent

# Create a temporary agent instance used only for collision checks during spawning.
dummy_agent_for_spawn_check = CockroachAgent(images=agent_images_list, simulation=sim)

# Spawn agents randomly, ensuring they don't start inside obstacles (if any were added)
while spawned_agents_count < num_agents_to_spawn:
    current_attempts = 0
    is_valid_spawn_position = False
    
    while current_attempts < max_spawn_attempts_per_agent:
        # Generate a random potential spawn position
        rand_x = random.randint(0, window_width)
        rand_y = random.randint(0, window_height)
        proposed_pos = Vector2(rand_x, rand_y)

        # Temporarily set the dummy agent's position to the proposed spot for collision checking
        dummy_agent_for_spawn_check.pos = proposed_pos
        dummy_agent_for_spawn_check.rect.center = proposed_pos.xy # Update rect center for collision detection

        is_colliding = False
        # Check for collisions with existing obstacles (no obstacles in this specific setup, but good practice)
        for obstacle in sim._obstacles:
            if pg.sprite.collide_mask(dummy_agent_for_spawn_check, obstacle):
                is_colliding = True
                break
        
        # No sites to check for collision during spawn here
        
        if not is_colliding:
            is_valid_spawn_position = True
            break # Found a valid position, exit inner loop
        
        current_attempts += 1
    
    if is_valid_spawn_position:
        CockroachAgent(images=agent_images_list, simulation=sim, pos=proposed_pos)
        spawned_agents_count += 1
    else:
        print(f"Warning: Could not find a non-colliding spawn position for agent {spawned_agents_count + 1} after {max_spawn_attempts_per_agent} attempts. Spawning fewer agents than requested.")
        break # Exit outer loop

# Run the simulation
sim.run()
