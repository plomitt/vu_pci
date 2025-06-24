import pygame as pg
import random
from dataclasses import dataclass

from vi import Agent, Config, Simulation
from pygame.math import Vector2


@dataclass
class FlockingConfig(Config):
    # TODO: Modify the weights and observe the change in behaviour.
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_weight: float = 1.0


class FlockingAgent(Agent[FlockingConfig]):
    # By overriding `change_position`, the default behaviour is overwritten.
    # Without making changes, the agents won't move.
    def change_position(self):
        self.there_is_no_escape()

        # Initialize vectors for flocking behaviors
        alignment_vector = Vector2(0, 0)
        cohesion_vector = Vector2(0, 0)
        separation_vector = Vector2(0, 0)
        
        neighbor_count = 0

        # Iterate over neighbors within proximity
        for neighbor, distance in self.in_proximity_accuracy():
            neighbor_count += 1
            
            # Alignment: Steer towards the average direction of local boids
            alignment_vector += neighbor.move.normalize() if neighbor.move.length() > 0 else Vector2(0,0)

            # Cohesion: Steer towards the average position (center of mass) of local boids
            cohesion_vector += neighbor.pos

            # Separation: Steer away to avoid being in too close proximity to local boids
            separation_vector += (self.pos - neighbor.pos).normalize() / (distance if distance > 0 else 1)
        
        if neighbor_count > 0:
            # Average the vectors
            alignment_vector = (alignment_vector / neighbor_count).normalize() if alignment_vector.length() > 0 else Vector2(0,0)

            cohesion_center = cohesion_vector / neighbor_count
            cohesion_vector = (cohesion_center - self.pos).normalize() if (cohesion_center - self.pos).length() > 0 else Vector2(0,0)
            
            separation_vector = (separation_vector / neighbor_count).normalize() if separation_vector.length() > 0 else Vector2(0,0)
        else:
            # If no neighbors, no flocking forces
            alignment_vector = Vector2(0,0)
            cohesion_vector = Vector2(0,0)
            separation_vector = Vector2(0,0)

        # Apply weights to each behavior and sum them to get the steering force
        steering_force = (
            self.config.separation_weight * separation_vector +
            self.config.cohesion_weight * cohesion_vector +
            self.config.alignment_weight * alignment_vector
        )

        # Update the velocity (self.move) with the steering force.
        self.move += steering_force

        # Limit the speed of the agent using the `movement_speed` from the config
        if self.move.length() > self.config.movement_speed:
            self.move.normalize_ip()
            self.move *= self.config.movement_speed
            
        # Update position based on the new move vector
        # Store the current position before updating, to revert if collision occurs
        old_pos = self.pos.copy()
        self.pos += self.move

        # Collision detection with obstacles
        # Access the obstacles group via the name-mangled attribute
        # self._Agent__simulation._obstacles: This is necessary due to Python's name mangling of __simulation in the base Agent class.
        for obstacle in self._Agent__simulation._obstacles:
            # Check for collision using pygame.sprite.collide_mask for pixel-perfect collision
            # This requires both sprites to have a mask. Agent has one, and _StaticSprite (obstacle) does too.
            if pg.sprite.collide_mask(self, obstacle):
                # Revert to the old position to prevent passing through
                self.pos = old_pos
                # Make a 180-degree turn
                self.move = -self.move
                # Add a slight random perturbation to avoid getting stuck or oscillating
                # Access prng_move from self._Agent__simulation.shared
                self.move = self.move.rotate(self._Agent__simulation.shared.prng_move.uniform(-10, 10))
                break # Only handle collision with one obstacle per tick

# Define simulation configuration
config = FlockingConfig(image_rotation=True, movement_speed=5, radius=50) # Increased movement_speed for more noticeable behavior

# Create the simulation instance
sim = Simulation(config)

# Stage 2: Add obstacles
# To make bubbles smaller, you would need to use a smaller image file for "bubble-full.png"
# or implement image scaling when the obstacle is spawned (which is not directly supported by spawn_obstacle).
obstacle_image_path = "images/bubble-small.png"
sim.spawn_obstacle(obstacle_image_path, x=300, y=300) 
sim.spawn_obstacle(obstacle_image_path, x=700, y=200) 
sim.spawn_obstacle(obstacle_image_path, x=500, y=500) 

# Spawn agents. The `vi` library's `batch_spawn_agents` or `spawn_agent` methods
# do not provide a way to specify initial positions to avoid obstacles.
# Agents will be spawned at random positions. However, the collision detection
# logic in the `change_position` method will ensure that if an agent spawns
# inside an obstacle, it will immediately detect the collision and turn around,
# moving out of the obstacle on the first frame.
sim.batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png"])

# Run the simulation
sim.run()