from dataclasses import dataclass
from typing import Generator

from vi import Agent, Config, Simulation
from pygame.math import Vector2


@dataclass
class FlockingConfig(Config):
    # TODO: Modify the weights and observe the change in behaviour.
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_weight: float = 1.0
    # delta_time: float = 3.0 # This was causing the error, and is not directly used for velocity update based on Reynolds' algorithm. Removed for now to fix the AttributeError.

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
            # Use neighbor.move directly as it represents the velocity/direction
            alignment_vector += neighbor.move.normalize() if neighbor.move.length() > 0 else Vector2(0,0)

            # Cohesion: Steer towards the average position (center of mass) of local boids
            cohesion_vector += neighbor.pos

            # Separation: Steer away to avoid being in too close proximity to local boids
            # Inversely proportional to distance
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
        # The equation for f_total is (alpha*s + beta*c + gamma*a) / M_boid
        # Assuming M_boid is 1 for simplicity, so f_total = alpha*s + beta*c + gamma*a
        # The `steering_force` represents the acceleration, which is added to the current velocity (`self.move`).
        
        steering_force = (
            self.config.separation_weight * separation_vector +
            self.config.cohesion_weight * cohesion_vector +
            self.config.alignment_weight * alignment_vector
        )

        # Update the velocity (self.move) with the steering force.
        # Removed `* self.config.delta_time` as it was causing the AttributeError and
        # `delta_time` in the tutorial was an example config parameter, not directly
        # used as a time step multiplier for velocity update in Reynolds' original algorithm,
        # where the forces directly affect velocity. The simulation's internal tick rate handles
        # time steps.
        self.move += steering_force

        # Limit the speed of the agent using the `movement_speed` from the config
        if self.move.length() > self.config.movement_speed:
            self.move.normalize_ip()
            self.move *= self.config.movement_speed
            
        # Update position based on the new move vector
        self.pos += self.move


(
    Simulation(
        # TODO: Modify `movement_speed` and `radius` and observe the change in behaviour.
        FlockingConfig(image_rotation=True, movement_speed=5, radius=50) # Increased movement_speed for more noticeable behavior
    )
    .batch_spawn_agents(100, FlockingAgent, images=["images/triangle.png"])
    .run()
)