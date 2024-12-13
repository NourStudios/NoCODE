import pygame
import os
import random

# Initialize pygame
pygame.init()

# Screen settings
width, height = 400, 400
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flappy Bird Dataset Generator")

# Colors
black = (255, 255, 255)
white = (0, 0, 0)
blue = (0, 0, 0)

# Game settings
bird_x, bird_y = 100, height // 2
bird_width, bird_height = 30, 30
gravity = 1
jump_force = -10
velocity = 0
pipe_width, pipe_gap = 70, 120
pipe_speed = 5

# Initialize pipes
pipes = [{"x": 300, "y": 200}, {"x": 600, "y": 150}]
score = 0

clock = pygame.time.Clock()

# Create output folders
output_folders = {"jump": "jump", "fall": "fall"}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Frame counters for each folder
frame_counters = {"jump": 1, "fall": 1}

def draw_bird(x, y):
    pygame.draw.rect(win, white, (x, y, bird_width, bird_height))

def draw_pipes(pipes):
    for pipe in pipes:
        pygame.draw.rect(win, blue, (pipe["x"], 0, pipe_width, pipe["y"]))  # Top pipe
        pygame.draw.rect(win, blue, (pipe["x"], pipe["y"] + pipe_gap, pipe_width, height - pipe["y"] - pipe_gap))  # Bottom pipe

def reset_game():
    """Reset the game variables and pipes."""
    global bird_y, velocity, pipes, score
    bird_y = height // 2
    velocity = 0
    pipes = [{"x": 300, "y": random.randint(100, height - pipe_gap - 100)},
             {"x": 600, "y": random.randint(100, height - pipe_gap - 100)}]
    score = 0

# Game loop
running = True
while running:
    jump_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            jump_pressed = True
            velocity = jump_force  # Bird jumps

    # Apply gravity
    velocity += gravity
    bird_y += velocity

    # Move pipes
    for pipe in pipes:
        pipe["x"] -= pipe_speed
        if pipe["x"] + pipe_width < 0:
            pipe["x"] = width
            # Increase the range for pipe generation so pipes appear higher
            pipe["y"] = random.randint(50, height - pipe_gap - 50)  # Adjusted range for higher pipes
            score += 1

    # Check collisions
    if bird_y < 0 or bird_y + bird_height > height:
        reset_game()  # Restart the game when the bird goes out of bounds
    for pipe in pipes:
        if bird_x + bird_width > pipe["x"] and bird_x < pipe["x"] + pipe_width:
            if bird_y < pipe["y"] or bird_y + bird_height > pipe["y"] + pipe_gap:
                reset_game()  # Restart the game if the bird collides with a pipe

    # Draw everything
    win.fill(black)
    draw_bird(bird_x, bird_y)
    draw_pipes(pipes)

    # Save frames based on motion (jump or fall)
    if velocity < 0:  # Bird moving upward (jump)
        folder = "jump"
        frame_path = os.path.join(output_folders[folder], f"{frame_counters[folder]}.png")
        pygame.image.save(win, frame_path)
        frame_counters[folder] += 1
    elif velocity > 0:  # Bird moving downward (fall)
        folder = "fall"
        frame_path = os.path.join(output_folders[folder], f"{frame_counters[folder]}.png")
        pygame.image.save(win, frame_path)
        frame_counters[folder] += 1

    pygame.display.update()
    clock.tick(10)

pygame.quit()
