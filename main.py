import pygame
import pygame.gfxdraw
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import cv2
from model import DoodleCNN
import time
from inference import DoodleInferencer

class QuickDrawGame:
    def __init__(self, model_path, window_size=(800, 600)):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up display
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Quick Draw Game")
        
        # Set up fonts
        self.title_font = pygame.font.SysFont('Arial', 36)
        self.normal_font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Drawing area
        self.drawing_area = pygame.Rect(50, 100, 700, 400)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (220, 220, 220)
        self.LIGHT_BLUE = (200, 230, 255)
        
        # Drawing variables
        self.drawing = False
        self.last_pos = None
        self.radius = 6
        self.canvas = pygame.Surface((self.drawing_area.width, self.drawing_area.height))
        self.canvas.fill(self.WHITE)
        
        # Game state
        self.current_challenge = None
        self.used_challenges = []  # Track used challenges to avoid repetition
        self.time_limit = 30  # seconds
        self.start_time = None
        self.score = 0
        self.rounds_played = 0
        self.max_rounds = 5
        self.game_over = False
        self.feedback = ""
        self.feedback_color = self.BLACK
        self.countdown = 3
        self.countdown_start = None
        self.state = "menu"  # menu, countdown, playing, feedback, gameover
    
        
        # Buttons
        # Buttons
        self.clear_button = pygame.Rect(50, 530, 100, 40)
        self.eraser_button = pygame.Rect(250, 530, 100, 40)  # New eraser button
        self.submit_button = pygame.Rect(450, 530, 100, 40)
        self.next_button = pygame.Rect(350, 530, 100, 40)
        self.play_button = pygame.Rect(window_size[0]//2 - 100, window_size[1]//2 + 50, 200, 50)

        # Drawing tool state
        self.using_eraser = False

        # model
        self.inferencer = DoodleInferencer(model_path = model_path)
        self.classes = self.inferencer.getClasses()
    
    def select_challenge(self):
        """Select a random drawing challenge from available classes that hasn't been used yet"""
        available_challenges = [c for c in self.classes if c not in self.used_challenges]
        
        # If we've used all challenges or not enough left for the game, reset
        if len(available_challenges) < (self.max_rounds - self.rounds_played):
            self.used_challenges = []
            available_challenges = self.classes
            
        # Select a random challenge
        self.current_challenge = random.choice(available_challenges)
        self.used_challenges.append(self.current_challenge)
        return self.current_challenge
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.fill(self.WHITE)
        
    def process_drawing(self):
        #read the image
        image = pygame.surfarray.array3d(self.canvas)
        image = np.transpose(image, (1, 0, 2))  # Convert to HxWxC

        
        return self.inferencer.inference(image)

    def draw_menu(self):
        """Draw the game menu screen"""
        self.screen.fill(self.WHITE)
        
        # Title
        title_text = self.title_font.render("Quick Draw Game", True, self.BLUE)
        self.screen.blit(title_text, (self.window_size[0]//2 - title_text.get_width()//2, 100))
        
        # Instructions
        instructions = [
            "Draw the requested object before time runs out!",
            f"You'll have {self.time_limit} seconds to draw each object.",
            f"Complete {self.max_rounds} rounds to win!",
            "\n"
        ]
        
        for i, line in enumerate(instructions):
            text = self.normal_font.render(line, True, self.BLACK)
            self.screen.blit(text, (self.window_size[0]//2 - text.get_width()//2, 200 + i*50))

        
        # Play button
        pygame.draw.rect(self.screen, self.GREEN, self.play_button, border_radius=10)
        play_text = self.normal_font.render("PLAY", True, self.WHITE)
        self.screen.blit(play_text, (self.play_button.centerx - play_text.get_width()//2, 
                                    self.play_button.centery - play_text.get_height()//2))
    
    def draw_countdown(self):
        """Draw the countdown before the game starts"""
        self.screen.fill(self.WHITE)
        
        seconds_left = 3 - int(time.time() - self.countdown_start)
        if seconds_left <= 0:
            self.state = "playing"
            self.start_time = time.time()
            return
        
        # Draw countdown number
        count_text = self.title_font.render(str(seconds_left), True, self.RED)
        self.screen.blit(count_text, (self.window_size[0]//2 - count_text.get_width()//2, 
                                     self.window_size[1]//2 - count_text.get_height()//2))
        
        # Draw challenge text
        challenge_text = self.normal_font.render(f"Get ready to draw: {self.current_challenge}", True, self.BLUE)
        self.screen.blit(challenge_text, (self.window_size[0]//2 - challenge_text.get_width()//2, 
                                         self.window_size[1]//2 - 100))
    
    def draw_game_screen(self):
        """Draw the main game screen with drawing canvas"""
        self.screen.fill(self.WHITE)
        
        # Draw the challenge
        challenge_text = self.normal_font.render(f"Draw: {self.current_challenge}", True, self.RED)
        self.screen.blit(challenge_text, (50, 30))
        
        # Draw time left
        time_passed = time.time() - self.start_time
        time_left = max(0, self.time_limit - time_passed)
        time_text = self.normal_font.render(f"Time: {int(time_left)}s", True, self.BLACK)
        self.screen.blit(time_text, (350, 30))
        
        # Draw score and round info
        score_text = self.normal_font.render(f"Score: {self.score}/{self.max_rounds}", True, self.BLACK)
        round_text = self.normal_font.render(f"Round: {self.rounds_played + 1}/{self.max_rounds}", True, self.BLACK)
        self.screen.blit(score_text, (650, 30))
        self.screen.blit(round_text, (650, 60))
        
        # Draw a highlighted area around the drawing area
        # First draw a slightly larger rectangle as background highlight
        highlight_rect = self.drawing_area.inflate(20, 20)
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, highlight_rect, border_radius=10)
        
        # Then draw the canvas border with thicker line
        pygame.draw.rect(self.screen, self.BLUE, self.drawing_area, 4)  # Thicker border (4px) with blue color
        
        # Draw canvas content
        self.screen.blit(self.canvas, (self.drawing_area.x, self.drawing_area.y))
        
        # Draw buttons
        pygame.draw.rect(self.screen, self.GRAY, self.clear_button, border_radius=5)
        pygame.draw.rect(self.screen, self.BLUE, self.submit_button, border_radius=5)
        pygame.draw.rect(self.screen, self.GRAY if not self.using_eraser else self.GREEN, self.eraser_button, border_radius=5)  # Eraser button

        clear_text = self.small_font.render("Clear", True, self.BLACK)
        submit_text = self.small_font.render("Submit", True, self.WHITE)
        eraser_text = self.small_font.render("Eraser", True, self.BLACK if not self.using_eraser else self.WHITE)

        self.screen.blit(clear_text, (self.clear_button.centerx - clear_text.get_width()//2, 
                                     self.clear_button.centery - clear_text.get_height()//2))
        self.screen.blit(submit_text, (self.submit_button.centerx - submit_text.get_width()//2, 
                                      self.submit_button.centery - submit_text.get_height()//2))
        self.screen.blit(eraser_text, (self.eraser_button.centerx - eraser_text.get_width()//2,
                                       self.eraser_button.centery - eraser_text.get_height()//2))
        
        # Check if time is up
        if time_left <= 0:
            self.submit_drawing()
    
    def draw_feedback_screen(self):
        """Draw the feedback screen after a drawing submission"""
        self.screen.fill(self.WHITE)
        
        # Draw main feedback
        feedback_text = self.normal_font.render(self.feedback, True, self.feedback_color)
        self.screen.blit(feedback_text, (self.window_size[0]//2 - feedback_text.get_width()//2, 30))
        
        # Draw the player's drawing
        canvas_pos_y = 120
        self.screen.blit(self.canvas, (self.window_size[0]//2 - self.canvas.get_width()//2, canvas_pos_y))
        
        # Calculate where the canvas bottom is
        canvas_bottom_y = canvas_pos_y + self.canvas.get_height() + 20
        
        # Draw the "Next" button if game is not over
        if not self.game_over:
            pygame.draw.rect(self.screen, self.GREEN, self.next_button, border_radius=5)
            next_text = self.small_font.render("Next", True, self.WHITE)
            self.screen.blit(next_text, (self.next_button.centerx - next_text.get_width()//2, 
                                       self.next_button.centery - next_text.get_height()//2))
        else:
            # Game over, draw final score below the canvas
            final_score_text = self.title_font.render(f"Final Score: {self.score}/{self.max_rounds}", True, self.BLACK)
            self.screen.blit(final_score_text, (self.window_size[0]//2 - final_score_text.get_width()//2, 500))
        
    
    def submit_drawing(self):
        """Submit the current drawing for classification"""
        predicted_clas, prob = self.process_drawing()
        
        # Check if the drawing matches the challenge
        if predicted_clas == self.current_challenge:
            self.score += 1
            self.feedback = f"Correct! The AI recognized your {self.current_challenge}! ({prob*100:.1f}%)"
            self.feedback_color = self.GREEN
        else:
            self.feedback = f"Not quite! The AI thought you drew a {predicted_clas} ({prob*100:.1f}%)"
            self.feedback_color = self.RED
        
        self.rounds_played += 1
        if self.rounds_played >= self.max_rounds:
            self.game_over = True
        
        self.state = "feedback"
    
    def next_round(self):
        """Set up the next round"""
        self.clear_canvas()
        self.select_challenge()
        self.state = "countdown"
        self.countdown_start = time.time()
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.score = 0
        self.rounds_played = 0
        self.game_over = False
        self.clear_canvas()
        self.used_challenges = []  # Reset used challenges
        self.select_challenge()
        self.state = "countdown"
        self.countdown_start = time.time()
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        self.select_challenge()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle mouse events
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    if self.state == "menu":
                        if self.play_button.collidepoint(mouse_pos):
                            self.reset_game()
                    
                    elif self.state == "playing":
                        if self.eraser_button.collidepoint(mouse_pos):
                            self.using_eraser = not self.using_eraser  # Toggle eraser
                        elif self.drawing_area.collidepoint(mouse_pos):
                            self.drawing = True
                            canvas_pos = (mouse_pos[0] - self.drawing_area.x, 
                                         mouse_pos[1] - self.drawing_area.y)
                            self.last_pos = canvas_pos
                            color = self.WHITE if self.using_eraser else self.BLACK
                            pygame.gfxdraw.filled_circle(self.canvas, canvas_pos[0], canvas_pos[1], 
                                                      self.radius, color)
                        elif self.clear_button.collidepoint(mouse_pos):
                            self.clear_canvas()
                        elif self.submit_button.collidepoint(mouse_pos):
                            self.submit_drawing()
                    
                    elif self.state == "feedback":
                        if not self.game_over and self.next_button.collidepoint(mouse_pos):
                            self.next_round()
                        elif self.game_over and self.play_button.collidepoint(mouse_pos):
                            self.reset_game()
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False
                    self.last_pos = None
                
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    if self.drawing_area.collidepoint(event.pos):
                        canvas_pos = (event.pos[0] - self.drawing_area.x, 
                                     event.pos[1] - self.drawing_area.y)
                        color = self.WHITE if self.using_eraser else self.BLACK
                        if self.last_pos:
                            pygame.draw.line(self.canvas, color, self.last_pos, canvas_pos, self.radius * 2)
                            pygame.gfxdraw.filled_circle(self.canvas, canvas_pos[0], canvas_pos[1], 
                                                      self.radius, color)
                        self.last_pos = canvas_pos
            
            # Update display based on current state
            if self.state == "menu":
                self.draw_menu()
            elif self.state == "countdown":
                self.draw_countdown()
            elif self.state == "playing":
                self.draw_game_screen()
            elif self.state == "feedback":
                self.draw_feedback_screen()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    game = QuickDrawGame(model_path='./doodle_classification/checkpoint/best.pt')
    game.run()