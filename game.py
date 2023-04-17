#----------------------------------------------------
#Garcia Junquera Luis Eduardo
#Fundamentos de inteligencia artificial
#5AV1
#-----------------------------------------------------

#--------------------------------------------------
#Importamos la librerias correspondientes
#---------------------------------------------------
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#---------------------------------------------------------------------------
#Definimos la clase  para las direcciones para ver donde se mueve
#---------------------------------------------------------------------------
class Direction(Enum):
    RIGHT = 1   #derecha
    LEFT = 2    #izquierda
    UP = 3      #arriba
    DOWN = 4    #abajo

#-----------------------------------------
#puntos
#------------------------------------------
Point = namedtuple('Point', 'x, y')

#-------------
# rgb colores
#-------------
WHITE = (255, 255, 255)   #blanco
RED = (200,0,0)   #rojo
BLUE1 = (0, 0, 255)   #azul
BLUE2 = (0, 100, 255)  #azul
BLACK = (0,0,0)  #negro

BLOCK_SIZE = 20   #bloques
SPEED = 40

#----------------------------------------
#definimos la clase para SnakeGameAI
#-----------------------------------------
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        #------------------
        # pantalla de inicio
        #-------------------
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
      #----------------------------------
        # estado inicial del juego
        #--------------------------------
        self.direction = Direction.RIGHT
        #direcciones

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    #-----------------------------------------------------
    #definimos como es que pondra la comida en el juego
    #-----------------------------------------------------
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
#----------------------------
#Paso del juego
#-----------------------------  
    def play_step(self, action):
        self.frame_iteration += 1
        #----------------------------------------------
        # 1. recopilar información del usuario
        #----------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #------------------
        # 2. Mover
        #------------------
        self._move(action) # actualizar la cabeza
        self.snake.insert(0, self.head)
        
        #-----------------------------------------
        # 3. comprobar si el juego ha terminado
        #------------------------------------------
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

#-------------------------------------------------
        # 4. coloque comida nueva o simplemente muévase
        #-----------------------------------------
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        #-----------------------------------
        # 5. actualizar la interfaz de usuario y el reloj
        #-----------------------------------
        self._update_ui()
        self.clock.tick(SPEED)
        #---------------------------------------
        #6. Devuelve el juego y anota
        #-----------------------------------------
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        #------------------------    
        # golpea el límite
        #-------------------------
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        #--------------------------
        # se golpea a sí mismo
        #-------------------------
        if pt in self.snake[1:]:
            return True

        return False

#----------------------------------------
#Definimos _update_ui para pygame
#-----------------------------------------
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
      #-------------------------------------------
        # [recta, derecha, izquierda]
      #-------------------------------------------  

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #ningún cambio
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            #-----------------------------------------------------
            ## giro a la derecha r -> d -> l -> u
            #------------------------------------------------------
            new_dir = clock_wise[next_idx] 
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            #--------------------------------------------------
            ## giro a la izquierda r -> u -> l -> d
            #--------------------------------------------------
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        #Fin de game.py