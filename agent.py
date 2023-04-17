#---------------------------------------
#Garcia Junquera Luis Eduardo
#Fundamentos de inteligencia artificial
#5AV1
#----------------------------------------
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

#--------------
#Clase agente
#--------------
class Agent:

#--------------------
#Constructor:
#--------------------
#model - red neuronal
#trainer -optimizador
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness      #juego al azar
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
#---------------------
#Estado del agente
#---------------------
#------obtener estado---get_state
#juego -game
    def obtener_estado(self, juego):
        head = juego.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        #---------------
        #PIXELES 20X20
        #---------------
        
        
        dir_l = juego.direction == Direction.LEFT
        dir_r = juego.direction == Direction.RIGHT
        dir_u = juego.direction == Direction.UP
        dir_d = juego.direction == Direction.DOWN

        state = [
            #------------------
            # Peligro directo
            #------------------
            (dir_r and juego.is_collision(point_r)) or 
            (dir_l and juego.is_collision(point_l)) or 
            (dir_u and juego.is_collision(point_u)) or 
            (dir_d and juego.is_collision(point_d)),
            #--------------------
            # Peligro derecho
            #--------------------
            
            (dir_u and juego.is_collision(point_r)) or 
            (dir_d and juego.is_collision(point_l)) or 
            (dir_l and juego.is_collision(point_u)) or 
            (dir_r and juego.is_collision(point_d)),
            #--------------------
            # Peligro izquierda
            #--------------------
            
            (dir_d and juego.is_collision(point_r)) or 
            (dir_u and juego.is_collision(point_l)) or 
            (dir_r and juego.is_collision(point_u)) or 
            (dir_l and juego.is_collision(point_d)),
            
            #--------------------------
            # Se mueve de dirección 
            #---------------------------
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #-----------------------------
            # Localización de la comida
            #-----------------------------
            #food-comida
            juego.food.x < juego.head.x,  # comida izquierda
            juego.food.x > juego.head.x,  # comida derecha
            juego.food.y < juego.head.y,  # comida arriba
            juego.food.y > juego.head.y  # comida abajo
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft si se alcanza MAX_MEMORY

#tren de memoria clarga---train_long_memory
    def tren_memoria_larga(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # lista de tuplas  #
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.trainer.train_step(np.array(states,dtype=np.float32),actions,rewards, np.array(next_states, dtype=np.float32),dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

#tren_corta_memoria----train_short_memory

    def tren_corta_memoria(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            # genera entero al azar entre 0 y 2
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #Dado state (R11) genera prediction (R3)
            prediction = self.model(state0)
            #move es entero entre 0 y 2
            #Es la entrada con valor máximo 
            move = torch.argmax(prediction).item()
            final_move[move] = 1
          # desicion es un vector en R3 de con 0 o 1
          
        return final_move
    

#-----------------------------------------------------------------------------------
#Definimos como es que se movera la viborita y los resultados que se obtendra
#----------------------------------------------------------------------------------
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #--------------------------
        # obtener estado antiguo
        #---------------------------
        state_old = agent.obtener_estado(game)
        #------------------
        # moverse
        #------------------
        final_move = agent.get_action(state_old)
        
        #----------------------------------------------------
        # realizar el movimiento y obtener un nuevo estado
        #-----------------------------------------------------
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.obtener_estado(game)
        #------------------------------
        # tren de corta memoria
        #-----------------------------
        
        agent.tren_corta_memoria(state_old, final_move, reward, state_new, done)
        #---------------------------------------------------------
        # RECUERDA PARA QUE NO VUELVA A COMETEREL MISMO ERROR
        #----------------------------------------------------------
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #------------------------------------------------
            # Entrenar la memoria larga, trazar el resultado
            #------------------------------------------------
            game.reset()
            agent.n_games += 1
            agent.tren_memoria_larga()
            
            #-----------------------------------------
            #Guarda el record con mayor puntuación 
            #-----------------------------------------

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            #--------------------------------------------------------------------
            # Nos dice el total de partidas y la puntuacion con mayor cantidad
            #--------------------------------------------------------------------

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            #--------------------
            #Muestra resultados
            #---------------------


if __name__ == '__main__':
    train()
    
    #----------------------
    #Fin de agente.py
    #----------------------
    