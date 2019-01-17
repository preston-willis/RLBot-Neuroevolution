import math
import random
import numpy
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class PythonExample(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.frame = 0  # frame counter for timed reset
        self.pos = 500 #ball position
        self.distance_to_ball = [10000]*10000 #set high for easy minumum

        #CREATE BOT AND NET
        self.bot = Individual()
        self.bot.create_node()

        #ASSIGN WEIGHTS

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        #INIT LOOP
        if self.frame == 0:
            self.reset() #reset at start
            #PRINT GENERATION INFO
            print("")
            print("     TEST = "+str(self.pos))
            print("-------------------------")
        self.frame = self.frame + 1


        #NEURAL NET INPUTS
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)
        my_car = packet.game_cars[self.index]
        car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
        car_direction = get_car_facing_vector(my_car)
        car_to_ball = ball_location - car_location
        self.distance_to_ball[self.frame] = math.sqrt(pow(my_car.physics.location.x-packet.game_ball.physics.location.x,2)+pow(my_car.physics.location.y-packet.game_ball.physics.location.y,2)+pow(my_car.physics.location.z-packet.game_ball.physics.location.z,2))
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z

        if self.frame > 5000:
            self.frame = 0

        #GAME STATE
        car_state = CarState(boost_amount=100)
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(500, -1000, 1200),angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state,cars={self.index: car_state})
        self.set_game_state(game_state)


        #NEURAL NET OUTPUTS
        hidden1 = self.bot.Nodes[0].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 0, 1, 2, 3)
        print()
        hidden2 = self.bot.Nodes[1].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 4, 5, 6, 7)
        hidden3 = self.bot.Nodes[2].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 8, 9, 10, 11)
        hidden4 = self.bot.Nodes[3].node(distance_to_ball_x, distance_to_ball_y, distance_to_ball_z, 12, 13, 14, 15)

        self.controller_state.pitch = self.bot.Nodes[4].node(hidden1, hidden2, hidden3, hidden4, 16, 17, 18, 19)
        self.controller_state.roll = self.bot.Nodes[7].node(hidden1, hidden2, hidden3, hidden4, 28, 29, 30, 31)
        self.controller_state.yaw = self.bot.Nodes[5].node(hidden1, hidden2, hidden3, hidden4, 20, 21, 22, 23)
        if self.bot.Nodes[6].node(hidden1, hidden2, hidden3, hidden4, 24, 25, 26, 27) > 0: self.controller_state.boost = True
        else: self.controller_state.boost = False


        #KILL
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or my_car.physics.location.x < -4000 or my_car.physics.location.x > 4000 or my_car.physics.location.y > 5000) and self.frame > 50:
            self.frame = 5000

        return self.controller_state

    def reset(self):
        #RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                            angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0,0,0),rotation=Rotator(45, 90, 0),location=Vector3(0.0, -4608,500),angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)

class Vector2:
    def __init__(self, x=0, y=0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, val):
        return Vector2(self.x + val.x, self.y + val.y)

    def __sub__(self, val):
        return Vector2(self.x - val.x, self.y - val.y)

    def correction_to(self, ideal):
        # The in-game axes are left handed, so use -x
        current_in_radians = math.atan2(self.y, -self.x)
        ideal_in_radians = math.atan2(ideal.y, -ideal.x)

        correction = ideal_in_radians - current_in_radians

        # Make sure we go the 'short way'
        if abs(correction) > math.pi:
            if correction < 0:
                correction += 2 * math.pi
            else:
                correction -= 2 * math.pi

        return correction


def get_car_facing_vector(car):
    pitch = float(car.physics.rotation.pitch)
    yaw = float(car.physics.rotation.yaw)

    facing_x = math.cos(pitch) * math.cos(yaw)
    facing_y = math.cos(pitch) * math.sin(yaw)

    return Vector2(facing_x, facing_y)

def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()

class Individual:
    def __init__(self):
        self.fitness = 0
        self.name = ""
        self.jump_finished = False
        #self.weights = [0] * 32
        self.nodeNum = 8
        self.Nodes = []

    def create_node(self):
        for i in range(self.nodeNum):
            self.Nodes.append(Node())

class Node(Individual):
    def __init__(self):
        self.weights = [-0.0900700056894154, 0.08163897763838893, 0.06754129450326049,
                    -0.017225056712242393,0.08923529977308461, 0.08352257552667955,
                    -0.024934058366163664, 0.03294766531173768,0.09441713908504601,
                    0.01909580010020842, 0.05435791963507006, -0.012677483965957276,
                    -0.039700261651211655, -0.06645082037532049, -0.00021644154190399167,
                    0.0718112151413976,0.04121459410006234, 0.08890238716333815,
                    -0.024496499334755548, -0.08462942705703125,-0.02313504718473136,
                    0.005266994606792544, 0.04263757239014759, -0.009904504339254092,
                    -0.04146424633203303, -0.07085939380448353, 0.08193406817475424,
                    -0.04485197550906331,0.05150452819914403, -0.09190340758590027,
                    0.03225712894268923, 0.07979613802060917]

    def node(self, input1, input2, input3, input4, weight1, weight2, weight3, weight4=0):
        out = numpy.tanh((input1 * self.weights[weight1])+(input2 * self.weights[weight2])+(input3 * self.weights[weight3])+(input4 * self.weights[weight4]))
        return out

