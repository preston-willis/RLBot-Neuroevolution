import math
import random
import numpy
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class PythonExample(BaseAgent):

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.i = 0  # frame counter for timed reset
        self.brain = -1  # bot counter for generation reset
        self.pop = 10  # population for bot looping
        self.out = [None] * self.pop  # output of nets
        self.gen = 0
        self.botList = []
        self.fittest = Fittest()
        self.pos = 2500
        self.mutRate = 0.2
        self.maxGen = 1000
        self.fitGraph = [""]*self.maxGen

        for i in range(self.pop):
            self.botList.append(Individual())
            self.botList[i].name = "Bot "+str(i)
        for i in self.botList:
            print("INIT: "+i.name)
            for p in range(0,len(i.weights)):
                i.weights[p] = random.uniform(0,0)
                print("----"+str(i.weights[p]))

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        #INIT LOOPS
        if self.i == 0:
            self.reset() #reset at start
            self.brain += 1 #change bot every reset
        if self.brain >= self.pop:
            self.gen += 1
            print("")
            print("     GEN = "+str(self.gen))
            print("-------------------------")
            print("FITTEST = "+str(self.botList[self.calcFittest()].name))
            print("------FITNESS = " + str(self.fittest.fitness*10000))
            print("------WEIGHTS = " + str(self.botList[self.fittest.index].weights))
            self.fitGraph[self.gen] = "#"*int(self.fittest.fitness*100000)
            for i in range(0,self.gen+1):
                print(self.fitGraph[i])

            self.selection()
            self.mutate()
            self.brain = 0 #reset bots after all have gone
        self.i = self.i + 10


        #CALCULATE ANN OUTPUT
        ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)
        my_car = packet.game_cars[self.index]
        car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
        car_direction = get_car_facing_vector(my_car)
        car_to_ball = ball_location - car_location
        steer_correction_radians = car_direction.correction_to(car_to_ball)
        distance_to_ball = (packet.game_ball.physics.location.y+packet.game_ball.physics.location.x) - (my_car.physics.location.y+my_car.physics.location.x)


        #DEFINE FITNESS
        self.botList[self.brain].fitness = (1/(distance_to_ball+1))


        #RENDER RESULTS
        action_display = "GEN: "+str(self.gen+1)+" | BOT: "+str(self.brain)+" \nTURN: "+str(self.botList[self.brain].node(steer_correction_radians,distance_to_ball,0,1))+" \nBOOST: "+str(self.botList[self.brain].node(steer_correction_radians,distance_to_ball,2,3))+" \nTHROTTLE: "+str(self.botList[self.brain].node(steer_correction_radians,distance_to_ball,4,5))
        if self.i > 1000:
            self.i = 0
        draw_debug(self.renderer, my_car, packet.game_ball, action_display)

        #CONST
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),angular_velocity=Vector3(0, 0, 0)))
        game_state = GameState(ball=ball_state)
        self.set_game_state(game_state)
        self.controller_state.throttle = self.botList[self.brain].node(steer_correction_radians,distance_to_ball,4,5)
        self.controller_state.steer = self.botList[self.brain].node(steer_correction_radians,distance_to_ball,0,1)
        self.controller_state.boost = self.botList[self.brain].node(steer_correction_radians,distance_to_ball,2,3)
        self.controller_state.jump = False

        return self.controller_state



    def calcFittest(self):
        temp = -100000
        count = -1
        for i in self.botList:
            count += 1
            if i.fitness > temp:
                temp = i.fitness
                self.fittest.index = count
        self.fittest.fitness = temp
        return self.fittest.index

    def reset(self):
        ball_state = BallState(Physics(velocity=Vector3(0, 0, 0), location=Vector3(self.pos, 5000, 3000),
                                            angular_velocity=Vector3(0, 0, 0)))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(velocity=Vector3(0,0,0),rotation=Rotator(0 , 90, 0),location=Vector3(0.0, -4608,16.5),angular_velocity=Vector3(0, 0, 0)))
        game_info_state = GameInfoState(game_speed=5)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)

    def selection(self):
        for i in self.botList:
            i.weights[0] = self.botList[self.fittest.index].weights[0]
            i.weights[1] = self.botList[self.fittest.index].weights[1]
            i.weights[2] = self.botList[self.fittest.index].weights[2]
            i.weights[3] = self.botList[self.fittest.index].weights[3]
            i.weights[4] = self.botList[self.fittest.index].weights[4]
            i.weights[5] = self.botList[self.fittest.index].weights[5]

    def mutate(self):
        for i in self.botList:
            if random.uniform(-1,1) > -0.5:
                mutWeight = random.randint(0, 5)
                i.weights[mutWeight] += random.uniform(-self.mutRate,self.mutRate)
                mutWeight = random.randint(0, 5)
                i.weights[mutWeight] += random.uniform(-self.mutRate, self.mutRate)
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

class Fittest:
    def __init__(self):
        self.weights = [0] * 10
        self.index = 0
        self.fitness = 0

class Individual:
    def __init__(self):
        self.weights = [0]*6
        self.fitness = 0
        self.name = ""

    def node(self,input1, input2,weight1,weight2):

        out = numpy.tanh((input1 * self.weights[weight1])+((1/input2 * self.weights[weight2])))
        return out