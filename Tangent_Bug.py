# Autor: Baruc Samuel Cabrera Garcia
from vispy import app
import sys
from vispy.scene import SceneCanvas
from vispy.scene.visuals import  Ellipse, Rectangle, RegularPolygon, Mesh, Markers, Text
from vispy.scene.visuals import Polygon as polygon_visual
from vispy.color import Color
from vispy.scene import visuals
import numpy as np
from vispy.visuals.transforms.linear import MatrixTransform
from shapely.geometry import Polygon, Point
import math
import copy
from shapely.geometry import Polygon as Polygon_geometry
#import numba

INF = float('inf')

#from shapely.geometry import Polygon, Point



# Dimensiones de la escena
scene_width = 600
scene_height = 500

tol = 50#Tolerancia para detectar discontinuidades
tol_error = 1 #Tolerancia para el error en comparacion
d = 50

#Colores
white = Color("#ecf0f1")
gray = Color("#121212")
red = Color("#e74c3c")
blue = Color("#2980b9")
orange = Color("#e88834")

#Variables a utilizar
center_u = scene_width/2
center_v = scene_height/2

centro = np.array([center_u,center_v])
robot_position = centro + np.array([-250,0])
goal_position = centro + np.array([150,70])
#robot_position_prev = copy.deepcopy(robot_position)

step = 1

"""
Las coordenadas son del tipo (u,v), 
donde comienzan desde la esquina superior izquierda.
u representa el movimineot hacia la derecha
v representa el movimiento hacia abajo

"""

# Crear una escena
canvas = SceneCanvas(keys='interactive', title='Window',
                     show=True, size = (scene_width, scene_height), autoswap=False, vsync=True)
view = canvas.central_widget.add_view() #Creamos la vista (ventana)
canvas.size = (scene_width, scene_height) #Le damos tamaño
view.bgcolor = gray #Le damos un color al fondo
view.camera.center = (0,0)


########################
#Fuciones para Tangent Bug, usando el entorno canvas y view


#@numba.jit
def move(theta):
    global view
    global robot_position
    global robot
    #global robot_position_prev

    translation = [step*math.cos(theta), step*math.sin(theta)]
    #robot_position_prev = copy.deepcopy(robot_position)
    robot_position += translation
    robot.transform.translate(translation)
    view.camera.center = (0,0)

#@numba.jit
def is_parallel(v_1, v_2, w_1, w_2, flag_check = False):
    m_v = (v_1[1] - v_2[1]) / (v_1[0] - v_2[0])
    m_w = (w_1[1] - w_2[1]) / (w_1[0] - w_2[0])

    if abs(m_v - m_w) < 0.1 or (abs(m_v) == INF and abs(m_w) == INF):#Para detectar pendientes iguales, o ambas infinitas
        return True
    else:
        return False

#Considerando un rayo que sale de v_1 a v_2, buscamos alguna intersección en w_1, w_2
def interseccion_rectas(v_1, v_2, w_1, w_2):
    """Se asume que las rectas (v_1 , v_2) y (w_1 y w_2) no son paralelas, 
    pero no sabemos si alguna tiene pendiente cero"""
    result = np.array([0,0])

    if (v_1[0] == v_2[0]):#Si la recta (v_1 , v_2) tiene pendiente cero
        x = v_1[0]
        #y = mx + b
        m_w = (w_1[1] - w_2[1]) / (w_1[0] - w_2[0])
        b_w = w_1[1]- (m_w * w_1[0])

        result[0] = x
        result[1] = m_w*x + b_w

    elif (w_1[0] == w_2[0]):#Si la recta (w_1 , w_2) tiene pendiente cero
        x = w_1[0]
        #y = mx + b
        m_v = (v_1[1] - v_2[1]) / (v_1[0] - v_2[0])
        b_v = v_1[1]- (m_v * v_1[0])

        result[0] = x
        result[1] = m_v*x + b_v

    else: #Ninguna recta es de pendiente cero

        #y = mx + b
        m_v = (v_1[1] - v_2[1]) / (v_1[0] - v_2[0])
        b_v = v_1[1] - (m_v * v_1[0])

        m_w = (w_1[1] - w_2[1]) / (w_1[0] - w_2[0])
        b_w = w_1[1]- (m_w * w_1[0])


        result[0] = (b_w - b_v) / (m_v - m_w)
        result[1] = m_v * result[0] + b_v

    return result, is_in_AB(w_1,result,w_2)

def is_in_AB(A,punto,B):#Determina si punto se encuentra en el segmento A,B
    distancia_total = distancia(A, B)
    dA = distancia(punto, A)
    dB = distancia(punto, B)

    if abs(dA + dB - distancia_total) < 1:
        return True
    else:
        return False

#@numba.jit
#Mandamos un rayo en la direccion theta hasta que choque con un obstaculo o rebase el limite
def Get_Point(theta, obstaculos, flag_check = False):
    pos_temp = robot_position + [math.cos(theta), math.sin(theta)]#Para formar una arista en direccion teta
    dist_min = INF

    for polygon in obstaculos:#Exploramos los obstaculos
        n_vertices = len(polygon) #Se añade uno extra
        
        for i in range(n_vertices - 1):
            if is_parallel(robot_position, pos_temp, polygon[i],polygon[i+1], flag_check) == False:#Si no son paralelas
                punto, flag = interseccion_rectas(robot_position, pos_temp, polygon[i],polygon[i+1])
                if flag:#Si el rayo intersecto una arista de polygon

                    dist  = distancia(robot_position, punto)
                    if dist < dist_min:#Buscamos el punto minimo para tomar ese punto como la colision
                        dist_min = dist
                        point = punto



    if dist_min < d:#Si el punto esta en el sensor, lo regresamos
        return point, dist_min, True

    #Si no se detecto nada, mandamos un punto maximo en la direccion dada
    return  robot_position + [d*math.cos(theta), d*math.sin(theta)], d, False
  
#@numba.jit
def sensor(obstaculos, flag_check = False):#Exploramos los alrededores del robot para encontrar los puntos end
    angulos = np.linspace(0, 2*np.pi, 360)
    end_points = np.empty((0, 2), dtype=int)


    for teta in angulos:#Exploramos los angulos desde 0 a 2*pi
        if teta == 0:#Caso especial, es una inicializacion
            punto_prev, d_prev, flag_prev = Get_Point(teta, obstaculos)

        else:
            punto_new, d_new, flag_new = Get_Point(teta, obstaculos)

            #Si hay discontinuidad, esta ocurre en dos casos:
            #Caso 1) flag_prev = flag_new = True y abs(d_new - d_prev) > tol, 
            #   i.e, Se detectaron dos puntos, pero su diferencia de distancias rebasa tol
            #Caso 2) flag_prev != flag_new Se detecto un punto cuando no se decto antes, o al revez.

            if flag_prev and flag_new and abs(d_new - d_prev) > tol:#Caso 1)
                #Añadimos el punto mas cercano
                if d_new < d_prev:
                    end_points = np.append(end_points, [punto_new], axis = 0)
                else:
                    end_points = np.append(end_points, [punto_prev], axis = 0)
                
            elif flag_prev != flag_new:#Caso 2)
                #Añadimos al punto que si detectamos
                if flag_new:
                    end_points = np.append(end_points, [punto_new], axis = 0)
                else:
                    end_points = np.append(end_points, [punto_prev], axis = 0)
        
            punto_prev = punto_new
            d_prev = d_new
            flag_prev = flag_new

    #Añadimos el punto T si existe
    teta_extra = angulo(goal_position)#Angulo para ir a goal
    punto_extra, distancia_extra, flag_extra = Get_Point(teta_extra, obstaculos)#Lanzamos el rayo hacia goal
    #punto_extra, distancia_extra, flag_extra = look_for_interseccion(teta_extra, obstaculos)#Lanzamos el rayo hacia goal

    if flag_extra == False:#Agregamos T, porque hay via libre hacia goal
        end_points = np.append(end_points, [punto_extra], axis = 0)
    return end_points

#@numba.jit
def sensor_frontera(obstaculo):#Similar a sensor, pero guarda los puntos de frontera vistos por el robot de un obstaculo
    angulos = np.linspace(0, 2*np.pi, 360)
    frontera =np.empty((0, 2), dtype=int)
    
    for teta in angulos:#Exploramos todos los angulos
        #Get_Point recibe un conjunto de obstaculos, como obstaculo es solo uno, debemos enviarlo entre []
        punto, dist, flag = Get_Point(teta, [obstaculo])
        #punto, dist, flag = look_for_interseccion(teta, [obstaculo])

        if flag:#Si lo detecta el sensor, lo regresamos
            frontera = np.append(frontera, [punto], axis = 0)

    return frontera

#@numba.jit
def angulo(Punto): #Angulo respecto del eje x del robot hacia el punto
    return np.arctan2(Punto[1] - robot_position[1], Punto[0] - robot_position[0])

#@numba.jit
def get_angulo(A, B):#Obtiene en angulo entre dos vectores 
  vector_A = np.append(A, [0])
  vector_B = np.append(B, [0])
  producto_cruz = np.cross(vector_A, vector_B)
  producto_punto = np.dot(vector_A, vector_B)
  magnitud_cruz = np.linalg.norm(producto_cruz)
  return np.arctan2(magnitud_cruz, producto_punto)

#@numba.jit
def get_angulo_x(p1,p2): #Angulo de p1 a p2 respecto al eje X (regresa valores negativos tambien)
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

#@numba.jit
def distancia(A, B):
    return np.sqrt( (B[0] - A[0])** 2 + (B[1] - A[1])**2)

#@numba.jit
def d_min(n):
    return distancia(robot_position, n) + distancia(n, goal_position)

#@numba.jit
def near_enough(A, B):#Nos dice si dos puntos estan lo suficientemente cerca para considerarse los mismos
    #Se usa para determinar cuando se dio una vuelta a un obstaculo
    if distancia(A,B) < tol_error:
        return True
    return False


"""
blocked obstaculo:  Primer Obstaculo detectado entre el robot y goal
d_followed:         Distancia mas corta entre la frontera detectada y goal
A_set     :         Conjunto de puntos vistos por el sensor de la frontera detectada
d_reach:            Distancia entre goal y el punto mas cercano en A_set a goal
"""
#@numba.jit
def get_blocked_obstacle(obstaculos):
    line = np.linspace(0,1,100)#Para explorar los puntos extre la posicion actual del robot hasta goal

    for lambd in line:
        punto = robot_position*(1-lambd) + goal_position*lambd
        for obstaculo in obstaculos:#Vemos si punto esta dentro de un obstaculo
            poligono = Polygon(obstaculo)
            if poligono.contains(Point(punto)):
                return obstaculo

    return []#No se encontro nada
    

#@numba.jit
#Obtenemos el conjunto \Lambda para determinar d_reach
def get_A_set(obstaculos):
    obstaculo = get_blocked_obstacle(obstaculos)
    if len(obstaculo) == 0:#Si no se encontro obstaculo
        return []
    
    return sensor_frontera(obstaculo)#Evaluamos la frontera visible y la regresamos

#@numba.jit
def d_reach(A_set):

    if len(A_set) == 0:#Si hay camino libre, activamos la condicion 3
        return -INF
    
    dist_min = INF
    for x in A_set:
        dist = distancia(x, goal_position)
        if dist < dist_min:
            dist_min = dist
            #x_opt = x

    return dist_min
    

#@numba.jit

def chose_boundary(n, obstaculos):
    Omega = sensor(obstaculos, True)

    #Debemos buscar el punto de Omega mas cercano a la direccion actual
    angulo_min = INF
    for v in Omega:
        angulo = abs(get_angulo_x(n - robot_position, v))
        if angulo < angulo_min:
            point_opt = v
            angulo_min = angulo

    return point_opt

def get_n(Omega):#Encontramos el n que minimiza a d_min en Omega, y su distancia minima
    min = INF
    for n_it in Omega:
        dist = d_min(n_it)
        if dist < min:
            min = dist
            n = n_it
    return n, min

#@numba.jit
def tangent_bug(obstaculos):
    flag = True #Indica si tangetn bug se detine o no
    while flag:

        #Motion to goal
        print("Se empezo el movimiento libre")
        print("Posición actual: ",robot_position)
        Omega = sensor(obstaculos)
        while flag:
            #Tomamos el punto n que minimiza a d_min antes de movernos
            n, min_prev = get_n(Omega)
            
            move(angulo(n))#Nos movemos a n
            Omega = sensor(obstaculos)
            n, min_new = get_n(Omega)#Actualizamos n, y tomamos min_new

            #Vemos si se cumple alguna condicion para romper el sub actual
            #Condicion 1
            if near_enough(robot_position, goal_position):#Si encontramos goal
                texto = Text("Se llego al destino", color='white', font_size=12, pos=centro)
                view.add(texto)
                flag = False
                break

            #Condicion 2    
            if min_new > min_prev:
                break

        print("Se empezo el movimiento por frontera")
        print("Posición actual: ",robot_position)
        #Elegimos una frontera basandonos en la direccion del n calculado previamente 
        n = chose_boundary(n, obstaculos)
        d_followed = INF
        punto_incio = copy.deepcopy(robot_position)#Para determinar si hubo un ciclo
        while flag:
            """Continuamente actualizaremos d_reach y nos moveremos 
                hacia n hasta que se cumpla una de las siguientes condiciones:
            1. alcancemos la meta
            2. Un robot termina un ciclo alrededor de un obstaculo
            3. d_reach < d_followed
            """
            move(angulo(n))#Nos movemos hacia n

            A_set = get_A_set(obstaculos)
            d_reac = d_reach(A_set)
            #Ya tenemos A_Set
            #Actualizamos d_followed si se puede
            if d_reac < d_followed:
                d_followed = d_reac

            #Ahora, revisamos las condiciones para ver si rompemos el while actual
            #Condicion 3
            if d_reac < d_followed:
                flag = False
                break
            
            #Condicion 1
            if near_enough(robot_position, goal_position):#Si encontramos goal
                texto = Text("Se llego al destino", color='white', font_size=12, pos=centro)
                view.add(texto)
                flag = False
                break

            #Condicion 2    
            if near_enough(robot_position, punto_incio):#Si dimos una vuelta al obstaculo
                texto = Text("Es imposible de resolver", color='white', font_size=12, pos=centro)
                view.add(texto)
                flag = False
                break

            n = chose_boundary(n, obstaculos)


#########################

# Obstaculos
vertices_1 = np.array([[100, 100], [200, 150], [250, 300], [150, 300],[100, 100]])
obstaculo_1 = polygon_visual(vertices_1, color=orange) #Para mostrar
view.add(obstaculo_1)

vertices_2 = np.array([[300, 350], [350, 300], [300, 250], [400, 300],[400, 350],[300, 350]])
obstaculo_2 = polygon_visual(vertices_2, color=orange) #Para mostrar
view.add(obstaculo_2)

obstaculos = [vertices_1,vertices_2]

#Robot a mover
robot = Rectangle(center = robot_position, width=5, height=5, color=blue)
robot.transform = MatrixTransform() #Pra usar matrices de transformación
view.add(robot)


#Objetivo a alcanzar
goal = Rectangle(center = goal_position, width=5, height=5, color=red)
view.add(goal)

#print(f"Punto de inicio: {robot_position}\tPunto final: {goal_position}")
tangent_bug(obstaculos)
# Configurar el temporizador para llamar a la función de actualización
#timer = app.Timer()
#timer.connect(tangent_bug(obstaculos))#Funcion a activar con el tiempo
#timer.start()#Cada cuantos segundos actuamos


if sys.flags.interactive != 1 or not hasattr(app, 'process_events'):
   app.run()