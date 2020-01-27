'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''
import time
import re
import random
import numpy as np
import math
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plp

num_neuronio_oculto = 100

centroides = []

cont = 0
x1 = []
x2 = []
y = []

mat_att_treino = np.ones((1001, 2))
mat_resp_treino = np.ones((1001, 1))
mat_plano_separador = np.ones((100, 2))


dados = open("twomoons.dat", "r")
for line in dados:
    # separando o que é x do que é d
    line = line.strip()  # quebra no \n
    line = re.sub('\s+', ',', line)  # trocando os espaços vazios por virgula
    xa, xb, y1 = line.split(",")  # quebra nas virgulas e retorna 3 valores
    x1.append(float(xa))
    x2.append(float(xb))
    y.append(float(y1))
dados.close()


def cria_mat_all():#pega os dados extraidos da base e monta uma matriz com todos eles
    mat = np.ones((1001, 3))

    for i in range(0, 1001):
        mat[i][0] = x1[i]
        mat[i][1] = x2[i]
        mat[i][2] = y[i]
    #mistura todas as linhas da matriz
    mat = np.random.permutation(mat)

    return mat


def cria_centroides():#cria os centroides usando kmeans
    global centroides
    mat_all = cria_mat_all()
    mat_dados = np.ones((1001, 2))
    #extraindo somente os dados da minha matriz com toda a base
    for i in range(0, 1001):
        for j in range(0, 2):
            mat_dados[i][j] = mat_all[i][j]

    kmeans = KMeans(n_clusters=num_neuronio_oculto,
                    random_state=0).fit(mat_dados)

    centroides = kmeans.cluster_centers_


def cria_mat_att_e_resp_treino_e_teste():
    global mat_att_treino, mat_resp_treino
    mat_all = cria_mat_all()
    # TREINO
    for i in range(0, 1001):
        for j in range(0, 2):
            mat_att_treino[i][j] = mat_all[i][j]

    for i in range(0, 1001):
        mat_resp_treino[i][0] = mat_all[i][2]


def neuronios_ocultos():#passa todas as entradas pelas funçoes de ativação e retorna uma matiz com os resultados
    global centroides
    G = np.ones((1001, num_neuronio_oculto+1))
    #Calculando as saidas dos neuronio ocultos
    for j in range(0, 1001):
        for k in range(1, num_neuronio_oculto+1):
            G[j][k] = math.exp((-1)*(((mat_att_treino[j][0] - centroides[k-1][0])**2
                                      + (mat_att_treino[j][1] - centroides[k-1][1])**2)))
    return G


def neuronio_saida_W():# gerando os W(pesos) da camada de saida
    global mat_resp_treino
    G = neuronios_ocultos()
    d = mat_resp_treino

    W = np.dot(np.dot(np.linalg.inv(
        np.dot(np.transpose(G), G)), np.transpose(G)), d)

    return W


def testa():
    global num_neuronio_oculto
    G = np.ones((1, num_neuronio_oculto+1))
    x = []
    y = []
    aux = 0
    print('Calculando valores para varrer a area!!!')
    while aux <= 7:
        aux += 0.028
        x.append(aux)
    aux = 2.3
    while aux <= 4.8:
        aux += 0.01
        y.append(aux)
    #time.sleep(0.8)
    print('Valores Calculados')
    print('Gerando grafico de saida...')
    
    W = neuronio_saida_W()
    x_ = []
    y_ = []
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            for k in range(1, num_neuronio_oculto+1):
                G[0][k] = math.exp((-1)*((x[i] - centroides[k-1][0])**2
                                         + (y[j] - centroides[k-1][1])**2))

            resp_rede = np.dot(G, W)
            # se entrar no if é pq a rede está em duvida sobre akele ponto entao eu o salvo para exibi-los depois
            if resp_rede <= 0.05 and resp_rede >= -0.05:
                x_.append(x[i])
                y_.append(y[j])

    #pegando os valores da base para a plotagem
    X1 = x1[0:501]
    Y1 = x2[0:501]
    X2 = x1[502:]
    Y2 = x2[502:]
    #Plotando o "hiperplano" separador
    plp.title("Num. neurônios ocultos: "+ str(num_neuronio_oculto))
    plp.plot(x_, y_, color='black')
    plp.scatter(X1, Y1, marker=".", color='red')
    plp.scatter(X2, Y2, marker=".")
    plp.show()


cria_centroides()

cria_mat_att_e_resp_treino_e_teste()

testa()
