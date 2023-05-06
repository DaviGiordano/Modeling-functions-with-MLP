import numpy as np
def redeMLP_21(x, d, eta, Nt, Nb, Ne, W01_1, W02_1, W01_2):
    """
    J_MSE, W1_1, W2_1, W1_2 = redeMLP(x, d, Nn, eta, Nt, Nb, Ne, W0)
    Saídas:
    J_MSE: valor da função custo ao longo das épocas
    W1_1: vetor de pesos do neurônio 1 da camada 1
    W2_1: vetor de pesos do neurônio 2 da camada 1
    W1_2: vetor de pesos do neurônio 1 da camada de saída
    Entradas:
    x: sinal de entrada
    d: sinal desejado
    eta: passo de adaptação
    Nt: número de dados de treinamento
    Nb: tamanho do mini-batch
    Ne: número de épocas
    W01_1: vetor de pesos do neurônio 1 da camada 1 (útlima iteração, inclui o bias)
    W02_1: vetor de pesos do neurônio 2 da camada 1 (útlima iteração, inclui o bias)
    W01_2: vetor de pesos o neurônio 1 da camada de saída (útlima iteração, inclui o bias)
    """

       
    # número de mini-batches por época
    Nmb = int(np.floor(Nt / Nb))
    
    # inicialização dos pesos
    W1_1 = W01_1.copy()
    W2_1 = W02_1.copy()
    W1_2 = W01_2.copy()
    
    # passo de adaptação dividido pelo tamanho do mini-batch
    eta = eta / Nb

    # inicialização do vetor que contém o valor da função custo
    J_MSE = np.zeros((Ne, 1))

    # Juntamos o vetor de entrada com o sinal desejado e inserimos
    # uma coluna de uns para levar em conta o bias
    Xd = np.hstack((np.ones((Nt, 1)), x, d))

    # vetor de uns para o bias no mini-batch
    b = np.ones((Nb, 1))

    # for das épocas
    for k in range(Ne): 
        np.random.shuffle(Xd)
        X = Xd[:, 0 : 3]
        d = Xd[:, [3]]

        # for dos mini-batches
        for l in range(Nmb):
            dmb = d[l * Nb : (l + 1) * Nb].reshape(-1, 1)
            X1mb = X[l * Nb : (l + 1) * Nb, :]
            
            # Cálculo Progressivo
            # Neurônio 1 da camada 1
            v1mb_1 = X1mb @ W1_1.T
            y1mb_1 = sigmoid(v1mb_1)
            dphi1_1 = y1mb_1 * (1 - y1mb_1)  # derivada da função sigmoide

            # Neurônio 2 da camada 1
            v2mb_1 = X1mb @ W2_1.T
            y2mb_1 = sigmoid(v2mb_1)
            dphi2_1 = y2mb_1 * (1 - y2mb_1) 
            
            # Neurônio de saída
            X2mb = np.hstack((b, y1mb_1, y2mb_1))
            v1mb_2 = X2mb @ W1_2.T
            y1mb_2 = sigmoid(v1mb_2)
            dphi1_2 = y1mb_2 * (1 - y1mb_2) 

            # erro da última camada                
            e1mb_2 = dmb - y1mb_2 

            #############
            # Complete o código
            
            delta1_2 = dphi1_2*e1mb_2
            delta1_1 = dphi1_1*delta1_2*W1_2[0, 1]
            delta2_1 = dphi2_1*delta1_2*W1_2[0, 2]

            # atualização dos pesos da camada

            W1_2 = W1_2 + eta*(delta1_2.T @ X2mb)
            W1_1 = W1_1 + eta*(delta1_1.T @ X1mb)
            W2_1 = W2_1 + eta*(delta2_1.T @ X1mb)
            
            #############

            # guarda no vetor J_MSE a norma do vertor de erros de saída ao quadrado
            J_MSE[k] = (J_MSE[k] + (np.linalg.norm(e1mb_2)) ** 2)

        # cálculo do MSE (divide o valor acumulado pelo número de
        # mini-batches x tamanho do batch x número de neurônios
        # da camada de saída)        
        J_MSE[k] = J_MSE[k] / (Nmb * Nb * 1)
        
        if k % 100 == 0:
            print(f"Época: {k}, MSE: {J_MSE[k]}")
          
    return J_MSE, W1_1, W2_1, W1_2