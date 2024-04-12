import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re

#=============Initialization=======================
fitness_count = 0
weight_Limit = 285
chromosome_Len = 100 #Length of gene(a solution length)
fitness_Limit = 10000 # If reach 10000 evaluate, stop
tourn_size = int(input("Input tournament size:"))
PM = int(input("Input integer as mutation rate:"))
Population_size = int(input("Input population size:"))

#Read weight and value data from BankProblem.txt, write in weightList and valueList
v = ''
w = ''
with open("BankProblem.txt") as f:
    data_list = f.readlines()
    i = 2
    while i<= 300:
        w = w + data_list[i]
           # print(data_list[i])
        i = i + 3
        #print(weight)
    weightList = re.findall(r"\d+\.?\d*",w)
    weightList = [ float(x) for x in weightList ] #type transfer

    j = 3
    while j<= 300:
        v = v + data_list[j]
        j = j + 3
    valueList = re.findall(r"\d+",v)
    valueList = [int(x) for x in valueList ] #type transfer

item = [[0]*2 for i in range(chromosome_Len)]
i = 0
while i < chromosome_Len:
    item [i][0] = weightList[i] # 0,0 front
    item [i][1] = valueList[i]  # 0,1 rear
    i+=1

#==============Initialize population ==================
def init(Population_size):
    Population = [[0]*chromosome_Len for i in range(Population_size)]
    for i in range(0,Population_size):
        for j in range(0,chromosome_Len):
            Population[i][j] = random.randint(0,1)
    return Population


#==============Evaluate Fitness ==================
def eval_Fitness(Population):
    global fitness_count
    fitness = []
    for ind in Population: #Scan all chromosome
        fitness_count += 1
        weight_ttl = 0
        value_ttl = 0
        for i, v in enumerate(ind):
            if int(v) == 1:
                weight_ttl += item[i][0]
                value_ttl += item[i][1]
        fitness.append([float('{:.1f}'.format(weight_ttl)), int(value_ttl)])
      
    return fitness
    
#============== Selection ======================
#Selection_filter include pop out not suitable data from initial population
def selection_Filter(Population,fitness):
    
    #remove pop if weight over 285 for generate initial population
    nsnum = 0
    index = len(fitness) - 1
    while index >=0:
        index -= 1
        if fitness[index][0] > weight_Limit:
            Population.pop(index) # Pop out not suitable individual
            fitness.pop(index)
            nsnum += 1
    return Population, fitness, nsnum

#=============Single-point Crossover=====================
def crossover(a,b):
    chromosome_Len = 100
   #Binary tournament selection, select 2 parents a & b to crossover and generate c and d
    cut_pos = random.randint(1, chromosome_Len-2)
    c = a[:cut_pos+1] + b[cut_pos+1:]
    d = b[:cut_pos+1] + a[cut_pos+1:]
    child_cd = [[0]*chromosome_Len for i in range(2)]
    child_cd[0] = c
    child_cd[1] = d

    return child_cd

#=================Mutation=====================
def mutation(child_cd, PM):
    #Choose PM number of positions to mutate
    global chromosome_Len
    mut_pos = random.sample(range(0,chromosome_Len), PM)
    
    for i in mut_pos:
        if child_cd[0][i] == 1:
            child_cd[0][i] = 0
        else:
            child_cd[0][i] = 1
                
        if child_cd[1][i] == 1:
            child_cd[1][i] = 0
        else:
            child_cd[1][i] = 1
            
    e = child_cd[0]
    f = child_cd[1]
    child_ef = [[0]*chromosome_Len for i in range(2)]
    child_ef[0] = e
    child_ef[1] = f
    
    return child_ef
    
#==================Weakest Replacement=============
def weakest_replacement(child_ef, fitness_e, fitness_f, Fitness_P, Population_P):

    v_list = [] #Only value of Fitness_P
    for i in range(len(Fitness_P)):
        v_list.append(Fitness_P[i][1]) #Fitness of Population_P's value
    
    index_1 = v_list.index(min(v_list))
    
    if fitness_e[1] > v_list[index_1]:
        v_list.pop(index_1)
        v_list.insert(index_1, fitness_e[1])
        Population_P.pop(index_1)
        Population_P.insert(index_1, child_ef[0])
        index_2 = v_list.index(min(v_list))
        
        if fitness_f[1] > v_list[index_2]:
            v_list.pop(index_2)
            v_list.insert(index_2, fitness_f[1])
            Population_P.pop(index_2)
            Population_P.insert(index_2, child_ef[1])
        
    elif fitness_f[1] > v_list[index_1]:
        Population_P.pop(index_1)
        Population_P.insert(index_1, child_ef[1])
        
    return Population_P

#==================Main=======================

def main():
    
    global weight_Limit
    global fitness_Limit
    global fitness_count
    global chromosome_Len
    global PM
   
    #Step1: Initial population and evaluate all fitness
    Population_P = init(Population_size)
    Fitness_P = eval_Fitness(Population_P)
    Population_P,Fitness_P,nsnum = selection_Filter(Population_P, Fitness_P)

    while nsnum != 0: # nsum means the amount of solution which over weight
        nsnum_P = init(nsnum)
        nsnum_Fitness = eval_Fitness(nsnum_P)
        nsnum_P,nsnum_Fitness,nsnum = selection_Filter(nsnum_P, nsnum_Fitness)
        Population_P = Population_P + nsnum_P
        Fitness_P = Fitness_P + nsnum_Fitness
    
    trial = 0
    fitness_count = 0
    plot_data = pd.DataFrame(columns=['Weight', 'Value'], index=list(range(1,)))
    plot_data.index = plot_data.index + 1
    
    #Step2: Do binary tournament to generate a & b
    while trial >= 0:
        champion =  [[0]*chromosome_Len for i in range(0,2)]
        k = 0
        while k <= 1: # a & b
            T_V = [[0]*1 for i in range(tourn_size)] #[[0]*2 for i in range(tourn_size)] # Store tournament value and it's
            T_I = [[0]*1 for i in range(tourn_size)]
            random_index_T = random.sample(range(0,len(Population_P)), tourn_size)
            for i in range(0,tourn_size):
                T_I[i] = random_index_T[i]
                T_V[i] = Fitness_P[random_index_T[i]][1]
            T_max_index = T_V.index(max(T_V))
            champion[k] = Population_P[T_I[T_max_index]]
            k = k+1
        
        a = champion[0]
        b = champion[1]
        
        #Step3: Crossover and generate c and d
        child_cd = crossover(a,b)
        c = child_cd[0]
        d = child_cd[1]
    
        #Step4: Mutation c & d to generate e & f
        child_ef = mutation(child_cd, PM)
        e = child_ef[0]
        f = child_ef[1]
        fitness_ef = eval_Fitness(child_ef)
        fitness_e = fitness_ef[0]
        fitness_f = fitness_ef[1]

        # If e or f is over weight, then re-mutation again from c and d to generate new solutions
        while fitness_e[0] > weight_Limit or fitness_f[0] > weight_Limit:
            child_ef = mutation(child_cd,PM)
            e = child_ef[0]
            f = child_ef[1]
            fitness_ef = eval_Fitness(child_ef)
            fitness_e = fitness_ef[0]
            fitness_f = fitness_ef[1]
        
        #Step5: Weakest replacement
        Population_P = weakest_replacement(child_ef, fitness_e, fitness_f, Fitness_P, Population_P) # Add replaced chromosome to original population
        Fitness_P = eval_Fitness(Population_P) #After replacement, re-evaluate the fitness of  Fitness_P)
        
        # check point
        if fitness_count > fitness_Limit:
            Final_solution = Best_solution
            Final_bag = []
            index = 0
            for i in range(0,len(Final_solution)):
                index += 1
                if Final_solution[i] == 1:
                    Final_bag.append('bag '+ str(index))
            Final_fitness = Fitness_P[Max_value_index]
            break
    
        # Step6: Best solution of this trial fromm Population_P
        v_list = []
        for i in range(len(Fitness_P)):
            v_list.append(Fitness_P[i][1]) #Fitness of Population_P's value
            
        Max_value_index = v_list.index(max(v_list))
        Best_solution = Population_P[Max_value_index]
        Best_fitness = Fitness_P[Max_value_index]
        weight_value = {"Weight":Best_fitness[0]*10,"Value":Best_fitness[1]}
        plot_data = plot_data.append(weight_value, ignore_index=True)
        trial += 1
    
    
    #=============Ptint best result=============
    plot_data = plot_data.drop(labels = 0)
    pd.set_option('display.max_rows', None)
    print(plot_data)
    #Final result, after 10000 fitness
   
    #Value of each generation
    plt.figure(1)
    plot_data.Value.plot()
    plt.title('Bank Problem',fontsize=24)
    plt.xlabel('Generation',fontsize=14)
    plt.ylabel('Value',fontsize=14)

    #Weight of each generation
    plt.figure(2)
    plot_data.Weight.plot(color='darkorange')
    plt.title('Bank Problem',fontsize=24)
    plt.xlabel('Generation',fontsize=14)
    plt.ylabel('Weight',fontsize=14)

    print('==================Final Solution===============')
    print('Generation count:', trial)
    print('fitness_count:', fitness_count)
    print('Best 0-1 solution is:', Final_solution)
    print('Best bag combination:', Final_bag)
    print('Best Max value:', Final_fitness[1])
    print('Weight of total bags:', Final_fitness[0])
    plt.show()
    
    
if __name__ == '__main__':
    main()
    
    
    
 


