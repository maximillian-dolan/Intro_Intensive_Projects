#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:37:04 2023

@author: maxdolan
"""

'''
This module defines a model by which we can investigate how the hawk and sparrow populations change as the 
two species interact both with eachother and with cats. It does this using the solve_ivp function from 
the scipy module.

Factors considered are:
The predation of Sparrows by Hawks and Cats
The birth rate of both Sparrows and Hawks
The starvation rate of Hawks
'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def population_change(t, populations, A = 0.1, B = 0.01, C = 0.01, D = 0.1, E = 0.001):
    
    '''
    This function defines the change in the population of Sparrows, Hawks and Foxes due to their interaction
    between eachother at each time step.
    
    Parameters
    ----------
    t: int/float - single time value
    populations : ndarray, shape (3,) - the current populations of Sparrows, Hawks and Foxes.
    A : (optional, default = 0.1) float - the Sparrow birth rate
    B : (optional, default = 0.01) float - the Sparrow-Hawk predated rate
    C : (optional, default = 0.01) float - the Sparrow-Cat predated rate
    D : (optional, default = 0.1) float - the Hawk starvation rate
    E : (optional, default = 0.001) float - the Hawk birth rate
   
    Returns
    -------
    change_in_x: float - the change in the Sparrow population per unit time
    change_in_y: float - the change in the Hawk population per unit time
    change_in_z: float - the change in the Fox population per unit time
    '''
    
    x = populations[0]
    y = populations[1]
    z = populations[2]
    
    change_in_x = A*x-B*x*y-C*x*z
    change_in_y = -D*y + E*x*y
    change_in_z = 0
    
    return change_in_x, change_in_y, change_in_z



def population(time_steps, sparrow_population, hawk_population, cat_population, A = 0.1, B = 0.01, C = 0.01, D = 0.1, E = 0.001, density = True, method = 'RK45', plot = True):
    
    '''
    This function calculates and plots the final population of the Sparrows, Hawks and Cats after the specified number 
    of timesteps, using the solve_ivp function from Scipy.

    Parameters
    ----------
    time_steps : int - the number of time steps
    sparrow_population : int - the Sparrow population
    hawk_population : int - the Hawk population
    cat_population : int - the Cat population
    A : (optional, default = 0.1) float - the Sparrow birth rate
    B : (optional, default = 0.01) float - the Sparrow-Hawk predated rate
    C : (optional, default = 0.01) float - the Sparrow-Cat predated rate
    D : (optional, default = 0.1) float - the Hawk starvation rate
    E : (optional, default = 0.001) float - the Hawk birth rate
    density : (optional, default = True) bool - whether to also compute a continuous solution, if true only continuous will be plotted
    method : (optional, default = 'RK45') string - allows the choice of integration method. Options can be seen on scipy.integrate.solve_ivp documentation.
    plot : (optional, default = True) bool - whether to automatically plot result


    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,) - Time points.
    y : ndarray, shape (n, n_points) - Values of the solution at `t`.
    sol : `OdeSolution` or None - Found solution as `OdeSolution` instance; None if `dense_output` was set to False.
    t_events : list of ndarray or None - Contains for each event type a list of arrays at which an event of that type event was detected. None if `events` was None.
    y_events : list of ndarray or None - For each value of `t_events`, the corresponding value of the solution. None if `events` was None.
    nfev : int - Number of evaluations of the right-hand side.
    njev : int - Number of evaluations of the Jacobian.
    nlu : int - Number of LU decompositions.
    status : int - Reason for algorithm termination:
    
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.
    
    message : string - Human-readable description of the termination reason.
    success : bool - True if the solver reached the interval end or a termination event occurred (``status >= 0``).

    '''
    
    #Set time spans and initial population array
    time_span = [0,time_steps]
    time_spans = np.linspace(0,time_steps,100)
    initial_population = [sparrow_population, hawk_population, cat_population]
    
    if density == True:
        
        #Run solve_ivp function on population
        new_populations = solve_ivp(population_change, time_span, initial_population, dense_output=True, args = (A,B,C,D,E), method = method)
        
        #Continous solutions
        sparrows = new_populations.sol(time_spans)[0]
        hawks = new_populations.sol(time_spans)[1]
        foxes = new_populations.sol(time_spans)[2]
        
        if plot == True:
            #Plots continuous solutions
            fig, ax = plt.subplots()
            ax.plot(time_spans,sparrows,label  = 'sparrows' , c = 'b')
            ax.plot(time_spans,hawks,label  = 'Hawks' , c = 'orange')
            ax.plot(time_spans,foxes,label  = 'foxes' , c = 'green')
            plt.legend()
            plt.show()

        return new_populations
    
    if density == False:
        
        #Run solve_ivp function on population
        new_populations = solve_ivp(population_change, time_span, initial_population, dense_output=False, args = (A,B,C,D,E), method = method)
        
        if plot == True:
            #Plot calculated solutions
            fig, ax = plt.subplots()
            ax.scatter(new_populations.t,new_populations.y[0],label  = 'Sparrows', marker = 'x', c = 'b')
            ax.scatter(new_populations.t,new_populations.y[1], label = 'Hawks', marker = 'x', c = 'orange')
            ax.scatter(new_populations.t,new_populations.y[2], label = 'Foxes', marker = 'x', c = 'green')
            plt.legend()
            plt.show()

        return new_populations



def iterative_population(time_steps, sparrow_population, hawk_population, cat_population, A = 0.1, B = 0.01, C = 0.01, D = 0.1, E = 0.001, plot = True):
    
    '''
    This function calculates and plots the final population of the Sparrows, Hawks and Cats after the specified number 
    of timesteps, using an iterative while loop method.

    Parameters
    ----------
    time_steps : int - the number of time steps
    sparrow_population : int - the Sparrow population
    hawk_population : int - the Hawk population
    cat_population : int - the Cat population
    A : (optional, default = 0.1) float - the Sparrow birth rate
    B : (optional, default = 0.01) float - the Sparrow-Hawk predated rate
    C : (optional, default = 0.01) float - the Sparrow-Cat predated rate
    D : (optional, default = 0.1) float - the Hawk starvation rate
    E : (optional, default = 0.001) float - the Hawk birth rate
    plot : (optional, default = True) bool - whether to automatically plot result


    Returns
    -------
    spopulatin : float - the final Sparrow population
    hpopulatin : float - the final Hawk population
    cpopulatin : float - the final Cat population
    '''
    
    #set individual arrays for population and time
    spopulation = np.array([sparrow_population])
    hpopulation = np.array([hawk_population])
    cpopulation = np.array([cat_population])
    time_span = np.linspace(0,time_steps,num = time_steps)
    
    i = 0
    
    #while loop calculates rate of change at every time step and adds respective new population to each population array
    while i <time_steps-1:
        populations = [spopulation[i],hpopulation[i],cpopulation[i]]
        changes = population_change(1, populations) #value for t doesn't matter as its not actually used
        spopulation = np.append(spopulation,spopulation[i]+changes[0])
        hpopulation = np.append(hpopulation,hpopulation[i]+changes[1])
        cpopulation = np.append(cpopulation,cpopulation[i]+changes[2])
        
        i+=1
     
    #Plot populations if plot = true
    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(time_span,spopulation, label  = 'Sparrows', marker = '1', c = 'b')
        ax.scatter(time_span, hpopulation, label = 'Hawks', marker = '1', c = 'orange')
        ax.scatter(time_span, cpopulation, label = 'Foxes', marker = '1', c = 'green')
        plt.legend()
        plt.show()    
    
    return spopulation, hpopulation, cpopulation








