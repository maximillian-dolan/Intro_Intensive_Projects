#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module introduces a class that can be used for orbital bodies, and also defines 
a function which caculates and saves the position of each body within an n-body system.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

#Define gravitational constant
G = 6.6743e-11

#Define orbital class
class Orbital:
    
    def __init__(self, name, mass, x, y, v_x, v_y):
        
        position = np.array([x,y])
        velocity = np.array([v_x,v_y])
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = 0
        self.old_acceleration = 0
        
    def __str__(self):
        return f'{self.name} has a mass of {self.mass}kg.'
    
    def calc_acceleration(self, orbitals):
        '''
        Calculates and updates acceleration of body.

        Parameters
        ----------
        orbitals : list
            list of the orbitals in the system. All orbitals must be of Orbital class.

        Returns
        -------
        acceleration : 2-dimensional vector
            The vector describing the acceleration of the body due to the graviational force of other bodies in the system.
        '''
        acceleration = 0
        
        for i in orbitals:   
            
            #Only run if not self, otherwise will get a 0 division error
            if i.name !=self.name:
                distance_vector = i.position - self.position
                acceleration += (G*i.mass*distance_vector)/(np.linalg.norm(distance_vector))**3
        
        self.acceleration = acceleration
            
        return acceleration

 
def position_update(orbitals, method = 'verlet', timestep = 3600, timelength = 3, save = False, filename = '/Users/maxdolan/Documents/Jupyter_lab_stuff/tests/assessment-and-data_2/data_test.csv', keep_original = True):
    '''
    This function calculates the 2-dimensional positions of each body within the system.

    Parameters
    ----------
    orbitals : LIST
        The list of the orbital bodies, note every body must be of Orbital class
    method : 'verlet' or 'runge-kutta', optional
        The method of iteration in calculating orbits. The default is 'verlet'.
    timestep : INT, optional
        The timestep used for calculating orbits, in seconds. The default is 3600, an hour.
    timelength : FLOAT, optional
        The length of the simulation, in Earth years. The default is 3.
    save : BOOL, optional
        Whether or not the data is saved in a CSV file. The default is False.
    filename : STR, optional
        The filename if saving in a CSV file. The default is '/Users/maxdolan/Documents/Jupyter_lab_stuff/tests/assessment-and-data_2/data_test.csv'.
    keep_original : BOOl, optional
        Whether or not to return the orbitals at the end as having their original position and velocity values. The default is True

    Returns
    -------
    positions : DataFrame
        The X and Y positions of each orbital body. x1/y1, x2/y2 etc. corrspond to the position of the body in the list.

    '''
    
    step_count = int((timelength*365.25*24*3600)/timestep)
    data = {}
    originals = np.zeros((len(orbitals),2,2))
    
    #Initialises each body's acceleration, the DataFrame, and saves the original values at timesteps = 0.
    for i in enumerate(orbitals):
        i[1].calc_acceleration(orbitals)
        data['x{0}'.format(i[0] + 1)] = np.zeros(step_count)
        data['y{0}'.format(i[0] + 1)] = np.zeros(step_count)
        originals[i[0],0] = i[1].position
        originals[i[0],1] = i[1].velocity
        
        
    positions = pd.DataFrame(data = data)
    
    
    if method =='verlet':
        for n in range(step_count):
            
            #Save old positions and calculate new positions
            for i in enumerate(orbitals):
                
                positions[positions.columns[2 * i[0]]][n] = i[1].position[0]
                positions[positions.columns[2 * i[0] + 1]][n] = i[1].position[1]
                
                i[1].position = i[1].position + i[1].velocity * timestep + 0.5*i[1].acceleration * timestep**2
            
            #calculate new acceleration and velocity
            for i in enumerate(orbitals):   
                i[1].old_acceleration = i[1].acceleration
                i[1].calc_acceleration(orbitals)
                i[1].velocity = i[1].velocity + (timestep/2)*(i[1].acceleration+i[1].old_acceleration)

        
    elif method == 'runge-kutta':
        for n in range(step_count):
            
            #Sets up arrays that store values needed only across a single timestep
            k_values = np.zeros((len(orbitals),4,2,2))
            rk_originals = np.zeros((len(orbitals),2,2))
            
            #Saves position and velocity to positions dataframe, and saves the position and velocity for use within timestep
            for i in enumerate(orbitals):
                positions[positions.columns[2 * i[0]]][n] = i[1].position[0]
                positions[positions.columns[2 * i[0] + 1]][n] = i[1].position[1]
                
                rk_originals[i[0],0] = i[1].position
                rk_originals[i[0],1] = i[1].velocity

            #Calculate k1 values
            for i in enumerate(orbitals):
                k_values[i[0],0,0] = rk_originals[i[0],1]
                k_values[i[0],0,1] = i[1].acceleration

            #calculate k2 values
            for i in enumerate(orbitals):
                
                #Position for all orbitals needs to be calculated before acceleration is calculated for each k value
                i[1].position = rk_originals[i[0],0] + 0.5 * timestep * k_values[i[0],0,0]
                
            for i in enumerate(orbitals): 
                
                #Calculates acceleration and velocity 0.5 timesteps into future, under gradient given by previous k velocity
                i[1].calc_acceleration(orbitals)
                
                k_values[i[0],1,0] = k_values[i[0],0,0] + 0.5 * timestep * k_values[i[0],0,1]
                k_values[i[0],1,1] = i[1].acceleration 

            #calculate k3 values
            for i in enumerate(orbitals):
                
                i[1].position = rk_originals[i[0],0] + 0.5 * timestep * k_values[i[0],1,0]
                
            for i in enumerate(orbitals):   
                i[1].calc_acceleration(orbitals)
                
                k_values[i[0],2,0] = k_values[i[0],0,0] + 0.5 * timestep * k_values[i[0],1,1]
                k_values[i[0],2,1] = i[1].acceleration

            #calculate k4 values
            for i in enumerate(orbitals):
                
                i[1].position = rk_originals[i[0],0] + timestep * k_values[i[0],2,0]
                
            for i in enumerate(orbitals):   
                i[1].calc_acceleration(orbitals)
                
                k_values[i[0],3,0] = k_values[i[0],0,0] + timestep * k_values[i[0],2,1]
                k_values[i[0],3,1] = i[1].acceleration


            #calculate actual next position and velocity
            for i in enumerate(orbitals):
                
                i[1].position = rk_originals[i[0],0] + (timestep/6)*(k_values[i[0],0,0]+ 2 * k_values[i[0],1,0] + 2 * k_values[i[0],2,0] + k_values[i[0],3,0])
                i[1].velocity = rk_originals[i[0],1] + (timestep/6)*(k_values[i[0],0,1]+ 2 * k_values[i[0],1,1] + 2 * k_values[i[0],2,1] + k_values[i[0],3,1])
   
    else:
        print("that is not a method. Methods are 'verlet' or 'runge-kutta'.")
        
    #Saves to CSV file
    if save == True:
        positions.to_csv(filename, index = False)
    
    #Sets velocity and position values back to their originals
    if keep_original == True:
        for i in enumerate(orbitals):
            i[1].position = originals[i[0],0]
            i[1].velocity = originals[i[0],1]
    
    return positions
  

  
def calc_eccentricity(orbital, centre, positions, orbitals, period_method = 'kepler', timestep = 3600):
    '''
    Caclulates the eccentricity of an orbit

    Parameters
    ----------
    orbital : INT
        Refers to the orbital body. Integer refers to x1/y1, x2/y2 etc. in positions.
    centre : INT
        Refers to the body at centre of orbit. Integer refers to x1/y1, x2/y2 etc. in positions.
    positions : DataFrame
        The DataFrame containing positions. Must be of column format x1,y1,x2,y2, etc.
    orbitals : LIST
        The list of the original orbital bodies, note every body must be of Orbital class and in the same order as originally used to calculate position.
    period_method : 'kepler' or 'manual'
        Dictates which period calculating method to use. Default is 'kepler'.
    timestep : INT, optional
        The timestep used for calculating orbits, as Kepler method returns periods in seconds, in seconds. The default is 3600, an hour.
        
    Returns
    -------
    eccentricity : FLOAT
        The eccentricity of the orbit.

    '''
    
    #find relative position of orbital to centre, and create array of its distance from the centre
    orbital_positions = np.array([positions[positions.columns[2*(orbital-1)]], positions[positions.columns[2*(orbital-1)+1]]])
    centre_positions = np.array([positions[positions.columns[2*(centre-1)]], positions[positions.columns[2*(centre-1)+1]]])
    
    orbital_distance = np.linalg.norm((orbital_positions-centre_positions).T, axis = -1)
    
    #find length of one period
    if period_method == 'kepler':
        period = kepler_period(orbital, centre, positions, orbitals)/timestep
    
    elif period_method == 'manual':
        period = manual_period(orbital ,centre, positions)
        
    else:
        print('That is not a period method')
        period = 0
    
    #analyse one period, and find apoapis and periapis, then use to find to eccentricity
    if period !=0:
        apoapis = max(orbital_distance[:int(period)])
        periapis = min(orbital_distance[:int(period)])
    
        eccentricity = (apoapis-periapis)/(apoapis+periapis)
        
        return eccentricity
    
    #if period function returns 0, orbital does not complete a period, and therefore eccentricity cannot be calculated
    else:
        return 0



def manual_period(orbital ,centre, positions):
    '''
    Caclulates the average period of an orbit by finding distance between maximum y coordinates

    Parameters
    ----------
    orbital : INT
        Refers to the orbital body. Integer refers to x1/y1, x2/y2 etc. in positions.
    centre : INT
        Refers to the body at centre of orbit. Integer refers to x1/y1, x2/y2 etc. in positions.
    positions : DataFrame
        The DataFrame containing positions. Must be of column format x1,y1,x2,y2, etc.

    Returns
    -------
    period : FLOAT
        The average period of the orbit, expressed in the timestep used in positions.

    '''
    
    #find vector describing position of orbital relative to centre
    orbital_positions = np.array([positions[positions.columns[2*(orbital-1)]], positions[positions.columns[2*(orbital-1)+1]]])
    centre_positions = np.array([positions[positions.columns[2*(centre-1)]], positions[positions.columns[2*(centre-1)+1]]])    
    
    orbital_relative_position = orbital_positions - centre_positions
    
    #finds index of each y-maximum in the orbit (relative to centre of orbit)    
    maximums = find_peaks(orbital_relative_position[1])[0]
    
    #if two maximums cant be found, orbital does not complete a full orbit
    if len(maximums) <=1:      
        print('The orbital does not complete a full orbit')       
        return 0 #Return 0 to not cause error in any further calculations
    
    #else find number of timesteps between peaks and use average for orbit
    else:              
        widths = np.diff(maximums)
        orbit = np.mean(widths)
        return orbit

def kepler_period(orbital ,centre, positions, orbitals):
    '''
    Caclulates the period of an orbit using Kepler's third law

    Parameters
    ----------
    orbital : INT
        Refers to the orbital body. Integer refers to x1/y1, x2/y2 etc. in positions.
    centre : INT
        Refers to the body at centre of orbit. Integer refers to x1/y1, x2/y2 etc. in positions.
    positions : DataFrame
        The DataFrame containing positions. Must be of column format x1,y1,x2,y2, etc.
    orbitals : LIST
        The list of the original orbital bodies, note every body must be of Orbital class and in the same order as originally used to calculate position.
   
    Returns
    -------
    period : FLOAT
        The average period of the orbit, expressed in seconds.

    '''
    
    #find relative position of orbital to centre, and create array of its distance from the centre
    orbital_positions = np.array([positions[positions.columns[2*(orbital-1)]], positions[positions.columns[2*(orbital-1)+1]]])
    centre_positions = np.array([positions[positions.columns[2*(centre-1)]], positions[positions.columns[2*(centre-1)+1]]])
    
    orbital_distance = np.linalg.norm((orbital_positions-centre_positions).T, axis = -1)
    
    #Find values for apoapsis and masses
    a = (max(orbital_distance)+min(orbital_distance))/2
    
    M = orbitals[centre-1].mass
    m = orbitals[orbital-1].mass
    
    #Calculates Kepler's third law
    period = np.sqrt((4*(np.pi**2)*(a**3))/(G*(M+m)))
    
    return period









