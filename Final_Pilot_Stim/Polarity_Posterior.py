# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:53:50 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:06:28 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:46:47 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:41:22 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:31:02 2024

@author: user
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:25:19 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:14:35 2024

@author: user
"""

import camstim
from psychopy import visual, event, core, monitors
import numpy as np
import random
from camstim import SweepStim,  Foraging
from camstim.sweepstim import StimulusArray
from camstim.sweepstim import Stimulus
import itertools
from psychopy.tools.arraytools import val2array

from camstim import Window, Warp

DIR_IND = 0
OPACITY_IND = 1
COHERENCE_IND = 2
DOT_SIZE_IND = 3
SPEED_IND = 4
FIELD_SIZE_IND = 5
NDOTS_IND = 6



DIR_CIRC_IND_TRIAL = 1
DIR_SQR_IND_TRIAL  = 0
OPACITY_SQR_IND_TRIAL  = 2
OPACITY_CIRC_IND_TRIAL  = 3
COHERENCE_IND_TRIAL = 4
DOT_SIZE_IND_TRIAL = 5
SPEED_IND_TRIAL = 6
FIELD_SIZE_SQR_IND_TRIAL = 7
FIELD_SIZE_CIRC_IND_TRIAL = 8

DOTSDensity_SQR_IND_TRIAL = 9
DOTSDensity_CIRC_IND_TRIAL = 10


NUM_REPS = 1


dev_mode = True
if dev_mode:
    my_monitor = monitors.Monitor(name='Test')
    my_monitor.setSizePix((1280,800))
    my_monitor.setWidth(20)
    my_monitor.setDistance(15)
    my_monitor.saveMon()
    win = Window(size=[1024,768],
        fullscr=False,
        screen=0,
        monitor= my_monitor,
        warp=Warp.Spherical,
        color= "white"
       # units='deg'
    )
else: 
    win = Window(
        fullscr=True,
        screen=1,
        monitor='Gamma1.Luminance50',
        warp=Warp.Spherical,
        units ='deg'
    )

# some constants
_piOver2 = np.pi / 2.
_piOver180 = np.pi / 180.
_2pi = 2 * np.pi

sweep_length_num       = 1
start_time_nun          = 0
blank_length_num        = 0.5 
blank_sweeps_num        = 0


# The following fixes the issue with the dots not being created on the edges of the screen. 
class FixedDotStim(visual.DotStim):
    def __init__(self, *args, **kwargs):
        nDots = kwargs.get('nDots')   
        self._deadDots = np.zeros(nDots, dtype=bool)

        # Initialize the dots density
        self.dotDensity = 1.0
        
        super(FixedDotStim, self).__init__(*args, **kwargs)
        # This is needed to avoid bar of dots. 
        self.refreshDots()

    def _newDotsXY(self, nDots):
        """Returns a uniform spread of dots, according to the `fieldShape` and
        `fieldSize`.

        Parameters
        ----------
        nDots : int
            Number of dots to sample.

        Returns
        -------
        ndarray
            Nx2 array of X and Y positions of dots.

        Examples
        --------
        Create a new array of dot positions::

            dots = self._newDots(nDots)

        """
        if self.fieldShape == 'circle':
            length = np.sqrt(np.random.uniform(0, 1, (nDots,)))
            angle = np.random.uniform(0., _2pi, (nDots,))

            newDots = np.zeros((nDots, 2))
            newDots[:, 0] = length * np.cos(angle)
            newDots[:, 1] = length * np.sin(angle)

            newDots *= self.fieldSize * .5
        else:
            newDots = np.random.uniform(-0.5, 0.5, size = (nDots, 2)) * self.fieldSize
            
        return newDots
      
    # The function belows are needed to allow the Stimulus object to 
    # overwrite the parameters of the dots.
    def setdotSize(self, dotSize):
        self.dotSize = dotSize
        self.refreshDots()

    def setopacity(self, opacity):
        self.opacity = opacity
        self.refreshDots()

    def setspeed(self, speed):
        self.speed = speed
        self.refreshDots()

    def setfieldSize(self, fieldSize):
        fieldSize = val2array((fieldSize, fieldSize), False)
        self.fieldSize = fieldSize
        self.size = fieldSize
        self.refreshDots()

    def setdotDensity(self, dotDensity):
        self.dotDensity = dotDensity
        self.refreshDots()

    def setnDots(self, nDots):
        self.nDots = nDots
        self.refreshDots()
        
        
    # updating the position of the dots for the square apparatus according to the direction of the movement
    def getRandPosInSquareSide(self,sideNum):
        xy = np.zeros((1, 2))
        if sideNum==0:
            xy[:,0] = -0.5*self.fieldSize[0] #set x
            xy[:,1] = np.random.uniform(-0.5, 0.5, size = None) * self.fieldSize[1] #set y
        elif sideNum==1:
            xy[:,0] = np.random.uniform(-0.5, 0.5, size = None) * self.fieldSize[0] #set x
            xy[:,1] = -0.5*self.fieldSize[1] #set y 
        if sideNum==2:
            xy[:,0] = 0.5*self.fieldSize[0] #set x
            xy[:,1] = np.random.uniform(-0.5, 0.5, size = None) * self.fieldSize[1] #set y
        elif sideNum==3:
            xy[:,0] = np.random.uniform(-0.5, 0.5, size = None) * self.fieldSize[0] #set x
            xy[:,1] = 0.5*self.fieldSize[1] #set y 
        return xy
            
    #  main for updating dots (also here is where the dot in circle are updated)
    def _update_OutOfBoundXY(self, outofbounds):
        nOutOfBounds = outofbounds.sum()
        allDir = self._dotsDir[outofbounds]
        newDots = np.zeros((nOutOfBounds, 2))
        if self.fieldShape=='sqr':
            for i in range(nOutOfBounds):
                currDir = allDir[i]%_2pi
                modAngle = currDir % _piOver2
                side = currDir//_piOver2
                oddsInFirstSide = 1/(1+np.tan(modAngle))
                isIn2ndSide = np.random.rand()>=oddsInFirstSide
                sideEnter = (side+isIn2ndSide) % 4; #which of 4 sides of the square a new dot enters
                newDots[i,:] = self.getRandPosInSquareSide(sideEnter)
            return newDots
        elif self.fieldShape=='circle':
            for i in range(nOutOfBounds):
                currDir = allDir[i]%_2pi
                entryAngle = currDir+np.pi
                ShiftFromEntryAngleOnEdge = np.arcsin(np.random.uniform(-1, 1, size = None))
                angleEntryOnCircEdge = entryAngle+ShiftFromEntryAngleOnEdge
                newDots[i,0] = np.cos(angleEntryOnCircEdge)*0.5*self.fieldSize[0]
                newDots[i,1] = np.sin(angleEntryOnCircEdge)*0.5*self.fieldSize[1]
            return newDots

    def refreshDots(self):
        """Callable user function to choose a new set of dots."""
        # We first calculate the number of dots from the density
        # this code take into account if a circle or a square is used as a shape
        if self.fieldShape in (None, 'square', 'sqr'):
            field_area = self.fieldSize[0] * self.fieldSize[1]
        else:
            field_area = np.pi * (self.fieldSize[0] / 2) ** 2

        # We derive the number of dots from the field_area
        self.nDots = int(np.ceil(self.dotDensity * field_area))

        # We then calculate the number of dots from the field size
        if self.dotSize is None:
            self.dotSize = 3.0
        self.vertices = self._verticesBase = self._dotsXY = self._newDotsXY(self.nDots)

        # all dots have the same speed
        if self.nDots != len(self._dotsSpeed):
            self._dotsSpeed = np.ones(self.nDots, dtype=float) * self.speed
            self._dotsLife = np.abs(self.dotLife) * np.random.rand(self.nDots)
            self._dotsDir = np.random.rand(self.nDots) * _2pi

        # Don't allocate another array if the new number of dots is equal to
        # the last.
        if self.nDots != len(self._deadDots):
            self._deadDots = np.zeros(self.nDots, dtype=bool)

    def _update_dotsXY(self):
        """The user shouldn't call this - its gets done within draw().
        """
        # Find dead dots, update positions, get new positions for
        # dead and out-of-bounds
        # renew dead dots
        if self.dotLife > 0:  # if less than zero ignore it
            # decrement. Then dots to be reborn will be negative
            self._dotsLife -= 1
            self._deadDots[:] = (self._dotsLife <= 0)
            self._dotsLife[self._deadDots] = self.dotLife
        else:
            self._deadDots[:] = False

        # update XY based on speed and dir
        # NB self._dotsDir is in radians, but self.dir is in degs
        # update which are the noise/signal dots
        if self.signalDots == 'different':
            #  **up to version 1.70.00 this was the other way around,
            # not in keeping with Scase et al**
            # noise and signal dots change identity constantly
            np.random.shuffle(self._dotsDir)
            # and then update _signalDots from that
            self._signalDots = (self._dotsDir == (self.dir * _piOver180))

        # update the locations of signal and noise; 0 radians=East!
        reshape = np.reshape
        if self.noiseDots == 'walk':
            # noise dots are ~self._signalDots
            sig = np.random.rand(np.sum(~self._signalDots))
            self._dotsDir[~self._signalDots] = sig * _2pi
            # then update all positions from dir*speed
            cosDots = reshape(np.cos(self._dotsDir), (self.nDots,))
            sinDots = reshape(np.sin(self._dotsDir), (self.nDots,))
            self._verticesBase[:, 0] += self.speed * cosDots
            self._verticesBase[:, 1] += self.speed * sinDots
        elif self.noiseDots == 'direction':
            # simply use the stored directions to update position
            cosDots = reshape(np.cos(self._dotsDir), (self.nDots,))
            sinDots = reshape(np.sin(self._dotsDir), (self.nDots,))
            self._verticesBase[:, 0] += self.speed * cosDots
            self._verticesBase[:, 1] += self.speed * sinDots
        elif self.noiseDots == 'position':
            # update signal dots
            sd = self._signalDots
            sdSum = self._signalDots.sum()
            cosDots = reshape(np.cos(self._dotsDir[sd]), (sdSum,))
            sinDots = reshape(np.sin(self._dotsDir[sd]), (sdSum,))
            self._verticesBase[sd, 0] += self.speed * cosDots
            self._verticesBase[sd, 1] += self.speed * sinDots
            # update noise dots
            self._deadDots[:] = self._deadDots + (~self._signalDots)

        # handle boundaries of the field
        if self.fieldShape in (None, 'square', 'sqr'):
            out0 = (np.abs(self._verticesBase[:, 0]) > .5 * self.fieldSize[0])
            out1 = (np.abs(self._verticesBase[:, 1]) > .5 * self.fieldSize[1])
            outofbounds = out0 + out1
        else:
            # transform to a normalised circle (radius = 1 all around)
            # then to polar coords to check
            # the normalised XY position (where radius should be < 1)
            normXY = self._verticesBase / .5 / self.fieldSize
            # add out-of-bounds to those that need replacing
            outofbounds = np.hypot(normXY[:, 0], normXY[:, 1]) > 1.
        # update any dead dots
        nDead = self._deadDots.sum()
        if nDead:
            self._verticesBase[self._deadDots, :] = self._newDotsXY(nDead)

        # Reposition any dots that have gone out of bounds. Net effect is to
        # place dot one step inside the boundary on the other side of the
        # aperture.
        nOutOfBounds = outofbounds.sum()
        if nOutOfBounds:
            # self._verticesBase[outofbounds, :] = self._newDotsXY(nOutOfBounds)
            self._verticesBase[outofbounds, :] = self._update_OutOfBoundXY(outofbounds)


        self.vertices = self._verticesBase / self.fieldSize

        # update the pixel XY coordinates in pixels (using _BaseVisual class)
        self._updateVertices()

# The functions below are necessary to generate the dots in the circle with older version of psychopy
def _calculateMinEdges(lineWidth, threshold=180):
    """
    Calculate how many points are needed in an equilateral polygon for the gap between line rects to be < 1px and
    for corner angles to exceed a threshold.

    In other words, how many edges does a polygon need to have to appear smooth?

    lineWidth : int, float, np.ndarray
        Width of the line in pixels

    threshold : int
        Maximum angle (degrees) for corners of the polygon, useful for drawing a circle. Supply 180 for no maximum
        angle.
    """
    # sin(theta) = opp / hyp, we want opp to be 1/8 (meaning gap between rects is 1/4px, 1/2px in retina)
    opp = 1.0/8
    hyp = lineWidth / 2
    thetaR = np.arcsin(opp / hyp)
    theta = np.degrees(thetaR)    
    # If theta is below threshold, use threshold instead
    theta = min(theta, threshold / 2)
    # Angles in a shape add up to 360, so theta is 360/2n, solve for n
    return int((360 / theta) / 2)

def _calcEquilateralVertices(edges, radius=0.5):
    """
    Get vertices for an equilateral shape with a given number of sides, will assume radius is 0.5 (relative) but
    can be manually specified
    """
    d = np.pi * 2.0 / edges
    vertices = np.asarray(
        [np.asarray((np.sin(e * d), np.cos(e * d))) * radius
            for e in range(int(round(edges)))])
    return vertices


#,nDotsSqr,nDotsirc,
# create a table with all the desired trials combination
def set_trials_sqr(n_reps, direction_vec,InnerdirVec, coherence_level,DotSpeed,DotSize,FieldSizeSqr,FieldSizeCirc,dotDensitysSqr,dotDensitysirc,shuff=True):
    
    opacitycirc=[0,1]
    opacitysqr = [0,1]
    
    combinations_with_fixed_opacity = list(itertools.product(direction_vec, InnerdirVec,[1], [1], coherence_level, DotSize, 
                                                             DotSpeed,FieldSizeSqr,FieldSizeCirc,dotDensitysSqr,dotDensitysirc,))
    my_list=combinations_with_fixed_opacity
    
    modified_list = []
    
    
  
    for tup in my_list:
            if tup[0] == tup[1]:
                new_tuple = (tup[0], tup[1], tup[2],0) + tup[4:]
                
            else:
                new_tuple = tup
    modified_list.append(new_tuple)
    


    combinations_with_variable_opacity = list(itertools.product(direction_vec, InnerdirVec, opacitycirc, opacitysqr,
                                                                coherence_level, DotSize, DotSpeed,FieldSizeSqr,FieldSizeCirc,dotDensitysSqr,dotDensitysirc,))
    
    filtered_combinations_with_variable_opacity = []
    for combo in combinations_with_variable_opacity:
        
        if combo[DIR_CIRC_IND_TRIAL] != combo[DIR_SQR_IND_TRIAL]:  # Skip combinations where directionvecinside == directionvecoutside
            continue
        if combo[OPACITY_SQR_IND_TRIAL] == 0 and combo[OPACITY_CIRC_IND_TRIAL] == 0:  # Skip combinations where both opacitycirc and opacitysqr are 0
            continue
        if combo[OPACITY_SQR_IND_TRIAL] == 1 and combo[OPACITY_CIRC_IND_TRIAL] == 1:  # Skip combinations where both opacitycirc and opacitysqr are 0
            continue
        filtered_combinations_with_variable_opacity.append(combo)

    # Combine both sets of combinations
    #all_combinations = combinations_with_fixed_opacity + filtered_combinations_with_variable_opacity
    all_combinations = modified_list+filtered_combinations_with_variable_opacity
    all_trials = []
    for combination in all_combinations:
        all_trials.extend([combination] * n_reps)
                
    random.shuffle(all_trials)     
    return(all_trials)




#NDOTS_CIRC_IND_TRIAL
    #NDOTS_SQR_IND_TRIAL
# set a similar order for the trials the dorCirc and dotSqr
def set_new_trial_orders (alltrial, circTable, sqrTable):

    circIndices = [DIR_CIRC_IND_TRIAL,OPACITY_CIRC_IND_TRIAL,COHERENCE_IND_TRIAL,DOT_SIZE_IND_TRIAL,SPEED_IND_TRIAL,FIELD_SIZE_CIRC_IND_TRIAL,DOTSDensity_CIRC_IND_TRIAL]
    sqrIndices = [DIR_SQR_IND_TRIAL,OPACITY_SQR_IND_TRIAL,COHERENCE_IND_TRIAL,DOT_SIZE_IND_TRIAL,SPEED_IND_TRIAL,FIELD_SIZE_SQR_IND_TRIAL,DOTSDensity_SQR_IND_TRIAL]

    sweepOrderCirc = list(range(0, len(alltrial)))
    sweepOrderSqr = list(range(0, len(alltrial)))
    sweepOrderCirc_all=[]
    sweepOrderSqr_all=[]
    for trial in alltrial:
        
        selected_elementsCircle = tuple([trial[i] for i in circIndices ])
        selected_elements1Sqr = tuple([trial[i] for i in sqrIndices ])
        #look for the index of the list to match the order between the circle and sqr

        sweepOrderCirc=([i for i, tpl in enumerate(circTable) if tpl == (selected_elementsCircle)])
        print('trial' ,trial)
        print('selected' ,selected_elementsCircle)
        print(sweepOrderCirc)
       
        sweepOrderCirc_all.append(int(sweepOrderCirc[0]))        

        sweepOrderSqr=([i for i, tpl in enumerate(sqrTable) if tpl == (selected_elements1Sqr)])
        sweepOrderSqr_all.append(int(sweepOrderSqr[0]))   
        

    return(sweepOrderCirc_all,sweepOrderSqr_all)

#field_size

# set dotSqr
def init_dot_stim(window,num_reps,field_size, n_dots,coher,field_shape, stim_name,sweep_params_exp_sqr):
#{ 'Dir': (dirVec, 0), 'FieldCoherence': (coherence_vec, 1),'dotSize': (dotsize_vec,2)}
    dot_stimuli = Stimulus(FixedDotStim(window, nDots=int(n_dots), 
                                        fieldPos=(0,0), units='deg',
                                        fieldSize=(field_size[0], field_size[0]), 
                                        fieldShape=field_shape,
                                        dir=90, coherence =coher,
                                        dotLife=-1, speed=0.01,  
                                        rgb=None, color=(0,0,0), 
                                        colorSpace='rgb255', opacity=1.0,
                                        contrast=1.0, depth=0, element=None, 
                                        signalDots='same', 
                                        noiseDots='direction', name='', 
                                        autoLog=True),
                            
                            sweep_params = sweep_params_exp_sqr,
                            sweep_length       = sweep_length_num,
                            start_time          = start_time_nun,
                            blank_length        = blank_length_num, 
                            blank_sweeps        = blank_sweeps_num,
                            runs                = num_reps,
                            shuffle             = True,
                           
                            )
    dot_stimuli.stim_path = r"C:\\not_a_stim_script\\"+stim_name+".stim"

    return dot_stimuli

# set dotCirc
def init_dot_stim_circ(window,num_reps,field_size, n_dots,coher,field_shape, stim_name,sweep_params_exp_circ):
    dot_stimuli_circ = Stimulus(FixedDotStim(window, nDots=int(n_dots), 
                                         fieldPos=(0,8), units='deg',
                                         fieldSize=(field_size[0], field_size[0]),
                                         fieldShape=field_shape, 
                                         dir=90, coherence =coher,
                                         dotLife=-1, speed=0.01,
                                         rgb=None, color=(0,0,0), 
                                         colorSpace='rgb255', opacity=1.0,
                                         contrast=1.0, depth=0, element=None, 
                                         signalDots='same', 
                                         noiseDots='direction', name='', 
                                         autoLog=True),
                            sweep_params        = sweep_params_exp_circ,
                            sweep_length        = sweep_length_num,
                            start_time          = start_time_nun,
                            blank_length        = blank_length_num,
                            blank_sweeps        = blank_sweeps_num,
                            runs                = num_reps,
                            shuffle             = True,
                            )

    dot_stimuli_circ.stim_path = r"C:\\not_a_stim_script\\"+stim_name+".stim"
   
    return dot_stimuli_circ


# set constant circ 
def init_circle(window, r=20, repetitions=10, sweep_param = {}):
    circle = visual.ShapeStim(
        win, vertices= _calcEquilateralVertices(_calculateMinEdges(1.5, threshold=5)),
        pos=(0,8), size=(r*2, r*2), units="deg",
        interpolate=True, fillColor="white",
        autoDraw=False, lineWidth=0, lineColor="white")
    circle_in_stim = Stimulus(circle, 
             sweep_params = sweep_param, 
             sweep_length = sweep_length_num, 
             start_time = start_time_nun, 
             blank_length = blank_length_num, 
             blank_sweeps = blank_sweeps_num, 
             runs = repetitions, 
             shuffle = False, 
             save_sweep_table = True)
    
    return circle_in_stim

# create the stimulus accroding to the desired  params and create a list of stimuli
def callAccParameter(win
                     ,num_reps_ex
                     ,fieldSize_Circle
                     ,fieldSize_Square
                     ,dotDensity_circ
                     ,dotDensity_sqr
                     ,sweep_params_circ
                     ,sweep_params_sqr
                     
                     ):
        
        alltrial = []        
        list_stimuli = []  
                
        rdkCircle = init_dot_stim_circ(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Circle
                    ,n_dots=1
                    ,coher= 1
                    ,field_shape='circle'
                    ,stim_name='rdkCircle'
                    ,sweep_params_exp_circ=sweep_params_circ
                    )
        
        # sweep order and sweep table    
        circle_sweepTable = rdkCircle.sweep_table
        
        rdkSqr = init_dot_stim(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Square
                    ,n_dots=1
                    ,coher= 1
                    ,field_shape='sqr'
                    ,stim_name='rdkSqr'
                    ,sweep_params_exp_sqr=sweep_params_sqr
                    )
    
        sqr_sweepTable = rdkSqr.sweep_table  

        # num_reps_ex/2
        if round(num_reps_ex/2)== 0 :
           n_reps_ex = 1
        else:
           n_reps_ex = round(n_reps_ex/2)
           
        alltrial= set_trials_sqr(
            n_reps=num_reps_ex
            ,direction_vec=sweep_params_sqr['Dir'][0]
            ,InnerdirVec=sweep_params_circ['Dir'][0]
            ,coherence_level=sweep_params_sqr['FieldCoherence'][0]
            ,DotSize=sweep_params_sqr['dotSize'][0]
            ,DotSpeed= sweep_params_sqr['speed'][0]
            ,FieldSizeSqr= sweep_params_sqr['fieldSize'][0]
            ,FieldSizeCirc= sweep_params_circ['fieldSize'][0]
            ,dotDensitysSqr= sweep_params_sqr['dotDensity'][0]
            ,dotDensitysirc= sweep_params_circ['dotDensity'][0]
            ,shuff=True
            ) 
        
        sweepOrderCirc,sweepOrderSqr = set_new_trial_orders(
            alltrial
            ,circle_sweepTable
            ,sqr_sweepTable
            )
 
        # sweep order and sweep table    
        rdkSqr.sweep_order = sweepOrderSqr
        rdkCircle.sweep_order = sweepOrderCirc

        list_diameters = np.array([[(float(indiv_sweep[FIELD_SIZE_IND])),(float(indiv_sweep[FIELD_SIZE_IND]))] for indiv_sweep in rdkCircle.sweep_table])
        list_opacity = np.array([(float(indiv_sweep[OPACITY_IND])) for indiv_sweep in rdkCircle.sweep_table])

        sweep_params_background_circle = { 'Size': (list_diameters[np.array(rdkCircle.sweep_order)], 0),
                                           'Opacity': ([1.0], 1)}
    
        circle = init_circle(win
                ,r=10
                ,repetitions=1
                ,sweep_param = sweep_params_background_circle
                )
        

        # We merge list_diameters and list_opacity into a single list where each element is a tuple of the form (diameters, opacity)
        list_circle = zip(list_diameters[np.array(rdkCircle.sweep_order)].tolist(), list_opacity[np.array(rdkCircle.sweep_order)].tolist())
        circle.sweep_table = list_circle

        # sweep order is the order in list_circle
        circle.sweep_order = np.arange(len(list_circle))

        list_stimuli.append(rdkSqr)
        list_stimuli.append(circle)
        list_stimuli.append(rdkCircle)
        
        both_stimuli = StimulusArray(list_stimuli, 
                            sweep_length = sweep_length_num,
                            blank_length = blank_length_num   )
        
        return both_stimuli    
    
def  createBlock(blockParameter
                    ,blockParameterName
                    ,blockParameterInd
                    ,sweep_params_int_sqr
                    ,sweep_params_int_circ
                    ,fieldSizeCircle
                    ,fieldSizeSquare
                    ,dotDensitysCircle
                    ,dotDensitysSquare
                    ,num_reps
                    ):
        
        nDotsPer1SqrArea = [0.3]
        fieldSizeCircle = [10]
        fieldSizeSquare = [100]
        dotDensitysCircle = nDotsPer1SqrArea
        dotDensitysSquare = nDotsPer1SqrArea

        num_reps = 1
        opacity_vec = [0,1]
        dirVecCirc = [0,180]
        dirVecSqr =[0,180,90]
        coherence_vec = [1]
        dotsize_vec = [25]
        dotspeed_vec = [0.1]
            
        sweep_params_block_circ = { 'Dir': (dirVecCirc, DIR_IND)
                             ,'opacity': (opacity_vec,OPACITY_IND)
                             ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                             ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                             ,'speed':(dotspeed_vec,SPEED_IND)
                             ,'fieldSize':(fieldSizeCircle,FIELD_SIZE_IND)
                             ,'dotDensity':(dotDensitysCircle,NDOTS_IND)
                             }
     
        sweep_params_block_sqr = { 'Dir': (dirVecSqr, DIR_IND)
                            ,'opacity': (opacity_vec,OPACITY_IND)
                            ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                            ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                            ,'speed':(dotspeed_vec,SPEED_IND)
                            ,'fieldSize':(fieldSizeSquare,FIELD_SIZE_IND)
                            ,'dotDensity':(dotDensitysSquare,NDOTS_IND)
                            } 
    
    
    
        sweep_params_block_circ[blockParameterName] =( blockParameter,blockParameterInd)
        sweep_params_block_sqr[blockParameterName] =( blockParameter,blockParameterInd)

        both_block= callAccParameter(win
                                     ,num_reps_ex=num_reps
                                     ,fieldSize_Circle=fieldSizeCircle
                                     ,fieldSize_Square=fieldSizeSquare
                                     ,dotDensity_circ=dotDensitysCircle
                                     ,dotDensity_sqr=dotDensitysSquare
                                     ,sweep_params_circ=sweep_params_block_circ
                                     ,sweep_params_sqr=sweep_params_block_sqr
                                     )
        return both_block
        
        
    
    

    
def main(): 
    nDotsPer1SqrArea = [0.3]

    fieldSizeCircle = [15]
    fieldSizeSquare = [100]
    dotDensitysCircle = nDotsPer1SqrArea
    dotDensitysSquare = nDotsPer1SqrArea

    
    num_reps = 1
    opacity_vec = [0,1]
    dirVecCirc = [0,180]
    dirVecSqr =[0,180,90]
    coherence_vec = [1]
    dotsize_vec = [25]
    dotspeed_vec = [0.1]
    
    
    sweep_params_int_circ = { 'Dir': (dirVecCirc, DIR_IND)
                             ,'opacity': (opacity_vec,OPACITY_IND)
                             ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                             ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                             ,'speed':(dotspeed_vec,SPEED_IND)
                             ,'fieldSize':(fieldSizeCircle,FIELD_SIZE_IND)
                             ,'dotDensity':(dotDensitysCircle,NDOTS_IND)
                             }
     
    sweep_params_int_sqr = { 'Dir': (dirVecSqr, DIR_IND)
                            ,'opacity': (opacity_vec,OPACITY_IND)
                            ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                            ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                            ,'speed':(dotspeed_vec,SPEED_IND)
                            ,'fieldSize':(fieldSizeSquare,FIELD_SIZE_IND)
                            ,'dotDensity':(dotDensitysSquare,NDOTS_IND)
                            } 
    
    

   
    both_stimuli_polarity= callAccParameter(win
                                             ,num_reps_ex=NUM_REPS
                                             ,fieldSize_Circle=fieldSizeCircle
                                             ,fieldSize_Square=fieldSizeSquare
                                             ,dotDensity_circ=dotDensitysCircle
                                             ,dotDensity_sqr=dotDensitysSquare
                                             ,sweep_params_circ=sweep_params_int_circ
                                             ,sweep_params_sqr=sweep_params_int_sqr
                                             )
    
    

    
    pre_blank = 0
    post_blank = 0
    ss  = SweepStim(win
                    ,stimuli = [both_stimuli_polarity]#need to be replaced with SessionA_stimuli
                    ,pre_blank_sec = pre_blank
                    ,post_blank_sec  = post_blank                 
                    ,params = {}  # will be set by MPE to work on the rig
                    )
    
    # add in foraging so we can track wheel, potentially give rewards, etc
    f = Foraging(window = win,
                auto_update = False,
                params      = {}
                )
    
    ss.add_item(f, "foraging")
    
    # run it
    ss.run()


if __name__ == "__main__":
    main()
