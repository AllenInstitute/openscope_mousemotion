
import camstim
from psychopy import visual, monitors
import numpy as np
import random
from camstim import SweepStim,  Foraging, MovieStim
from camstim.sweepstim import StimulusArray
from camstim.sweepstim import Stimulus
import itertools
from psychopy.tools.arraytools import val2array
from camstim import Window, Warp
import argparse
import logging 
import yaml
import os 
from psychopy import monitors, visual

# All CONSTANTS below are NOT to be changed
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
DOTSDENSITY_IND_TRIAL = 9
#DOTSDENSITY_SQR_IND_TRIAL = 9
#@DOTSDENSITY_CIRC_IND_TRIAL = 10
SWEEP_LENGTH_NUM       = 1
START_TIME_NUM          = 0
BLANK_LENGTH_NUM        = 0.5 
BLANK_SWEEPS_NUM        = 0
PIOVER2 = np.pi / 2.
PIOVER180 = np.pi / 180.
PI_2 = 2 * np.pi

# The following fixes the issue with the dots not being created on the edges of the screen. 
class FixedDotStim(visual.DotStim):
    def __init__(self, *args, **kwargs):
        nDots = kwargs.get('nDots')   
        self._deadDots = np.zeros(nDots, dtype=bool)
        # Initialize the dots density
        self.dotDensity = 1.0
        
        self.background_color = kwargs.pop('background_color', None)
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
            angle = np.random.uniform(0., PI_2, (nDots,))
            newDots = np.zeros((nDots, 2))
            newDots[:, 0] = length * np.cos(angle)
            newDots[:, 1] = length * np.sin(angle)
            newDots *= self.fieldSize * .5
        else:
            newDots = np.random.uniform(-0.5, 0.5, size = (nDots, 2)) * self.fieldSize
            
        return newDots
      
    # The function belows are needed to allow the Stimulus object to 
    # overwrite the parameters of the dots and to make sure new dots are displayed
    # on new sweeps. 
    def setdotSize(self, dotSize):
        self.dotSize = dotSize
        self.refreshDots()

    def setDir(self, Dir):
        self.dir = Dir
        self.refreshDots()

    def setopacity(self, opacity):
        self.opacity = opacity
        self.refreshDots()

    def setFieldCoherence(self, coherence):
        self.coherence = coherence
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
        
    def _selectWindow(self, win):
        """We modify this function to change background color of the window on the first draw."""
        # We first call the parent class method
        super(FixedDotStim, self)._selectWindow(win)
        
        if self.background_color is not None:
            if self.background_color != win.color:
                self.old_background_color = win.color
                win.color = self.background_color

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
                currDir = allDir[i]%PI_2
                modAngle = currDir % PIOVER2
                side = currDir//PIOVER2
                oddsInFirstSide = 1/(1+np.tan(modAngle))
                isIn2ndSide = np.random.rand()>=oddsInFirstSide
                sideEnter = (side+isIn2ndSide) % 4; #which of 4 sides of the square a new dot enters
                newDots[i,:] = self.getRandPosInSquareSide(sideEnter)
            return newDots
        elif self.fieldShape=='circle':
            for i in range(nOutOfBounds):
                currDir = allDir[i]%PI_2
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
        # We recreate all dots on every refreshs
        self._dotsSpeed = np.ones(self.nDots, dtype=float) * self.speed
        self._dotsLife = np.abs(self.dotLife) * np.random.rand(self.nDots)
        self._dotsDir = np.random.rand(self.nDots) * PI_2
        self._deadDots = np.zeros(self.nDots, dtype=bool)
        self.coherence = self.coherence
        
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
            self._signalDots = (self._dotsDir == (self.dir * PIOVER180))
        # update the locations of signal and noise; 0 radians=East!
        reshape = np.reshape
        if self.noiseDots == 'walk':
            # noise dots are ~self._signalDots
            sig = np.random.rand(np.sum(~self._signalDots))
            self._dotsDir[~self._signalDots] = sig * PI_2
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
#,nDotsSqr,nDotsirc,

# create a table with all the desired trials combination
def set_trials_sqr(n_reps, direction_vec,InnerdirVec, coherence_level,DotSpeed,DotSize,FieldSizeSqr,FieldSizeCirc,DotDensity,shuff=True):

    opacitycirc=[0,1]
    opacitysqr = [0,1]
    FieldSizeSqr1 = [FieldSizeSqr[0]]
    combinations_with_fixed_opacity = list(itertools.product(direction_vec, InnerdirVec,[1], [1], coherence_level, DotSize, 
                                                             DotSpeed,FieldSizeSqr1,FieldSizeCirc,DotDensity,))
    my_list=combinations_with_fixed_opacity
    
    modified_list = []
    
    if len(FieldSizeCirc)>1:
        
        modified_list = []
        seen_tuples = set()

        for tup in my_list:
            if (tup[0], tup[1]) in seen_tuples:
        # Skip processing if the tuple has already been modified
                continue
            if tup[0] == tup[1]:
                new_tuple = (tup[0], tup[1], tup[2], 0) + tup[4:]
                seen_tuples.add((tup[0], tup[1]))
                modified_list.append(new_tuple)
        # Skip further processing for this pair
                continue
            else:
                new_tuple = tup
                modified_list.append(new_tuple)
    else:
  
        for tup in my_list:
            if tup[0] == tup[1]:
                new_tuple = (tup[0], tup[1], tup[2],0) + tup[4:]     
            else:
                new_tuple = tup
            modified_list.append(new_tuple)
    

    combinations_with_variable_opacity_new = list(itertools.product([direction_vec[0]], InnerdirVec, [0], [1],
                                                                coherence_level, DotSize, DotSpeed,FieldSizeSqr1,FieldSizeCirc,DotDensity,)) 
    
    combinations_with_variable_opacity_90 = list(itertools.product([direction_vec[2]], [InnerdirVec[0]], [1], [0],
                                                                coherence_level, DotSize, DotSpeed,FieldSizeSqr1,FieldSizeCirc,DotDensity,)) 
    
    
    if len(FieldSizeCirc)>1:
        combinations_with_variable_opacity_90 = list(itertools.product([direction_vec[2]], [InnerdirVec[0]], [1], [0],
                                                                coherence_level, DotSize, DotSpeed,FieldSizeSqr1,[FieldSizeCirc[0]],DotDensity,)) 
    

    # Combine both sets of combinations
    #all_combinations = combinations_with_fixed_opacity + filtered_combinations_with_variable_opacity
    all_combinations = modified_list+combinations_with_variable_opacity_new+combinations_with_variable_opacity_90
    all_trials = []
    for combination in all_combinations:
        all_trials.extend([combination] * n_reps)
                
    random.shuffle(all_trials)     
    return(all_trials)

# set a similar order for the trials the dorCirc and dotSqr
def set_new_trial_orders (alltrial, circTable, sqrTable):
    circIndices = [DIR_CIRC_IND_TRIAL,OPACITY_CIRC_IND_TRIAL,COHERENCE_IND_TRIAL,DOT_SIZE_IND_TRIAL,SPEED_IND_TRIAL,FIELD_SIZE_CIRC_IND_TRIAL,DOTSDENSITY_IND_TRIAL]
    sqrIndices = [DIR_SQR_IND_TRIAL,OPACITY_SQR_IND_TRIAL,COHERENCE_IND_TRIAL,DOT_SIZE_IND_TRIAL,SPEED_IND_TRIAL,FIELD_SIZE_SQR_IND_TRIAL,DOTSDENSITY_IND_TRIAL]
    sweepOrderCirc = list(range(0, len(alltrial)))
    sweepOrderSqr = list(range(0, len(alltrial)))
    sweepOrderCirc_all=[]
    sweepOrderSqr_all=[]
    for trial in alltrial:
        
        selected_elementsCircle = tuple([trial[i] for i in circIndices ])
        selected_elements1Sqr = tuple([trial[i] for i in sqrIndices ])
        #look for the index of the list to match the order between the circle and sqr
        sweepOrderCirc=([i for i, tpl in enumerate(circTable) if tpl == (selected_elementsCircle)])
        sweepOrderCirc_all.append(int(sweepOrderCirc[0]))        
        sweepOrderSqr=([i for i, tpl in enumerate(sqrTable) if tpl == (selected_elements1Sqr)])
        sweepOrderSqr_all.append(int(sweepOrderSqr[0]))   
        
    return(sweepOrderCirc_all,sweepOrderSqr_all)

# set dotSqr
def init_dot_stim(window,
                  num_reps,
                  field_size, 
                  n_dots,
                  coher,
                  field_shape, 
                  stim_name,
                  sweep_params_exp_sqr, 
                  background_color=None, 
                  dot_color=(255,255,255)
                  ):
    dot_stimuli = Stimulus(FixedDotStim(window, nDots=int(n_dots), 
                                        fieldPos=(0,0), units='pix',
                                        fieldSize=(field_size[0], field_size[0]), 
                                        fieldShape=field_shape,
                                        dir=90, coherence =coher,
                                        dotLife=-1, speed=0.01,  
                                        rgb=None, color=dot_color, 
                                        colorSpace='rgb255', opacity=1.0,
                                        contrast=1.0, depth=0, element=None, 
                                        signalDots='same', 
                                        noiseDots='direction', name='', 
                                        autoLog=True,
                                        background_color=background_color
                                        ),
                            
                            sweep_params = sweep_params_exp_sqr,
                            sweep_length       = SWEEP_LENGTH_NUM,
                            start_time          = START_TIME_NUM,
                            blank_length        = BLANK_LENGTH_NUM, 
                            blank_sweeps        = BLANK_SWEEPS_NUM,
                            runs                = num_reps,
                            shuffle             = True,
                           
                            )
    dot_stimuli.stim_path = r"C:\\not_a_stim_script\\"+stim_name+".stim"
    return dot_stimuli

# set dotCirc
def init_dot_stim_circ(window,
                       num_reps,
                       field_size, 
                       n_dots,
                       coher,
                       field_shape, 
                       stim_name,
                       sweep_params_exp_circ, 
                       background_color=None, 
                       dot_color=(255,255,255),
                       vertical_pos = 8
                       ):
    dot_stimuli_circ = Stimulus(FixedDotStim(window, nDots=int(n_dots), 
                                         fieldPos=(0,vertical_pos), units='pix',
                                         fieldSize=(field_size[0], field_size[0]),
                                         fieldShape=field_shape, 
                                         dir=90, coherence =coher,
                                         dotLife=-1, speed=0.01,
                                         rgb=None, color=dot_color, 
                                         colorSpace='rgb255', opacity=1.0,
                                         contrast=1.0, depth=0, element=None, 
                                         signalDots='same', 
                                         noiseDots='direction', name='', 
                                         autoLog=True,
                                         background_color=background_color  
                                         ),
                            sweep_params        = sweep_params_exp_circ,
                            sweep_length        = SWEEP_LENGTH_NUM,
                            start_time          = START_TIME_NUM,
                            blank_length        = BLANK_LENGTH_NUM,
                            blank_sweeps        = BLANK_SWEEPS_NUM,
                            runs                = num_reps,
                            shuffle             = True,
                            )
    dot_stimuli_circ.stim_path = r"C:\\not_a_stim_script\\"+stim_name+".stim"
   
    return dot_stimuli_circ

# set constant circ 
def init_circle(win, r=20, repetitions=10, sweep_param = {}, color='black', vertical_pos = 8):
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
    circle = visual.ShapeStim(
        win, vertices= _calcEquilateralVertices(_calculateMinEdges(1.5, threshold=5)),
        pos=(0,vertical_pos), size=(r*2, r*2), units="pix",
        interpolate=True, fillColor=color,
        autoDraw=False, lineWidth=0, lineColor=color)
    
    circle_in_stim = Stimulus(circle, 
             sweep_params = sweep_param, 
             sweep_length = SWEEP_LENGTH_NUM, 
             start_time = START_TIME_NUM, 
             blank_length = BLANK_LENGTH_NUM, 
             blank_sweeps = BLANK_SWEEPS_NUM, 
             runs = repetitions, 
             shuffle = False, 
             save_sweep_table = True)
    
    return circle_in_stim

# create the stimulus accroding to the desired  params and create a list of stimuli
def callAccParameter(win, num_reps_ex
                     ,fieldSize_Circle
                     ,fieldSize_Square
                     ,sweep_params_circ
                     ,sweep_params_sqr
                     ,color_background
                     ,color_dots
                     ,vertical_pos
                     ):
        
        alltrial = []        
        list_stimuli = []  
                
        rdkCircle = init_dot_stim_circ(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Circle
                    ,n_dots= 1
                    ,coher= 1
                    ,field_shape='circle'
                    ,stim_name='rdkCircle'
                    ,sweep_params_exp_circ=sweep_params_circ
                    ,background_color=color_background
                    ,dot_color=color_dots
                    ,vertical_pos=vertical_pos
                    )
        
        # sweep order and sweep table    
        circle_sweepTable = rdkCircle.sweep_table
        
        rdkSqr = init_dot_stim(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Square
                    ,n_dots= 1
                    ,coher= 1
                    ,field_shape='sqr'
                    ,stim_name='rdkSqr'
                    ,sweep_params_exp_sqr=sweep_params_sqr
                    ,background_color=color_background
                    ,dot_color=color_dots
                    )
    
        sqr_sweepTable = rdkSqr.sweep_table  

           
        alltrial= set_trials_sqr(
            n_reps=num_reps_ex
            ,direction_vec=sweep_params_sqr['Dir'][0]
            ,InnerdirVec=sweep_params_circ['Dir'][0]
            ,coherence_level=sweep_params_sqr['FieldCoherence'][0]
            ,DotSize=sweep_params_sqr['dotSize'][0]
            ,DotSpeed= sweep_params_sqr['speed'][0]
            ,FieldSizeSqr= sweep_params_sqr['fieldSize'][0]
            ,FieldSizeCirc= sweep_params_circ['fieldSize'][0]
            ,DotDensity= sweep_params_sqr['dotDensity'][0]
            
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
                ,color=color_background
                ,vertical_pos=vertical_pos
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
                            sweep_length = SWEEP_LENGTH_NUM,
                            blank_length = BLANK_LENGTH_NUM   )
        
        return both_stimuli

def createBlock(win, blockParameterCircl, blockParameterSqr
                    ,blockParameterName
                    ,blockParameterInd
                    ,fieldSizeCircle
                    ,fieldSizeSquare
                    ,sweep_params_block_circ
                    ,sweep_params_block_sqr
                    ,num_reps
                    ,color_background 
                    ,color_dots
                    ,vertical_pos 
                    ):
    # We modify a subset of the parameters for the block
    if blockParameterName != 'None':
        sweep_params_block_circ[blockParameterName] =(blockParameterCircl,blockParameterInd)
        sweep_params_block_sqr[blockParameterName] =(blockParameterSqr,blockParameterInd)
    both_block = callAccParameter(win, num_reps_ex=num_reps
                                    ,fieldSize_Circle=fieldSizeCircle
                                    ,fieldSize_Square=fieldSizeSquare
                                    ,sweep_params_circ=sweep_params_block_circ
                                    ,sweep_params_sqr=sweep_params_block_sqr
                                    ,color_background=color_background
                                    ,color_dots=color_dots
                                    ,vertical_pos=vertical_pos
                                    )
    return both_block

def create_receptive_field_mapping(window, number_runs = 15):
    x = np.arange(-40,45,10)
    y = np.arange(-40,45,10)
    position = []
    for i in x:
        for j in y:
            position.append([i,j])

    stimulus = Stimulus(visual.GratingStim(window,
                        units='deg',
                        size=20,
                        mask="circle",
                        texRes=256,
                        sf=0.1,
                        ),
        sweep_params={
                'Pos':(position, 0),
                'Contrast': ([0.8], 4),
                'TF': ([4.0], 1),
                'SF': ([0.08], 2),
                'Ori': ([0,45,90, ], 3),
                },
        sweep_length=0.25,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=number_runs,
        shuffle=True,
        save_sweep_table=True,
        )
    stimulus.stim_path = r"C:\\not_a_stim_script\\receptive_field_block.stim"

    return stimulus

def create_gratingStim(window, number_runs = 15):
    stimulus_grating = Stimulus(visual.GratingStim(window,
                        pos=(0, 0),
                        units='deg',
                        size=(250, 250),
                        mask="None",
                        texRes=256,
                        sf=0.1,
                        ),
        sweep_params={
                'Contrast': ([0.8], 0),
                'TF': ([4.0], 1),
                'SF': ([0.08], 2),
                'Ori': (range(0, 360, 45), 3),
                },
        sweep_length=1,
        start_time=0.0,
        blank_length=0.5,
        blank_sweeps=0,
        runs=number_runs,
        shuffle=True,
        save_sweep_table=True,
        )
    stimulus_grating.stim_path = r"C:\\not_a_stim_script\\drifting_gratings_field_block.stim"
    return stimulus_grating

def create_homogeneous_background(window, duration, color):
    # Create an homogeneous background with a color
    # color is 1 for white and 0 for black
    stimulus = Stimulus(visual.GratingStim(window,
                        pos=(0, 0),
                        units='deg',
                        size=(250, 250),
                        mask="None",
                        texRes=256,
                        sf=0,
                        ),
        sweep_params={
                'Ori': ([color], 0),
                },
        sweep_length=duration,
        start_time=0.0,
        blank_length=0.0,
        blank_sweeps=0,
        runs=1,
        shuffle=True,
        save_sweep_table=True,
        )
    stimulus.stim_path = r"C:\\not_a_stim_script\\homogeneous_background.stim"

    return stimulus
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("mtrain")
    parser.add_argument("json_path", nargs="?", type=str, default="")
    args, _ = parser.parse_known_args() # <- this ensures that we ignore other arguments that might be needed by camstim
    
    # print args
    if args.json_path == "":
        logging.warning("No json path provided, using default parameters. THIS IS NOT THE EXPECTED BEHAVIOR FOR PRODUCTION RUNS")
        json_params = {}
    else:
        with open(args.json_path, 'r') as f:
            # we use the yaml package here because the json package loads as unicode, which prevents using the keys as parameters later
            json_params = yaml.load(f)
            logging.info("Loaded json parameters from mtrain")
            # end of mtrain part
    
    dist = 15.0
    wid = 52.0

    # mtrain should be providing : a path to a network folder or a local folder with the entire repo pulled
    vertical_pos = json_params.get('vertical_pos', 8)
    num_reps = json_params.get('num_reps', 1)
    dev_mode = json_params.get('dev_mode', True)
    inter_block_interval = json_params.get('inter_block_interval', 10)
    
    # We get the current script path 
    script_path = os.path.abspath(os.path.dirname(__file__))

    data_folder = json_params.get('data_folder', os.path.abspath(
        os.path.join(script_path, '..', "data")))

    nDotsPer1SqrArea = [0.0002]
    fieldSizeCircle_default = [196] # For varying do [5,20,40]
    fieldSizeSquare_default = [2000] # For varying do [100,100,100]
    dotDensity_default = nDotsPer1SqrArea
    dotDensitysCircle = nDotsPer1SqrArea
    dotDensitysSquare = nDotsPer1SqrArea
    opacity_vec = [0,1]
    dirVecCirc = [0,180]
    dirVecSqr =[0,180,90]
    coherence_vec = [1]
    dotsize_vec = [40] 
    dotspeed_vec = [5]

    if dev_mode:
        my_monitor = monitors.Monitor(name='Test')
        my_monitor.setSizePix((800,600))
        my_monitor.setWidth(wid)
        my_monitor.setDistance(dist)
        my_monitor.saveMon()
        win = Window(size=[800,600], # [1024,768],
            fullscr=True,
            screen=0,
            monitor= my_monitor,
            warp=Warp.Spherical,
            color= "gray",
            units = 'deg'
        )
    else: 
        win = Window(
            fullscr=True,
            screen=0,
            monitor='Gamma1.Luminance50',
            warp=Warp.Spherical,
        )
    
    # Below are the basic parameters for all blocks
    sweep_params_block_circ = { 'Dir': (dirVecCirc, DIR_IND)
                            ,'opacity': (opacity_vec,OPACITY_IND)
                            ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                            ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                            ,'speed':(dotspeed_vec,SPEED_IND)
                            ,'fieldSize':(fieldSizeCircle_default,FIELD_SIZE_IND)
                            ,'dotDensity':(dotDensity_default,NDOTS_IND)
                            }
    
    sweep_params_block_sqr = { 'Dir': (dirVecSqr, DIR_IND)
                        ,'opacity': (opacity_vec,OPACITY_IND)
                        ,'FieldCoherence': (coherence_vec, COHERENCE_IND)
                        ,'dotSize': (dotsize_vec,DOT_SIZE_IND)
                        ,'speed':(dotspeed_vec,SPEED_IND)
                        ,'fieldSize':(fieldSizeSquare_default,FIELD_SIZE_IND)
                        ,'dotDensity':(dotDensity_default,NDOTS_IND)
                        }
        
    color_background = 'black'
    color_dots = (255,255,255)

    # COHERENCE block 
    coherence_vec_exp = [1, 0.9, 0.75, 0.6, 0.4] 

    both_stimuli_coherence = createBlock(win, coherence_vec_exp,coherence_vec_exp,
                                         'FieldCoherence'
                                         ,COHERENCE_IND
                                         ,fieldSizeCircle_default
                                         ,fieldSizeSquare_default
                                         ,sweep_params_block_circ.copy()
                                         ,sweep_params_block_sqr.copy()
                                         ,num_reps
                                         ,color_background 
                                         ,color_dots
                                         ,vertical_pos
                                         )
    
    # DOT dot density
    nDotsPer1SqrArea_vec = [0.0001,0.0002,0.0003]
    dotDensitysCircle = nDotsPer1SqrArea
    dotDensitysSquare = nDotsPer1SqrArea
    both_stimuli_Dotdensity = createBlock(win, nDotsPer1SqrArea_vec,nDotsPer1SqrArea_vec,
                                            'dotDensity'
                                            ,DOTSDENSITY_IND_TRIAL
                                            ,fieldSizeCircle_default
                                            ,fieldSizeSquare_default
                                            ,sweep_params_block_circ.copy()
                                            ,sweep_params_block_sqr.copy()
                                            ,num_reps
                                            ,color_background
                                            ,color_dots
                                            ,vertical_pos
                                            )
    # DOT speed block
    dotspeed_vec_exp = [3,5,7]
    both_stimuli_speed = createBlock(win, dotspeed_vec_exp,dotspeed_vec_exp,
                                         'speed'
                                         ,SPEED_IND
                                         ,fieldSizeCircle_default
                                         ,fieldSizeSquare_default
                                         ,sweep_params_block_circ.copy()
                                         ,sweep_params_block_sqr.copy()
                                         ,num_reps
                                         ,color_background
                                         ,color_dots
                                         ,vertical_pos
                                         )
        
    # DOT fieldsize block
    fieldSizeCircle_exp = [146,196,298]
    fieldSizeSquare_exp  = [2000,2000,2000] # Both should be the same size
    both_stimuli_Fieldsize = createBlock(win, fieldSizeCircle_exp,fieldSizeSquare_exp,
                                            'fieldSize'
                                            ,FIELD_SIZE_IND
                                            ,fieldSizeCircle_default
                                            ,fieldSizeSquare_default
                                            ,sweep_params_block_circ.copy()
                                            ,sweep_params_block_sqr.copy()
                                            ,num_reps
                                            ,color_background
                                            ,color_dots
                                            ,vertical_pos
                                            )
        
    moviepath = os.path.join(data_folder, "sparse_noise_8x14.npy")

    lsn_stim = MovieStim(movie_path=moviepath,
                    window=win,
                    frame_length=0.25,
                    size=(1024,768), #(1260, 720),
                    start_time=0.0,
                    stop_time=None,
                    runs=1
                    )

    nb_runs_ephys_rf = 12
    ephys_rf_stim = create_receptive_field_mapping(win, number_runs=nb_runs_ephys_rf)
    drifting_grating_stim = create_gratingStim(win, number_runs=10)

    All_stim = []

    # Add LSN
    current_time = 0

    # this is to make the LSN block shorter for the test mode
    if num_reps == 1:
        length_lsn_seconds = 10
    else:
        length_lsn_seconds = 740
    lsn_stim.set_display_sequence([(current_time, current_time+length_lsn_seconds)])
    All_stim.append(lsn_stim)
    print("length_lsn_seconds: ",length_lsn_seconds)
    
    # Add RF code from ephys
    current_time = current_time+length_lsn_seconds+inter_block_interval
    if num_reps == 1:
        length_rf_seconds = 10
    else: 
        length_rf_seconds = 60*nb_runs_ephys_rf
    
    ephys_rf_stim.set_display_sequence([(current_time, current_time+length_rf_seconds)])
    All_stim.append(ephys_rf_stim)
    print("length_rf_seconds: ",length_rf_seconds)
    

    if num_reps == 1:
        delay_luminance = 10
    else:   
        delay_luminance = 120

    # Here we add 2 min long of delay to accomodate change in luminance
    current_time = current_time+length_rf_seconds+delay_luminance 
    background_homogeneous = create_homogeneous_background(win, duration=delay_luminance, color=0)
    background_homogeneous.set_display_sequence([(current_time, current_time+delay_luminance)])
    All_stim.append(background_homogeneous)
    print("length_delay_luminance_seconds: ",delay_luminance)

    # Add blockFieldSize        
    current_time = current_time+delay_luminance
    fps = both_stimuli_coherence.stimuli[0].fps
    length_fieldsize_frames = both_stimuli_Fieldsize.get_total_frames()
    length_fieldsize_seconds = float(length_fieldsize_frames) / float(fps)    
    blockFieldSize = [(current_time, current_time+length_fieldsize_seconds)] 
    both_stimuli_Fieldsize.set_display_sequence(blockFieldSize)
    All_stim.append(both_stimuli_Fieldsize)
    print("length_fieldsize_seconds: ",length_fieldsize_seconds)
    
    # Add blockCoherence
    current_time = current_time+length_fieldsize_seconds+inter_block_interval 
    length_coherence_frames = both_stimuli_coherence.get_total_frames()
    length_coherence_seconds = float(length_coherence_frames) / float(fps)
    blockCoherence = [(current_time, current_time+length_coherence_seconds)]    
    both_stimuli_coherence.set_display_sequence(blockCoherence)
    All_stim.append(both_stimuli_coherence)
    print("length_coherence_seconds: ",length_coherence_seconds)
    
    # Add blockDotdensity  
    current_time = current_time+length_coherence_seconds+inter_block_interval
    length_Dotdensity_frames = both_stimuli_Dotdensity.get_total_frames()
    length_Dotdensity_seconds = float(length_Dotdensity_frames) / float(fps)    
    blockDotdensity = [(current_time, current_time+length_Dotdensity_seconds)] 
    both_stimuli_Dotdensity.set_display_sequence(blockDotdensity)
    All_stim.append(both_stimuli_Dotdensity)
    print("length_fieldsize_seconds: ",length_Dotdensity_seconds)
    
    # Add blockSpeed 
    current_time = current_time+inter_block_interval+length_Dotdensity_seconds
    length_speed_frames = both_stimuli_speed.get_total_frames()
    length_speed_seconds = float(length_speed_frames) / float(fps)    
    blockSpeed = [(current_time, current_time+length_speed_seconds)]
    both_stimuli_speed.set_display_sequence(blockSpeed)
    All_stim.append(both_stimuli_speed)
    print("length_speed_seconds: ",length_speed_seconds)

    # Here we add 2 min long of delay to accomodate change in luminance
    current_time = current_time+length_speed_seconds+inter_block_interval
    background_homogeneous_2 = create_homogeneous_background(win, duration=delay_luminance, color=0.5)
    background_homogeneous_2.set_display_sequence([(current_time, current_time+delay_luminance)])
    All_stim.append(background_homogeneous_2)
    print("length_delay_luminance_seconds: ",delay_luminance)

    # drifting_grating_stim
    current_time = current_time+delay_luminance

    if num_reps == 1:
        length_drifting_grating_seconds = 10
    else:
        length_drifting_grating_seconds = 8*1.5*10

    drifting_grating_stim.set_display_sequence([(current_time, current_time+length_drifting_grating_seconds)])
    All_stim.append(drifting_grating_stim)
    print("length_drifting_grating_seconds: ",length_drifting_grating_seconds)

    pre_blank = 0
    post_blank = 0
    ss  = SweepStim(win
                    ,stimuli = All_stim
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
