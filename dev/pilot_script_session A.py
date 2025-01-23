
import camstim
from psychopy import visual, event, core, monitors
import numpy as np
import random
from camstim import SweepStim,  Foraging
from camstim.sweepstim import StimulusArray
from camstim.sweepstim import Stimulus
import itertools

from camstim import Window, Warp

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
        warp=Warp.Spherical
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

sweep_length_num       = 0.5
start_time_nun          = 0
blank_length_num        = 0.5 
blank_sweeps_num        = 0


# The following fixes the issue with the dots not being created on the edges of the screen. 
class FixedDotStim(visual.DotStim):
    def __init__(self, *args, **kwargs):
        nDots = kwargs.get('nDots')   
        self._deadDots = np.zeros(nDots, dtype=bool)
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
        self.fieldSize = (fieldSize,fieldSize)
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
        self.vertices = self._verticesBase = self._dotsXY = self._newDotsXY(self.nDots)

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

# create a table with all the desired trials combination
def set_trials_sqr(n_reps, direction_vec,InnerdirVec, coherence_level,DotSpeed,DotSize,shuff=True):
    
    opacitycirc=[0,1]
    opacitysqr = [0,1]
    
    combinations_with_fixed_opacity = list(itertools.product(direction_vec, InnerdirVec,[1], [1], coherence_level, DotSize, DotSpeed))

    combinations_with_variable_opacity = list(itertools.product(direction_vec, InnerdirVec, opacitycirc, opacitysqr, coherence_level, DotSize, DotSpeed))
    
    filtered_combinations_with_variable_opacity = []
    for combo in combinations_with_variable_opacity:
        if combo[0] == combo[1]:  # Skip combinations where directionvecinside == directionvecoutside
            continue
        if combo[5] == 0 and combo[6] == 0:  # Skip combinations where both opacitycirc and opacitysqr are 0
            continue
        filtered_combinations_with_variable_opacity.append(combo)

    # Combine both sets of combinations
    all_combinations = combinations_with_fixed_opacity + filtered_combinations_with_variable_opacity
    
    all_trials = []
    for combination in all_combinations:
        all_trials.extend([combination] * n_reps)
                
    random.shuffle(all_trials)     
    return(all_trials)

# set a similar order for the trials the dorCirc and dotSqr
def set_new_trial_orders (alltrial, circTable, sqrTable):
   # paramertLength = [len(alltrial[0])]
    #circIndices = list(range(1, len(alltrial[0])-1))+ paramertLength
    circIndices = [1]+[2]+ list(range(4, len(alltrial[0])))
    sqrIndices = [0] + list(range(3, len(alltrial[0])))

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
def init_dot_stim(window,num_reps, field_size,ndots,field_shape, stim_name,sweep_params_exp_sqr):
#{ 'Dir': (dirVec, 0), 'FieldCoherence': (coherence_vec, 1),'dotSize': (dotsize_vec,2)}
    dot_stimuli = Stimulus(FixedDotStim(window, nDots=int(ndots), 
                                        fieldPos=(0,0), 
                                        fieldSize=(field_size,field_size), 
                                        fieldShape=field_shape, 
                                        dotLife=-1, #speed=0.01,  
                                        rgb=None, color=(255,255,255), 
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
def init_dot_stim_circ(window,num_reps,field_size,ndots, field_shape, stim_name,sweep_params_exp_circ):

    dot_stimuli_circ = Stimulus(FixedDotStim(window, nDots=int(ndots), 
                                         fieldPos=(0,0), 
                                         fieldSize=(field_size,field_size), 
                                         fieldShape=field_shape, 
                                         dotLife=-1, 
                                         rgb=None, color=(255,255,255), 
                                         colorSpace='rgb255', 
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
def init_circle(window, r=128, repetitions=10):
    circle = visual.ShapeStim(
        win, vertices= _calcEquilateralVertices(_calculateMinEdges(1.5, threshold=5)),
        pos=(0.5, 0.5), size=(r*2, r*2), units="pix",
        fillColor="gray",  interpolate=True,
        autoDraw=False, lineWidth=0, lineColor="gray")
    circle_in_stim = Stimulus(circle, 
             sweep_params = {}, 
             sweep_length = sweep_length_num, 
             start_time = start_time_nun, 
             blank_length = blank_length_num, 
             blank_sweeps = blank_sweeps_num, 
             runs = repetitions, 
             shuffle = True, 
             save_sweep_table = True)
    
    return circle_in_stim

# create the stimulus accroding to the desired  params and create a list of stimuli
def callAccParameter(win
                     ,num_reps_ex
                     ,fieldSize_Circle
                     ,fieldSize_Square
                     ,ndots_circ
                     ,ndots_sqr
                     ,sweep_params_circ
                     ,sweep_params_sqr
                     ):
        
        alltrial = []        
        list_stimuli = []  
                
        rdkCircle = init_dot_stim_circ(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Circle
                    ,ndots=ndots_circ
                    ,field_shape='circle'
                    ,stim_name='rdkCircle'
                    ,sweep_params_exp_circ=sweep_params_circ
                    )
        
        # sweep order and sweep table    
        circle_sweepTable = rdkCircle.sweep_table
        
        rdkSqr = init_dot_stim(win
                    ,num_reps_ex
                    ,field_size=fieldSize_Square
                    ,ndots=ndots_sqr
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
            ,InnerdirVec=sweep_params_sqr['Dir'][0]
            ,coherence_level=sweep_params_sqr['FieldCoherence'][0]
            ,DotSize=sweep_params_sqr['dotSize'][0]
            ,DotSpeed= sweep_params_sqr['speed'][0]
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

        nb_sweeps = len(rdkSqr.sweep_order)
        circle = init_circle(win
                ,r=128
                ,repetitions=nb_sweeps*num_reps_ex
                )
        
        list_stimuli.append(rdkSqr)
        list_stimuli.append(circle)
        list_stimuli.append(rdkCircle)
        
        both_stimuli = StimulusArray(list_stimuli, 
                            sweep_length = sweep_length_num,
                            blank_length = blank_length_num                            )
        
        return both_stimuli    
    
def main(): 
    nDotsPer1SqrArea = 200
    fieldSizeCircle = 0.5
    fieldSizeSquare = 2.1
    areaCircle = (fieldSizeCircle/2)**2*np.pi
    areaSquare = fieldSizeSquare**2
    nDotsCircle = round(areaCircle*nDotsPer1SqrArea)
    nDotsSquare = round(areaSquare*nDotsPer1SqrArea)   
    
    num_reps = 1
    #dirVec=[0, 45 ,90, 135, 180 ,225, 270 ,315]
    opacity_vec = [0,1]
    dirVecCirc = [0,180,90]
    dirVecSqr =[0,180,90]
    coherence_vec = [1]
    dotsize_vec = [12]
    dotspeed_vec = [0.01]
    
    sweep_params_exp_circ = { 'Dir': (dirVecCirc, 0)
                             ,'opacity': (opacity_vec,1)
                             ,'FieldCoherence': (coherence_vec, 2)
                             ,'dotSize': (dotsize_vec,3)
                             ,'speed':(dotspeed_vec,4)
                             }
     
    sweep_params_exp_sqr = { 'Dir': (dirVecSqr, 0)
                            ,'opacity': (opacity_vec,1)
                            ,'FieldCoherence': (coherence_vec, 2)
                            ,'dotSize': (dotsize_vec,3)
                            ,'speed':(dotspeed_vec,4)
                            } 

    # COHERENCE
    coherence_vec_exp = [1,0.5,0.3,0.6]
    sweep_params_exp_sqr['FieldCoherence'] = (coherence_vec_exp,2)
    sweep_params_exp_circ['FieldCoherence'] = (coherence_vec_exp,2)

    
    both_stimuli_coherence = callAccParameter(win
            ,num_reps_ex=num_reps
            ,fieldSize_Circle=fieldSizeCircle
            ,fieldSize_Square=fieldSizeSquare
            ,ndots_circ=nDotsCircle
            ,ndots_sqr=nDotsSquare
            ,sweep_params_circ=sweep_params_exp_circ
            ,sweep_params_sqr=sweep_params_exp_sqr
            )
                                           
    # DOT SPEED
    dotspeed_vec_exp = [0.009,0.01,0.026]  
    sweep_params_exp_sqr['speed'] = (dotspeed_vec_exp,4)
    sweep_params_exp_circ['speed'] = (dotspeed_vec_exp,4)

    # YAEL: ISN'T THIS NECESSARY?
    # We reset Coherence to default value
    sweep_params_exp_sqr['FieldCoherence'] = ([1],2)
    sweep_params_exp_circ['FieldCoherence'] = ([1],2)

    both_stimuli_speed = callAccParameter(win
            ,num_reps_ex=num_reps
            ,fieldSize_Circle=fieldSizeCircle
            ,fieldSize_Square=fieldSizeSquare
            ,ndots_circ=nDotsCircle
            ,ndots_sqr=nDotsSquare
            ,sweep_params_circ=sweep_params_exp_circ
            ,sweep_params_sqr=sweep_params_exp_sqr
            )
     
    # DOT Size  
    dotsize_vec_exp = [5,10,12,30]
    sweep_params_exp_sqr['dotSize'] =( dotsize_vec_exp,3)
    sweep_params_exp_circ['dotSize'] =( dotsize_vec_exp,3)
    # YAEL: ISN'T THIS NECESSARY?
    # We reset Speed to default value
    sweep_params_exp_sqr['speed'] = ([0.01],4)
    sweep_params_exp_circ['speed'] = ([0.01],4)

    both_stimuli_size = callAccParameter(win
            ,num_reps_ex=num_reps
            ,fieldSize_Circle=fieldSizeCircle
            ,fieldSize_Square=fieldSizeSquare
            ,ndots_circ=nDotsCircle
            ,ndots_sqr=nDotsSquare
            ,sweep_params_circ=sweep_params_exp_circ
            ,sweep_params_sqr=sweep_params_exp_sqr
            )
    
    SessionA=[]
    
    SessionA.append(both_stimuli_coherence)
    SessionA.append(both_stimuli_speed)
    SessionA.append(both_stimuli_size)
    
    # an attempt to create a session where you see first coherence than dot speed and than dot size
    pre_blank = 0
    post_blank = 0
    ss  = SweepStim(win
                    ,stimuli = [both_stimuli_size] #need to be replaced with SessionA_stimuli
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


