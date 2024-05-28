from psychopy import visual, event, core
import numpy as np
import random



win = visual.Window([1000,1000], fullscr=False,color="black", winType='pyglet', waitBlanking=False)
win.clearBuffer()

# some constants
_piOver2 = np.pi / 2.
_piOver180 = np.pi / 180.
_2pi = 2 * np.pi

# The following fixes the issue with the dots not being created on the edges of the screen. 
class FixedDotStim(visual.DotStim):
    def __init__(self, *args, **kwargs):
        nDots = kwargs.get('nDots')          
        self._deadDots = np.zeros(nDots, dtype=bool)
        super(FixedDotStim, self).__init__(*args, **kwargs)

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
      
#updating the position of the dots for the square aparatus according to the direction of the movement
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
        allDir = self._dotsDir[outofbounds];
        newDots = np.zeros((nOutOfBounds, 2))
        if self.fieldShape=='sqr':
            for i in range(nOutOfBounds):
                currDir = allDir[i]%_2pi;
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

       

def set_trials_sqr(n_reps, direction_vec, coherence_level,n_dotsPerTrail,shuff=True):
       """ creates vector of all motion directions """
       ncohernece = list(range(0, len(coherence_level)))
       nAngle = list(range(0, len(direction_vec)))
       nDots = list(range(0, len(n_dotsPerTrail)))

       listLength = (len(direction_vec)*len(direction_vec)*len(ncohernece)*len(nDots))*n_reps

       all_trials = [[0 for i in range(2)] for j in range(listLength)]

       countInd = 0

       for k in range(n_reps):
           for DirOutsideInd in nAngle:
               count = 0
               for DirInsideInd in nAngle:
                   for CoherInd in ncohernece:
                 
                       for NdotsInd in nDots:
                           new_column = [direction_vec[DirOutsideInd],direction_vec[DirInsideInd],coherence_level[CoherInd],n_dotsPerTrail[NdotsInd]]
                           all_trials[countInd] = new_column
             
                           count= count+1
                           countInd = countInd +1
                   
       random.shuffle(all_trials)           
       return(all_trials)

# The function below are necesasry to generate the dots in the circle with older version of psychopy
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


dirVec=[0, 45 ,90, 135, 180 ,225, 270 ,315] 
num_reps = 10
coherence_vec = [1,0.5,0]
ndotsPerTrail = [1,2,3]

alltrials = set_trials_sqr(n_reps=num_reps, direction_vec = dirVec ,
                            coherence_level=coherence_vec,n_dotsPerTrail = ndotsPerTrail)

for dot_info in zip(alltrials):
     
  trailInfo = dot_info[(0)]
  coherencePerTrial = trailInfo[2]
  n_dots  = trailInfo[3]

  dirSqr = trailInfo[0]
  dirCirc = trailInfo[1]
  
  nDotsPer1SqrArea = 200
  fieldSizeCircle = 0.5
  fieldSizeSquare = 2.1
  areaCircle = (fieldSizeCircle/2)**2*np.pi
  areaSquare = fieldSizeSquare**2
      
  if n_dots == 1:
                nDotsCircle = 1
                nDotsSquare = round(areaSquare*nDotsPer1SqrArea)   
  elif n_dots == 2:
                 nDotsSquare = 1
                 nDotsCircle = round(areaCircle*nDotsPer1SqrArea)
    
  elif n_dots == 3 :
                 nDotsCircle = round(areaCircle*nDotsPer1SqrArea)
                 nDotsSquare = round(areaSquare*nDotsPer1SqrArea)   
  r = 128
  circle = visual.ShapeStim(
        win, vertices= _calcEquilateralVertices(_calculateMinEdges(1.5, threshold=5)),
        pos=(0.5, 0.5), size=(r*2, r*2), units="pix",
        fillColor="black",  opacity=1, interpolate=True,
        autoDraw=False)
    
  rdkCircle = FixedDotStim(win, nDots=int(nDotsCircle), coherence=coherencePerTrial,
                        fieldPos=(0,0), 
                        fieldSize=(fieldSizeCircle,fieldSizeCircle), 
                        fieldShape='circle', dotSize=15.0, 
                        dotLife=-1, dir=dirCirc, speed=0.01, 
                        rgb=None, color=(255,255,255), 
                        colorSpace='rgb255', opacity=1,
                        contrast=1.0, depth=0, element=None, 
                        signalDots='same', 
                        noiseDots='direction', name='', 
                        autoLog=True)
  rdkSqr = FixedDotStim(win, nDots=int(nDotsSquare), coherence=coherencePerTrial, 
                        fieldPos=(0,0), 
                        fieldSize=(fieldSizeSquare,fieldSizeSquare), 
                        fieldShape='sqr', dotSize=15.0, 
                        dotLife=-1, dir=dirSqr, speed=0.01, 
                        rgb=None, color=(255,255,255), 
                        colorSpace='rgb255', opacity=1.0,
                        contrast=1.0, depth=0, element=None, 
                        signalDots='same', 
                        noiseDots='direction', name='', 
                        autoLog=True)
    
    
    
    
    
    
  stop = False
  timer = core.Clock()
  timer = core.CountdownTimer(3)
  
 # while stop == False: 
  while timer.getTime() > 0:
    
      
      if n_dots == 1:
               rdkSqr.draw()  
               circle.draw()
               win.flip()

      elif n_dots == 2:
              circle.draw()
              rdkCircle.draw()
              win.flip()

      elif n_dots == 3 :
                
             rdkSqr.draw()  
             circle.draw()
             rdkCircle.draw()
             win.flip()

     
    
       
    
  if event.getKeys("a"):
            win.close()
            stop = True