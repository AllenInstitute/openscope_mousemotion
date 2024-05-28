from psychopy import visual, event, core
import numpy as np
import random



win = visual.Window([1000,1000], fullscr=False,color="black", winType='pyglet', waitBlanking=False)
win.clearBuffer()


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
    print(thetaR)
    print(threshold)

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

  #nDotsCircle = round(areaCircle*nDotsPer1SqrArea)
  #nDotsSquare = round(areaSquare*nDotsPer1SqrArea)   
      
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
    
  rdkCircle = visual.DotStim(win, nDots=int(nDotsCircle), coherence=coherencePerTrial,
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
  rdkSqr = visual.DotStim(win, nDots=int(nDotsSquare), coherence=coherencePerTrial, 
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