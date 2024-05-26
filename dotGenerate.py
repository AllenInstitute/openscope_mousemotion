from psychopy import visual, event, core
import numpy as np
import random



win = visual.Window([1000,1000], fullscr=False,color="black", winType='pyglet', waitBlanking=False)
win.clearBuffer(color=True)


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
        win, vertices="circle",
        pos=(0.5, 0.5), size=(r*2, r*2), units="pix",
        fillColor="black",  opacity=1, interpolate=True,
        autoDraw=False)
    
  rdkCircle = visual.DotStim(win, nDots=nDotsCircle, coherence=coherencePerTrial,
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
  rdkSqr = visual.DotStim(win, nDots=nDotsSquare, coherence=coherencePerTrial, 
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