#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:16:41 2019

@author: xiaorli
"""

import numpy as np
import pandas as pd

from matplotlib.path import Path
import matplotlib.pyplot as plt

import os.path as osPath

import os as OS
import fnmatch
import csv

###############################################################################
#  DELETE .npz file first
###############################################################################

pitch = 34.0e-3 #m rod's pitch
L1 = 0.136 #m dimension along X
L2 = L1 #m dimension along Y
L3 = 0.96 #m dimension along Z

nSPE = 4

###############################################################################
def modifyCFDfilecoordinate(filename):  
    # function that moves the coordinate system
    # parameter1: abs total pressure
    filenamenew = filename[:-4] + 'modified.csv'
    if osPath.isfile(filenamenew) == True:
        return 0
        
    tableData = pd.read_csv(filename, header = 0, delimiter = ',')
    columnnames = tableData.columns
    
    if len(columnnames) == 5: # file represents data for gaps: turb tracer flux
        cellX = tableData['X (m)']
        cellY = tableData['Y (m)']
        cellZ = tableData['Z (m)']
        weight = tableData[columnnames[0]]  # area as weighing factor
        parameter1 = tableData[tableData.columns[1]]
    elif len(columnnames) == 6: # file represents data for sunChs: v and tracer
        cellX = tableData['X (m)']
        cellY = tableData['Y (m)']
        cellZ = tableData['Z (m)']
        weight = tableData[columnnames[2]] # volume as weighing factor
        parameter1 = tableData[tableData.columns[0]] # mean vk
        parameter2 = tableData[tableData.columns[1]] # tracer
        
        
        
    cellXTmp = cellX.values - 0.068
    cellYTmp = cellY.values - 0.068
    cellZmin = min(cellZ.values)
    cellZTmp = cellZ.values - cellZmin
    
    if len(columnnames) == 5:
        d = {'X (m)': cellXTmp, 'Y (m)': cellYTmp, 'Z (m)': cellZTmp,
             'weight': weight, 'parameter1': parameter1}
        df = pd.DataFrame(data = d)
    if len(columnnames) == 6:
        d = {'X (m)': cellXTmp, 'Y (m)': cellYTmp, 'Z (m)': cellZTmp,
             'weight': weight, 'parameter1': parameter1, 'parameter2': parameter2}
        df = pd.DataFrame(data = d)
    # save the dataframe structure to csv file
    
    df.to_csv(filenamenew, sep = ',', header = True, index = False)
    return 0
    
###############################################################################
def concatFiles(file1, file2):
    if osPath.isfile('CFDmodifiedGap.csv') == True:
        return 0
    combinedCFD = pd.concat([pd.read_csv(file1), \
                             pd.read_csv(file2)])
    combinedCFD.to_csv('CFDmodifiedGap.csv', index=False)
    return 0

###############################################################################
def subChannelGeom(p, L):
  #generate the coordinate for the CTF model

    rodP = p;
    channelL = L;

    #definition of cordinates of points defining subchannels
    xNodes = np.array([-channelL / 2.0, \
              -channelL / 2.0 + 1.0 * rodP, \
              -channelL / 2.0 + 2.0 * rodP, \
              -channelL / 2.0 + 3.0 * rodP, \
              -channelL / 2.0 + 4.0 * rodP]);

    yNodes = np.array([-channelL / 2.0, \
              -channelL / 2.0 + 1.0 * rodP, \
              -channelL / 2.0 + 2.0 * rodP, \
              -channelL / 2.0 + 3.0 * rodP, \
              -channelL / 2.0 + 4.0 * rodP]);

    xN, yN = np.meshgrid(xNodes, yNodes)

    return xN, yN
###############################################################################
def subChannelPartitioning(fileName, nSubChPerEdge, xN, yN):
    
    subChFileList = ['cfdData_SubCh1.csv', 'cfdData_SubCh2.csv', 'cfdData_SubCh3.csv', 
                 'cfdData_SubCh4.csv', 'cfdData_SubCh5.csv', 'cfdData_SubCh6.csv', 
                 'cfdData_SubCh7.csv', 'cfdData_SubCh8.csv', 'cfdData_SubCh9.csv', 
                 'cfdData_SubCh10.csv', 'cfdData_SubCh11.csv', 'cfdData_SubCh12.csv', 
                 'cfdData_SubCh13.csv', 'cfdData_SubCh14.csv', 'cfdData_SubCh15.csv', 
                 'cfdData_SubCh16.csv', ]
    
    indicator = 1
    for each in subChFileList:
       if osPath.isfile(each) == True:
           indicator = indicator * 1
       else:
           indicator = indicator * 0
    
    if indicator:
        return 0
    else:
        el = []
    
        tableData = pd.read_csv(fileName, header = 0, delimiter = ',')
    
        cellX = tableData['X (m)']
        cellY = tableData['Y (m)']
    
        cellXTmp = cellX.values
        cellYTmp = cellY.values
        tableDataTmp = tableData
    
        idSubCh = 0
        for idCol in [4, 3, 2, 1]:
            for idRow in [1, 2, 3, 4]:
                idSubCh = idSubCh + 1
    
                xV1 = xN[0 + idCol - 1, 0 + idRow - 1]
                xV2 = xN[0 + idCol - 1, 1 + idRow - 1]
                xV3 = xN[1 + idCol - 1, 0 + idRow - 1]
                xV4 = xN[1 + idCol - 1, 1 + idRow - 1]
    
                yV1 = yN[0 + idCol - 1, 0 + idRow - 1]
                yV2 = yN[0 + idCol - 1, 1 + idRow - 1]
                yV3 = yN[1 + idCol - 1, 0 + idRow - 1]
                yV4 = yN[1 + idCol - 1, 1 + idRow - 1]
    
                points = [(xV1, yV1),
                          (xV2, yV2),
                          (xV4, yV4),
                          (xV3, yV3),
                          (xV1, yV1)]
    
                codes = [Path.MOVETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.LINETO,
                         Path.CLOSEPOLY,
                         ]
    
                polyPath = Path(points, codes)
    
    
                pCloud2D = np.column_stack((cellXTmp, cellYTmp))
                vecTmp = np.where(polyPath.contains_points(pCloud2D) == True)[0]
                #vecTmp -- indices in pCloud2D where the condition is fulfilled!!
                el.append([idSubCh, vecTmp.shape[0]])
    
                fileOutput = 'cfdData_SubCh' + str(idSubCh) + '.csv'
    
                if osPath.isfile(fileOutput) == False:
                    tableFrame = tableDataTmp.iloc[vecTmp,:]
                    tableFrame.to_csv(fileOutput, index = False, header = True,
                                      sep = ',')
                else:
                    continue
                    #tableFrameOld = pd.read_csv(fileOutput, header = 0,
                    #                            delimiter = ',')
                    #tableFrameNew = pd.concat([tableFrameOld,
                    #                           tableData.iloc[vecTmp,:]])
                    #tableFrameNew.to_csv(fileOutput, index = False, header = True,
                    #                     sep = ',')
    
                #Remove elements to avoid duplication
                cellXT = np.delete(cellXTmp, vecTmp)
                cellYT = np.delete(cellYTmp, vecTmp)
    
                cellXTmp = cellXT
                cellYTmp = cellYT
    
                indexesToKeep = set(range(tableDataTmp.shape[0])) - set(vecTmp)
                tableDataT = tableDataTmp.take(list(indexesToKeep))
                tableDataTmp = tableDataT
    
        sumElm = 0
        for elmnt in el:
    
            print elmnt[0], elmnt[1]
            sumElm = sumElm + elmnt[1]
    
        print cellX.shape[0], sumElm, cellX.shape[0] - sumElm
    
        return 0

###############################################################################
def gapPartitioning(combinedCFD):
    
    gapFileList = ['cfdData_gap1.csv', 'cfdData_gap2.csv', 'cfdData_gap3.csv', 'cfdData_gap4.csv',
                    'cfdData_gap5.csv', 'cfdData_gap6.csv', 'cfdData_gap7.csv', 'cfdData_gap8.csv',
                    'cfdData_gap9.csv', 'cfdData_gap10.csv', 'cfdData_gap11.csv', 'cfdData_gap12.csv',
                    'cfdData_gap13.csv', 'cfdData_gap14.csv', 'cfdData_gap15.csv', 'cfdData_gap16.csv',
                    'cfdData_gap17.csv', 'cfdData_gap18.csv', 'cfdData_gap19.csv', 'cfdData_gap20.csv',
                    'cfdData_gap21.csv', 'cfdData_gap22.csv', 'cfdData_gap23.csv', 'cfdData_gap24.csv',]
                
    indicator = 1
    for each in gapFileList:
       if osPath.isfile(each) == True:
           indicator = indicator * 1
       else:
           indicator = indicator * 0
    
    if indicator:
        return 0
    else:
        tableCFD = pd.read_csv(combinedCFD, header = 0, delimiter = ',')
    #define gaps positions
    gappos = {'gap1':[-0.034, (0.0465, 0.0555)], 'gap2':[(-0.0555, -0.0465), 0.034], 
              'gap3':[0.0, (0.0465, 0.0555)], 'gap4':[(-0.0215, -0.0125), 0.034],
              'gap5':[0.034, (0.0465, 0.0555)], 'gap6':[(0.0125, 0.0215), 0.034], 
              'gap7':[(0.0465, 0.0555), 0.034], 'gap8':[-0.034, (0.0125, 0.0215)],
              'gap9':[(-0.0555, -0.0465), 0.0], 'gap10':[0.0, (0.0125, 0.0215)], 
              'gap11':[(-0.0215, -0.0125), 0.0], 'gap12':[0.034, (0.0125, 0.0215)],
              'gap13':[(0.0125, 0.0215), 0.0], 'gap14':[(0.0465, 0.0555), 0.0], 
              'gap15':[-0.034, (-0.0215, 0.0125)], 'gap16':[(-0.0555, -0.0465), -0.034],
              'gap17':[0.0, (-0.0215, -0.0125)], 'gap18':[(-0.0215, -0.0125), -0.034], 
              'gap19':[0.034, (-0.0215, -0.0125)], 'gap20':[(0.0125, 0.0215), -0.034],
              'gap21':[(0.0465, 0.0555), -0.034], 'gap22':[-0.034, (-0.0555, -0.0465)], 
              'gap23':[0.0, (-0.0555, -0.0465)], 'gap24':[0.034, (-0.0555, -0.0465)]}
    
    nogaps = len(gappos)
    gaps = []
#    idVolAve = np.zeros((nCell, 2)) - 1
    for idZ in range(0, nogaps):
        key = 'gap' + str((idZ + 1))
        if not isinstance(gappos[key][0], tuple):
            aZ = np.where((abs(tableCFD['X (m)'] - gappos[key][0]) <= 0.0001) &
                          (tableCFD['Y (m)'] >= gappos[key][1][0]) &
                          (tableCFD['Y (m)'] <= gappos[key][1][1]))[0]
        else:
            aZ = np.where((abs(tableCFD['Y (m)'] - gappos[key][1]) <= 0.0001) &
                          (tableCFD['X (m)'] >= gappos[key][0][0]) &
                          (tableCFD['X (m)'] <= gappos[key][0][1]))[0]


        gaps.append(aZ)
#        idVolAve[aZ, 1] = idZ
        fileOutput = 'cfdData_gap' + str(idZ+1) + '.csv'
    
        if osPath.isfile(fileOutput) == False:
            tableFrame = tableCFD.iloc[aZ,:]
            tableFrame.to_csv(fileOutput, index = False, header = True,
                              sep = ',')
        else:
            continue
        
    return 0    

###############################################################################
def axialSubChPartitioning(FileList, fileNameCTF, newName):

    if osPath.isfile(newName) == True:
        return 0
    
    tableDataCBTF = pd.read_csv(fileNameCTF, header = 0, delimiter = '\t')

    nChannel = np.max(tableDataCBTF['channel'].astype(int))
    nAxialNodes = np.max(tableDataCBTF['node'].astype(int))

    newTableAxialNodePos = np.reshape(tableDataCBTF['node_pos'].values,
                                      [nChannel, nAxialNodes])
    # note: only coordinates are reshaped
    newTableSortedAxialNodePos = np.fliplr(newTableAxialNodePos[:])
    newArraySortedAxialNodePos = newTableSortedAxialNodePos[0]

    cbtfNZCell = np.size(newArraySortedAxialNodePos)

    #Variable initialization
    aveparameter1 = np.zeros([nChannel, np.size(newArraySortedAxialNodePos)-1])
    aveparameter2 = np.zeros([nChannel, np.size(newArraySortedAxialNodePos)-1])
    aveX   = np.zeros([nChannel, np.size(newArraySortedAxialNodePos)-1])
    aveY   = np.zeros([nChannel, np.size(newArraySortedAxialNodePos)-1])
    aveZ   = np.zeros([nChannel, np.size(newArraySortedAxialNodePos)-1])

    #loop on all files that have to be subdivided in the axial direction
    idSubCh = 0
    for i in FileList:

        idSubCh = idSubCh + 1

        tableCFD = pd.read_csv(i, header = 0, delimiter = ',')
        nCell = tableCFD.shape[0]
        if nCell == 0:
            continue

        isInLevelN = []
        idVolAve = np.zeros((nCell, 2)) - 1

        #axial subdivision
        for idZ in range(0, cbtfNZCell-1):
            aZ = np.where((tableCFD['Z (m)'] >= newArraySortedAxialNodePos[idZ]) &
                          (tableCFD['Z (m)'] < newArraySortedAxialNodePos[idZ + 1]))[0]
            # aZ = np.where*****[0]: only index in the table is returned
            isInLevelN.append(aZ)
            idVolAve[aZ, 1] = idZ
            

        idLevel = 0
        for idArr in isInLevelN:

            totweight = np.sum(tableCFD['weight'][idArr])

            totparameter1 = np.sum(tableCFD['parameter1'][idArr] *
                     tableCFD['weight'][idArr])
            
            totparameter2 = np.sum(tableCFD['parameter2'][idArr] *
                     tableCFD['weight'][idArr])


            totX   = np.sum(tableCFD['X (m)'][idArr] *
                     tableCFD['weight'][idArr])

            totY   = np.sum(tableCFD['Y (m)'][idArr] *
                     tableCFD['weight'][idArr])

            totZ   = np.sum(tableCFD['Z (m)'][idArr] *
                     tableCFD['weight'][idArr])


            aveX[idSubCh-1][idLevel]   = totX / totweight
            aveY[idSubCh-1][idLevel]   = totY / totweight
            aveZ[idSubCh-1][idLevel]   = totZ / totweight

            aveparameter1[idSubCh-1][idLevel] = totparameter1 / totweight
            aveparameter2[idSubCh-1][idLevel] = totparameter2 / totweight

            idLevel = idLevel + 1

    np.savez(newName, aveX = aveX, aveY = aveY, aveZ = aveZ, 
             aveparameter1 = aveparameter1,
             aveparameter2 = aveparameter2) 

    return 0

###############################################################################
    
def axialGapPartitioning_averageCal(FileList, newName):

    if osPath.isfile(newName) == True:
        return 0
    # creat matrix for gap data
    nGap = 24
    nAxialNodes = 15

    gap = np.linspace(1, nGap, nGap, dtype = int)
    node    = np.linspace(1, nAxialNodes, nAxialNodes, dtype = int)
    
    #Axial nodalization in CTF model
    z = [0.0, 0.01, 0.06, 0.11, 0.16, 0.21, 0.26, 0.31, 0.36, 0.41,
         0.46, 0.51, 0.56, 0.61, 0.66, 0.71]

    #manipulation of variables to comply to CTF output format
    tmpGap     = np.repeat(gap, nAxialNodes)
    tmpNode    = np.tile(np.flipud(node), nGap)
    tmpZ       = np.tile(np.flipud(z[:-1]), nGap)

    newTableAxialNodePos = np.reshape(tmpZ,[nGap, nAxialNodes])
    # note: only coordinates are reshaped
    newTableSortedAxialNodePos = np.fliplr(newTableAxialNodePos[:])
    newArraySortedAxialNodePos = newTableSortedAxialNodePos[0]

    cbtfNZCell = np.size(newArraySortedAxialNodePos)

    #Variable initialization
    aveparameter1 = np.zeros([nGap, np.size(newArraySortedAxialNodePos)])
    aveX   = np.zeros([nGap, np.size(newArraySortedAxialNodePos)])
    aveY   = np.zeros([nGap, np.size(newArraySortedAxialNodePos)])
    aveZ   = np.zeros([nGap, np.size(newArraySortedAxialNodePos)])

    #loop on all files that have to be subdivided in the axial direction
    idGap = 0
    for i in FileList:

        idGap = idGap + 1

        tableCFD = pd.read_csv(i, header = 0, delimiter = ',')
        nCell = tableCFD.shape[0]
        if nCell == 0:
            continue

        isInLevelN = []
        idVolAve = np.zeros((nCell, 2)) - 1

        #axial subdivision
        for idZ in range(0, cbtfNZCell - 1):
            aZ = np.where((tableCFD['Z (m)'] >= newArraySortedAxialNodePos[idZ]) &
                          (tableCFD['Z (m)'] < newArraySortedAxialNodePos[idZ + 1]))[0]
            # aZ = np.where*****[0]: only index in the table is returned
            isInLevelN.append(aZ)
            idVolAve[aZ, 1] = idZ
            

        idLevel = 0
        for idArr in isInLevelN:

            totweight = np.sum(tableCFD['weight'][idArr])

            totparameter1 = np.sum(tableCFD['parameter1'][idArr] *
                     tableCFD['weight'][idArr])


            totX   = np.sum(tableCFD['X (m)'][idArr] *
                     tableCFD['weight'][idArr])

            totY   = np.sum(tableCFD['Y (m)'][idArr] *
                     tableCFD['weight'][idArr])

            totZ   = np.sum(tableCFD['Z (m)'][idArr] *
                     tableCFD['weight'][idArr])

            aveparameter1[idGap-1][idLevel] = totparameter1 / totweight

            aveX[idGap-1][idLevel]   = totX / totweight
            aveY[idGap-1][idLevel]   = totY / totweight
            aveZ[idGap-1][idLevel]   = totZ / totweight

            idLevel = idLevel + 1

    np.savez(newName, aveX = aveX, aveY = aveY, aveZ = aveZ, 
             aveparameter1 = aveparameter1) 

    return 0
###############################################################################
def subExtract(npzName):
    ''' Extract data from sub-channelized file xxx.npz '''
    myData = np.load(npzName)
    
    # generate columns
    mystring = 'Node'
    clmnames = []
    for i in range(0, nAxialNode-1):
        clmnames.append(mystring + str(i))  

    if 'Gap' in npzName:
        # generate indices        
        mystring = 'gap'
        indices = []
        for i in range(0, nGap):
            indices.append(mystring + str(i)) 
        
        GapData = np.zeros([nGap,nAxialNode-1])
        for gap in range(0, nGap):
            GapData[gap][:] = myData['aveparameter1'][gap][0:-1] # last element is 0
        GapData = pd.DataFrame(GapData, columns = clmnames, index = indices)
        GapData.to_csv('Data_crossflow.csv')
    if 'Subch' in npzName:
        # generate indices        
        mystring = 'Subch'
        indices = []
        for i in range(0, nSubCh):
            indices.append(mystring + str(i)) 
            
        SubData1 = np.zeros([nSubCh,nAxialNode-1])
        SubData2 = np.zeros([nSubCh,nAxialNode-1])
        for SubCh in range(0, nSubCh):
            SubData1[SubCh][:] = myData['aveparameter1'][SubCh] # mean vk
            SubData2[SubCh][:] = myData['aveparameter2'][SubCh] # tracer
        vData = pd.DataFrame(SubData1, columns = clmnames, index = indices)
        trData = pd.DataFrame(SubData2, columns = clmnames, index = indices)
        vData.to_csv('Data_subchanvk.csv')
        trData.to_csv('Data_subchantr.csv')
        

###############################################################################

def calEachCrossflow():    
    """ calculate cross flow factor for each gap, including the second peak."""
    
    crossFlow = pd.read_csv('Data_crossflow.csv', index_col = 'Unnamed: 0')
    peakCross = crossFlow['Node2']
    crossFlowPeakFactor = peakCross/0.8
    #original_factor = peakCross/0.8
    #need to judge the sign of lateral flow according to CTF rule!!
    gapsToFlip = [2,4,6,7,9,11,13,14,16,18,20,21] #gaps in y direction
    gapsToFlipIndex = [x - 1 for x in gapsToFlip]
    for index in gapsToFlipIndex:
        crossFlowPeakFactor[index] = -crossFlowPeakFactor[index] 
     
    return crossFlowPeakFactor


###############################################################################

def calEachCrossflow2peak():    
    """ calculate cross flow factor for each gap."""
    
    crossFlow = pd.read_csv('Data_crossflow.csv', index_col = 'Unnamed: 0')
    peakCross = crossFlow['Node2']
    crossFlowPeakFactor = peakCross/0.8
    
    peakCross2 = crossFlow['Node6']
    crossFlowPeakFactor2 = peakCross2/0.8
    #original_factor = peakCross/0.8
    #need to judge the sign of lateral flow according to CTF rule!!
    gapsToFlip = [2,4,6,7,9,11,13,14,16,18,20,21] #gaps in y direction
    gapsToFlipIndex = [x - 1 for x in gapsToFlip]
    for index in gapsToFlipIndex:
        crossFlowPeakFactor[index] = -crossFlowPeakFactor[index] 
        crossFlowPeakFactor2[index] = -crossFlowPeakFactor2[index]
     
    return crossFlowPeakFactor, crossFlowPeakFactor2

###############################################################################

def calEachCrossflowAllAxialNode():    
    """ calculate cross flow factor for each gap, including all axial nodes.
    Node in CFD file the node lable starts from 'Node0'. """
    AxialNodeno = 14 # axial node number in CFD data
    Nodes = []
    base = 'Node'
    for i in range(0, AxialNodeno):
        Nodes.append(base+str(i))
    
    crossFlow = pd.read_csv('Data_crossflow.csv', index_col = 'Unnamed: 0')
    lateralFactors = []
    for node in Nodes:
        lateralFactors.append(crossFlow[node]/0.8)
    #need to judge the sign of lateral flow according to CTF rule!!
    gapsToFlip = [2,4,6,7,9,11,13,14,16,18,20,21] #gaps in y direction
    gapsToFlipIndex = [x - 1 for x in gapsToFlip]
    for factors in lateralFactors:
        for index in gapsToFlipIndex:
            factors[index] = -factors[index] 
    #note: lateralFactors is a list of list
    
    #below calculate factors averaged over all subchannels
    crossFlowAveFactor = crossFlow.apply(abs).mean(axis = 0)/0.8
    lateralFactorsAvelist = []
    for i in range(0,14):
        base = []
        for j in range(0,24):
            base.append(crossFlowAveFactor[i])
        lateralFactorsAvelist.append(base)
            
        
    for i in range(0, 14):
        for j in range(0, 24):
            #note, in the original model there is only one sign for all source
            #terms in one sub-channel. therefore -- sign(crossFlow.iloc[j,2])
            lateralFactorsAvelist[i][j] = lateralFactorsAvelist[i][j] *sign(crossFlow.iloc[j,2])    
    for each in lateralFactorsAvelist:
        for index in gapsToFlipIndex:
            each[index] = -each[index] 
    
    
    return lateralFactors, lateralFactorsAvelist

###############################################################################

def plotCrossflows():    
    """ plot the crossflows from CFD data at each gap
    can also manually plot directly from the csv file"""
    
    crossFlow = pd.read_csv('Data_crossflow.csv', index_col = 'Unnamed: 0')
    axialno = [i for i in range(1,15)]
    gapno = len(crossFlow.index)
    gapscatter = []
    gapplots = [i for i in range(0, gapno)]
    for i in gapplots:
        leg = "gap" + str(i+1)
        gapscatter.append(plt.plot(axialno, crossFlow.iloc[i,:],\
                                   label=leg, linewidth=0.5))
    plt.legend(loc='lower right', ncol=4, fontsize=7)
    plt.grid(True, linestyle=':')
    plt.xticks(axialno)
    plt.xlabel('Axial node number')
    plt.ylabel('lateral velocity, m/s')
    plt.savefig("lateral velocity in gaps CFD.jpg", dpi=300)
    plt.close()     
        
    return 0

###############################################################################
def CrossFactorToCTFFile(peakFactors):
    basefilename = 'C:\\Users\\xiaorli\\CTF40\\simulations\\xflow_data'
    gapno = 24
    axialNode = 21 # number of axial nodes in CTF model
    crossFactors = [0] * axialNode # potential use in future, not necessary now.
    basehd = open(basefilename, 'r')
    lines =[line.rstrip('\n') for line in basehd.readlines()]
    for i in range(0, gapno):
        crossFactors[3] = peakFactors[i]
        filename = basefilename + str(i+1)
        filehd = open(filename, 'w') # open a new file, or replace an old one
        index = 0
        while index < 9: # the first 8 lines remain untouched
            filehd.write(lines[index]+'\n')
            index += 1
        while index < 30 :
            line = lines[index]
            parsedline = line.split()
            parsedline[1] = str(crossFactors[index-9])
            newline = parsedline[0] + '  ' + parsedline[1]
            filehd.write(newline+'\n')
            index += 1
        filehd.close()
    basehd.close()
    return 0

###############################################################################
def CrossFactorToCTFFile2peak(Factors1, Factors2):
    basefilename = 'C:\\Users\\xiaorli\\CTF40\\simulations\\xflow_data'
    gapno = 24
    axialNode = 21 # number of axial nodes in CTF model
    crossFactors = [0] * axialNode # potential use in future, not necessary now.
    basehd = open(basefilename, 'r')
    lines =[line.rstrip('\n') for line in basehd.readlines()]
    for i in range(0, gapno):
        crossFactors[3] = Factors1[i]
        crossFactors[7] = Factors2[i]
        filename = basefilename + str(i+1)
        filehd = open(filename, 'w') # open a new file, or replace an old one
        index = 0
        while index < 9: # the first 8 lines remain untouched
            filehd.write(lines[index]+'\n')
            index += 1
        while index < 30 :
            line = lines[index]
            parsedline = line.split()
            parsedline[1] = str(crossFactors[index-9])
            newline = parsedline[0] + '  ' + parsedline[1]
            filehd.write(newline+'\n')
            index += 1
        filehd.close()
    basehd.close()
    return 0

###############################################################################
def CrossFactorToCTFFileAllAxial(lateralFactorsAllAxial):
    basefilename = 'C:\\Users\\xiaorli\\CTF40\\simulations\\xflow_data'
    gapno = 24
    axialNodeCTF = 21 # number of axial nodes in CTF model
    axialNodeCFD = 14
    crossFactors = [0] * axialNodeCTF # potential use in future, not necessary now.
    basehd = open(basefilename, 'r')
    lines =[line.rstrip('\n') for line in basehd.readlines()]
    for i in range(0, gapno):
        for j in range(0, axialNodeCFD):
            crossFactors[j+1] = lateralFactorsAllAxial[j][i]
            filename = basefilename + str(i+1)
            filehd = open(filename, 'w') # open a new file, or replace an old one
            index = 0
            while index < 9: # the first 8 lines remain untouched
                filehd.write(lines[index]+'\n')
                index += 1
            while index < 30 :
                line = lines[index]
                parsedline = line.split()
                parsedline[1] = str(crossFactors[index-9])
                newline = parsedline[0] + '  ' + parsedline[1]
                filehd.write(newline+'\n')
                index += 1
            filehd.close()
    basehd.close()
    return 0

###############################################################################
###############################################################################


nSubCh = 16
nGap = 24
nAxialNode = 15

# =============================================================================
xAS, yAS = subChannelGeom(pitch, L1) #generates coordinates for subchannel 
CFDfilename1 = 'velocity_x.csv'
CFDfilename2 = 'velocity_y.csv'
modifyCFDfilecoordinate(CFDfilename1)
modifyCFDfilecoordinate(CFDfilename2)
 
#concatinate two csv files together, fluxes in x and y derections, respectively
concatFiles('velocity_xmodified.csv', 'velocity_ymodified.csv')
 
fileNameCFDgap = 'CFDmodifiedGap.csv'
gapPartitioning(fileNameCFDgap)
fileNameCFDsubCh = 'vandtracerslopeRegionsmodified.csv'
#tmpRes = subChannelPartitioning(fileNameCFDsubCh, nSPE, xAS, yAS) # done on Euler
 
createCTFDummyCSVFile()
fileNameCTF = 'channel_passivescalar12DummyCSVFile.csv'
 
 
gapFileList = ['cfdData_gap1.csv', 'cfdData_gap2.csv', 'cfdData_gap3.csv', 'cfdData_gap4.csv',
                 'cfdData_gap5.csv', 'cfdData_gap6.csv', 'cfdData_gap7.csv', 'cfdData_gap8.csv',
                 'cfdData_gap9.csv', 'cfdData_gap10.csv', 'cfdData_gap11.csv', 'cfdData_gap12.csv',
                 'cfdData_gap13.csv', 'cfdData_gap14.csv', 'cfdData_gap15.csv', 'cfdData_gap16.csv',
                 'cfdData_gap17.csv', 'cfdData_gap18.csv', 'cfdData_gap19.csv', 'cfdData_gap20.csv',
                 'cfdData_gap21.csv', 'cfdData_gap22.csv', 'cfdData_gap23.csv', 'cfdData_gap24.csv',]
 
 
axialGapPartitioning_averageCal(gapFileList, 'cfdDataGapNodalSubdivision.npz')
# calculate averaged variables from CFD data, for each subchannel and axial node
SubNodedata = subExtract('cfdDataGapNodalSubdivision.npz')
# =============================================================================

factors = calEachCrossflow()
CrossFactorToCTFFile(factors*2.1) #2.2 resulted slightly larger lateral peaks

#factors1, factors2 = calEachCrossflow2peak()
lateralFactorsAllAxial = calEachCrossflowAllAxialNode()[1] # 1: averaged factors
#CrossFactorToCTFFile2peak(factors1*2, factors2*4) #3 times factors2 seems too small
CrossFactorToCTFFileAllAxial(lateralFactorsAllAxial)