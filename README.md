# CFD_data_processing
Processing of CFD data obtained on tens of millions computational cells 

## Background

As a fundamental part of my PhD project, I built a huge Computational Fluid Dynamics (CFD) model to calculate the fluid field in a complex flow channel, as shown below. It's a heating rod bundle, where the coolant flows through tha space between the rods and cool them down. The channel is called computational domain in the context of numerical modelling. 

<img src="/CFDbundle.png" height="400"  width="400" >

The CFD model is huge, since it discretizes the flow channel into tens of millions cells. Actually, the picture above can also show the CFD-discretized representation of the flow bundle. Because the cells are very small, especially in the region near the solid surfaces of the channel, the CFD nodalization conforms very well to the real geometry of the channel. Since the cells are too small, I prefer to not display them on the geometry in the picture, but you can imagine the whole geometry is piled up by these small cells. A CFD software solves for the fluid field parameters (e.g. temperature, velocity, etc) on each of the small cells. These data are exported as csv files, which also contain the 3-dimensional coordinate information.

Processing of such huge CFD data is necessary, because in the subsequent work the CFD data has to be compared to a much more simplified model with much fewer nodes/cells. Therefore the huge amount of data on tens of millions computational cells must be summarized according to the coarse nodalization in the simplified model, as shown below. In this case, the nodes/cells in the nodalization are represented by the cuboids that are clearly visible.

<img src="/CTFbundle.png" height="400" >

Both with the CFD model and the simplified model, the computational domain itself stays identical. The difference is the discretization as mentioned before and shown with the two pictures: with CFD the domain is discretized into tens of millions of cells, whereas in the simplified model there are much fewer of them (around 200, in current case). When processing the CFD data, the geometric mapping between the CFD cells and the nodalization in the simplified model must be considered - this is the main task for this part of job and is described in detail in the following section.


## Python functions for data processing

Various Python functions have to be created for the aforementioned purpose of data processing (see the "crossvelocityCFD_xflowfactor_github.py" file in current repo). The most important packages used are numpy and pandas. In this section, I'll briefly describe a few key functions involved and show the workflow. In order to make the discussion flow smooth and concise, I'll have to skip a lot very specific technical details, which are nevertheless all included in the "crossvelocityCFD_xflowfactor_github.py". So let's take the processing of lateral velocities in two directions (x and y) as the example data. The startpoint is the raw CFD data stored in two csv files - in x and y directions, respectively.

- modifyCFDfilecoordinate(filename) : This function modifies the coordinates in the raw CFD data. When exported, these data natually take the coordinates that have been determined by the inner CFD model. However in the simpified system a linearly shifted coordinate system is preferred. Therefore, this function shifts the raw data to a new coordinate system.

- concatFiles(file1, file2): In the data processing stage, I don't need to differentiate between x and y direction of the flow anymore for some reason. So I used this function to concatinate the (modified) CFD raw data together.

- subChannelGeom(p, L): This function generates and returns the coordinates for the simplified system, i.e., determines the positions for the cuboids as shown in the second figure.

- subChannelPartitioning(fileName, nSubChPerEdge, xN, yN): This function does the first-step mapping from the CFD nodalization (shown in first figure) to the simplified nodalization (shown in second figure). It reads in the raw CFD data, which is a single csv file that stores velocity data for all the CFD cells in the whole computational domain, and partitions them into sub-channels corresponding to the cuboids stacks/columns shown in the second figure. There are in total 16 of these stacks/columns. This function writes each part of the partitioned data into separate files named "cfdData_SubChx.csv", where "x" indicates the sub-channel index. Since the raw data is big, the partitioning process takes some computational power. I did it on a cluster in my university, rather than on my own PC.

- axialSubChPartitioning(FileList, fileNameCTF, newName): This function fulfills the second-step mapping from the CFD nodalization (shown in first figure) to the simplified nodalization (shown in second figure). It further partitions the files returned by the previous function ("subChannelPartitioning") in the vertical/axial direction. As a result, the raw CFD data can now be averaged for each cuboids (shown in second figure) and stored in an npz file.


- gapPartitioning(combinedCFD) and axialGapPartitioning_averageCal(FileList, newName): The two functions are pretty similar to "subChannelPartitioning" and "axialSubChPartitioning". Instead of partitioning out the sub-channels, they work on the interfaces between adjacent sub-channels. We call these interfaces "gaps". The velocity data through these gaps are of great interest for the purpose of my research.

Till now, the major part of the data processing is accomplished. This lays a fundation for the succeeding work: [development of an iterative learning algorithm!](https://github.com/XiaorongLi/Momentum_Source_Iteration)
