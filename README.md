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

Various Python functions have to be created for the aforementioned purpose of data processing. In this section we walk through these functions.

- 
