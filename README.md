# CFD_data_processing
Processing of CFD data obtained on tens of millions computational cells 

## Background

As a fundamental part of my PhD project, I built a huge Computational Fluid Dynamics (CFD) model to calculate the fluid field in a complex flow channel. The CFD model is huge, since it discretizes the flow channel into tens of millions cells and solves for the fluid field parameters on each of them, including fluid temperature, velocities and turbulence. These data are exported as csv files, which also contain the 3-dimensional coordinate information.

Processing of such huge CFD data is necessary, because in the subsequent work the CFD data has to be compared to a much more simplified model with much fewer nodes/cells. Therefore the huge amount of data on tens of millions computational cells must be summarized according to the coarse nodalization in the simplified model. The following picture shows the flow channel under consideration. It's a heating rod bundle, where the coolant flows through tha space between the rods and cool them down. The channel is called computational domain in the context of numerical modelling. Both with the CFD model and the simplified model, the computational domain itself stays identical. The difference is the discretization as mentioned before: with CFD the domain is discretized into tens of millions of cells, whereas in the simplified model there are much fewer of them (around 200, in current case). When processing the CFD data, the geometric mapping between the CFD cells and the nodalization in the simplified model must be considered - this is the main task for this part of job and is described in detail in the following section.


## Python functions for data processing

Various Python functions have to be created for the aforementioned purpose of data processing. In this section we walk through these functions.

- 
