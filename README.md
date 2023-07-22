# Deep Spectrum Cartography: Completing Radio Map Tensors Using Learned Neural Model

Implementation of the paper, "[Deep Spectrum Cartography: Completing Radio Map Tensors Using Learned Neural Model](https://ieeexplore.ieee.org/document/9693274)", published in IEEE Transactions on Signal Processing. A shorter, conference version of this paper is published in IEEE ICASSP 2021, available [here](https://ieeexplore.ieee.org/document/9413382).

## Spectrum Cartography (SC) Problem:

<p align='center' style="background-color:white;" >
    <img src='img/spectrum3d.png' style="width: 25%; margin-right:20px" />
    <img src='img/SC.png' style='width:50%; margin-left:20px'>
    </br>
    </br>
    <figurecaption style='color: black'>[Left] SC environment. [Right] SC task.</figurecaption>
</p>
</br>

## Proposed Method:
<p align='center' style="background-color:white;" >
    <img src='img/emitter_disaggregate.png' style='width:70%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Radio map model.</figurecaption>
</p>
</br>

<p align='center' style="background-color:white;" >
    <img src='img/Nasdac_process_long.png' style='width:100%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Method 1: Nonnegative matrix factorization ASsisted Deep emitter spAtial loss Completion (Nasdac) .</figurecaption>
</p>
</br>

<p align='center' style="background-color:white;" >
    <img src='img/dowjons_illus2.png' style='width:100%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Method 2: Deep generative priOr With Joint OptimizatioN for Spectrum cartography (Dowjons). 
    </figurecaption>
</p>

## Results:

<p align='center' style="background-color:white;" >
    <img src='img/sep_performance_compare.png' style='width:70%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Completed Radio maps using different methods.</figurecaption>
</p>
</br>

<p align='center' style="background-color:white;" >
    <img src='img/sep_psd_disaggregate_all.png' style='width:50%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Recovered PSD by the proposed methods.</figurecaption>
</p>
</br>

<p align='center' style="background-color:white;" >
    <img src='img/sep_slf_disaggregate_all4.png' style='width:70%; margin-left:10px'>
    </br>
    <figurecaption style='color: black'>Recovered SLF by the proposed methods.</figurecaption>
</p>
</br>


## Installation:
The code was built with the `python3.6`, `matlab2020b`, and `torch=1.10.2`

To run the code follow the followign installation instructions:

    1. Install all python packages located in requirements.txt.

    2. Download tensorlab from  https://www.tensorlab.net

    3. Make sure that the above packages are in your environment path.

    4. To interface the python models from matlab use instructions provided here: https://www.mathworks.com/help/matlab/matlab_external/create-object-from-python-class.html.

## Usage:
Sample demonstration of the proposed method in the paper is available in `experiments/demo.m`. 

