.. _task_hi:

############
Health index
############

Dataset
"""""""

The dataset includes the result of an experiment conducted on a Matsuura machining
centre (MC-510V) that performed dry, rough milling of cast iron or J45 stainless 
steel workpieces with a six-tooth face mill with KC710 carbide inserts at various 
cutting parameters [1]. The dataset contains 16 sub-datasets 
in which the cutting depths, feed rates and material types were changed. The sensors 
used to record the milling process are two vibroaccelerometers, two "AE" sensors, 
and two "CTA" current sensors.

Task
""""

The dataset's target space represents the level of sawing surface wear measured 
based on optical microscopy. Of the total 16 cases in the dataset, 17 steps, 
including five in case 6, six in case 8, and six in case 16, contain no tool wear 
level information. The following dataset parameters are used for prediction: AC 
spindle motor current (smcAC), DC spindle motor current (smcDC), Table vibration 
(vib\_table), Spindle vibration (vib\_spindle), Acoustic emission at the table 
(AE\_table), Acoustic emission at spindle (AE\_spindle). The main idea of the dataset 
is to build a process-independent feature extraction model for predicting the equipment 
wear value. Due to this, the dataset is divided by processes: numbers 1,3,5 for testing 
and others for training and validation.

The runs always start with a perfectly operable device state, confirmed by microscopy. 
To construct a process-based HI on wear, we use the Taylor equation proposed in the 
dataset documentation:

.. math::

  \begin{equation}\label{eq:Milling}
  T = C_{v}v_{c}^{k}
  \end{equation}

where :math:`T` is tool life, :math:`C_{v}` is the constant related to 1 min tool life, 
:math:`v_{c}` is cutting speed, :math:`k` is the Taylor exponent.

We performed an approximation (Fig. 1) of the wear measured at the end of 
each cut using the Nelder Mead method from SciPy library [2]. The 
obtained equation parameters characterize the process itself by definition, and the 
character of the curve behavior between the measured points corresponds to the empirically 
verified equation.

.. figure:: ../../_static/hi/mil_data.png
   :align: center

   Figure 1: Milling dataset. A – Sensor reading examples of the first dataset case, used as input for
   pipeline. B – Target tool wear constructed using the Taylor equation to approximate missing sawtooth
   wear values, used for HI construction.

Next, we transformed the curves characterizing the process and the state of the device 
simultaneously to the device Health Index. To do this, we performed min-max scaling 
(using train subdatasets), where the minimum value of wear is 0 mm, and the maximum 
corresponds to the maximum allowable wear of 1.58 mm, and subtracted the obtained curves 
from 1 to reduce the impact of implicit weighting of healthy states.

.. figure:: ../../_static/hi/target_milling.jpg
   :align: center

   Figure 2: Milling dataset trajectory targets.

Metrics
"""""""

The performance of the models on the milling dataset HI problem was evaluated using RMSE 
on 1,3,5 cuts chosen for evaluation. RMSE metric is applying both to wear and acquired HI 
values:

.. math::

  \begin{equation}
  RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
  \end{equation}

where :math:`\hat{y}_i` and :math:`y_i` are predicted and true value of HI targets during 
equipment :math:`i`, :math:`N` is number of test samples. 

References
""""""""""

.. The reference results of state-of-the-art papers for the original [1] 
   dataset are presented in table 1.

   Since the test dataset involves processes with unknown characteristics, it limits most 
   related HI-based work approaches, especially similarity-based techniques, which require 
   information about similar process trajectories. 

   .. table:: Table 1: References for C-Milling dataset, cuts ‘1’, ‘3’, ‘7’ testing, with the original dataset wear target.

   +--------------------------------+-----------------------+
   | Model                          | RMSE (cycles)         | 
   +================================+=======================+
   |MapReduce-based PRFs (2020)     | 0.0531                |
   +--------------------------------+-----------------------+
   |LS-SVM (2020)                   | 0.0254                |
   +--------------------------------+-----------------------+
   |TAKELM (2020)                   |0.0134                 |
   +--------------------------------+-----------------------+
   |LSTM-based (2018)               | 0.0148                |
   +--------------------------------+-----------------------+

[1]   Agogino, A.; Goebel, K. Milling Data Set. NASA Prognostics Data Repository 2007.

[2]   Virtanen, P. et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature
Methods 2020, 17, 261–272.