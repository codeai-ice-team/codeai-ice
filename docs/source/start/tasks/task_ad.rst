.. _task_ad:

#################
Anomaly detection
#################

Dataset
"""""""

The TEP [1] :ref:`tep-dataset` is used for Anomaly detection task.

Task
""""

In anomaly detection within the Tennessee Eastman Process (TEP) dataset, the focus 
is on monitoring a set of process variables to define the operational state of the 
chemical process and classify them as normal or faulty. These variables, such as 
temperature, pressure, flow rates, level indicators, and chemical compositions, are 
key indicators of the process's health and efficiency. Anomalies or faults in the 
process are typically signaled by deviations in these variables from their normal 
operating ranges.

The mathematical formulation for anomaly detection in this context involves 
creating a function that maps the process variables to a binary output, indicating 
the normal or anomalous state. Let :math:`X(t)` represent the vector of process 
variables at time :math:`t`:

.. math::

  \begin{equation}
    X(t) = [x_1(t), x_2(t), \ldots, x_n(t)],
  \end{equation}

where :math:`x_i(t)` denotes the value of the $i$-th process variable at time 
:math:`t`, and :math:`n` is the total number of monitored process variables.

The anomaly detection function :math:`f` can then be defined as:

.. math::

  \begin{equation}
      f(X(t)) = 
    \begin{cases} 
      1 & \text{if } X(t) \text{ indicates an anomaly}, \\
      0 & \text{otherwise} 
    \end{cases}
  \end{equation}

In this setup, :math:`f` could be a machine learning model trained on historical 
data from the TEP dataset. This model learns to classify the operational state as 
normal or anomalous based on the values and relationships of these process 
variables. Advanced machine learning algorithms, such as neural networks [2, 3], decision 
trees, or support vector machines, are employed to capture the complex 
interdependencies between these variables and accurately identify patterns 
indicative of anomalies.

Metrics
"""""""

In anomaly detection, several metrics are commonly used to evaluate the performance 
of models. These metrics help in understanding how well the model is identifying 
anomalies (true positives) and avoiding false alarms (false positives). The choice 
of metrics often depends on the specific requirements of the task, such as the 
importance of detecting all anomalies versus the cost of false alarms. Common 
metrics include:

**Accuracy** measures the overall correctness of the model, calculated as the 
number of correct predictions divided by the total number of cases.

.. math::

  \begin{equation}
    \text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Number of Cases}}
  \end{equation}

**Recall (True Positive Rate, TPR)** measures the proportion of actual positives 
(anomalies) correctly identified as such (true positives). It is particularly 
important in scenarios where missing an anomaly can have serious consequences.

.. math::

  \begin{equation}
    \text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  \end{equation}

**False Positive Rate (FPR)** measures the proportion of negative instances that 
are incorrectly classified as positive (anomalies). A lower FPR is desirable as it 
indicates fewer false alarms. 

.. math::

  \begin{equation}
    \text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
  \end{equation} 

References
""""""""""

In recent years, there has been a lot of research on neural network models for the task of 
anomaly detection. The most cited of them are presented in Table 1. The models were trained 
and tested on various datasets, but all of them can be adapted to work with TEP.

.. table:: Table 1: Deep learning models for Anomaly detection.

   +--------------------------------+-----------------------+-----------------------+
   | Model                          | Datasets              | F1-score              |
   +================================+=======================+=======================+
   |DAGMM (2018) [4]                | KDDCUP                | 0.9369                |
   |                                +-----------------------+-----------------------+
   |                                | Thyroid               | 0.4782                |
   |                                +-----------------------+-----------------------+
   |                                | Arrhuthmia            | 0.4983                |
   |                                +-----------------------+-----------------------+
   |                                | KDDCUP-Rev            | 0.9380                |
   +--------------------------------+-----------------------+-----------------------+
   |GDN (2021) [5]                  | SWaT                  | 0.81                  |
   |                                +-----------------------+-----------------------+
   |                                | WADI                  | 0.57                  |
   +--------------------------------+-----------------------+-----------------------+
   |GTA (2022) [6]                  | SWaT                  | 0.84                  |
   |                                +-----------------------+-----------------------+
   |                                | WADI                  | 0.82                  |
   |                                +-----------------------+-----------------------+
   |                                | SMAP                  | 0.9041                |
   |                                +-----------------------+-----------------------+
   |                                | MSL                   | 0.9111                |
   +--------------------------------+-----------------------+-----------------------+
   |MAD-GAN (2019) [7]              | KDDCUP99              | 0.94                  |
   |                                +-----------------------+-----------------------+
   |                                | SWaT                  | 0.77                  |
   |                                +-----------------------+-----------------------+
   |                                | WADI                  | 0.37                  |
   +--------------------------------+-----------------------+-----------------------+
   |MSCRED (2020) [8]               | Synthetic data        | 0.89                  |
   |                                +-----------------------+-----------------------+
   |                                | Power Plant data      | 0.82                  |
   +--------------------------------+-----------------------+-----------------------+
   |OmniAnomaly (2019) [9]          | SMAP                  | 0.8434                |
   |                                +-----------------------+-----------------------+
   |                                | MSL                   | 0.8989                |
   |                                +-----------------------+-----------------------+
   |                                | SMD                   | 0.8857                |
   +--------------------------------+-----------------------+-----------------------+
   |THOC (2020) [10]                | 2D-gesture            | 0.6331                |
   |                                +-----------------------+-----------------------+
   |                                | power-demand          | 0.4568                |
   |                                +-----------------------+-----------------------+
   |                                | KDDCUP99              | 0.9886                |
   |                                +-----------------------+-----------------------+
   |                                | SWaT                  | 0.8809                |
   |                                +-----------------------+-----------------------+
   |                                | SMAP                  | 0.9518                |
   |                                +-----------------------+-----------------------+
   |                                | MSL                   | 0.9367                |
   +--------------------------------+-----------------------+-----------------------+
   |USAD (2020) [11]                | SWaT                  | 0.8460                |
   |                                +-----------------------+-----------------------+
   |                                | WADI                  | 0.4296                |
   |                                +-----------------------+-----------------------+
   |                                | SMAP                  | 0.8634                |
   |                                +-----------------------+-----------------------+
   |                                | MSL                   | 0.9272                |
   |                                +-----------------------+-----------------------+
   |                                | SMD                   | 0.9463                |
   +--------------------------------+-----------------------+-----------------------+

[1]   Reinartz, Christopher, Murat Kulahci, and Ole Ravn. "An extended Tennessee 
Eastman simulation dataset for fault-detection and decision support systems." 
Computers & Chemical Engineering 149 (2021): 107281

[2]   K. Choi, J. Yi, C. Park and S. Yoon, "Deep Learning for Anomaly Detection in 
Time-Series Data: Review, Analysis, and Guidelines," in IEEE Access, vol. 9, pp. 
120043-120065, 2021

[3]   Gen Li, Jason J. Jung, "Deep learning for anomaly detection in multivariate 
time series: Approaches, applications, and challenges," Information Fusion, Volume 
91, 2023

[4]   B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and
H. Chen, "Deep autoencoding Gaussian mixture model for unsupervised
anomaly detection," in Proc. Int. Conf. Learn. Represent., pp. 1–19, 2018

[5]   A. Deng and B. Hooi, "Graph neural network-based anomaly detection
in multivariate time series," in Proc. Conf. Artif. Intell. (AAAI), vol. 35, pp. 
4027–4035, 2021

[6]   Z. Chen, D. Chen, X. Zhang, Z. Yuan, and X. Cheng, "Learning graph structures 
with transformer for multivariate time series anomaly detection in IoT," 
arXiv:2104.03466, 2022

[7]   D. Li, D. Chen, B. Jin, L. Shi, J. Goh, and S.-K. Ng, "MAD-GAN: Multivariate 
anomaly detection for time series data with generative adversarial networks," in Proc. 
Int. Conf. Artif. Neural Netw. (ICANN). Munich, Germany: Springer, pp. 703–716, 2019

[8]   J. Sipple, "Interpretable, multidimensional, multimodal anomaly detection with 
negative sampling for detection of device failure," in Proc. Int. Conf. Mach. Learn., 
pp. 9016–9025, 2020

[9]   Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust anomaly detection 
for multivariate time series through stochastic recurrent neural network,"
in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discovery Data
Mining, pp. 2828–2837, 2019

[10]  L. Shen, Z. Li, and J. Kwok, "Timeseries anomaly detection using temporal 
hierarchical one-class network," in Proc. Adv. Neural Inf. Process. Syst., vol. 33, 
pp. 13016–13026, 2020

[11]  J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga,
"USAD: Unsupervised anomaly detection on multivariate time series,"
in Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining,
pp. 3395–3404, 2020
