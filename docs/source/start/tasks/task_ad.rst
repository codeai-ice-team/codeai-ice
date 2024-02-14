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

[1]   Reinartz, Christopher, Murat Kulahci, and Ole Ravn. "An extended Tennessee 
Eastman simulation dataset for fault-detection and decision support systems." 
Computers & Chemical Engineering 149 (2021): 107281

[2]   K. Choi, J. Yi, C. Park and S. Yoon, "Deep Learning for Anomaly Detection in 
Time-Series Data: Review, Analysis, and Guidelines," in IEEE Access, vol. 9, pp. 
120043-120065, 2021

[3]   Gen Li, Jason J. Jung, “Deep learning for anomaly detection in multivariate 
time series: Approaches, applications, and challenges,” Information Fusion, Volume 
91, 2023