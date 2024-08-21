.. _custom_model:

############
Custom Model
############

Users can independently add models according to the inheritance scheme.
The user model should be described by a class that inherits from one of
the base classes of the task model: :ref:`base_ad`, :ref:`base_fd`,
:ref:`base_hi`, :ref:`base_rul`. The class should implement two methods:
**__init__** and **_create_model**.

The **__init__** method should include parameters of the base task class, as well
as parameters specific to this model, such as **hidden_dim** for an 
:ref:`mlp` model. The method should call the **__init__** method of the
parent class, passing all necessary parameters.

The **_create_model** method should include the parameters **input_dim**
and **output_dim**, which define the number of sensors and the number of
output values, respectively. For :ref:`task_fd` tasks, the number of 
output values corresponds to the number of faults, while for other tasks,
this number equals 1. The method should assign an object of the 
`PyTorch <https://pytorch.org/>`_ neural network model of type **torch.nn.Module** to the model 
attribute of the class.

The source code of the user model should be located in the directory 
**ice/task_name/models**, where **task_name** should be replaced with 
the specific task being solved, for example, **fault_diagnosis**.