.. _custom_data:

###########
Custom Data
###########

Users can independently add datasets according to the 
inheritance scheme. The user dataset should be described by a class 
that inherits from the base dataset class :ref:`base_dataset`. The class should 
implement the method **set_name_public_link**. The method does not take any 
parameters other than the standard self-parameter. The method should 
assign the name of the dataset to the class attribute **name** and 
assign a download link to the class attribute **public_link**. Currently, 
only download links from the cloud service 
`Yandex Disk <https://360.yandex.ru/disk/>`_ are supported. 
The download link should point to a **zip** archive that contains files 
specific to each task.

Fault Diagnosis
"""""""""""""""

Dataset should be provided as ZIP archive with a set of files in Comma-Separated 
Values (CSV) format. The set of files includes:
1. df.csv - a file with a table of sensor values at each point in time. The 
rows contain a sequence of values. The first two columns of the table are 
run_id and sample. The run_id column contains the run identifiers of the 
industrial process. The sample column contains the sequence element numbers. 
All other columns contain sensor values.
2. target.csv - file with the table of industrial process states. Table 
columns: run_id, sample, target. The run_id column contains the run identifier 
of the industrial process. The sample column contains the number of the 
sequence element. The target column contains the state number of the 
industrial process, where 0 is the normal state.
3. train_mask.csv - file with the table of the mask of the training 
sample. Columns of the table: run_id, sample, train_mask. The run_id column 
contains the identifier of the industrial process start. The sample column 
contains the sequence element number. The train_mask column contains the 
value of the training sample attribute, where 0 is not a training sample, 
1 is a training sample.

Anomaly Detection
"""""""""""""""""

Dataset should be provided in the same way as for the Fault Diagnosis task.

RUL
"""

HI 
""

Additional preprocessing steps
""""""""""""""""""""""""""""""

In the case of a structurally complex dataset or its parts that need to 
be processed simultaneously according to a specific rule, it is 
permissible to override the **_load** method with the attributes 
**num_chunks** and **force_download** in the user class that inherits 
from :ref:`base_dataset`. The **num_chunks** attribute is responsible 
for the number of loaded data fragments, and **force_download** controls 
the mandatory download of the file from the cloud even if the directory 
contains a folder with the data.