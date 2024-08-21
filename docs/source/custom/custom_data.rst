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

In the case of a structurally complex dataset or its parts that need to 
be processed simultaneously according to a specific rule, it is 
permissible to override the **_load** method with the attributes 
**num_chunks** and **force_download** in the user class that inherits 
from :ref:`base_dataset`. The **num_chunks** attribute is responsible 
for the number of loaded data fragments, and **force_download** controls 
the mandatory download of the file from the cloud even if the directory 
contains a folder with the data.