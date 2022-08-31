# wildai-fish-detection-project
This is a project for detecting fish for the wildlife.ai organization.

We trained a 'FasterRCNN' on the data wildlife.ai supplied us to detect several species.
As well, we used methods like augmentations, style transfer to improve our model results.

We have two dataset, one is smaller which called 'data' and the second dataset is called data_0808 which contains larger dataset and more images.

For the results of the project and more general information you can look on the report -https://docs.google.com/document/d/1me7qQJ8Ltosk5hY5faO3QiFOmuirvS71/edit?usp=sharing&ouid=100668548382721803471&rtpof=true&sd=true

General information about the files in the project:

1. In order to train the model you should run the train.py with the required parameters.
2. To test the model you should run the file test.py
3. In order to load the data and transform the data you should run the SpyFishAotearoaDatahandler - this should create a folder with the formatted data and the CSV files.
4. To get general information about the data you should run the file utils/eda_lib.py
5. To transfer the data style you need to use the utils/transfer_dataset
6. To plot the images with the labels you should run the file plot_boxes_on_images.py

For more information you can contact us:
Adi Gabay - adi.gabay@mail.huji.ac.il
Ohad Tayler - ohad.tayler@mail.huji.ac.il

We want to thank the organization wildlife.ai, Eran Paz and Victor Anton for the help in the project