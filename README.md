## How to setup Google Colab

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUser/YourRepo/blob/main/notebook.ipynb) 

**Requirements** To use Colab, you must have a Google account with an associated Google Drive.

**Reminder** Ressources on colab are not guaranteed and therefore there might be times where some ressources cannot get allocated. If you're idle for 90 minutes or your connection time exceeds the maximum of 12 hours, the colab virtual machine will disconnect. This means that unsaved progress such as model parameters are lost.

**GPU access** Before you run the notebook, you need to enable GPU access. This is done by clicking on the "Runtime" menu and then on "Change runtime type". You can then select "GPU" and click on "Save".

**Upload the data**
You need to upload your data on the remote machine. Make sure to upload the zip and not the extracted data, this is significantly faster. There are two primary options:  
You can upload the data directly into colab (Click on the Files icon on the left side and then on upload). This is the most straightforward way, but you have to do it every time you start a new colab session.  
The second option is to use Google Drive and import the data from there into Colab.
First, you need to upload the provided zip file (data.zip) to your Google Drive. Next, you mount your Google Drive on the remote machine. In order to do so, you can use the cell below.  
In all case you then have to execute the "extract data" cell to unpack the zip file (You might have to change the path_to_zip variable).  
Use the "verify" cell to make sure that the data is accessible.
