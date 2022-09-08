In this repo we have:


- WavAndLabelCreator.py: This code creates a wav and Label raw objects. They are the processed audio (an ndArray) and the labels with "who talked when" (an array of 
touples)
- WavAndLabelProcessor.py: This code will process the wav and Label to for example, create a wav file for each person with all they said

- GenderDetectorTrainer: With this one we create the pre-trained-gender-detector
- GenderDetectionTester: With this one we test the gmm's with wav examples

- rawObjects: We will create the rawObjects here (Labelling and WAV)
- pre-trained-gender-detectors: Here we have the gmm's for the gender detection. It took about 6 min to train with 80% of the AudioSet database, and generated an accurracy of 0.87 in the 20% left.  


The data used will be in the following drive folder:

https://drive.google.com/drive/folders/1TpXdHmjglwLXyioNUKnr8edXmmdXblbW?usp=sharing

