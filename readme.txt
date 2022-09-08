In this repo we have:


- WavAndLabelCreator.py: This code creates a wav and Label raw objects. They are the processed audio (an ndArray) and the labels with "who talked when" (an array of 
touples)
- WavAndLabelProcessor.py: This code will process the wav and Label to for example, create a wav file for each person with all they said

- GenderDetectorTrainer: With this one we create the pre-trained-gender-detector
- GenderDetectionTester: With this one we test the gmm's with wav examples

- audio_res: File sources (some random mp3/wav files to test with)
- processesed_audio_res: We will put the processed audio here

- rawObjects: We will create the rawObjects here (Labelling and WAV)
- pre-trained-gender-detectors: Here we have the gmm's for the gender detection. It has an accurracy of about 80% with a relatively small training database



