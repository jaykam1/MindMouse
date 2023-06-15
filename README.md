To run this program, install the required dependencies (in requirements.txt) and run main.py

By default, it will use the most recent model I trained on myself with 12,553 datapoints giving 96% validation-set accuracy (eye_tracking_modelB4-20.h5).

If you would like to train the model on yourself follow the below instructions:

1) Run datacollection.py, a video feed will pop up and will collect your left and right eye and facial landmarks and your current cursor position
2) Move your cursor around your screen, keeping your eyes fixed on the cursor (move your head as well but only naturally)
3) When you have finished (recommended to do this for ~10 minutes for 10,000+ datapoints) press 'q'
4) Run model.py and change the text inside model.save() to 'NAME_YOUR_MODEL.h5'
5) Run main.py and change the text inside load_model() to 'NAME_YOUR_MODEL.h5'

You should now be able to use the model created to control your cursor. You can blink to click and press caps lock to pause tracking.
