# Stepfile Generation
## Generating Steps
Clone this repo into `StepMania/Songs/Generated/generation` to start generating stepfiles

To generate steps for songs add them to `/stepping_songs` and run
```
python Stepping.py --songs {name}.mp3 {name}.mp3...
```
If `--songs` is not specified all songs in `/stepping_songs` will be stepped


To generate steps with a certain number of steps, jumps etc run
```
python Stepping.py --class_ammounts {steps} {jumps} {hands} {holds} {rolls} {mines}
```
where each amount is the fraction of steps that will be of that type eg
```
python Stepping.py --class_ammounts 0.15 0.04 0.0 0.01 0.0 0.01
```
## Retraining Models
To retrain models start by running
```
python FilterTrainingSongs.py
python ExtractMusicFeatures.py
python ExtractStepfileFeatures.py
```
This will create a list of songs to train on from your downloaded packs and then generate some features from the music and stepfile for each (This takes a long time to run)

You can then run
```
python TrainNoteClassModel.py
```
or 
```
python TrainNoteModels.py
```
Once a model has been retrained, you can run `Stepping.py` on a song that has already been stepped with `--regenerate_note_classes` or `--regenerate_notes` to ignore previously generated steps
