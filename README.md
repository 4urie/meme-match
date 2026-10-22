#  Meme Matcher - Real-time Facial Expression to Meme Matching

  A real-time computer vision application that matches your facial expressions and hand gestures to famous internet memes using MediaPipe's face and hand detection.

##  What It Does

Point your webcam at yourself, make different facial expressions and hand gestures, and watch as the app finds the meme that best matches your expression in real-time! The matched meme appears side-by-side with your camera feed.

##  Features

- **Real-time Face Detection**: Uses MediaPipe Face Landmarker to track 478 facial landmarks
- **Hand Gesture Detection**: Tracks hand positions to distinguish similar expressions (e.g., Leo's cheers vs Disaster Girl's smirk)
- **Advanced Expression Analysis**:
  - Eye openness (surprise, wide eyes)
  - Eyebrow position (raised, furrowed)
  - Mouth shape (smiling, open, concerned)
  - Hand gestures (raised hands, fist pumps)
- **Smart Matching Algorithm**: Weighted similarity scoring with exponential decay for accurate matching
- **7 Meme Images**: Three human face matches and four expression or gesture matches in the current assets folder


##  Installation

### Prerequisites

- Python 3.11
- Webcam

### Setup

1. Clone the repository:
```bash
git clone 
cd make_me_a_meme
```

2. Install dependencies:
```bash
pip install mediapipe opencv-python numpy
```

3. Run the application:
```bash
python3 main.py
```

The first time you run it, the app will automatically download the required MediaPipe models (~7MB total).

## How to Use

The current image set includes four direct matches: point with your index finger
for **Cat Pointing Laugh**, smile with your mouth mostly closed for **Cat Smile**,
smile with your eyes closed for **Speed My Mom Is Kinda Homeless Face**, or
cover your mouth with your hand for **Cover Mouth**. The other memes use facial
feature similarity.

1. Run `python3 main.py`
2. Your webcam will activate
3. Make different expressions and gestures. The app shows "No close match" when the best human match is too weak or ambiguous.
4. Press **'d'** to show live smile, blink, and eye measurements when tuning the Speed expression. Press **'q'** to quit.

### Improve matching with your own examples

Record about 20-30 varied examples for each meme you want to recognize, plus
neutral expressions. Face the camera, make the expression, and press **Space**
to save each example. Move or change your expression slightly between saves.
Only numeric face and hand features are stored; camera images are not saved.

```bash
python main.py --collect "Cat Pointing Laugh"
python main.py --collect "Cat Smile"
python main.py --collect "Cover Mouth"
python main.py --collect "Speed My Mom Is Kinda Homeless Face"
python main.py --collect "Can Wearing Hodie"
python main.py --collect "Joshua Garcia"
python main.py --collect "Snake Arm Meme"
python main.py --collect neutral
python main.py --calibrate
python main.py --evaluate
python main.py
```

The examples go into `match_samples.jsonl`, and calibration writes
`match_profile.json`. Calibration starts working with 3 examples for one meme;
for useful weight and rejection tuning, collect examples for multiple human
memes and at least 3 neutral expressions. Run `--calibrate` again after adding
examples. If you change the feature extraction code, old examples are ignored
and need to be collected again.
The evaluation reports matches on the same examples used for calibration, so
try fresh expressions on camera to judge how well it works in practice.

## How It Works

### 1. Face & Hand Detection
- Uses MediaPipe Face Landmarker (478 landmarks per face)
- Uses MediaPipe Hand Landmarker (21 landmarks per hand, up to 2 hands)
- Detects facial features and hand positions in real-time

### 2. Feature Extraction
For each frame, the app calculates:
- **Eye features**: Openness, symmetry
- **Eyebrow features**: Height, position relative to eyes
- **Mouth features**: Openness, width ratio, elevation
- **Hand features**: Number of hands, raised/lowered position
- **Expression scores**: Surprise, smile, concern, cheers

### 3. Similarity Matching
- Compares your features against pre-loaded meme features
- Uses weighted exponential decay scoring
- Higher weights for distinctive features (cheers score: 30 points, hand_raised: 25 points)
- Rejects weak or ambiguous matches and waits for a stable result before switching memes

##  Contributing

Feel free to:
- Add more memes (with hand gestures for better accuracy!)
- Improve the matching algorithm
- Enhance the UI
- Optimize performance

##  License

This project is for educational and entertainment purposes.

## Credits

- **MediaPipe**: Google's ML framework for face and hand detection
- **Meme Images**: Fair use, iconic internet memes
- **OpenCV**: Open source computer vision library

##  Future Improvements

- [ ] Add more memes (target: 10-15)
- [ ] GUI for meme selection
- [ ] Save matched screenshots
- [ ] Expression history/statistics
- [ ] Multiple face support
- [ ] Custom meme upload via UI
- [ ] Performance optimizations
- [ ] Mobile app version

---

