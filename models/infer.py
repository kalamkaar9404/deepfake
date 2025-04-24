{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7598d2f-76a5-445c-9362-6e7e036634d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from models.deepfake_detector import DeepfakeDetector\n",
    "from data.preprocess_frames import extract_frames\n",
    "from utils.attention_viz import visualize_attention\n",
    "\n",
    "model = DeepfakeDetector()\n",
    "model.eval()\n",
    "\n",
    "frames = extract_frames(\"data/sample_video.mp4\")\n",
    "output, attn_maps = model(frames[:10])\n",
    "print(\"Prediction Score:\", output.item())\n",
    "visualize_attention(attn_maps[0], frames[0])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
