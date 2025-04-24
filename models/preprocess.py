{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f307e042-a928-47e7-9b91-7f00f4d43c4f",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'cv2'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mcv2\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mos\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtorch\u001b[39;00m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'cv2'"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import os\n",
    "import torch\n",
    "from torchvision import transforms\n",
    "from PIL import Image\n",
    "\n",
    "def extract_frames(video_path, frame_rate=5):\n",
    "    vidcap = cv2.VideoCapture(video_path)\n",
    "    frames = []\n",
    "    transform = transforms.Compose([\n",
    "        transforms.ToPILImage(),\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor()\n",
    "    ])\n",
    "    count = 0\n",
    "    while True:\n",
    "        success, image = vidcap.read()\n",
    "        if not success:\n",
    "            break\n",
    "        if count % frame_rate == 0:\n",
    "            frame_tensor = transform(image)\n",
    "            frames.append(frame_tensor)\n",
    "        count += 1\n",
    "    return torch.stack(frames)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d786f227-9aba-42bd-acb8-0582271152b0",
   "metadata": {},
   "outputs": [],
   "source": []
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
