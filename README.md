# deepfake
```
deepfake_attention_detector/
│
├── data/
│   ├── sample_video.mp4                
│   └── preprocess_frames.py           # preprocess frames from videos
│
├── models/
│   ├── cnn_transformer.py             # EfficientNet + Vision Transformer hybrid feature extractor
│   ├── lstm_temporal.py               # LSTM for temporal analysis
│   └── deepfake_detector.py           # Full model combining spatial + temporal modules
│
├── utils/
│   ├── attention_viz.py               # visualizing transformer attention maps
│   └── train_utils.py                 # Training and evaluation 
│
├── train.py                           # Training script
├── infer.py                           # Inference script for testing on a video
├── requirements.txt                   # Required packages
└── README.md                          # Project description and usage
```
