{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d22b9005-77ca-477b-bd1f-1bfb5425c088",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class TemporalLSTM(nn.Module):\n",
    "    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2):\n",
    "        super(TemporalLSTM, self).__init__()\n",
    "        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)\n",
    "        self.classifier = nn.Linear(hidden_dim, 1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        lstm_out, _ = self.lstm(x)\n",
    "        output = self.classifier(lstm_out[:, -1, :])\n",
    "        return torch.sigmoid(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fe58d34-230c-4b2d-a6b7-8bf5461ed30d",
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
