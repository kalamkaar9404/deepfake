{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "53fd74a4-7823-4719-80a7-779329a7ff99",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torchvision'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 4\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtorch\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtorch\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mnn\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mnn\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtorchvision\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mmodels\u001b[39;00m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mtransformers\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m ViTModel\n\u001b[1;32m      7\u001b[0m \u001b[38;5;28;01mclass\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mCNNTransformerAttention\u001b[39;00m(nn\u001b[38;5;241m.\u001b[39mModule):\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'torchvision'"
     ]
    }
   ],
   "source": [
    "# EfficientNet + Vision Transformer block for explainable attention\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torchvision.models as models\n",
    "from transformers import ViTModel\n",
    "\n",
    "class CNNTransformerAttention(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(CNNTransformerAttention, self).__init__()\n",
    "        self.efficientnet = models.efficientnet_b0(pretrained=True)\n",
    "        self.efficientnet.classifier = nn.Identity()\n",
    "\n",
    "        self.transformer = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')\n",
    "\n",
    "        self.fc = nn.Linear(self.transformer.config.hidden_size, 512)\n",
    "\n",
    "    def forward(self, x):\n",
    "        cnn_features = self.efficientnet(x)  # shape: (B, 1280)\n",
    "        vit_input = nn.functional.interpolate(x, size=(224, 224))\n",
    "        transformer_outputs = self.transformer(pixel_values=vit_input)\n",
    "        transformer_features = transformer_outputs.last_hidden_state[:, 0, :]  # CLS token\n",
    "        x = self.fc(transformer_features)\n",
    "        return x, transformer_outputs.attentions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4fa6122c-9e36-4324-88e1-2a74b5094258",
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
