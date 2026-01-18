# Emoji Classification - Kaggle Competition

This project focuses on fine-tuning state-of-the-art vision models for emoji classification. It employs a sophisticated multi-stage pipeline involving transfer learning, hybrid model architectures, k-fold ensembling, and meta-learning stacking.

## Project Architecture

The project follows a 4-phase training strategy and a robust inference pipeline, as detailed in the architectural schemas.

### 1. Training Pipeline

The training process is divided into four distinct phases:

* **Phase 1: Pre-training**: Leveraging external HuggingFace datasets to pre-train five different architectures (EfficientNet B0/B1/B2, ConvNeXt V2, and DINOv2) on a mapped set of labels.
* **Phase 2: K-Fold Training**: Implementing a stratified 3-fold split on the target dataset. The models use a **Hybrid Architecture** that combines visual features from a backbone with image metadata (size, alpha channel presence).
* **Phase 3: Meta-Learning**: Generating Out-Of-Fold (OOF) predictions and extracting image statistics (color, brightness, etc.) to train a **LightGBM Stacking Model**.
* **Phase 4: Pseudo-Labeling**: Using the meta-model to predict labels for the test set, filtering for high-confidence predictions (>0.95), and re-training the ensemble with the combined dataset.

### 2. Inference Pipeline

A robust inference flow ensures maximum accuracy:

* **Multi-Scale TTA**: 12 variations per image including flips, rotations, and various crops.
* **Ensemble Inference**: Averaging predictions from 15 trained models (5 architectures × 3 folds).
* **Feature Engineering**: Injecting statistical features (Width, Height, RGB mean/std, Brightness, etc.) into the final stacking layer.
* **Meta-Model Stacking**: A final LightGBM model processes the concatenated model probabilities and statistical features for the final prediction.

## Tech Stack

* **Deep Learning**: PyTorch, Torchvision, HuggingFace Transformers, Accelerate
* **Machine Learning**: Scikit-learn, LightGBM, XGBoost
* **Data Processing**: Pandas, NumPy, Pillow, Datasets
* **Visualization**: Matplotlib, Seaborn
* **Utilities**: Kagglehub, tqdm

## Project Structure

* `finetune_swin_emoji_V*.ipynb`: Iterative development notebooks (V1 to V17).
* `V16_Architecture_Schemas.md`: Detailed Mermaid diagrams of the V16 architecture.
* `translator.py`: Utility script for CSV label normalization.
* `visualisations.ipynb`: Notebook for data exploration and result visualization.
* `requirements.txt`: Environment dependencies.
* `data/`: Directory containing training and testing datasets.

## Key Features

* **Hybrid Models**: Integration of visual backbones with metadata features.
* **Cross-Architecture Ensemble**: Combining CNNs (EfficientNet) with Transformers (Swin, DINOv2, ConvNeXt).
* **Advanced Data Augmentation**: Including Test Time Augmentation (TTA) for robust inference.
* **Stacking Generalization**: Using LightGBM as a meta-classifier to learn how to best combine base model predictions.
* **Semi-Supervised Learning**: Improving performance through iterative pseudo-labeling.

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```
2. **Run Notebooks**: Start with `finetune_swin_emoji_V16.ipynb` for the most efficient implementation.

## Zoom on V16 Approach

### V16 Training Approach & Architecture

This schema outlines the comprehensive training strategy implemented in `finetune_swin_emoji_V16.ipynb`. It highlights the **Hybrid Architecture** (combining visual backbones with metadata), the **Transfer Learning** flow (HuggingFace $\to$ Target), **Ensembling** (K-Fold), and **Semi-Supervised Learning** (Pseudo-labeling).

```mermaid

flowchart TD
    %% Global Styles
    classDef dataset fill:#e1f5fe,stroke:#01579b,color:black;
    classDef process fill:#fff3e0,stroke:#ff6f00,color:black;
    classDef model fill:#e8f5e9,stroke:#2e7d32,color:black;
    classDef artifact fill:#f3e5f5,stroke:#7b1fa2,color:black;
    classDef meta fill:#fff9c4,stroke:#fbc02d,color:black;

    subgraph "Phase 1: Pre-training"
        HF_DS[("HuggingFace Dataset\n(11 Classes)")]:::dataset --> Mapping[("Label Mapping\n(11 -> 7 Classes)")]:::process
        Mapping --> PreTrain{{"Pre-train 5 Architectures"}}:::process

    subgraph "Architectures"
            M1[EfficientNet B0]:::model
            M2[EfficientNet B1]:::model
            M3[EfficientNet B2]:::model
            M4[ConvNeXt V2]:::model
            M5[DINOv2]:::model
        end

    PreTrain --> M1 & M2 & M3 & M4 & M5
        M1 & M2 & M3 & M4 & M5 --> PT_Weights[("Pre-trained Weights")]:::artifact
    end

    subgraph "Phase 2: K-Fold Training"
        Target_DS[("Target Dataset\n(Train.csv)")]:::dataset --> KFold{{"Stratified K-Fold Split\n(3 Folds)"}}:::process
        PT_Weights --> FoldLoop
        KFold --> FoldLoop

    subgraph "Training Loop (Per Fold & Model)"
            FoldLoop[("For Each Fold (1-3)")]

    subgraph "Hybrid Model Structure"
                Input[Image] --> Backbone[("Visual Backbone\n(Frozen -> Unfrozen)")]:::model
                Input --> MetaExtract[("Metadata Extraction\n(Size, Alpha)")]:::meta

    Backbone --> VisualFeats[Pooled Features]
                MetaExtract --> MetaFeats[Metadata Vector]

    VisualFeats & MetaFeats --> Fusion{{"Fusion Layer\n(Concat)"}}:::process
                Fusion --> Head[("Classifier Head\n(Label Smooth CrossEntropy)")]:::model
            end
        end

    FoldLoop --> TrainedModels[("15 Trained Models\n(5 Archs x 3 Folds)")]:::artifact
    end

    subgraph "Phase 3: Meta-Learning"
        TrainedModels --> FeatGen{{"Generate Probabilities\n(OOF Predictions)"}}:::process
        Target_DS --> ImageStats[("Extract Image Statistics\n(Color, Brightness, etc.)")]:::meta

    FeatGen & ImageStats --> MetaFeatures[("Meta-Feature Matrix")]:::artifact
        MetaFeatures --> LGBM_Train{{"Train LightGBM\nStacking Model"}}:::process
        LGBM_Train --> MetaModel[("LightGBM Meta-Model")]:::artifact
    end

    subgraph "Phase 4: Pseudo-Labeling Loop"
        Test_DS[("Test Dataset")]:::dataset --> Inf_PL{{"Inference w/ Meta-Model"}}:::process
        MetaModel --> Inf_PL
        Inf_PL --> ConfCheck{{"Confidence > 0.95?"}}:::process
        ConfCheck -- Yes --> PseudoLabels[("Pseudo-Labeled Data")]:::dataset
        PseudoLabels --> Combine{{"Combine with Train Data"}}:::process
        Target_DS --> Combine
        Combine --> Retrain{{"Retrain Models\n(Fine-tune)"}}:::process
        Retrain --> UpdatedModels[("Final 15 Models")]:::artifact
    end
```

### Final Inference Pipeline

This schema details the flow of a single test image through the complex inference pipeline, including **Multi-Scale Test Time Augmentation (TTA)**, **Feature Engineering**, and the **Stacking Ensemble**.

```mermaid

flowchart TD
    %% Styles
    classDef input fill:#e1f5fe,stroke:#01579b,color:black;
    classDef tta fill:#f3e5f5,stroke:#7b1fa2,color:black;
    classDef model_inf fill:#e8f5e9,stroke:#2e7d32,color:black;
    classDef features fill:#fff9c4,stroke:#fbc02d,color:black;
    classDef final fill:#fff3e0,stroke:#ff6f00,color:black;

    InputImage[("Test Image")]:::input --> StatsExtract[("1. Statistical Extraction")]:::features
    InputImage --> TTA[("2. Test Time Augmentation (TTA)")]:::tta

    subgraph "Feature Engineering (Stats)"
        StatsExtract --> StatFeats["Width, Height, Aspect Ratio`<br/>`Pixel Count, Mean/Std (RGB)`<br/>`Brightness, White%, Mode"]:::features
    end

    subgraph "Multi-Scale TTA (12 Variations)"
        TTA --> Aug1[Original]
        TTA --> Aug2[H-Flip]
        TTA --> Aug3[V-Flip]
        TTA --> Aug4["Rotations (+/- 5, 10)"]
        TTA --> Aug5[Corner Crops]
        TTA --> Aug6[Center Crop]
    end

    subgraph "Ensemble Inference (15 Models)"
        Aug1 & Aug2 & Aug3 & Aug4 & Aug5 & Aug6 --> Batching{{"Batch Processing"}}

    Batching --> Models_EffNet["EfficientNet (B0, B1, B2)`<br/>`(x3 Folds)"]:::model_inf
        Batching --> Models_Conv["ConvNeXt V2`<br/>`(x3 Folds)"]:::model_inf
        Batching --> Models_Dino["DINOv2`<br/>`(x3 Folds)"]:::model_inf

    %% Metadata injection into models
        %%MetaInput["Metadata (Norm Size + Alpha)"] -.-> Models_EffNet
        %%MetaInput -.-> Models_Conv
        %%MetaInput -.-> Models_Dino
    end

    Models_EffNet & Models_Conv & Models_Dino --> RawProbs["Raw Probabilities`<br/>`(15 Models x 12 Augs x 7 Classes)"]:::features

    subgraph "Meta-Model Stacking"
        RawProbs --> Flatten{{"Flatten & Concat"}}
        StatFeats --> Flatten

    Flatten --> FeatureMatrix[("Final Feature Matrix`<br/>`(~1200+ features)")]:::features
        FeatureMatrix --> LGBM[("LightGBM Meta-Model")]:::final
    end

    LGBM --> Softmax{{"Softmax"}}
    Softmax --> FinalPred[("Final Class Prediction")]:::final
    Softmax --> Confidence[("Confidence Score")]:::final
```
