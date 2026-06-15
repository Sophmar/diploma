"""
Модулі та допоміжні функції для класифікації відеоданих.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_env(env_path="key.env"):
    """Завантаження змінних середовища з файлу key.env."""
    if _load_dotenv is not None:
        _load_dotenv(env_path)

    if not os.getenv("KAGGLE_USERNAME"):
        raise ValueError("KAGGLE_USERNAME не вказаний в .env")

    if not os.getenv("KAGGLE_KEY"):
        raise ValueError("KAGGLE_KEY не вказаний в .env")

def get_env_int(name, default):
    return int(os.getenv(name, default))


def get_env_float(name, default):
    return float(os.getenv(name, default))
def get_dataset_path(env_var, kaggle_dataset):
    local_path = os.getenv(env_var)
    if local_path:
        return local_path
    import kagglehub
    return kagglehub.dataset_download(kaggle_dataset)


def sample_center_sequence_indices(total_frames, num_frames = 16):
    """
    Вибір num_frames послідовних кадрів із центральної частини відео.
    Якщо відео містить менше кадрів, ніж потрібно, останній кадр повторюється до досягнення необхідної довжини послідовності.
    """
    if total_frames <= 0:
        return [0] * num_frames

    if total_frames >= num_frames:
        start = (total_frames - num_frames) // 2
        return list(range(start, start + num_frames))

    indices = list(range(total_frames))
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices


def sample_tsn_frame_indices(total_frames, num_segments = 16, mode = "train"):
    """
    Рівномірний вибір кадрів за методом TSN: відео ділиться на сегменти,
    з кожного сегмента вибирається один кадр (випадковий під час навчання
    або центральний під час оцінювання).
    """
    if total_frames <= 0:
        return [0] * num_segments

    segment_edges = np.linspace(0, total_frames, num_segments + 1)
    indices = []
    for i in range(num_segments):
        start = int(segment_edges[i])
        end = int(segment_edges[i + 1])

        if end <= start:
            idx = min(start, total_frames - 1)
        elif mode == "train":
            idx = np.random.randint(start, end)
        else:
            idx = (start + end - 1) // 2
        indices.append(idx)
    return indices


def sample_frame_indices(total_frames, num_frames = 16):
    """
    Рівномірний вибір кадрів по всій довжині відео.
    """
    if total_frames <= 0:
        return [0] * num_frames
    if total_frames == 1:
        return [0] * num_frames
    return [int(round(i * (total_frames - 1) / (num_frames - 1))) for i in range(num_frames)]


class ClipTrainTransform:
    """
    Перетворення кадрів для навчальної вибірки.
    До всіх кадрів застосовуються однакові операції:
    зміна розміру до 256 пікселів, випадкове обрізання до size×size,
    випадкове горизонтальне віддзеркалення, перетворення у тензор
    та нормалізація за статистиками ImageNet.
    """
    def __init__(self, size = 112, mean = None, std = None):
        self.size = size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD

    def __call__(self, frames):
        do_flip = random.random() < 0.5
        resized_frames = [F.resize(img, 256) for img in frames]
        i, j, h, w = T.RandomCrop.get_params(resized_frames[0], output_size=(self.size, self.size))

        processed = []
        for img in resized_frames:
            img = F.crop(img, i, j, h, w)
            if do_flip:
                img = F.hflip(img)
            img = F.to_tensor(img)
            img = F.normalize(img, mean=self.mean, std=self.std)
            processed.append(img)
        return processed


class ClipTrainTransform2:
    """
    Перетворення кадрів для навчальної вибірки.
    До всіх кадрів застосовуються однакові операції:
    зміна розміру, випадкове обрізання, корекція яскравості,
    контрастності, насиченості та відтінку кольору, після чого
    виконується перетворення у тензор та нормалізація.
    """
    def __init__(self, size = 112, mean = None, std = None):
        self.size = size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.05

    def __call__(self, frames):
        resized_frames = [F.resize(img, 256) for img in frames]
        i, j, h, w = T.RandomCrop.get_params(resized_frames[0], output_size=(self.size, self.size))

        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        processed = []
        for img in resized_frames:
            img = F.crop(img, i, j, h, w)
            img = F.adjust_brightness(img, brightness_factor)
            img = F.adjust_contrast(img, contrast_factor)
            img = F.adjust_saturation(img, saturation_factor)
            img = F.adjust_hue(img, hue_factor)
            img = F.to_tensor(img)
            img = F.normalize(img, mean=self.mean, std=self.std)
            processed.append(img)
        return processed


class ClipEvalTransform:
    """
    Перетворення кадрів для валідаційної та тестової вибірок.
    До кожного кадру застосовуються зміна розміру до 256 пікселів,
    центральне обрізання до size×size, перетворення у тензор та
    нормалізація за статистиками ImageNet.
    """
    def __init__(self, size = 112, mean = None, std = None):
        self.size = size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD

    def __call__(self, frames):
        processed = []
        for img in frames:
            img = F.resize(img, 256)
            img = F.center_crop(img, (self.size, self.size))
            img = F.to_tensor(img)
            img = F.normalize(img, mean=self.mean, std=self.std)
            processed.append(img)
        return processed

def extract_frames_from_video(clip_path, output_dir, num_frames = 8, img_size = 112):
    """
    Виділення, масштабування та збереження фіксованої кількості
    рівномірно розподілених кадрів із відео.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, num_frames)

    saved_frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            fallback_idx = max(0, frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
            success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        saved_frames.append(frame)

    cap.release()

    if not saved_frames:
        return 0

    while len(saved_frames) < num_frames:
        saved_frames.append(saved_frames[-1].copy())

    for i, frame in enumerate(saved_frames[:num_frames]):
        out_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return min(len(saved_frames), num_frames)


def process_ucf_split(split, dataset_root, output_root, selected_classes, num_frames = 8, img_size = 112):
    """
    Обробка вибірки UCF101: фільтрація класів, виділення кадрів
    із відео та збереження їх у структурованій директорії.
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    df = pd.read_csv(dataset_root / f"{split}.csv")
    df = df[df["label"].isin(selected_classes)].reset_index(drop=True)

    print(f"{split}: {len(df)} відео після фільтрації")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        rel_clip_path = str(row["clip_path"])[1:]
        label = str(row["label"])
        clip_path = dataset_root / rel_clip_path
        video_id = clip_path.stem
        out_dir = output_root / split / label / video_id

        existing = list(out_dir.glob("frame_*.jpg"))
        if len(existing) == num_frames:
            continue

        if not clip_path.exists():
            print(f"[WARNING] Немає файлу: {clip_path}")
            continue

        saved = extract_frames_from_video(clip_path, out_dir, num_frames=num_frames, img_size=img_size)
        if saved != num_frames:
            print(f"[WARNING] {clip_path.name}: {saved}/{num_frames}")


def prepare_balanced_split(df, selected_classes, samples_per_class, test_size=0.3, random_state=42, label_col="label"):
    """
    Створює піднабір даних із вибраних класів
    та розбиває його на навчальну, валідаційну і тестову вибірки.
    Для кожного класу відбирається однакова кількість прикладів,
    після чого формуються нові числові мітки класів і виконується
    стратифікований поділ даних.
    """
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    df_small = df[df[label_col].isin(selected_classes)].copy()

    df_small = (
        df_small
        .groupby(label_col, group_keys=False)
        .sample(n=samples_per_class, random_state=random_state)
        .reset_index(drop=True)
    )

    df_small["new_label_id"] = df_small[label_col].map(class_to_idx)

    train_df, temp_df = train_test_split(
        df_small,
        test_size=test_size,
        stratify=df_small["new_label_id"],
        random_state=random_state,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["new_label_id"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return df_small, train_df, val_df, test_df, class_to_idx, idx_to_class


def print_split_info(df_small, train_df, val_df, test_df):
    """
    Виводить інформацію про сформовані вибірки:
    приклади записів, розподіл класів та розміри train, val і test.
    """
    print(df_small[["video_id", "label", "label_id", "new_label_id"]].head())
    print(df_small["new_label_id"].value_counts().sort_index())

    print(train_df["new_label_id"].value_counts().sort_index())
    print(val_df["new_label_id"].value_counts().sort_index())
    print(test_df["new_label_id"].value_counts().sort_index())

    print("Train_df shape:", train_df.shape)
    print("Val_df shape:", val_df.shape)
    print("Test_df shape:", test_df.shape)

class VideoFramesDataset(Dataset):
    """
    Dataset для роботи з попередньо збереженими кадрами відео набору даних UCF101.
    Для кожного відео зчитує відповідну папку з кадрами, перевіряє
    їх кількість, застосовує перетворення та повертає тензор відеокліпу
    разом із числовою міткою класу.
    """
    def __init__(self, dataframe, frames_root, num_frames = 8, transform=None, class_to_idx=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.frames_root = Path(frames_root)
        self.num_frames = num_frames
        self.transform = transform
        self.class_to_idx = class_to_idx or self._build_class_mapping()

    def _build_class_mapping(self):
        if "label" not in self.dataframe.columns:
            raise ValueError("У dataframe має бути колонка 'label'")
        classes = sorted(self.dataframe["label"].astype(str).unique())
        return {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        label_name = str(row["label"])
        video_id = Path(str(row["clip_path"])).stem
        clip_dir = self.frames_root / label_name / video_id
        frame_paths = sorted(clip_dir.glob("frame_*.jpg"))

        if len(frame_paths) != self.num_frames:
            raise ValueError(f"Очікувалось {self.num_frames} кадрів, але знайдено {len(frame_paths)} у {clip_dir}")

        frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
        if self.transform is not None:
            frames = self.transform(frames)

        clip = torch.stack(frames, dim=0)
        label = self.class_to_idx[label_name]
        return clip, label


class JesterDataset(Dataset):
    """
    Dataset для набору даних Jester.
    Зчитує кадри відео з відповідної папки, вибирає потрібну кількість
    кадрів центральним або рівномірним способом, застосовує перетворення
    та повертає тензор відеокліпу разом із міткою класу.
    """
    def __init__(self, df, root_dir, split="Train", num_frames=8, transform=None, mode="train", frame_division="center"):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode
        self.frame_division = frame_division

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = str(row["video_id"])
        label = int(row["new_label_id"])
        video_folder = os.path.join(self.root_dir, self.split, video_id)
        frame_files = sorted(os.listdir(video_folder))
        total_frames = len(frame_files)

        if self.frame_division == "center":
            frame_indices = sample_center_sequence_indices(total_frames, self.num_frames)
        elif self.frame_division == "uniform":
            frame_indices = sample_tsn_frame_indices(total_frames, self.num_frames, self.mode)
        else:
            raise ValueError("frame_division має бути 'center' або 'uniform'")

        frames = []
        for frame_idx in frame_indices:
            frame_path = os.path.join(video_folder, frame_files[frame_idx])
            frames.append(Image.open(frame_path).convert("RGB"))

        if self.transform is not None:
            frames = self.transform(frames)

        return torch.stack(frames), label


class VideoClassifierBase(nn.Module):
    """
    Базовий клас для моделей класифікації відеоданих.
    Реалізує спільну логіку навчання, валідації, ранньої зупинки,
    збереження найкращої моделі, отримання прогнозів та підрахунку параметрів.
    """

    def _init_training_state(self, lr=1e-4, patience=5, checkpoint_path="best_model.pth"):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.patience = patience
        self.checkpoint_path = checkpoint_path

    def fit(self, train_loader, val_loader=None, epochs=10):
        best_val_loss = float("inf")
        counter = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for clips, labels in train_loader:
                outputs = self(clips)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train loss: {avg_loss:.4f} | Train acc: {train_acc:.4f}"
            )

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)

                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(self.state_dict(), self.checkpoint_path)
                else:
                    counter += 1

                if counter > self.patience:
                    print("Рання зупинка.")
                    break

    def evaluate(self, loader):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for clips, labels in loader:
                outputs = self(clips)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(loader)
        acc = total_correct / total_samples

        return avg_loss, acc

    def predict(self, loader):
        self.eval()
        all_preds = []

        with torch.no_grad():
            for clips, _ in loader:
                outputs = self(clips)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.tolist())

        return all_preds

    def get_predictions_and_labels(self, loader):
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for clips, labels in loader:
                outputs = self(clips)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        return all_labels, all_preds

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_params, trainable_params


def build_resnet18_feature_extractor(freeze_backbone = False):
    """
    Створення екстрактора ознак на основі попередньо навченої ResNet18
    з можливістю заморожування параметрів згорткової частини.
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_dim = backbone.fc.in_features

    if freeze_backbone:
        for param in feature_extractor.parameters():
            param.requires_grad = False

    return feature_extractor, feature_dim


class CNNMeanPoolingClassifier(VideoClassifierBase):
    """
    Покадрова модель класифікації відео: ResNet18 + Mean Pooling + класифікатор.
    """
    def __init__(self, num_classes, lr=1e-4, patience=5, freeze_backbone=False, checkpoint_path="best_model.pth"):
        super().__init__()
        self.feature_extractor, self.feature_dim = build_resnet18_feature_extractor(freeze_backbone)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_training_state(lr=lr, patience=patience, checkpoint_path=checkpoint_path)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.flatten(1).view(B, T, self.feature_dim)
        video_features = features.mean(dim=1)
        return self.classifier(video_features)


class CNNLSTMClassifier(VideoClassifierBase):
    """
    Послідовна модель класифікації відео: ResNet18 + LSTM + класифікатор.
    """
    def __init__(self, num_classes, lr=1e-4, hidden_size=256, num_layers=1, dropout=0.2, patience=5, freeze_backbone=False, checkpoint_path="best_lstm_model.pth"):
        super().__init__()
        self.feature_extractor, self.feature_dim = build_resnet18_feature_extractor(freeze_backbone)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._init_training_state(lr=lr, patience=patience, checkpoint_path=checkpoint_path)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.flatten(1).view(B, T, self.feature_dim)
        _, (h_n, _) = self.lstm(features)
        video_features = self.dropout(h_n[-1])
        return self.classifier(video_features)


def load_video_frames(video_path, num_frames=24):
    """
    Завантаження та підготовка кадрів із відео.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = sample_frame_indices(total_frames, num_frames)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Не вдалося зчитати кадри з відео: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames


def predict_custom_video(video_path, model, transform, num_frames, idx_to_class, device="cpu"):
    """
    Класифікація окремого відеофайлу.
    """
    frames = load_video_frames(video_path, num_frames=num_frames)
    frames = transform(frames)

    clip = torch.stack(frames).unsqueeze(0)
    clip = clip.to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(clip)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_idx].item()

    return idx_to_class[pred_idx], confidence


def evaluate_custom_videos(root_dir, model, transform, num_frames, idx_to_class, selected_classes, device="cpu"):
    """
    Тестування моделі на власних відео.
    """
    root_dir = Path(root_dir)
    results = []

    for true_class in selected_classes:
        class_dir = root_dir / true_class

        if not class_dir.exists():
            print(f"Warning: folder not found: {class_dir}")
            continue

        video_files = (list(class_dir.glob("*.mp4")))

        for video_path in video_files:
            pred_class, confidence = predict_custom_video(video_path=video_path, model=model, transform=transform, num_frames=num_frames, idx_to_class=idx_to_class, device=device)

            results.append({
                "video": video_path.name,
                "true_class": true_class,
                "predicted_class": pred_class,
                "confidence": confidence,
                "correct": true_class == pred_class,
            })
    return pd.DataFrame(results)