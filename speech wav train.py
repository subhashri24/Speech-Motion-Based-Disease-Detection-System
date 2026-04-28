import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# -------------------------
# 📂 DATA PATH
# -------------------------
DATA_PATH = r"C:\Users\shris\PycharmProjects\speech and motion\dataset"

# -------------------------
# 🎧 FEATURE EXTRACTION
# -------------------------
def extract_features_from_audio(y, sr):
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc_mean, mfcc_std, chroma_mean, contrast_mean, zcr])


# -------------------------
# 🔄 AUDIO AUGMENTATION
# -------------------------
def augment_audio(y, sr):
    augmented = []

    # Original
    augmented.append(y)

    # Noise
    noise = 0.005 * np.random.randn(len(y))
    augmented.append(y + noise)

    # Pitch shift
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    augmented.append(y_pitch)

    return augmented


# -------------------------
# 📊 LOAD DATA
# -------------------------
X = []
y = []

print("🔍 Loading dataset...")

for label, folder in enumerate(["HC_AH", "PD_AH"]):
    folder_path = os.path.join(DATA_PATH, folder)

    if not os.path.exists(folder_path):
        print(f"❌ Missing folder: {folder_path}")
        continue

    files = os.listdir(folder_path)
    print(f"\n📁 {folder} → {len(files)} files")

    for file in files:
        if file.lower().endswith(".wav"):
            path = os.path.join(folder_path, file)

            try:
                y_audio, sr = librosa.load(path, duration=5)

                augmented_versions = augment_audio(y_audio, sr)

                for y_aug in augmented_versions:
                    features = extract_features_from_audio(y_aug, sr)
                    X.append(features)
                    y.append(label)

            except Exception as e:
                print("❌ Error:", path, e)

print("\n✅ Total samples after augmentation:", len(X))

# -------------------------
# ❌ SAFETY CHECK
# -------------------------
if len(X) == 0:
    raise ValueError("❌ No audio data loaded")

X = np.array(X)
y = np.array(y)

# -------------------------
# 🔀 TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# ⚖️ CLASS BALANCING
# -------------------------
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# -------------------------
# 🤖 MODEL (SVM)
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight=class_weights
    ))
])

print("\n🚀 Training model...")
pipeline.fit(X_train, y_train)

# -------------------------
# 📈 EVALUATION
# -------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", accuracy)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# 💾 SAVE MODEL
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/parkinson_model.pkl")

print("\n✅ Model saved → models/parkinson_model.pkl")