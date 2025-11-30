"""
CIFAR-10 Image Classification Application using TensorFlow and Streamlit.

This application provides a web interface for:
- Classifying images using a pre-trained CNN model
- Training and fine-tuning the model with custom parameters
- Visualizing training progress and metrics with TensorBoard

The CNN architecture includes Conv2D layers with BatchNormalization,
LeakyReLU activations, MaxPooling, and Dense layers for classification.

Usage:
    streamlit run main.py
"""

from __future__ import annotations

import os
import socket
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from numpy.typing import NDArray
from PIL import Image

# =============================================================================
# Constants
# =============================================================================

MODEL_CHECKPOINT_PATH: str = "model_checkpoint.keras"
TRAINING_HISTORY_PATH: str = "training_history.csv"
TENSORBOARD_LOG_DIR: str = "./logs"
IMAGE_SIZE: tuple[int, int] = (32, 32)
CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)
SUPPORTED_IMAGE_TYPES: list[str] = ["jpg", "jpeg", "png", "webp"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration dataclass for model training parameters.

    Attributes:
        learning_rate: The learning rate for the optimizer.
        epochs: Number of training epochs.
        batch_size: Size of training batches.
        optimizer_name: Name of the optimizer to use.
        early_stopping_patience: Patience for early stopping callback.
        reduce_lr_patience: Patience for learning rate reduction callback.
    """

    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 32
    optimizer_name: str = "Adam"
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 15


# =============================================================================
# Data Loader
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of the CIFAR-10 dataset.

    This class provides methods to load the CIFAR-10 dataset with caching
    to avoid reloading on every Streamlit rerun.
    """

    @staticmethod
    @st.cache_data(show_spinner="📦 Loading CIFAR-10 dataset...")
    def load_cifar10() -> tuple[NDArray[np.float32], NDArray[np.uint8],
                                 NDArray[np.float32], NDArray[np.uint8]]:
        """Load and normalize the CIFAR-10 dataset.

        Returns:
            A tuple containing (x_train, y_train, x_test, y_test) arrays.
            Images are normalized to [0, 1] range as float32.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        return x_train, y_train, x_test, y_test


# =============================================================================
# Model Builder
# =============================================================================

class ModelBuilder:
    """Handles CNN model construction and optimizer creation.

    This class provides static methods for building the CNN architecture
    and creating optimizer instances.
    """

    @staticmethod
    def build_cnn() -> tf.keras.Model:
        """Build the CNN model architecture for CIFAR-10 classification.

        Returns:
            A compiled Keras Sequential model with CNN architecture.
        """
        model = tf.keras.models.Sequential([
            # Input Layer
            tf.keras.layers.Input(shape=(32, 32, 3)),

            # First Conv Block
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Second Conv Block
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Third Conv Block
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope=0.1),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])

        # Default compilation - will be recompiled before training
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
        """Create an optimizer instance by name.

        Args:
            name: Name of the optimizer (Adam, SGD, RMSprop, etc.).
            learning_rate: Learning rate for the optimizer.

        Returns:
            A Keras optimizer instance.

        Raises:
            ValueError: If the optimizer name is not supported.
        """
        optimizers: dict[str, type[tf.keras.optimizers.Optimizer]] = {
            "Adam": tf.keras.optimizers.Adam,
            "SGD": tf.keras.optimizers.SGD,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "Adagrad": tf.keras.optimizers.Adagrad,
            "Adadelta": tf.keras.optimizers.Adadelta,
            "FTRL": tf.keras.optimizers.Ftrl,
            "Nadam": tf.keras.optimizers.Nadam
        }

        if name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {name}")

        return optimizers[name](learning_rate=learning_rate)


# =============================================================================
# TensorBoard Manager
# =============================================================================

class TensorBoardManager:
    """Manages TensorBoard subprocess lifecycle.

    This class provides methods to start and stop TensorBoard server,
    ensuring proper cleanup of subprocess resources.

    Attributes:
        process: The subprocess.Popen instance for TensorBoard.
        port: The port number TensorBoard is running on.
    """

    def __init__(self) -> None:
        """Initialize TensorBoardManager with no active process."""
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None

    def _get_free_port(self) -> int:
        """Find an available port for TensorBoard.

        Returns:
            An available port number.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start(self, log_dir: str = TENSORBOARD_LOG_DIR) -> str:
        """Start TensorBoard server.

        Args:
            log_dir: Directory containing TensorBoard logs.

        Returns:
            The URL where TensorBoard is accessible.
        """
        # Stop existing process if running
        self.stop()

        self.port = self._get_free_port()
        self.process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--host', 'localhost', '--port', str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return f"http://localhost:{self.port}"

    def stop(self) -> None:
        """Stop the TensorBoard server if running."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.port = None

    def is_running(self) -> bool:
        """Check if TensorBoard is currently running.

        Returns:
            True if TensorBoard is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None


# =============================================================================
# Main Application
# =============================================================================

class ImageClassifierApp:
    """Main Streamlit application for CIFAR-10 image classification.

    This class orchestrates the entire application including:
    - Model loading and training
    - Image upload and classification
    - Training visualization
    - TensorBoard integration

    Attributes:
        tensorboard_manager: Manager for TensorBoard subprocess.
    """

    def __init__(self) -> None:
        """Initialize and run the Image Classifier application."""
        st.set_page_config(
            page_title="🖼️ Image Classification App",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._init_session_state()
        self._init_tensorboard_manager()
        self._init_environment()
        self._load_data()
        self._load_model()
        self._run_app()

    def _init_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'training_history' not in st.session_state:
            st.session_state.training_history = None
        if 'dataset_loaded' not in st.session_state:
            st.session_state.dataset_loaded = False

    def _init_tensorboard_manager(self) -> None:
        """Initialize TensorBoard manager in session state."""
        if 'tensorboard_manager' not in st.session_state:
            st.session_state.tensorboard_manager = TensorBoardManager()
        self.tensorboard_manager = st.session_state.tensorboard_manager

    def _init_environment(self) -> None:
        """Initialize environment settings and display configuration sidebar."""
        st.sidebar.title("⚙️ Configuration")
        use_onednn = st.sidebar.checkbox(
            "Enable ONEDNN Optimizations",
            value=True,
            help="TF_ENABLE_ONEDNN_OPTS=1"
        )
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' if use_onednn else '0'

        status_emoji = "✅" if use_onednn else "❌"
        status_text = "enabled" if use_onednn else "disabled"
        st.sidebar.caption(f"{status_emoji} ONEDNN optimizations {status_text}")

    def _load_data(self) -> None:
        """Load CIFAR-10 dataset using cached DataLoader."""
        if not st.session_state.dataset_loaded:
            data = DataLoader.load_cifar10()
            st.session_state.x_train = data[0]
            st.session_state.y_train = data[1]
            st.session_state.x_test = data[2]
            st.session_state.y_test = data[3]
            st.session_state.dataset_loaded = True
            st.sidebar.success("✅ Dataset loaded successfully!")

    def _load_model(self) -> None:
        """Load or build the CNN model."""
        if st.session_state.model is None:
            if os.path.exists(MODEL_CHECKPOINT_PATH):
                st.sidebar.info("📂 Loading pre-trained model...")
                st.session_state.model = tf.keras.models.load_model(MODEL_CHECKPOINT_PATH)
                st.sidebar.success("✅ Model loaded!")
            else:
                st.sidebar.info("🔨 Building new model...")
                st.session_state.model = ModelBuilder.build_cnn()
                st.sidebar.success("✅ Model built!")

    def _train_model(self, config: TrainingConfig) -> None:
        """Train the model with the given configuration.

        Args:
            config: TrainingConfig instance with training parameters.
        """
        try:
            model = st.session_state.model
            if model is None:
                st.error("❌ Model is not initialized!")
                return

            # Recompile model with selected optimizer
            optimizer = ModelBuilder.get_optimizer(config.optimizer_name, config.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=MODEL_CHECKPOINT_PATH,
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=config.early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=TENSORBOARD_LOG_DIR,
                    histogram_freq=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.1,
                    patience=config.reduce_lr_patience
                )
            ]

            st.sidebar.info("🏋️ Training model...")

            history = model.fit(
                st.session_state.x_train,
                st.session_state.y_train,
                epochs=config.epochs,
                validation_data=(st.session_state.x_test, st.session_state.y_test),
                callbacks=callbacks,
                batch_size=config.batch_size
            )

            # Save training history
            st.session_state.training_history = history.history
            pd.DataFrame(history.history).to_csv(TRAINING_HISTORY_PATH, index=False)

            st.sidebar.success("✅ Training complete!")
            self._display_training_history()

        except Exception as e:
            st.error(f"❌ Training error: {e}")

    def _display_training_history(self) -> None:
        """Display training history charts and metrics."""
        history = st.session_state.training_history
        if history is None:
            return

        st.subheader("📊 Training Progress")

        # Create two columns for accuracy and loss
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Accuracy")
            accuracy_df = pd.DataFrame({
                'Training': history['accuracy'],
                'Validation': history['val_accuracy']
            })
            st.line_chart(accuracy_df, use_container_width=True)

            # Display best accuracy metrics
            best_train_acc = max(history['accuracy'])
            best_val_acc = max(history['val_accuracy'])
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("🏆 Best Training", f"{best_train_acc:.4f}")
            with metric_col2:
                st.metric("🏆 Best Validation", f"{best_val_acc:.4f}")

        with col2:
            st.markdown("#### 📉 Loss")
            loss_df = pd.DataFrame({
                'Training': history['loss'],
                'Validation': history['val_loss']
            })
            st.line_chart(loss_df, use_container_width=True)

            # Display best loss metrics
            best_train_loss = min(history['loss'])
            best_val_loss = min(history['val_loss'])
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("🎯 Best Training", f"{best_train_loss:.4f}")
            with metric_col2:
                st.metric("🎯 Best Validation", f"{best_val_loss:.4f}")

    def _display_tensorboard(self) -> None:
        """Start TensorBoard and display access link."""
        url = self.tensorboard_manager.start(TENSORBOARD_LOG_DIR)
        st.success(f"🚀 TensorBoard started!")
        st.markdown(f"[📊 Open TensorBoard]({url})")

    def _stop_tensorboard(self) -> None:
        """Stop TensorBoard server."""
        self.tensorboard_manager.stop()
        st.info("🛑 TensorBoard stopped.")

    def _reset_model(self) -> None:
        """Reset model and training history."""
        st.session_state.model = ModelBuilder.build_cnn()
        st.session_state.training_history = None
        self.tensorboard_manager.stop()
        st.success("🔄 Model reset successfully!")

    def _classify_image(self, uploaded_file: Any) -> tuple[str, float, NDArray[Any]]:
        """Classify an uploaded image.

        Args:
            uploaded_file: The uploaded file from Streamlit file uploader.

        Returns:
            A tuple of (class_name, confidence_percentage, score_array).
        """
        img = Image.open(uploaded_file).resize(IMAGE_SIZE, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)

        predictions = st.session_state.model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0]).numpy()
        class_idx = np.argmax(score)
        class_name = CIFAR10_CLASSES[class_idx]
        confidence = float(np.max(score) * 100)

        return class_name, confidence, score

    def _run_app(self) -> None:
        """Run the main Streamlit application interface."""
        st.title("🖼️ Image Classification App")
        st.info(
            "🎯 Classify images into CIFAR-10 categories: "
            f"{', '.join(CIFAR10_CLASSES)}"
        )

        # Image upload section
        uploaded_files = st.file_uploader(
            "📤 Upload images:",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )

        if uploaded_files:
            # Display images in columns
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, uploaded_file in enumerate(uploaded_files):
                col = cols[idx % 3]
                with col:
                    try:
                        img = Image.open(uploaded_file)
                        st.image(img, caption=uploaded_file.name, use_container_width=True)

                        class_name, confidence, score = self._classify_image(uploaded_file)

                        st.markdown(f"**🏷️ Predicted:** {class_name}")
                        st.progress(confidence / 100)
                        st.caption(f"Confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        # Sidebar - Training Configuration
        st.sidebar.header("🎛️ Training Configuration")

        learning_rate = st.sidebar.number_input(
            "📊 Learning Rate",
            min_value=1e-6,
            max_value=1.0,
            value=0.001,
            format="%.6f"
        )

        epochs = st.sidebar.slider(
            "🔄 Epochs",
            min_value=1,
            max_value=100,
            value=20
        )

        batch_size = st.sidebar.slider(
            "📦 Batch Size",
            min_value=16,
            max_value=128,
            value=32,
            step=16
        )

        optimizer_name = st.sidebar.selectbox(
            "⚡ Optimizer",
            ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "FTRL", "Nadam"]
        )

        # Training buttons
        st.sidebar.divider()

        if st.sidebar.button("🚀 Train Model", use_container_width=True):
            config = TrainingConfig(
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                optimizer_name=optimizer_name
            )
            self._train_model(config)

        # TensorBoard buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📊 Start TB", use_container_width=True):
                self._display_tensorboard()
        with col2:
            if st.button("🛑 Stop TB", use_container_width=True):
                self._stop_tensorboard()

        # Reset button
        st.sidebar.divider()
        if st.sidebar.button("🔄 Reset Model", use_container_width=True, type="secondary"):
            self._reset_model()

        # Display training history if available
        if st.session_state.training_history is not None:
            self._display_training_history()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    ImageClassifierApp()
