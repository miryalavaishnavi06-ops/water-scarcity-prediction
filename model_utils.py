"""
model_utils.py

Attention-augmented LSTM, matching the base paper's approach
(Tripathy et al. 2026 -- attention-augmented LSTM + gradient-based
explainability), adapted here for water scarcity classification.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="phase1")
class SumOverTime(layers.Layer):
    """Collapses (batch, time, features) into (batch, features) by summing
    over time, after attention weighting. A registered custom layer (not a
    Lambda) so the model saves/loads cleanly."""

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def build_lstm_model(window_size, n_features, n_classes):
    inputs = layers.Input(shape=(window_size, n_features))

    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)

    # Attention: learns how much weight to give each time step in the window
    attention_scores = layers.Dense(1, activation="tanh")(x)
    attention_weights = layers.Softmax(axis=1)(attention_scores)
    context = layers.Multiply()([x, attention_weights])
    context = SumOverTime()(context)

    x = layers.Dense(32, activation="relu")(context)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_phase1_attention_lstm(window_size, n_features, n_classes=5, compile_model=True):
    """Small temporal-attention classifier for the documented Phase 1 task.

    Uses the project's Attention-LSTM pattern with fewer units for the 180
    available sequences. The continuous SPEI-3 target is mapped to documented
    severity classes in phase1_data.py; the original SPEI target is untouched.
    """
    inputs = layers.Input(shape=(window_size, n_features), name="historical_predictors")
    sequence = layers.LSTM(16, return_sequences=True, unroll=True, name="lstm")(inputs)
    attention_scores = layers.Dense(1, activation="tanh", name="attention_score")(sequence)
    attention_weights = layers.Softmax(axis=1, name="temporal_attention")(attention_scores)
    weighted = layers.Multiply(name="attention_weighted_sequence")(
        [sequence, attention_weights]
    )
    context = SumOverTime(name="attention_context")(weighted)
    hidden = layers.Dense(8, activation="relu", name="dense_hidden")(context)
    hidden = layers.Dropout(0.25, name="dropout")(hidden)
    outputs = layers.Dense(n_classes, activation="softmax", name="severity_class")(hidden)
    model = models.Model(inputs, outputs, name="phase1_attention_lstm")
    if not compile_model:
        return model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        run_eagerly=True,
    )
    return model
