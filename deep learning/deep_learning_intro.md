# Introduction to Deep Learning

Deep Learning is a subfield of Machine Learning that uses **artificial neural networks with many layers** to learn patterns from large datasets. It mimics the human brain using interconnected neurons that automatically extract features from raw data such as images, text, audio, and video.

Deep learning models improve with more data, compute power (GPUs/TPUs), and optimized architectures.

---

# Parts of Deep Learning

Deep Learning mainly consists of **three important neural network architectures**:

## 1. Artificial Neural Network (ANN)

**What ANN is:**
- Simplest form of deep learning architecture.
- Used for **tabular data**, **binary/multi-class classification**, and **regression** tasks.

**Structure:**
- **Input Layer** → Receives features
- **Hidden Layers** → Learn internal representations
- **Output Layer** → Produces final prediction

**When to Use ANN:**
- Customer churn prediction
- Loan default prediction
- Fraud detection
- Temperature/stock prediction

**Limitations of ANN:**
- Cannot handle spatial or sequential data well
- Requires feature engineering
- Computationally expensive with large datasets
- Prone to overfitting
- Not suitable for high-dimensional inputs
- Difficult to interpret

---

## 2. Convolutional Neural Network (CNN)

**What CNN is:**
- Specialized for **image and video data**.
- Extracts spatial features like **edges**, **shapes**, and **textures**.

**Key Components:**
- **Convolution Layer** → Feature extraction
- **ReLU Activation** → Non-linearity
- **Pooling Layer** → Dimension reduction
- **Fully Connected Layer** → Final classification

**When to Use CNN:**
- Image classification (cats vs dogs)
- Face recognition
- Medical imaging
- Object detection (YOLO, SSD)
- Image segmentation (U-Net)

**Limitations of CNN:**
- Cannot handle sequential/time-dependent data
- Computationally expensive
- Needs large labeled datasets
- Not rotation or scale invariant by default
- Difficult to interpret learned filters
- Poor performance on non-spatial data

---

## 3. Recurrent Neural Network (RNN)

**What RNN is:**
- Designed for **sequence data**.
- Remembers previous inputs via a **hidden state**.

**Why RNNs:**
- Captures temporal patterns such as word order, time series, and audio patterns.

**Types:**
- Simple RNN → Short memory
- LSTM → Long memory, stable training
- GRU → Faster than LSTM, similar performance

**When to Use RNN:**
- Text generation
- Language modeling
- Time series forecasting
- Speech recognition
- Chatbots

**Limitations of RNN:**
- Vanishing and exploding gradient problem
- Cannot learn very long-range dependencies well
- Slow training time
- Difficult to handle very long sequences
- Not good for spatial data (images)
- High computational cost

---

## LSTM (Long Short-Term Memory)

**Key Idea:**
- Solves vanishing gradient problem in RNNs
- Introduces gates to control information flow: Forget, Input, Output

**Why LSTM:**
- Remembers long sequences
- Stable training
- Works well for text, speech, and time series

**Applications:**
- Machine translation
- Chatbots
- Stock prediction
- Speech recognition
- Text generation

**Limitations:**
- Heavy computation
- High memory usage
- Slow training

---

## GRU (Gated Recurrent Unit)

**Key Idea:**
- Simplified version of LSTM
- Uses two gates: Update and Reset

**Why GRU:**
- Fewer parameters, faster training
- Works well on small datasets
- Performs similarly to LSTM

**Applications:**
- Time series forecasting
- Sentiment analysis
- Sequence classification
- Language modeling

**Limitations:**
- Slightly less expressive than LSTM
- May not capture very long dependencies as well

---

## LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|--------|------|-----|
| Number of Gates | 3 (Forget, Input, Output) | 2 (Update, Reset) |
| Complexity | High | Medium |
| Parameters | More | Fewer |
| Speed | Slower | Faster |
| Memory Need | High | Lower |
| Long-term Memory | Very strong | Strong |
| Performance | Great for large data | Great for small/medium data |
| Parallelization | Harder | Easier |

---

**Summary:**
- LSTM → More powerful, better for long dependencies
- GRU → Faster, simpler, similar accuracy

