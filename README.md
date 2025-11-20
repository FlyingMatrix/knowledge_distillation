# Knowledge Distillation

### 📦 Definition

Knowledge distillation is a machine learning technique where a smaller, simpler model (called the student) is trained to mimic a larger, more complex model (called the teacher). The goal is to compress knowledge from the teacher into the student, making the student model faster and lighter while retaining high performance.

### 🎯 Core Idea

Instead of just training the student on the original dataset with hard labels (0 or 1 for classification), it is trained on soft labels — the output probabilities from the teacher model.

### 🔍 Why Use Knowledge Distillation?

1. Model compression: Reduce size and inference time.
2. Deployment efficiency: Useful for deploying models to resource-constrained environments (mobile, embedded, etc.).
3. Better generalization: Student can sometimes generalize better than a small model trained directly on data.

### 🛠️ How It Works

1. Train the Teacher Model: A large model is trained on the original data.
2. Generate Soft Targets: Pass data through the teacher to get output logits/probabilities.
3. Train the Student Model:
   - Use both the soft targets from the teacher and the hard labels from the data.
   - Use a combined loss function.

### 🚀 Knowldege Distillation Loss Function - Mathematical Explanation

In knowledge distillation, we want the **student model** to learn from both:

- The true labels (**hard targets**), and

- The **soft predictions** of the teacher model (i.e., the probability distribution over classes).

#### Step 1: Temperature Scaling

Let:

- $z_t$ = logits from teacher model

- $z_s$ = logits from student model

- $T$ = temperature (a scalar > 1 used to soften logits)

The **softened probabilities** are:

$$
{p_t} = \mathrm{Softmax}\left(\frac{z_t}{T}\right), \quad {p_s} = \mathrm{Softmax}\left(\frac{z_s}{T}\right)
$$

#### Step 2: Loss Components

We define the total loss as a weighted sum of two losses:

1. **Distillation loss (soft targets)**:
   
   - This measures how well the student mimics the teacher.
   
   - Use **KL divergence** (see 4. in this section) or **cross-entropy**:

$$
L_{\mathrm{KD}} = T^2 \cdot \mathrm{KL}(p_t \parallel p_s)
$$

             or:

$$
L_{\mathrm{KD}} = T^2 \cdot \mathrm{CrossEntropy}(p_t, p_s)
$$

          The $T^2$ term ensures the gradient magnitudes are balanced.

2. **Standard classification loss (hard targets)**:

$$
L_{\mathrm{CE}} = \mathrm{CrossEntropy}(y, \mathrm{Softmax}(z_s))
$$

3. **Total loss**:

$$
L =  \alpha \cdot T^2 \cdot KL(p_t \parallel p_s) + (1 - \alpha) \cdot \mathrm{CrossEntropy}(y, \mathrm{Softmax}(z_s))
$$

            Where $α∈[0,1]$ controls how much weight to give to the hard vs soft loss.

4. **KL divergence**:

        **Kullback–Leibler (KL) divergence** is a measure from information theory that         quantifies how one probability distribution is different from a second, reference         probability distribution.

        For discrete probability distributions $P$ and $Q$ defined over the same support:

$$
D_{KL}(P \parallel Q) = \sum_x P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

        For continuous distributions:

$$
D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{p(x)}{q(x)}\right) \, dx
$$

        Specifically, for the KL divergence between two **normal distributions**:

$$
P(x) = \mathcal{N}(\mu_1, \sigma_1^2), \quad Q(x) = \mathcal{N}(\mu_2, \sigma_2^2)
$$

        Then the KL divergence from P to Q is:

$$
D_{KL}(P \parallel Q) = \log\left(\frac{\sigma_2}{\sigma_1}\right) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

        Where:

            $P$: The true distribution (often called the "prior" or "real" distribution)

            $Q$: The approximation or "guessed" distribution

### 🏗️ Effects of Temperature

In **knowledge distillation**, the **temperature $T$** plays a crucial role in controlling how **soft** or **hard** the predicted probabilities are when distilling knowledge from a **teacher model** to a **student model**.

During distillation, we modify the **softmax function** used to compute class probabilities:

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Where:

- $z_i$ is the logit (raw output) for class $i$,

- $T$ is the **temperature**.

##### ✅ 1. **When T=1**:

- You get the standard softmax distribution.

- Probabilities are sharp: typically, one class has high probability, others are near zero.

##### ✅ 2. **When T>1**:

- The output probabilities become **softer** (more uniform).

- Higher T means **more smoothed** distributions.

- This exposes **dark knowledge**: relative similarities between classes learned by the teacher.

##### ✅ 3. **When T→∞**:

- All classes tend to have **equal** probability → total smoothing.

- Too high T may make the signal too weak for the student to learn useful differences.

##### ✅ 4. **When T<1**:

- The output becomes **sharper**, approaching a **one-hot** vector.

- Less softening → fewer insights into class relationships.

##### ✅ 5. **Summary Table**:

| Temperature \( $T$ \) | Effect on Logits  | Effect on Softmax Output            |
| --------------------- | ----------------- | ----------------------------------- |
| \( $T > 1$ \)         | Compresses logits | Softer, more uniform probabilities  |
| \( $T = 1$ \)         | No change         | Normal softmax                      |
| \( $T < 1$ \)         | Expands logits    | Sharper, more confident predictions |

##### ✅ 6. **Typical Practice**:

In Knowledge Distillation:

- You use a high temperature (e.g., $T∈[2,5]$) on both teacher and student.

- The softened targets help the student learn **not just what is right**, but also **what is similar**.

- After training, during **inference**, you use $T=1$ (standard softmax).

### 💻 General PyTorch Knowledge Distillation Loss Function

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):

    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        T = self.temperature
        alpha = self.alpha

        # Soft targets: distillation loss
        soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

        # Hard targets: standard classification loss
        ce_loss = self.ce_loss(student_logits, true_labels)

        # Total loss
        return alpha * kd_loss + (1. - alpha) * ce_loss
```

### 📝 Frequently Asked Questions (FAQ)

1. **Why Do We Need Knowledge Distillation?**
   
   (1) Model Compression (without huge drops in performance)
   
   (2) Training Efficiency
   
   - Student models can be trained faster than training from scratch.
   
   - Especially useful when labeled data is scarce, the teacher model can generate rich supervision.
   
   (3) Improved Generalization (how well a model performs on unseen data)
   
   - The student learns smoother, more informative distributions (soft targets) instead of just hard labels.
   
   - Knowledge Distillation acts as a regularizer, reducing overfitting.

2. **What will the student model learn from the teacher model?**
   
   (1) Soft Targets (Soft Labels)
   
   Instead of only learning from the **hard labels** (e.g., class 3 is correct), the student also learns from the teacher’s **output probabilities** (e.g., 0.2 for class 1, 0.7 for class 3, etc.).
   
   - These soft targets contain **dark knowledge** — relationships between classes.
   
   - Example:
     
     - Hard label: `dog`
     
     - Teacher output: `[dog: 0.85, wolf: 0.10, cat: 0.03]`
     
     - Student learns that **wolf** is more similar to **dog** than **cat** is — a nuance not visible in the hard label.
   
   (2) Intermediate Representations
   
   In advanced setups, the student can mimic:
   
   - Hidden layers
   
   - Attention maps
   
   - Feature representations
   
   This allows the student to match not just **outputs**, but also the **process** of the teacher.
   
   (3) Logits Instead of Argmax
   
   The student is trained to match the **logits** (pre-softmax values) or softened probabilities from the teacher. These provide richer supervision than just "the correct answer".

3. **Can the teacher model help train the student model if the dataset was never used to train the teacher model?**
   
   **Yes**, the teacher model can still help train the student **even if the dataset was never used to train the teacher.** Because **knowledge distillation doesn't require the dataset used to train the teacher model.** What matters is that:
   
   (1) The **teacher is well-trained** on a similar or related domain.
   
   (2) The **teacher's outputs** on the new dataset still contain meaningful information (even if it's never seen those exact samples) by producing **soft predictions** (probability distributions over classes). These outputs:
   
   - **Capture the teacher's generalization ability**.
   
   - Include information about **class relationships**, which aren’t visible from hard labels.
   
   - Help the student **learn better representations**, even if ground-truth labels are sparse or noisy.
   
   This process can:
   
   - Improve student accuracy.
   
   - Act as a form of **regularization** (preventing overfitting).
   
   - Be helpful even with **unlabeled data** (i.e., pseudo-labeling via the teacher).
   
   In conclusion, **A well-trained teacher can guide the student even on unseen data**, as long as the input domain is similar or related. This is one of the powers of knowledge distillation — it doesn't need the teacher to have seen your exact dataset before.

4. **For example, if I would like to train a DistilBERT model (student) via knowledge distillation from a BERT model (teacher), which tokenizer should I choose for the dataset?**
   
   Knowledge distillation involves the student (DistilBERT) learning from the teacher's outputs. If the tokenization is different, the input tokens won't align, and the student can't learn effectively. For this reason, always **use the same tokenizer as the teacher model (BERT tokenizer)** and this guarantees smooth transfer of learned knowledge and ensures same vocabulary, same token splitting, same input IDs and attention masks.
