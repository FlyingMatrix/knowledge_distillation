## Knowledge Distillation

#### Definition

Knowledge distillation is a machine learning technique where a smaller, simpler model (called the student) is trained to mimic a larger, more complex model (called the teacher). The goal is to compress knowledge from the teacher into the student, making the student model faster and lighter while retaining high performance.

#### Core Idea

Instead of just training the student on the original dataset with hard labels (0 or 1 for classification), it is trained on soft labels — the output probabilities from the teacher model.

#### Why Use Knowledge Distillation?

1. Model compression: Reduce size and inference time.
2. Deployment efficiency: Useful for deploying models to resource-constrained environments (mobile, embedded, etc.).
3. Better generalization: Student can sometimes generalize better than a small model trained directly on data.

#### How It Works

1. Train the Teacher Model: A large model is trained on the original data.
2. Generate Soft Targets: Pass data through the teacher to get output logits/probabilities.
3. Train the Student Model:
   - Use both the soft targets from the teacher and the hard labels from the data.
   - Use a combined loss function.

#### Knowldege Distillation Loss Function - Mathematical Explanation

In knowledge distillation, we want the **student model** to learn from both:

- The true labels (**hard targets**), and

- The **soft predictions** of the teacher model (i.e., the probability distribution over classes).

##### Step 1: Temperature Scaling

Let:

- `z_t` = logits from teacher model

- `z_s` = logits from student model

- `T` = temperature (a scalar > 1 used to soften logits)

The **softened probabilities** are:

$$
{p_t} = \mathrm{Softmax}\left(\frac{z_t}{T}\right), \quad {p_s} = \mathrm{Softmax}\left(\frac{z_s}{T}\right)
$$

##### Step 2: Loss Components

We define the total loss as a weighted sum of two losses:

1. **Distillation loss (soft targets)**:
   
   - This measures how well the student mimics the teacher.
   
   - Use **KL divergence** or **cross-entropy**:

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

3. **Total Loss**:

$$
L_{\text{total}} = \alpha \cdot L_{\mathrm{CE}} + (1 - \alpha) \cdot L_{\mathrm{KD}}
$$

Where $α∈[0,1]$ controls how much weight to give to the hard vs soft loss.







