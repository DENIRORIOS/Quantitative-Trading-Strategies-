//+------------------------------------------------------------------+
//|                                               NeuralNetwork.mqh  |
//|                   3-Layer Feedforward Neural Network from Scratch|
//|                                                                  |
//| Implements a 3-layer ANN (512-256-128 neurons) with:            |
//| - ReLU activation for hidden layers                             |
//| - Sigmoid activation for output layer                           |
//| - Adam optimizer for weight updates                             |
//| - Dropout regularization during training                        |
//| - He initialization for weights                                 |
//+------------------------------------------------------------------+

#property copyright "BTC Trading ANN"
#property strict

//+------------------------------------------------------------------+
//| Neural Network Class                                             |
//| Implements feedforward and backpropagation from scratch          |
//+------------------------------------------------------------------+
class CNeuralNetwork
{
private:
    // Network architecture
    int m_input_size;      // 120 inputs (20 periods × 6 features)
    int m_hidden1_size;    // 512 neurons
    int m_hidden2_size;    // 256 neurons
    int m_hidden3_size;    // 128 neurons
    int m_output_size;     // 1 output (probability)
    
    // Weights and biases
    double m_W1[][];       // Weights layer 1: [input_size × hidden1_size]
    double m_b1[];         // Biases layer 1: [hidden1_size]
    double m_W2[][];       // Weights layer 2: [hidden1_size × hidden2_size]
    double m_b2[];         // Biases layer 2: [hidden2_size]
    double m_W3[][];       // Weights layer 3: [hidden2_size × hidden3_size]
    double m_b3[];         // Biases layer 3: [hidden3_size]
    double m_W4[][];       // Weights layer 4: [hidden3_size × output_size]
    double m_b4[];         // Biases layer 4: [output_size]
    
    // Activations (stored for backpropagation)
    double m_z1[];         // Pre-activation layer 1
    double m_a1[];         // Activation layer 1 (after ReLU)
    double m_z2[];         // Pre-activation layer 2
    double m_a2[];         // Activation layer 2 (after ReLU)
    double m_z3[];         // Pre-activation layer 3
    double m_a3[];         // Activation layer 3 (after ReLU)
    double m_z4[];         // Pre-activation output layer
    double m_output[];     // Final output (after Sigmoid)
    
    // Dropout masks
    double m_dropout1[];   // Dropout mask for layer 1
    double m_dropout2[];   // Dropout mask for layer 2
    double m_dropout3[];   // Dropout mask for layer 3
    
    // Adam optimizer parameters
    double m_learning_rate;
    double m_beta1;        // First moment decay (default: 0.9)
    double m_beta2;        // Second moment decay (default: 0.999)
    double m_epsilon;      // Small constant for numerical stability (default: 1e-8)
    int m_timestep;        // Current timestep for Adam
    
    // Adam moment estimates for weights
    double m_mW1[][], m_vW1[][];  // First and second moments for W1
    double m_mb1[], m_vb1[];      // First and second moments for b1
    double m_mW2[][], m_vW2[][];  // First and second moments for W2
    double m_mb2[], m_vb2[];      // First and second moments for b2
    double m_mW3[][], m_vW3[][];  // First and second moments for W3
    double m_mb3[], m_vb3[];      // First and second moments for b3
    double m_mW4[][], m_vW4[][];  // First and second moments for W4
    double m_mb4[], m_vb4[];      // First and second moments for b4
    
    // Dropout rates
    double m_dropout_rate1;
    double m_dropout_rate2;
    double m_dropout_rate3;
    
    // Training mode flag
    bool m_is_training;
    
public:
    //+------------------------------------------------------------------+
    //| Constructor                                                       |
    //+------------------------------------------------------------------+
    CNeuralNetwork(int input_size=120, int h1=512, int h2=256, int h3=128,
                   double lr=0.001, double dropout1=0.3, double dropout2=0.3, double dropout3=0.2)
    {
        m_input_size = input_size;
        m_hidden1_size = h1;
        m_hidden2_size = h2;
        m_hidden3_size = h3;
        m_output_size = 1;
        
        m_learning_rate = lr;
        m_beta1 = 0.9;
        m_beta2 = 0.999;
        m_epsilon = 1e-8;
        m_timestep = 0;
        
        m_dropout_rate1 = dropout1;
        m_dropout_rate2 = dropout2;
        m_dropout_rate3 = dropout3;
        
        m_is_training = false;
        
        InitializeWeights();
        InitializeAdamMoments();
    }
    
    //+------------------------------------------------------------------+
    //| Initialize weights using He initialization                       |
    //| He init: weights ~ N(0, sqrt(2/n_in)) for ReLU activation       |
    //+------------------------------------------------------------------+
    void InitializeWeights()
    {
        // Seed random number generator
        MathSrand((int)TimeLocal());
        
        // Layer 1: input -> hidden1
        ArrayResize(m_W1, m_input_size);
        for(int i = 0; i < m_input_size; i++)
        {
            ArrayResize(m_W1[i], m_hidden1_size);
            double std = MathSqrt(2.0 / m_input_size);  // He initialization
            for(int j = 0; j < m_hidden1_size; j++)
                m_W1[i][j] = RandomNormal() * std;
        }
        ArrayResize(m_b1, m_hidden1_size);
        ArrayInitialize(m_b1, 0.0);
        
        // Layer 2: hidden1 -> hidden2
        ArrayResize(m_W2, m_hidden1_size);
        for(int i = 0; i < m_hidden1_size; i++)
        {
            ArrayResize(m_W2[i], m_hidden2_size);
            double std = MathSqrt(2.0 / m_hidden1_size);
            for(int j = 0; j < m_hidden2_size; j++)
                m_W2[i][j] = RandomNormal() * std;
        }
        ArrayResize(m_b2, m_hidden2_size);
        ArrayInitialize(m_b2, 0.0);
        
        // Layer 3: hidden2 -> hidden3
        ArrayResize(m_W3, m_hidden2_size);
        for(int i = 0; i < m_hidden2_size; i++)
        {
            ArrayResize(m_W3[i], m_hidden3_size);
            double std = MathSqrt(2.0 / m_hidden2_size);
            for(int j = 0; j < m_hidden3_size; j++)
                m_W3[i][j] = RandomNormal() * std;
        }
        ArrayResize(m_b3, m_hidden3_size);
        ArrayInitialize(m_b3, 0.0);
        
        // Layer 4: hidden3 -> output
        ArrayResize(m_W4, m_hidden3_size);
        for(int i = 0; i < m_hidden3_size; i++)
        {
            ArrayResize(m_W4[i], m_output_size);
            double std = MathSqrt(2.0 / m_hidden3_size);
            for(int j = 0; j < m_output_size; j++)
                m_W4[i][j] = RandomNormal() * std;
        }
        ArrayResize(m_b4, m_output_size);
        ArrayInitialize(m_b4, 0.0);
        
        Print("Neural network weights initialized with He initialization");
    }
    
    //+------------------------------------------------------------------+
    //| Initialize Adam optimizer moment estimates                       |
    //+------------------------------------------------------------------+
    void InitializeAdamMoments()
    {
        // Initialize all moment estimates to zero
        ArrayResize(m_mW1, m_input_size);
        ArrayResize(m_vW1, m_input_size);
        for(int i = 0; i < m_input_size; i++)
        {
            ArrayResize(m_mW1[i], m_hidden1_size);
            ArrayResize(m_vW1[i], m_hidden1_size);
            ArrayInitialize(m_mW1[i], 0.0);
            ArrayInitialize(m_vW1[i], 0.0);
        }
        ArrayResize(m_mb1, m_hidden1_size);
        ArrayResize(m_vb1, m_hidden1_size);
        ArrayInitialize(m_mb1, 0.0);
        ArrayInitialize(m_vb1, 0.0);
        
        ArrayResize(m_mW2, m_hidden1_size);
        ArrayResize(m_vW2, m_hidden1_size);
        for(int i = 0; i < m_hidden1_size; i++)
        {
            ArrayResize(m_mW2[i], m_hidden2_size);
            ArrayResize(m_vW2[i], m_hidden2_size);
            ArrayInitialize(m_mW2[i], 0.0);
            ArrayInitialize(m_vW2[i], 0.0);
        }
        ArrayResize(m_mb2, m_hidden2_size);
        ArrayResize(m_vb2, m_hidden2_size);
        ArrayInitialize(m_mb2, 0.0);
        ArrayInitialize(m_vb2, 0.0);
        
        ArrayResize(m_mW3, m_hidden2_size);
        ArrayResize(m_vW3, m_hidden2_size);
        for(int i = 0; i < m_hidden2_size; i++)
        {
            ArrayResize(m_mW3[i], m_hidden3_size);
            ArrayResize(m_vW3[i], m_hidden3_size);
            ArrayInitialize(m_mW3[i], 0.0);
            ArrayInitialize(m_vW3[i], 0.0);
        }
        ArrayResize(m_mb3, m_hidden3_size);
        ArrayResize(m_vb3, m_hidden3_size);
        ArrayInitialize(m_mb3, 0.0);
        ArrayInitialize(m_vb3, 0.0);
        
        ArrayResize(m_mW4, m_hidden3_size);
        ArrayResize(m_vW4, m_hidden3_size);
        for(int i = 0; i < m_hidden3_size; i++)
        {
            ArrayResize(m_mW4[i], m_output_size);
            ArrayResize(m_vW4[i], m_output_size);
            ArrayInitialize(m_mW4[i], 0.0);
            ArrayInitialize(m_vW4[i], 0.0);
        }
        ArrayResize(m_mb4, m_output_size);
        ArrayResize(m_vb4, m_output_size);
        ArrayInitialize(m_mb4, 0.0);
        ArrayInitialize(m_vb4, 0.0);
    }
    
    //+------------------------------------------------------------------+
    //| Generate random number from standard normal distribution         |
    //| Using Box-Muller transform                                       |
    //+------------------------------------------------------------------+
    double RandomNormal()
    {
        double u1 = MathRand() / 32767.0;  // Uniform [0,1]
        double u2 = MathRand() / 32767.0;
        if(u1 < 1e-10) u1 = 1e-10;  // Avoid log(0)
        return MathSqrt(-2.0 * MathLog(u1)) * MathCos(2.0 * M_PI * u2);
    }
    
    //+------------------------------------------------------------------+
    //| ReLU activation function: max(0, x)                             |
    //+------------------------------------------------------------------+
    double ReLU(double x)
    {
        return MathMax(0.0, x);
    }
    
    //+------------------------------------------------------------------+
    //| ReLU derivative: 1 if x > 0, else 0                             |
    //+------------------------------------------------------------------+
    double ReLU_derivative(double x)
    {
        return (x > 0) ? 1.0 : 0.0;
    }
    
    //+------------------------------------------------------------------+
    //| Sigmoid activation function: 1 / (1 + exp(-x))                  |
    //+------------------------------------------------------------------+
    double Sigmoid(double x)
    {
        // Clip to prevent overflow
        if(x > 20.0) return 1.0;
        if(x < -20.0) return 0.0;
        return 1.0 / (1.0 + MathExp(-x));
    }
    
    //+------------------------------------------------------------------+
    //| Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))               |
    //+------------------------------------------------------------------+
    double Sigmoid_derivative(double x)
    {
        double sig = Sigmoid(x);
        return sig * (1.0 - sig);
    }
    
    //+------------------------------------------------------------------+
    //| Generate dropout mask                                            |
    //+------------------------------------------------------------------+
    void GenerateDropoutMask(double &mask[], int size, double dropout_rate)
    {
        ArrayResize(mask, size);
        double keep_prob = 1.0 - dropout_rate;
        for(int i = 0; i < size; i++)
        {
            // Binary mask: 1/keep_prob if kept, 0 if dropped
            // Scale by 1/keep_prob to maintain expected value during inference
            mask[i] = (MathRand() / 32767.0 > dropout_rate) ? (1.0 / keep_prob) : 0.0;
        }
    }
    
    //+------------------------------------------------------------------+
    //| Forward propagation                                              |
    //| Input: features[120] -> Output: probability[1]                   |
    //+------------------------------------------------------------------+
    double Forward(double &input[])
    {
        // Resize activation arrays
        ArrayResize(m_z1, m_hidden1_size);
        ArrayResize(m_a1, m_hidden1_size);
        ArrayResize(m_z2, m_hidden2_size);
        ArrayResize(m_a2, m_hidden2_size);
        ArrayResize(m_z3, m_hidden3_size);
        ArrayResize(m_a3, m_hidden3_size);
        ArrayResize(m_z4, m_output_size);
        ArrayResize(m_output, m_output_size);
        
        // Layer 1: input -> hidden1 (ReLU)
        // z1 = W1^T * input + b1
        for(int j = 0; j < m_hidden1_size; j++)
        {
            m_z1[j] = m_b1[j];
            for(int i = 0; i < m_input_size; i++)
                m_z1[j] += m_W1[i][j] * input[i];
            m_a1[j] = ReLU(m_z1[j]);
        }
        
        // Apply dropout during training
        if(m_is_training)
        {
            GenerateDropoutMask(m_dropout1, m_hidden1_size, m_dropout_rate1);
            for(int j = 0; j < m_hidden1_size; j++)
                m_a1[j] *= m_dropout1[j];
        }
        
        // Layer 2: hidden1 -> hidden2 (ReLU)
        // z2 = W2^T * a1 + b2
        for(int j = 0; j < m_hidden2_size; j++)
        {
            m_z2[j] = m_b2[j];
            for(int i = 0; i < m_hidden1_size; i++)
                m_z2[j] += m_W2[i][j] * m_a1[i];
            m_a2[j] = ReLU(m_z2[j]);
        }
        
        // Apply dropout during training
        if(m_is_training)
        {
            GenerateDropoutMask(m_dropout2, m_hidden2_size, m_dropout_rate2);
            for(int j = 0; j < m_hidden2_size; j++)
                m_a2[j] *= m_dropout2[j];
        }
        
        // Layer 3: hidden2 -> hidden3 (ReLU)
        // z3 = W3^T * a2 + b3
        for(int j = 0; j < m_hidden3_size; j++)
        {
            m_z3[j] = m_b3[j];
            for(int i = 0; i < m_hidden2_size; i++)
                m_z3[j] += m_W3[i][j] * m_a2[i];
            m_a3[j] = ReLU(m_z3[j]);
        }
        
        // Apply dropout during training
        if(m_is_training)
        {
            GenerateDropoutMask(m_dropout3, m_hidden3_size, m_dropout_rate3);
            for(int j = 0; j < m_hidden3_size; j++)
                m_a3[j] *= m_dropout3[j];
        }
        
        // Layer 4: hidden3 -> output (Sigmoid)
        // z4 = W4^T * a3 + b4
        for(int j = 0; j < m_output_size; j++)
        {
            m_z4[j] = m_b4[j];
            for(int i = 0; i < m_hidden3_size; i++)
                m_z4[j] += m_W4[i][j] * m_a3[i];
            m_output[j] = Sigmoid(m_z4[j]);
        }
        
        return m_output[0];
    }
    
    //+------------------------------------------------------------------+
    //| Backpropagation with gradient calculation                        |
    //| Uses stored activations from forward pass                        |
    //+------------------------------------------------------------------+
    void Backward(double &input[], double target, double &dW1[][], double &db1[],
                  double &dW2[][], double &db2[], double &dW3[][], double &db3[],
                  double &dW4[][], double &db4[])
    {
        // Resize gradient arrays
        ArrayResize(dW1, m_input_size);
        for(int i = 0; i < m_input_size; i++)
            ArrayResize(dW1[i], m_hidden1_size);
        ArrayResize(db1, m_hidden1_size);
        
        ArrayResize(dW2, m_hidden1_size);
        for(int i = 0; i < m_hidden1_size; i++)
            ArrayResize(dW2[i], m_hidden2_size);
        ArrayResize(db2, m_hidden2_size);
        
        ArrayResize(dW3, m_hidden2_size);
        for(int i = 0; i < m_hidden2_size; i++)
            ArrayResize(dW3[i], m_hidden3_size);
        ArrayResize(db3, m_hidden3_size);
        
        ArrayResize(dW4, m_hidden3_size);
        for(int i = 0; i < m_hidden3_size; i++)
            ArrayResize(dW4[i], m_output_size);
        ArrayResize(db4, m_output_size);
        
        // Output layer gradient
        // For binary cross-entropy: dL/dz4 = output - target
        double dz4[];
        ArrayResize(dz4, m_output_size);
        dz4[0] = m_output[0] - target;
        
        // Gradients for W4 and b4
        for(int i = 0; i < m_hidden3_size; i++)
            for(int j = 0; j < m_output_size; j++)
                dW4[i][j] = m_a3[i] * dz4[j];
        
        for(int j = 0; j < m_output_size; j++)
            db4[j] = dz4[j];
        
        // Backprop to hidden layer 3
        double dz3[];
        ArrayResize(dz3, m_hidden3_size);
        for(int i = 0; i < m_hidden3_size; i++)
        {
            dz3[i] = 0.0;
            for(int j = 0; j < m_output_size; j++)
                dz3[i] += m_W4[i][j] * dz4[j];
            dz3[i] *= ReLU_derivative(m_z3[i]);
            // Apply dropout mask
            if(m_is_training)
                dz3[i] *= m_dropout3[i];
        }
        
        // Gradients for W3 and b3
        for(int i = 0; i < m_hidden2_size; i++)
            for(int j = 0; j < m_hidden3_size; j++)
                dW3[i][j] = m_a2[i] * dz3[j];
        
        for(int j = 0; j < m_hidden3_size; j++)
            db3[j] = dz3[j];
        
        // Backprop to hidden layer 2
        double dz2[];
        ArrayResize(dz2, m_hidden2_size);
        for(int i = 0; i < m_hidden2_size; i++)
        {
            dz2[i] = 0.0;
            for(int j = 0; j < m_hidden3_size; j++)
                dz2[i] += m_W3[i][j] * dz3[j];
            dz2[i] *= ReLU_derivative(m_z2[i]);
            // Apply dropout mask
            if(m_is_training)
                dz2[i] *= m_dropout2[i];
        }
        
        // Gradients for W2 and b2
        for(int i = 0; i < m_hidden1_size; i++)
            for(int j = 0; j < m_hidden2_size; j++)
                dW2[i][j] = m_a1[i] * dz2[j];
        
        for(int j = 0; j < m_hidden2_size; j++)
            db2[j] = dz2[j];
        
        // Backprop to hidden layer 1
        double dz1[];
        ArrayResize(dz1, m_hidden1_size);
        for(int i = 0; i < m_hidden1_size; i++)
        {
            dz1[i] = 0.0;
            for(int j = 0; j < m_hidden2_size; j++)
                dz1[i] += m_W2[i][j] * dz2[j];
            dz1[i] *= ReLU_derivative(m_z1[i]);
            // Apply dropout mask
            if(m_is_training)
                dz1[i] *= m_dropout1[i];
        }
        
        // Gradients for W1 and b1
        for(int i = 0; i < m_input_size; i++)
            for(int j = 0; j < m_hidden1_size; j++)
                dW1[i][j] = input[i] * dz1[j];
        
        for(int j = 0; j < m_hidden1_size; j++)
            db1[j] = dz1[j];
    }
    
    //+------------------------------------------------------------------+
    //| Update weights using Adam optimizer                              |
    //| Adam: Adaptive Moment Estimation                                 |
    //| Updates first moment (mean) and second moment (variance)         |
    //+------------------------------------------------------------------+
    void UpdateWeightsAdam(double &dW1[][], double &db1[], double &dW2[][], double &db2[],
                           double &dW3[][], double &db3[], double &dW4[][], double &db4[])
    {
        m_timestep++;
        
        // Bias correction factors
        double bias_correction1 = 1.0 - MathPow(m_beta1, m_timestep);
        double bias_correction2 = 1.0 - MathPow(m_beta2, m_timestep);
        
        // Update W1 and b1
        for(int i = 0; i < m_input_size; i++)
        {
            for(int j = 0; j < m_hidden1_size; j++)
            {
                // Update biased first moment estimate
                m_mW1[i][j] = m_beta1 * m_mW1[i][j] + (1.0 - m_beta1) * dW1[i][j];
                // Update biased second moment estimate
                m_vW1[i][j] = m_beta2 * m_vW1[i][j] + (1.0 - m_beta2) * dW1[i][j] * dW1[i][j];
                // Compute bias-corrected moment estimates
                double m_hat = m_mW1[i][j] / bias_correction1;
                double v_hat = m_vW1[i][j] / bias_correction2;
                // Update weights
                m_W1[i][j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
            }
        }
        
        for(int j = 0; j < m_hidden1_size; j++)
        {
            m_mb1[j] = m_beta1 * m_mb1[j] + (1.0 - m_beta1) * db1[j];
            m_vb1[j] = m_beta2 * m_vb1[j] + (1.0 - m_beta2) * db1[j] * db1[j];
            double m_hat = m_mb1[j] / bias_correction1;
            double v_hat = m_vb1[j] / bias_correction2;
            m_b1[j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
        }
        
        // Update W2 and b2
        for(int i = 0; i < m_hidden1_size; i++)
        {
            for(int j = 0; j < m_hidden2_size; j++)
            {
                m_mW2[i][j] = m_beta1 * m_mW2[i][j] + (1.0 - m_beta1) * dW2[i][j];
                m_vW2[i][j] = m_beta2 * m_vW2[i][j] + (1.0 - m_beta2) * dW2[i][j] * dW2[i][j];
                double m_hat = m_mW2[i][j] / bias_correction1;
                double v_hat = m_vW2[i][j] / bias_correction2;
                m_W2[i][j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
            }
        }
        
        for(int j = 0; j < m_hidden2_size; j++)
        {
            m_mb2[j] = m_beta1 * m_mb2[j] + (1.0 - m_beta1) * db2[j];
            m_vb2[j] = m_beta2 * m_vb2[j] + (1.0 - m_beta2) * db2[j] * db2[j];
            double m_hat = m_mb2[j] / bias_correction1;
            double v_hat = m_vb2[j] / bias_correction2;
            m_b2[j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
        }
        
        // Update W3 and b3
        for(int i = 0; i < m_hidden2_size; i++)
        {
            for(int j = 0; j < m_hidden3_size; j++)
            {
                m_mW3[i][j] = m_beta1 * m_mW3[i][j] + (1.0 - m_beta1) * dW3[i][j];
                m_vW3[i][j] = m_beta2 * m_vW3[i][j] + (1.0 - m_beta2) * dW3[i][j] * dW3[i][j];
                double m_hat = m_mW3[i][j] / bias_correction1;
                double v_hat = m_vW3[i][j] / bias_correction2;
                m_W3[i][j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
            }
        }
        
        for(int j = 0; j < m_hidden3_size; j++)
        {
            m_mb3[j] = m_beta1 * m_mb3[j] + (1.0 - m_beta1) * db3[j];
            m_vb3[j] = m_beta2 * m_vb3[j] + (1.0 - m_beta2) * db3[j] * db3[j];
            double m_hat = m_mb3[j] / bias_correction1;
            double v_hat = m_vb3[j] / bias_correction2;
            m_b3[j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
        }
        
        // Update W4 and b4
        for(int i = 0; i < m_hidden3_size; i++)
        {
            for(int j = 0; j < m_output_size; j++)
            {
                m_mW4[i][j] = m_beta1 * m_mW4[i][j] + (1.0 - m_beta1) * dW4[i][j];
                m_vW4[i][j] = m_beta2 * m_vW4[i][j] + (1.0 - m_beta2) * dW4[i][j] * dW4[i][j];
                double m_hat = m_mW4[i][j] / bias_correction1;
                double v_hat = m_vW4[i][j] / bias_correction2;
                m_W4[i][j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
            }
        }
        
        for(int j = 0; j < m_output_size; j++)
        {
            m_mb4[j] = m_beta1 * m_mb4[j] + (1.0 - m_beta1) * db4[j];
            m_vb4[j] = m_beta2 * m_vb4[j] + (1.0 - m_beta2) * db4[j] * db4[j];
            double m_hat = m_mb4[j] / bias_correction1;
            double v_hat = m_vb4[j] / bias_correction2;
            m_b4[j] -= m_learning_rate * m_hat / (MathSqrt(v_hat) + m_epsilon);
        }
    }
    
    //+------------------------------------------------------------------+
    //| Set training mode (enables/disables dropout)                     |
    //+------------------------------------------------------------------+
    void SetTrainingMode(bool is_training)
    {
        m_is_training = is_training;
    }
    
    //+------------------------------------------------------------------+
    //| Predict (forward pass in inference mode)                         |
    //+------------------------------------------------------------------+
    double Predict(double &input[])
    {
        SetTrainingMode(false);
        return Forward(input);
    }
    
    //+------------------------------------------------------------------+
    //| Calculate binary cross-entropy loss                              |
    //+------------------------------------------------------------------+
    double CalculateLoss(double prediction, double target)
    {
        // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        // Clip predictions to avoid log(0)
        double p = MathMax(1e-7, MathMin(1.0 - 1e-7, prediction));
        return -(target * MathLog(p) + (1.0 - target) * MathLog(1.0 - p));
    }
    
    //+------------------------------------------------------------------+
    //| Save weights to binary file                                      |
    //+------------------------------------------------------------------+
    bool SaveWeights(string filename)
    {
        int handle = FileOpen(filename, FILE_WRITE | FILE_BIN);
        if(handle == INVALID_HANDLE)
        {
            Print("Failed to open file for writing: ", filename);
            return false;
        }
        
        // Write layer sizes
        FileWriteInteger(handle, m_input_size);
        FileWriteInteger(handle, m_hidden1_size);
        FileWriteInteger(handle, m_hidden2_size);
        FileWriteInteger(handle, m_hidden3_size);
        FileWriteInteger(handle, m_output_size);
        
        // Write W1
        for(int i = 0; i < m_input_size; i++)
            for(int j = 0; j < m_hidden1_size; j++)
                FileWriteDouble(handle, m_W1[i][j]);
        // Write b1
        for(int j = 0; j < m_hidden1_size; j++)
            FileWriteDouble(handle, m_b1[j]);
        
        // Write W2
        for(int i = 0; i < m_hidden1_size; i++)
            for(int j = 0; j < m_hidden2_size; j++)
                FileWriteDouble(handle, m_W2[i][j]);
        // Write b2
        for(int j = 0; j < m_hidden2_size; j++)
            FileWriteDouble(handle, m_b2[j]);
        
        // Write W3
        for(int i = 0; i < m_hidden2_size; i++)
            for(int j = 0; j < m_hidden3_size; j++)
                FileWriteDouble(handle, m_W3[i][j]);
        // Write b3
        for(int j = 0; j < m_hidden3_size; j++)
            FileWriteDouble(handle, m_b3[j]);
        
        // Write W4
        for(int i = 0; i < m_hidden3_size; i++)
            for(int j = 0; j < m_output_size; j++)
                FileWriteDouble(handle, m_W4[i][j]);
        // Write b4
        for(int j = 0; j < m_output_size; j++)
            FileWriteDouble(handle, m_b4[j]);
        
        FileClose(handle);
        Print("Weights saved successfully to: ", filename);
        return true;
    }
    
    //+------------------------------------------------------------------+
    //| Load weights from binary file                                    |
    //+------------------------------------------------------------------+
    bool LoadWeights(string filename)
    {
        int handle = FileOpen(filename, FILE_READ | FILE_BIN);
        if(handle == INVALID_HANDLE)
        {
            Print("Failed to open file for reading: ", filename, " - will use initialized weights");
            return false;
        }
        
        // Read and verify layer sizes
        int input_size = FileReadInteger(handle);
        int hidden1_size = FileReadInteger(handle);
        int hidden2_size = FileReadInteger(handle);
        int hidden3_size = FileReadInteger(handle);
        int output_size = FileReadInteger(handle);
        
        if(input_size != m_input_size || hidden1_size != m_hidden1_size ||
           hidden2_size != m_hidden2_size || hidden3_size != m_hidden3_size ||
           output_size != m_output_size)
        {
            Print("Weight file has incompatible architecture, using initialized weights");
            FileClose(handle);
            return false;
        }
        
        // Read W1
        for(int i = 0; i < m_input_size; i++)
            for(int j = 0; j < m_hidden1_size; j++)
                m_W1[i][j] = FileReadDouble(handle);
        // Read b1
        for(int j = 0; j < m_hidden1_size; j++)
            m_b1[j] = FileReadDouble(handle);
        
        // Read W2
        for(int i = 0; i < m_hidden1_size; i++)
            for(int j = 0; j < m_hidden2_size; j++)
                m_W2[i][j] = FileReadDouble(handle);
        // Read b2
        for(int j = 0; j < m_hidden2_size; j++)
            m_b2[j] = FileReadDouble(handle);
        
        // Read W3
        for(int i = 0; i < m_hidden2_size; i++)
            for(int j = 0; j < m_hidden3_size; j++)
                m_W3[i][j] = FileReadDouble(handle);
        // Read b3
        for(int j = 0; j < m_hidden3_size; j++)
            m_b3[j] = FileReadDouble(handle);
        
        // Read W4
        for(int i = 0; i < m_hidden3_size; i++)
            for(int j = 0; j < m_output_size; j++)
                m_W4[i][j] = FileReadDouble(handle);
        // Read b4
        for(int j = 0; j < m_output_size; j++)
            m_b4[j] = FileReadDouble(handle);
        
        FileClose(handle);
        Print("Weights loaded successfully from: ", filename);
        return true;
    }
};
