# ASSINGMENT-1
19TD608-Prompt Engineering

ASSIGNMENT 1

NAME: DINESH KUMAR

REG.NO: 212222060053

SLOT:4X2-55



List The different types of generative models and explain their key characteristics.
1. Types of Generative Models and Their Key Characteristics
Generative models are AI systems that learn to generate new data samples that resemble a given dataset (e.g., text, images, music, etc.). The main types include:
Type	Key Characteristics	Example Applications
1. Generative Adversarial Networks (GANs)	Comprise two neural networks — a generator that creates data and a discriminator that evaluates its authenticity. They train in a competitive process.	Image synthesis, face generation, super-resolution.
2. Variational Autoencoders (VAEs)	Encode data into a latent space and then decode it back to generate new samples. They learn probabilistic representations.	Image denoising, anomaly detection, synthetic data generation.
3. Autoregressive Models	Generate data one step (token or pixel) at a time, predicting the next element based on previous ones.	Text generation (GPT models), music composition.
4. Diffusion Models	Learn to reverse a gradual noise process applied to data, effectively “denoising” to create new samples.	Image synthesis (e.g., DALL·E 3, Stable Diffusion).
5. Flow-based Models	Learn invertible transformations between data and latent space, allowing exact likelihood estimation.	High-quality image generation, density estimation.
6. Energy-Based Models (EBMs)	Learn an energy function that assigns low energy (high probability) to real data and high energy to fake data.	Structured prediction, generative modeling of physical systems.
2. Explain what a large language model is and how it functions at a high level.
2. What is a Large Language Model (LLM)?
A Large Language Model (LLM) is a type of autoregressive transformer-based neural network trained on vast text datasets to understand and generate human-like language.
High-Level Functioning:
1.	Pre-training:
o	The model is trained on massive text corpora (e.g., books, websites) using self-supervised learning.
o	Objective: Predict the next word or token given previous context (called language modeling).
o	Through this, it learns grammar, facts, reasoning, and world knowledge.
2.	Architecture:
o	Built using the Transformer architecture.
o	Uses self-attention mechanisms to understand contextual relationships between words in a sequence.
3.	Fine-tuning (optional):
o	After pre-training, the model is fine-tuned for specific tasks such as summarization, translation, or question-answering.
4.	Inference:
o	When prompted with input text, the model generates coherent, contextually relevant text outputs by predicting tokens sequentially.
3. Select a real-world problem that could benefit from the application of a large language model. Outline the steps you would take to pre-train and fine-tune a model to address this problem, highlighting the specific benefits that an LLM would offer over traditional programming methods. Include potential limitations or ethical considerations that might arise.

3. Real-World Problem Example
Problem: Medical Chat Assistant for Preliminary Health Guidance
A healthcare organization wants to deploy an AI-powered chatbot that provides reliable, conversational responses to patient queries (symptoms, medication info, appointment guidance).

Step 1: Data Collection for Pre-training
•	Gather large-scale, publicly available text data (medical journals, health forums, Wikipedia, etc.).
•	Clean and tokenize text for training.
•	Use general-purpose data (e.g., Common Crawl, books) to help the model understand language structure.
Step 2: Pre-training
•	Train a transformer-based model (like GPT architecture) using unsupervised learning.
•	Objective: Predict the next token (word/phrase) given prior context.
•	Outcome: The model learns general language and reasoning patterns.
Step 3: Fine-tuning
•	Fine-tune the model using domain-specific datasets, such as:
o	Medical QA datasets (e.g., PubMedQA)
o	Doctor–patient conversation transcripts
o	Clinical guidelines (e.g., WHO, CDC)
•	Apply reinforcement learning from human feedback (RLHF) to improve factual accuracy and empathy in responses.
Step 4: Evaluation
•	Test on held-out validation sets for:
o	Accuracy of medical information
o	Clarity and readability
o	Bias and misinformation checks
Step 5: Deployment
•	Deploy via a secure cloud service or API with access restrictions.
•	Add user disclaimers: “This is not a substitute for professional medical advice.”

Benefits of LLMs Over Traditional Programming
Aspect	LLMs	Traditional Programming
Flexibility	Understands natural language queries; adapts to diverse inputs.	Requires strict, rule-based logic.
Knowledge Integration	Leverages vast textual knowledge.	Must be explicitly programmed with knowledge.
Scalability	Handles multilingual or unstructured data easily.	Requires custom modules for each new feature.
Context Awareness	Remembers and interprets conversation flow.	No built-in contextual understanding.

