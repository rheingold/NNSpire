1. Primitive Math Blocks Index

These are the finest-grained reusable “mathematical LEGO bricks”.

Primitive ID	Primitive	Main Mathematical Function
Prim.1	Dense / Linear	weighted vector transform
Prim.2	Embedding Lookup	token → vector projection
Prim.3	ReLU	nonlinear positive activation
Prim.4	GELU	smooth probabilistic activation
Prim.5	Sigmoid	bounded gating/probability
Prim.6	Softmax	probability normalization
Prim.7	Multi-Head Attention	contextual weighted mixing
Prim.8	Cross Attention	inter-modal/context attention
Prim.9	Residual Add	information preservation
Prim.10	LayerNorm	activation normalization
Prim.11	BatchNorm	batch variance stabilization
Prim.12	Convolution	spatial feature extraction
Prim.13	MaxPooling	strongest-feature selection
Prim.14	Positional Encoding	sequence order injection
Prim.15	Top-k / Top-p Sampling	probabilistic output filtering
Prim.16	Contrastive Loss	latent distance optimization
Prim.17	PPO/RL Optimization	reward-based policy optimization
Prim.18	Routing/Gating	expert/path selection
Prim.19	Diffusion Denoising Step	iterative noise removal
Prim.20	Recurrence/Loop	iterative refinement


2. Semantic Behaviors Index
Behavior ID	Semantic Behavior	What Emerges Functionally
Beh.1	Token Recognition	symbols become vectors
Beh.2	Positional Awareness	order/sequence preserved
Beh.3	Syntax Learning	local grammatical structure
Beh.4	Semantic Association	concept relationships
Beh.5	Long-range Dependency Tracking	distant context linkage
Beh.6	Concept Abstraction	higher-level latent structure
Beh.7	Hypothesis Generation	candidate continuation creation
Beh.8	Candidate Competition	output scoring
Beh.9	Probabilistic Selection	choosing outputs
Beh.10	Memory Preservation	context retention
Beh.11	Training Stabilization	gradient/activation control
Beh.12	Query Expansion	semantic reformulation
Beh.13	Ranking/Relevance Scoring	best-result ordering
Beh.14	Image Feature Detection	visual primitive extraction
Beh.15	Visual Abstraction	object/scene representation
Beh.16	Cross-modal Alignment	image↔text mapping
Beh.17	Latent Denoising	iterative image refinement
Beh.18	Reward Optimization	preference shaping
Beh.19	Planning/Decomposition	multi-step task structure
Beh.20	Tool/API Invocation	structured action generation
Beh.21	Entity Extraction	structured information tagging
3. Hierarchical Architecture Table
NN.1 — TE (Text Embedding Networks)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.1 TE-1 (Basic Text Embedding)	Beh.1 Token Recognition	Basic Embedding Encoder	EmbeddingLookup (Prim.2) → Dense/Linear (Prim.1)	1×
NN.1 TE-2 (Semantic Embedding)	Beh.4 Semantic Association	Siamese Encoder Stack	Dense (Prim.1) → GELU (Prim.4) → Dense (Prim.1)	2–12×
NN.1 TE-3 (Embedding Stabilization)	Beh.11 Training Stabilization	Normalization Projection	Dense (Prim.1) → LayerNorm (Prim.10)	1–4×
NN.1 TE-4 (Contrastive Embedding)	Beh.4 Semantic Association	Dual-Tower Contrastive Encoder	Dense (Prim.1) → GELU (Prim.4) → ContrastiveLoss (Prim.16)	4–48×
NN.2 — IA (Instruction/Inference Assistant LLM)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.2 IA-1 (Token Embedding)	Beh.1 Token Recognition	Token Embedding Layer	EmbeddingLookup (Prim.2) → Dense (Prim.1)	1×
NN.2 IA-2 (Position Encoding)	Beh.2 Positional Awareness	Positional Injection	PositionalEncoding (Prim.14) → ResidualAdd (Prim.9)	1×
NN.2 IA-3 (Syntax Attention)	Beh.3 Syntax Learning	Early Transformer Attention Block	LayerNorm (Prim.10) → MultiHeadAttention (Prim.7) → ResidualAdd (Prim.9)	2–12×
NN.2 IA-4 (Semantic Transformer)	Beh.4 Semantic Association	Transformer Semantic Block	MultiHeadAttention (Prim.7) → Dense (Prim.1) → GELU (Prim.4) → Dense (Prim.1)	12–96×
NN.2 IA-5 (Long-Context Tracking)	Beh.5 Long-range Dependency Tracking	Deep Attention Stack	MultiHeadAttention (Prim.7) → ResidualAdd (Prim.9) → LayerNorm (Prim.10)	24–128×
NN.2 IA-6 (Concept Abstraction)	Beh.6 Concept Abstraction	Feedforward Latent Synthesizer	Dense (Prim.1) → GELU (Prim.4) → Dense (Prim.1)	24–128×
NN.2 IA-7 (Hypothesis Generator)	Beh.7 Hypothesis Generation	Autoregressive Decoder	MultiHeadAttention (Prim.7) → Dense (Prim.1) → Softmax (Prim.6)	24–128×
NN.2 IA-8 (Candidate Competition)	Beh.8 Candidate Competition	Logits Projection Head	Dense (Prim.1) → Softmax (Prim.6)	1×
NN.2 IA-9 (Probabilistic Selector)	Beh.9 Probabilistic Selection	Decoding Filter	Softmax (Prim.6) → Top-k/Top-p Sampling (Prim.15)	runtime only
NN.2 IA-10 (Context Preservation)	Beh.10 Memory Preservation	Residual Context Pathways	ResidualAdd (Prim.9)	throughout network
NN.2 IA-11 (Training Stabilizer)	Beh.11 Training Stabilization	Normalization Layers	LayerNorm (Prim.10)	throughout network
NN.2 IA-12 (Reasoning/Planning Loop)	Beh.19 Planning/Decomposition	Recurrent Inference Loop	RecurrenceLoop (Prim.20) → MultiHeadAttention (Prim.7) → Dense (Prim.1)	2–32 inference passes
NN.2 IA-13 (Tool Invocation Decoder)	Beh.20 Tool/API Invocation	Structured Output Decoder	MultiHeadAttention (Prim.7) → Dense (Prim.1) → Softmax (Prim.6)	2–24×
NN.3 — QR (Query Rewrite / Ranking Networks)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.3 QR-1 (Semantic Query Rewriter)	Beh.12 Query Expansion	Semantic Reformulation Encoder	MultiHeadAttention (Prim.7) → Dense (Prim.1) → GELU (Prim.4)	6–48×
NN.3 QR-2 (Cross-Attention Reranker)	Beh.13 Ranking/Relevance Scoring	Cross-Attention Comparator	CrossAttention (Prim.8) → Dense (Prim.1) → Sigmoid (Prim.5)	4–32×
NN.3 QR-3 (Pairwise Ranking Head)	Beh.13 Ranking/Relevance Scoring	Pairwise Ranking Projection	Dense (Prim.1) → GELU (Prim.4) → Softmax (Prim.6)	1–8×
NN.4 — MP (Multimodal Perception Networks)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.4 MP-1 (CNN Feature Extractor)	Beh.14 Image Feature Detection	CNN Feature Stack	Convolution (Prim.12) → ReLU (Prim.3) → MaxPooling (Prim.13)	8–200×
NN.4 MP-2 (Hierarchical Visual Encoder)	Beh.15 Visual Abstraction	Deep CNN Hierarchy	Convolution (Prim.12) → BatchNorm (Prim.11) → ReLU (Prim.3)	20–400×
NN.4 MP-3 (CLIP-like Crossmodal Encoder)	Beh.16 Cross-modal Alignment	Dual Encoder Alignment Stack	Dense (Prim.1) → GELU (Prim.4) → ContrastiveLoss (Prim.16)	12–96×
NN.4 MP-4 (Vision Transformer)	Beh.15 Visual Abstraction	ViT Attention Encoder	PositionalEncoding (Prim.14) → MultiHeadAttention (Prim.7) → Dense (Prim.1)	12–128×
NN.5 — GC (Generative Creation / Diffusion)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.5 GC-1 (Diffusion Denoiser)	Beh.17 Latent Denoising	Diffusion U-Net	Convolution (Prim.12) → MultiHeadAttention (Prim.7) → DiffusionStep (Prim.19)	50–150 denoising iterations
NN.5 GC-2 (Prompt Conditioning)	Beh.16 Cross-modal Alignment	Text-Image Fusion Stack	CrossAttention (Prim.8) → Dense (Prim.1)	12–64×
NN.5 GC-3 (Latent Visual Abstraction)	Beh.6 Concept Abstraction	Latent Diffusion Encoder	Convolution (Prim.12) → GELU (Prim.4) → Dense (Prim.1)	20–200×
NN.6 — DS (Decision / RL Systems)
NN Type	Semantic Behavior	Architectural Structure	Primitive/Math Sandwich	Typical Repetition
NN.6 DS-1 (Policy Network)	Beh.18 Reward Optimization	RL Policy Head	Dense (Prim.1) → GELU (Prim.4) → Softmax (Prim.6)	4–48×
NN.6 DS-2 (PPO Optimizer Loop)	Beh.18 Reward Optimization	Reinforcement Optimization Loop	PPO/RL Optimization (Prim.17)	millions of episodes
NN.6 DS-3 (Action Router)	Beh.20 Tool/API Invocation	Routing/Gating Policy	Routing/Gating (Prim.18) → Softmax (Prim.6)	1–8×
NN.7 — DS (Decision Systems / RL Agents)
NN ID	Variant	Behavior ID	Semantic Behavior	Architectural Structure	Primitive Sequence	Typical Repetition
NN.7	DS-1	Beh.18	Reward Optimization	policy network	Prim.1 → Prim.4 → Prim.6	4–48×
NN.7	DS-2	Beh.18	RL Policy Refinement	PPO optimization loop	Prim.17	millions of episodes
NN.7	DS-3	Beh.20	Action Selection	routing policy head	Prim.18 → Prim.6	1–8×
NN.8 — PD (Planning / Decomposition)
NN ID	Variant	Behavior ID	Semantic Behavior	Architectural Structure	Primitive Sequence	Typical Repetition
NN.8	PD-1	Beh.19	Task Decomposition	scratchpad transformer	Prim.7 → Prim.1 → Prim.9	12–96×
NN.8	PD-2	Beh.19	Sequential Planning	recurrent planner loop	Prim.20 → Prim.7	2–64 planning cycles