# LLM Architecture Comparison Notes

This repository provides supplemental notes and detailed explanations for key architectural concepts covered in Sebastian Raschka's comprehensive article: **["The Big LLM Architecture Comparison"](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)**.

## 📖 Purpose

This repository serves as an educational companion to Sebastian Raschka's analysis, offering:
- **Deep-dive notes** on individual LLM architectures
- **Cross-cutting concept explanations** for techniques used across multiple models
- **Comparative analysis** highlighting architectural evolution from 2019-2025
- **Technical implementation details** and performance insights
- **Reference material** for students and practitioners studying modern LLM architectures

## 🏗️ Repository Structure

```
llm-architecture-comparison-notes/
├── README.md
├── Generic/                    # Cross-cutting architectural concepts
├── DeepSeek-V3-R1/            # Multi-Head Latent Attention (MLA) + MoE
├── OLMo-2/                    # Post-Norm + QK-Norm innovations
├── Gemma-3/                   # Sliding Window Attention optimizations
├── Mistral-Small-3.1/         # Performance-focused standard architecture
├── Llama-4/                   # MoE adoption in mainstream models
├── Qwen3/                     # Dense + MoE variants (0.6B to 235B)
├── SmolLM3/                   # NoPE (No Positional Embeddings)
├── Kimi-2/                    # Trillion-parameter architecture scaling
├── GPT-OSS/                   # Open-source GPT implementations
├── Grok-2.5/                  # Latest architectural developments
├── GLM-4.5/                   # Bidirectional + autoregressive design
└── Qwen3-Next/                # Next-generation improvements
```

## 🤖 LLM Models Covered

### **Architectures from the Original Article**

**DeepSeek V3/R1** (671B total, 37B active)
- Multi-Head Latent Attention (MLA) for memory efficiency
- Mixture-of-Experts (MoE) with shared expert design
- KV cache compression techniques

**OLMo 2** (Various sizes)
- Post-Norm layer placement innovations
- QK-Norm for training stability
- Transparent development approach

**Gemma 3** (Multiple sizes, focus on 27B)
- Sliding Window Attention (5:1 ratio with global attention)
- Hybrid normalization (Pre-Norm + Post-Norm)
- Memory-efficient local attention patterns

**Mistral Small 3.1** (24B)
- Standard architecture optimized for inference speed
- Custom tokenizer improvements
- Simplified attention without sliding windows

**Llama 4 Maverick** (400B total, 17B active)
- MoE adoption with alternating dense/sparse layers
- Grouped-Query Attention (GQA)
- Balanced expert utilization

**Qwen3** (Dense: 0.6B-32B, MoE: 30B-A3B, 235B-A22B)
- Comprehensive size range from mobile to datacenter
- Both dense and MoE architectural variants
- No shared expert in MoE design

**SmolLM3** (3B)
- NoPE (No Positional Embeddings) experiments
- Compact architecture with strong performance
- Length generalization improvements

**Kimi 2** (1T parameters)
- Largest open-weight model of current generation
- DeepSeek V3-based architecture at massive scale
- Advanced MoE scaling techniques

### **Additional Models**

**GPT-OSS** - Open-source GPT implementations and variants

**Grok 2.5** - Latest architectural developments and innovations

**GLM-4.5** - Bidirectional encoder + autoregressive decoder design

**Qwen3-Next** - Next-generation improvements and optimizations

## 🧠 Generic Concepts

The `Generic/` folder will contain detailed explanations of architectural concepts used across multiple models:

### **Attention Mechanisms**
- **Multi-Head Attention (MHA)** - Original transformer attention
- **Grouped-Query Attention (GQA)** - Memory-efficient key-value sharing
- **Multi-Head Latent Attention (MLA)** - DeepSeek's compression approach
- **Sliding Window Attention** - Local attention for efficiency

### **Architectural Components**
- **Mixture-of-Experts (MoE)** - Sparse expert routing and scaling
- **Normalization Techniques** - LayerNorm, RMSNorm, Pre/Post-Norm placement
- **Positional Embeddings** - Absolute, RoPE, NoPE comparisons
- **KV Caching** - Memory optimization strategies

### **Training & Efficiency**
- **Memory Optimization** - Techniques for reducing computational costs
- **Scaling Laws** - Parameter efficiency and performance relationships
- **Inference Optimization** - Speed vs. quality tradeoffs

## 📊 Architectural Evolution Timeline

**2019-2020**: GPT-2, basic transformer architecture
**2021-2022**: Scale-up era, efficiency improvements
**2023**: Attention mechanism refinements (GQA adoption)
**2024**: MoE mainstream adoption, memory optimization focus
**2025**: Hybrid approaches, efficiency at scale, architectural diversity

## 🎯 Key Trends Highlighted

1. **Efficiency Revolution**: Movement from dense to sparse (MoE) architectures
2. **Attention Evolution**: MHA → GQA → MLA progression
3. **Memory Optimization**: KV cache compression, sliding windows
4. **Normalization Innovations**: QK-Norm, placement variations
5. **Scale Diversity**: From 0.6B (Qwen3) to 1T parameters (Kimi 2)

## 📚 References

**Primary Source**: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) by Sebastian Raschka

**Related Resources**:
- Original model papers and technical reports
- Sebastian Raschka's [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) book
- Community implementations and discussions

## ⚖️ License

This repository is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sebastian Raschka** for the comprehensive original analysis
  - [Ahead of AI Magazine](https://magazine.sebastianraschka.com/) - Sebastian's Substack publication
  - [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) - Foundational book for understanding LLM implementation
- **Model development teams** at DeepSeek, Allen Institute (OLMo), Google (Gemma), Mistral, Meta (Llama), Alibaba (Qwen), Hugging Face (SmolLM), Moonshot AI (Kimi), and other organizations
- **Open-source community** for making these architectures accessible for study and research